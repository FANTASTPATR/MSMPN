'''
此版本增加了并行
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from scipy import sparse as sp
import dgl

from STSMPN import STSMPN

import pickle


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self._args = args
        self._chps = self._generate_hierarchical_checkpoints(dilation_sets=self._args.dilations,
                                                             his_len=self._args.his_len)
        self._adj_mx = self._args.adj_mx
        self._pearson_mx = self._generate_sparse_pearson_mx(file_path=self._args.pearson_path)
        self._adj_supports = self._generate_categoty_dgl(adj_mx=self._adj_mx, snapshot=2, device=self._args.device)
        self._pearson_supports = self._generate_categoty_dgl(adj_mx=self._pearson_mx, snapshot=2,
                                                             device=self._args.device)
        self._stsmps_adj = self._generate_stsmpns()
        self._stsmps_pearson = self._generate_stsmpns()

        self._in_linear = nn.Linear(self._args.in_dim, self._args.hid_dim)
        self._residual_convs = self._generate_residual_conv()
        self._gate_convs = self._generate_residual_conv()
        self._gate_linear_1 = nn.Linear(self._args.hid_dim, self._args.hid_dim)
        self._gate_linear_2 = nn.Linear(self._args.hid_dim, self._args.hid_dim)
        self._layer_norms = self._generate_layer_norms()
        self._decoder_conv = nn.Conv2d(in_channels=1, out_channels=self._args.pred_len, kernel_size=(1, 1),
                                       stride=(1, 1), padding=0)
        self._decoder_linear = nn.Linear(self._args.hid_dim, 1)

    def forward(self, inputs):
        x = self._in_linear(inputs)  # [b,seq_in,num_node,d]
        last_res = x
        for idx, layer_group in enumerate(self._chps):
            residual = self._residual_convs[idx](last_res)  # [b,new_seq,num_node,d]
            gate_features = self._gate_convs[idx](last_res)  # [b,new_seq,num_node,d]
            gate_features = F.relu(self._gate_linear_1(gate_features))  # [b,new_seq,num_node,d]
            gate = F.sigmoid(gate_features)  # [b,new_seq,num_node,d]
            adj_ret = self._stsmps_adj[idx](last_res)  # [b,new_seq,num_node,d]
            pearson_ret = self._stsmps_pearson[idx](last_res)  # [b,new_seq,num_node,d]
            gate_res = gate_features * adj_ret + (1 - gate) * pearson_ret  # [b,new_seq,num_node,d]
            conv_res = gate_res + residual
            last_res = self._layer_norms[idx](conv_res)
        out = F.relu(self._decoder_conv(last_res))
        out = self._decoder_linear(out)
        return out

    def _generate_stsmpns(self):
        stsmps = nn.ModuleList()
        for idx, chp in enumerate(self._chps):
            stsmps.append(STSMPN(args=self._args, ckps=chp, supports=self._adj_supports))
        return stsmps

    def _generate_residual_conv(self):
        residual_convs = nn.ModuleList()
        for idx, chp in enumerate(self._chps):
            if idx == 0:
                residual_convs.append(
                    nn.Conv2d(in_channels=self._args.his_len, out_channels=len(chp), kernel_size=(1, 1), stride=(1, 1),
                              padding=0))
                residual_convs.append(
                    nn.Conv2d(in_channels=len(chp), out_channels=len(self._chps[idx + 1]), kernel_size=(1, 1),
                              stride=(1, 1), padding=0))
            elif idx == len(self._chps) - 1:
                break
            else:
                residual_convs.append(
                    nn.Conv2d(in_channels=len(chp), out_channels=len(self._chps[idx + 1]), kernel_size=(1, 1),
                              stride=(1, 1),
                              padding=0))
        return residual_convs

    def _generate_layer_norms(self):
        layer_norms = nn.ModuleList()
        for idx, chp in enumerate(self._chps):
            layer_norms.append(nn.LayerNorm([self._args.batch_size, len(chp), self._args.num_node, self._args.hid_dim]))
        return layer_norms

    @staticmethod
    def _combine_adj_forward(adj_mx, snapsht_len):
        '''
        链接不同时间戳上对应的点
        :param adj_mx: 邻接矩阵
        :param snapsht_len: 需要链接的时间戳
        :return: 链接后的时间戳
        '''
        rst = torch.zeros(adj_mx.shape[0] * snapsht_len, adj_mx.shape[0] * snapsht_len)
        offset = 0
        for v_idx in range(snapsht_len):
            if v_idx == snapsht_len - 1:
                rst[-adj_mx.shape[0]:, -adj_mx.shape[1]:] = adj_mx
                offset += adj_mx.shape[0]
            else:
                hstack_unit = torch.cat((adj_mx, torch.eye(adj_mx.shape[0])), dim=-1)
                rst[offset:offset + hstack_unit.shape[0], offset:offset + hstack_unit.shape[1]] = hstack_unit
                offset += adj_mx.shape[0]
        return rst

    @staticmethod
    def _combine_adj_backward(adj_mx, snapsht_len):
        '''
        将邻接矩阵放到一起，之间不相连
        :param adj_mx:
        :param snapsht_len:
        :return:
        '''
        rst = torch.zeros(adj_mx.shape[0] * snapsht_len, adj_mx.shape[0] * snapsht_len)
        offset = 0
        for v_idx in range(snapsht_len):
            rst[offset:offset + adj_mx.shape[0], offset:offset + adj_mx.shape[1]] = adj_mx
            offset += adj_mx.shape[0]
        return rst

    @staticmethod
    def _generate_dgl_graphs(mx, device):
        '''
        根据邻接矩阵生成dgl图
        :param mx:邻接矩阵
        :param device:放置特征的设备
        :param batch_size:
        :return:
        '''
        adj_mx_unwei = np.where(mx > 0, 1, 0)
        adj_mx_unwei = sp.csr_matrix(adj_mx_unwei)
        g = dgl.DGLGraph(adj_mx_unwei)
        g.edata['w'] = torch.randn(g.number_of_edges(), 1).to(device)
        for u_idx in range(mx.shape[0]):
            for v_idx in range(mx.shape[1]):
                if mx[u_idx][v_idx] != 0:
                    g.edata['w'][g.edge_id(u_idx, v_idx)] = mx[u_idx][v_idx]
        return g

    @staticmethod
    def _generate_categoty_dgl(adj_mx, snapshot, device):
        '''
        生成一类邻接矩阵的正反两项dgl图
        :param adj_mx: 邻接矩阵
        :param napshot: 连接片个数
        :return: 【反链接图,正链接图】
        '''
        backward_adj_mx = Model._combine_adj_backward(adj_mx=adj_mx, snapsht_len=snapshot)
        forward_adj_mx = Model._combine_adj_forward(adj_mx=adj_mx, snapsht_len=snapshot)
        return [Model._generate_dgl_graphs(mx=backward_adj_mx, device=device),
                Model._generate_dgl_graphs(mx=forward_adj_mx, device=device)]

    @staticmethod
    def _generate_checkpoints_for_sequence(input_seq, dilation):
        '''
        为单个序列按照dilation生成卷积索引
        :param input_seq: [0,1,2,3,4,5,6,7,8]
        :param dilation: 1
        :return:combination_rst:[(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,8)],
                idx_rst:[0,1,2,3,4,5,6,7]
        '''
        combination_rst = []
        idx_rst = []
        left_idx = input_seq[0]
        right_idx = left_idx + dilation
        idx = 0
        while right_idx <= input_seq[-1]:
            combination_rst.append((left_idx, right_idx))
            idx_rst.append(idx)
            left_idx += 1
            right_idx += 1
            idx += 1
        return combination_rst, idx_rst

    @staticmethod
    def _generate_hierarchical_checkpoints(dilation_sets, his_len):
        '''
        生成层级化的检查点
        :param dilation_sets: [1,2,1,2,1,2,1,2]
        :param his_len: 12
        :return: [[(0,1),(1,2),...,(7,8)],
                [(0,2),...,(5,7)]]
        '''
        rst = []
        input_seq = [i for i in range(his_len)]
        for idx, dilation in enumerate(dilation_sets):
            combination_rst, idx_rst = Model._generate_checkpoints_for_sequence(input_seq=input_seq, dilation=dilation)
            rst.append(combination_rst)
            input_seq = idx_rst
        return rst

    @staticmethod
    def _generate_sparse_pearson_mx(file_path, ratio=0.95):
        with open(file_path, "rb") as f:
            pearson_adj = pickle.load(f)
        all_corr = []
        for row_idx in range(pearson_adj.shape[0]):
            for col_idx in range(pearson_adj.shape[1]):
                all_corr.append(pearson_adj[col_idx][row_idx])
        all_corr = sorted(all_corr)
        ths = all_corr[int(len(all_corr) * ratio)]
        sparse_adj = np.where(pearson_adj > ths, pearson_adj, 0)
        for idx, item in enumerate(sparse_adj):
            if np.sum(item) == 1:
                max_idx1 = np.argsort(pearson_adj[idx])[-3]
                max_idx2 = np.argsort(pearson_adj[idx])[-2]
                sparse_adj[idx][max_idx1] = pearson_adj[idx][max_idx1]
                sparse_adj[idx][max_idx2] = pearson_adj[idx][max_idx2]
        return torch.Tensor(sparse_adj)
