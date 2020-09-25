'''
modelv6备胎（添加并行）
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from scipy import sparse as sp
import dgl

from GCNv3 import GCN

import pickle


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self._args = args
        self._chps = self._generate_hierarchical_checkpoints(dilation_sets=self._args.dilations,
                                                             his_len=self._args.his_len)
        self._pre_graphs = self._generate_graphs()
        self._in_linear = nn.Linear(self._args.in_dim, self._args.hid_dim)



    def forward(self, inputs):
        x = self._in_linear(inputs)  # [b,seq_in,n,d]
        last_res = None
        for idx, layer in enumerate(self._chps):
            if idx == 0:
                features_conv = []
                for field in layer:  # (节点1，节点2)
                    features = [x[:, field[0], :, :], x[:, field[1], :, :]]
                    features = torch.stack(features, dim=1)  # [b,2,n,d]
                    features_conv.append(features)
                features_conv = torch.stack(features_conv).transpose(0, 1)  # [b,parral_num,2,n,d]
                conv_res_adj = []



        out = self.out_linear1(last_res.transpose(-1, 1)).transpose(-1, 1)  # [b,pred,n,d]
        out = F.relu(out)
        out = self.out_linear2(out)
        return out

    def _generate_graphs(self):
        '''
        生成虚拟的图
        :return: batch*num*g
        '''
        rst = []
        adj_mx = self._args.adj_mx
        pearson_mx = self._generate_sparse_pearson_mx(self._args.pearson_path)
        adj_combine_forward = self._combine_adj_forward(adj_mx, self._args.snpsts_len)
        adj_combine_backward = self._combine_adj_backward(adj_mx, self._args.snpsts_len)
        adj_comb_f_g = self._generate_dgl_graphs(mx=adj_combine_forward, device=self._args.device)
        adj_comb_b_g = self._generate_dgl_graphs(mx=adj_combine_backward, device=self._args.device)
        pearson_combine_forward = self._combine_adj_forward(pearson_mx, self._args.snpsts_len)
        pearson_combine_backward = self._combine_adj_backward(pearson_mx, self._args.snpsts_len)
        pearson_comb_f_g = self._generate_dgl_graphs(mx=pearson_combine_forward, device=self._args.device)
        pearson_comb_b_g = self._generate_dgl_graphs(mx=pearson_combine_backward, device=self._args.device)
        for idx, chps in enumerate(self._chps):
            parral_num = len(chps)
            graphs = {}
            parral_adj_f_g = self._generate_batch_dgl_graphs(adj_comb_f_g, parral_num)
            graphs['adj_forward'] = self._generate_batch_dgl_graphs(parral_adj_f_g, self._args.batch_size)
            parral_adj_b_g = self._generate_batch_dgl_graphs(adj_comb_b_g, parral_num)
            graphs['adj_back'] = self._generate_batch_dgl_graphs(parral_adj_b_g, self._args.batch_size)
            parral_pearson_f_g = self._generate_batch_dgl_graphs(pearson_comb_f_g, parral_num)
            graphs['pearson_forward'] = self._generate_batch_dgl_graphs(parral_pearson_f_g, self._args.batch_size)
            parral_pearson_b_g = self._generate_batch_dgl_graphs(pearson_comb_b_g, parral_num)
            graphs['pearson_back'] = self._generate_batch_dgl_graphs(parral_pearson_b_g, self._args.batch_size)
            rst.append(graphs)

        return rst

    def _generate_gcns(self):
        gcns = nn.ModuleList()
        for _ in range(self._args.channels):
            gcns.append(GCN(in_feats=self._args.hid_dim, hid_feats=self._args.hid_dim, layers=self._args.layers,
                            dropout=self._args.dropout, num_node=self._args.num_node, support_len=2,
                            snpsht_len=self._args.snpsts_len, batch_size=self._args.batch_size))
        return gcns

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
    def _generate_batch_dgl_graphs(g, batch_num):
        return dgl.batch([g for _ in range(batch_num)])

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
