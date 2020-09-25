'''
此版本增加了pearson附属矩阵
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
        self.args = args
        self._chp = self._generate_checkpoints(self.args.snpsts_len, self.args.his_len)
        self._pre_graphs = self._generate_graphs()
        self.in_linear = nn.Linear(self.args.in_dim, self.args.hid_dim)
        self.residual_conv = nn.Conv2d(in_channels=self.args.snpsts_len,
                                       out_channels=1, kernel_size=(1, 1), stride=(1, 1), padding=0)
        self.gcns_adj = self._generate_gcns()
        self.gcns_pearson = self._generate_gcns()
        self.gate_conv = nn.Conv2d(in_channels=self.args.snpsts_len,
                                   out_channels=1, kernel_size=(1, 1), stride=(1, 1), padding=0)
        self.gate_1 = nn.Linear(self.args.hid_dim, self.args.hid_dim)
        self.gate_2 = nn.Linear(self.args.hid_dim, self.args.hid_dim * self.args.channels)
        self.reduce_dim = nn.Linear(self.args.channels * self.args.hid_dim, self.args.hid_dim)
        self.out_conv = nn.Conv2d(in_channels=1, out_channels=self.args.pred_len, kernel_size=(1, 1), stride=(1, 1),
                                  padding=0)
        self.out_linear = nn.Linear(self.args.hid_dim, 1)
        self.layer_norm = nn.LayerNorm([self.args.batch_size, self.args.num_node, self.args.hid_dim],
                                       elementwise_affine=False)

    def forward(self, inputs):
        x = self.in_linear(inputs)  # [batch_size,seq_in,num_node,features]
        last_res = None
        left = 0
        for idx, right in enumerate(self._chp):
            feature_slice = x[:, left:right, :, :].contiguous()  # [b,snapshot,n,d]
            if idx != 0:
                feature_slice = torch.cat([last_res, feature_slice], dim=1)  # [b,snapshot,n,d]
            gate_feature = self.gate_conv(feature_slice).view(self.args.batch_size * self.args.num_node,
                                                              self.args.hid_dim).contiguous()  # [b,1,n,d]->[b*n,d]
            residual = feature_slice
            residual = self.residual_conv(residual).squeeze()
            feature_slice = feature_slice.view(x.shape[0] * self.args.snpsts_len * x.shape[2],
                                               x.shape[-1]).contiguous()  # [b*snapshot*n,d]
            conv_res_adj = []
            for conv_idx, conv_channel in enumerate(self.gcns_adj):
                channel_res = conv_channel(supports=self._pre_graphs['adj_graphs'],
                                           features=feature_slice)  # [b*1*n,d]
                conv_res_adj.append(channel_res)
            conv_res_adj = torch.cat(conv_res_adj, dim=-1)
            conv_res_pearson = []
            for conv_idx, conv_channel in enumerate(self.gcns_pearson):
                channel_res = conv_channel(supports=self._pre_graphs['pearson_graphs'],
                                           features=feature_slice)  # [b*1*n,d]
                conv_res_pearson.append(channel_res)
            conv_res_pearson = torch.cat(conv_res_pearson, dim=-1)
            gate = F.relu(self.gate_1(gate_feature))
            gate = F.sigmoid(self.gate_2(gate))
            gate_ret = gate * conv_res_adj + (1 - gate) * conv_res_pearson
            gate_ret = self.reduce_dim(gate_ret)  # [batch_size*num_node,hid_dim]
            gate_ret = gate_ret.view(self.args.batch_size, self.args.num_node, self.args.hid_dim)
            conv_res = gate_ret + residual  # [b,n,d]
            last_res = self.layer_norm(conv_res).unsqueeze(1)  # [b,1,n,d]
            left = right
        out = self.out_conv(last_res)  # [b,pred,n,d]
        out = F.relu(out)
        out = self.out_linear(out)
        return out

    def _generate_graphs(self):
        rst = {}
        adj_mx = self.args.adj_mx
        pearson_mx = self._generate_sparse_pearson_mx(self.args.pearson_path)
        adj_combine_forward = self._combine_adj_forward(adj_mx, self.args.snpsts_len)
        adj_combine_backward = self._combine_adj_backward(adj_mx, self.args.snpsts_len)
        rst['adj_graphs'] = {'backward': self._generate_dgl_graphs(mx=adj_combine_backward, device=self.args.device,
                                                                   batch_size=self.args.batch_size),
                             'forward': self._generate_dgl_graphs(mx=adj_combine_forward, device=self.args.device,
                                                                  batch_size=self.args.batch_size)}
        pearson_combine_forward = self._combine_adj_forward(pearson_mx, self.args.snpsts_len)
        pearson_combine_backward = self._combine_adj_backward(pearson_mx, self.args.snpsts_len)
        rst['pearson_graphs'] = {
            'backward': self._generate_dgl_graphs(mx=pearson_combine_backward, device=self.args.device,
                                                  batch_size=self.args.batch_size),
            'forward': self._generate_dgl_graphs(mx=pearson_combine_forward, device=self.args.device,
                                                 batch_size=self.args.batch_size)}
        return rst

    def _generate_gcns(self):
        gcns = nn.ModuleList()
        for _ in range(self.args.channels):
            gcns.append(GCN(in_feats=self.args.hid_dim, hid_feats=self.args.hid_dim, layers=self.args.layers,
                            dropout=self.args.dropout, num_node=self.args.num_node, support_len=2,
                            snpsht_len=self.args.snpsts_len, batch_size=self.args.batch_size))
        return gcns

    @staticmethod
    def _generate_checkpoints(snapsht_len, his_len):
        mark = snapsht_len
        checkpoints = []
        while mark < his_len:
            checkpoints.append(mark)
            mark += (snapsht_len - 1)
        if checkpoints[-1] != his_len:
            checkpoints.append(his_len)
        return checkpoints

    @staticmethod
    def _combine_adj_forward(adj_mx, snapsht_len):
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
        with open('./adj_mx.pkl', "wb") as f:
            pickle.dump(rst, f)
        return rst

    @staticmethod
    def _combine_adj_backward(adj_mx, snapsht_len):
        rst = torch.zeros(adj_mx.shape[0] * snapsht_len, adj_mx.shape[0] * snapsht_len)
        offset = 0
        for v_idx in range(snapsht_len):
            rst[offset:offset + adj_mx.shape[0], offset:offset + adj_mx.shape[1]] = adj_mx
            offset += adj_mx.shape[0]
        with open('./adj_mx.pkl', "wb") as f:
            pickle.dump(rst, f)
        return rst

    @staticmethod
    def _generate_dgl_graphs(mx, device, batch_size):
        adj_mx_unwei = np.where(mx > 0, 1, 0)
        adj_mx_unwei = sp.csr_matrix(adj_mx_unwei)
        g = dgl.DGLGraph(adj_mx_unwei)
        g.edata['w'] = torch.randn(g.number_of_edges(), 1).to(device)
        for u_idx in range(mx.shape[0]):
            for v_idx in range(mx.shape[1]):
                if mx[u_idx][v_idx] != 0:
                    g.edata['w'][g.edge_id(u_idx, v_idx)] = mx[u_idx][v_idx]
        batch_g = dgl.batch([g for _ in range(batch_size)])
        return batch_g

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
