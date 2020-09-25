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
        self.residual_conv_adj = nn.Conv1d(in_channels=self.args.snpsts_len * self.args.num_node,
                                           out_channels=self.args.num_node, kernel_size=1)
        self.residual_conv_pearson = nn.Conv1d(in_channels=self.args.snpsts_len * self.args.num_node,
                                               out_channels=self.args.num_node, kernel_size=1)
        self.gcns_adj = self._generate_gcns()
        self.gcns_pearson = self._generate_gcns()
        self.layer_norm_adj = nn.LayerNorm([self.args.batch_size, self.args.num_node, self.args.hid_dim],
                                           elementwise_affine=False)
        self.layer_norm_pearson = nn.LayerNorm([self.args.batch_size, self.args.num_node, self.args.hid_dim],
                                               elementwise_affine=False)

    def forward(self, inputs):
        x = self.in_linear(inputs)
        last_res_adj = None
        last_res_pearson = None
        left = 0
        for idx, right in enumerate(self._chp):
            pearson_feature_slice = adj_feature_slice = x[:, left:right, :, :].contiguous()  # [b,snapshot,n,d]
            if idx != 0:
                adj_feature_slice = torch.cat([last_res_adj, adj_feature_slice], dim=1)  # [b,snapshot,n,d]
                pearson_feature_slice = torch.cat([last_res_pearson, pearson_feature_slice],
                                                  dim=1)  # [b,snapshot,n,d]
            last_res_adj = self._round_conv(field="adj", features=adj_feature_slice)
            last_res_pearson = self._round_conv(field="pearson", features=pearson_feature_slice)
            left = right
        print(last_res_adj.shape)
        print(last_res_pearson.shape)

    def _round_conv(self, field, features):
        features = features.view(self.args.batch_size, self.args.snpsts_len * self.args.num_node, self.args.hid_dim)
        residual = features
        if field == "adj":
            residual = self.residual_conv_adj(residual)
        elif field == "pearson":
            residual = self.residual_conv_pearson(residual)
        features = features.view(self.args.batch_size * self.args.snpsts_len * self.args.num_node, self.args.hid_dim)
        conv_res = []
        if field == "adj":
            for conv_idx, conv_channel in enumerate(self.gcns_adj):
                channel_res = conv_channel(supports=self._pre_graphs['adj_graphs'],
                                           features=features)  # [b*1*n,d]
                conv_res.append(channel_res)
        elif field == "pearson":
            for conv_idx, conv_channel in enumerate(self.gcns_pearson):
                channel_res = conv_channel(supports=self._pre_graphs['pearson_graphs'],
                                           features=features)  # [b*1*n,d]
                features.append(channel_res)
        conv_res = torch.cat(conv_res, dim=-1)
        conv_res = conv_res + residual
        if field == "adj":
            return self.layer_norm_adj(conv_res).unsqueeze(1)
        elif field == "pearson":
            return self.layer_norm_pearson(conv_res).unsqueeze(1)

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
        return rst

    @staticmethod
    def _combine_adj_backward(adj_mx, snapsht_len):
        rst = torch.zeros(adj_mx.shape[0] * snapsht_len, adj_mx.shape[0] * snapsht_len)
        offset = 0
        for v_idx in range(snapsht_len):
            rst[offset:offset + adj_mx.shape[0], offset:offset + adj_mx.shape[1]] = adj_mx
            offset += adj_mx.shape[0]
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
