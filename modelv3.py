import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from scipy import sparse as sp
from dgl import DGLGraph
import dgl

from GCNv3 import GCN

import pickle


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.in_linear = nn.Linear(self.args.in_dim, self.args.hid_dim)
        self.chp = self._generate_checkpoints()
        self.virtual_gs = self._generate_dgl_graphs()
        self.gcns = self._generate_gcns()
        self.reduce_conv_dim = nn.Linear(self.args.hid_dim * self.args.channels, self.args.hid_dim)
        self.layer_norm = nn.LayerNorm([self.args.batch_size, self.args.num_node,
                                        self.args.hid_dim], elementwise_affine=False)
        self.residual_conv = nn.Conv1d(in_channels=self.args.snpsts_len * self.args.num_node,
                                       out_channels=self.args.num_node,
                                       kernel_size=1)
        self.out_linear1 = nn.Linear(1, self.args.pred_len)
        self.out_linear2 = nn.Linear(self.args.hid_dim, 1)

    def forward(self, inputs):
        x = self.in_linear(inputs)  # [b,s,n,d]
        last_res = None
        left = 0
        for idx, right in enumerate(self.chp):
            feature_slice = x[:, left:right, :, :].contiguous()  # [b,snapshot,n,d]
            if idx == 0:
                feature_slice = feature_slice.view(self.args.batch_size, self.args.snpsts_len * self.args.num_node,
                                                   self.args.hid_dim).contiguous()  # [b,snapshot*n,d]
                residual = feature_slice  # [b,snapshot*n,d]
                residual = self.residual_conv(residual)  # [b,1*n,d]
                feature_slice = feature_slice.view(x.shape[0] * self.args.snpsts_len * x.shape[2],
                                                   x.shape[-1]).contiguous()  # [b*snapshot*n,d]
                conv_res = []
                for conv_idx, channel in enumerate(self.gcns):
                    channel_res = channel(supports=self.virtual_gs, features=feature_slice)  # [b*1*n,d]
                    conv_res.append(channel_res)
                conv_res = torch.cat(conv_res, dim=-1)  # [b*1*n,d*channel]
                conv_res = self.reduce_conv_dim(conv_res)  # [b*1*n,d]
                conv_res = conv_res.view(self.args.batch_size, self.args.num_node,
                                         self.args.hid_dim)  # [b,1*n,d]
                conv_res = conv_res + residual  # [b,1*n,d]
                last_res = self.layer_norm(conv_res).unsqueeze(1)  # [b,1,n,d]

                left = right
            else:
                feature_slice = torch.cat([last_res, feature_slice], dim=1)  # [b,snapshot,n,d]
                feature_slice = feature_slice.view(self.args.batch_size, self.args.snpsts_len * self.args.num_node,
                                                   self.args.hid_dim).contiguous()  # [b,snapshot*n,d]
                residual = feature_slice  # [b,snapshot*n,d]
                residual = self.residual_conv(residual)  # [b,1*n,d]
                feature_slice = feature_slice.view(x.shape[0] * self.args.snpsts_len * x.shape[2],
                                                   x.shape[-1]).contiguous()  # [b*snapshot*n,d]
                conv_res = []
                for conv_idx, channel in enumerate(self.gcns):
                    channel_res = channel(supports=self.virtual_gs, features=feature_slice)  # [b*1*n,d]
                    conv_res.append(channel_res)
                conv_res = torch.cat(conv_res, dim=-1)  # [b*1*n,d*channel]
                conv_res = self.reduce_conv_dim(conv_res)  # [b*1*n,d]
                conv_res = conv_res.view(self.args.batch_size, self.args.num_node,
                                         self.args.hid_dim)  # [b,1*n,d]
                conv_res = conv_res + residual  # [b,1*n,d]
                last_res = self.layer_norm(conv_res).unsqueeze(1)  # [b,1,n,d]

                left = right
        last_res = last_res.transpose(-1, 1)  # [b,d,n,1]
        out = self.out_linear1(last_res).transpose(-1, 1)  # [b,pred_len,n,d]
        out = F.relu(out)
        out = self.out_linear2(out)  # [b,pred_len,n,1]
        return out

    def _generate_checkpoints(self):
        '''
        generate check points that compose connected graphs
        :return:
        '''
        mark = self.args.snpsts_len
        checkpoints = []
        while mark < self.args.his_len:
            checkpoints.append(mark)
            mark += (self.args.snpsts_len - 1)
        if checkpoints[-1] != self.args.his_len:
            checkpoints.append(self.args.his_len)
        return checkpoints

    def _generate_gcns(self):
        gcns = nn.ModuleList()
        for _ in range(self.args.channels):
            gcns.append(GCN(in_feats=self.args.hid_dim, hid_feats=self.args.hid_dim, layers=self.args.layers,
                            dropout=self.args.dropout, num_node=self.args.num_node, support_len=len(self.virtual_gs),
                            snpsht_len=self.args.snpsts_len, batch_size=self.args.batch_size))
        return gcns

    def _combine_graph_forward(self):
        rst = torch.zeros(self.args.adj_mx.shape[0] * self.args.snpsts_len,
                          self.args.adj_mx.shape[0] * self.args.snpsts_len)
        offset = 0
        for v_idx in range(self.args.snpsts_len):
            if v_idx == self.args.snpsts_len - 1:
                rst[-self.args.adj_mx.shape[0]:, -self.args.adj_mx.shape[1]:] = self.args.adj_mx
                offset += self.args.adj_mx.shape[0]
            else:
                hstack_unit = torch.cat((self.args.adj_mx, torch.eye(self.args.adj_mx.shape[0])), dim=-1)
                rst[offset:offset + hstack_unit.shape[0], offset:offset + hstack_unit.shape[1]] = hstack_unit
                offset += self.args.adj_mx.shape[0]
        return torch.Tensor(rst)

    def _combind_graph_backward(self):
        rst = torch.zeros(self.args.adj_mx.T.shape[0] * self.args.snpsts_len,
                          self.args.adj_mx.T.shape[0] * self.args.snpsts_len)
        offset = 0
        for v_idx in range(self.args.snpsts_len):
            rst[offset:offset + self.args.adj_mx.T.shape[0],
            offset:offset + self.args.adj_mx.T.shape[1]] = self.args.adj_mx.T
            offset += self.args.adj_mx.shape[0]

        return torch.Tensor(rst)

    def _generate_dgl_graphs(self):
        adj_mx_forward = self._combine_graph_forward()
        adj_mx_f_unwei = np.where(adj_mx_forward > 0, 1, 0)
        adj_mx_f_unwei = sp.csr_matrix(adj_mx_f_unwei)
        f_g = dgl.DGLGraph(adj_mx_f_unwei)
        f_g.edata['w'] = torch.randn(f_g.number_of_edges(), 1).to(self.args.device)
        for u_idx in range(adj_mx_forward.shape[0]):
            for v_idx in range(adj_mx_forward.shape[1]):
                if adj_mx_forward[u_idx][v_idx] != 0:
                    f_g.edata['w'][f_g.edge_id(u_idx, v_idx)] = adj_mx_forward[u_idx][v_idx]
        batch_g_f = dgl.batch([f_g for _ in range(self.args.batch_size)])

        adj_mx_back = self._combind_graph_backward()
        adj_mx_b_unwei = np.where(adj_mx_back > 0, 1, 0)
        adj_mx_b_unwei = sp.csr_matrix(adj_mx_b_unwei)
        b_g = dgl.DGLGraph(adj_mx_b_unwei)
        b_g.edata['w'] = torch.randn(b_g.number_of_edges(), 1).to(self.args.device)
        for u_idx in range(adj_mx_back.shape[0]):
            for v_idx in range(adj_mx_back.shape[1]):
                if adj_mx_back[u_idx][v_idx] != 0:
                    b_g.edata['w'][b_g.edge_id(u_idx, v_idx)] = adj_mx_back[u_idx][v_idx]
        batch_g_b = dgl.batch([b_g for _ in range(self.args.batch_size)])

        return [batch_g_b, batch_g_f]
