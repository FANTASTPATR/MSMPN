import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from scipy import sparse as sp
from dgl import DGLGraph
import dgl

from GCN import GCN


# import pickle


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.in_linear = nn.Linear(self.args.in_dim, self.args.hid_dim)
        self.chp = self._generate_checkpoints()
        self.GCN = GCN(in_feats=self.args.hid_dim, hid_feats=self.args.hid_dim, out_feats=self.args.hid_dim,
                       channels=self.args.channels, layers=self.args.layers, dropout=self.args.dropout)
        self.virtual_g = self._generate_dgl_graphs()
        # self.layer_norms = nn.ModuleList()
        #         # # for _ in range(len(self.chp)):
        #         # #     self.layer_norms.append(
        #         # #         nn.LayerNorm([self.args.batch_size * self.args.snpsts_len * self.args.num_node, self.args.hid_dim],
        #         # #                      elementwise_affine=False))
        self.layer_norms = nn.LayerNorm(
            [self.args.batch_size * self.args.snpsts_len * self.args.num_node, self.args.hid_dim],
            elementwise_affine=False)
        self.out_linear1 = nn.Linear(1, self.args.pred_len)
        self.out_linear2 = nn.Linear(self.args.hid_dim, 1)

    def forward(self, inputs):
        x = self.in_linear(inputs)  # [b,s,n,d]
        last_res = None
        left = 0
        for idx, right in enumerate(self.chp):
            feature_slice = x[:, left:right, :, :].contiguous()  # [b,snapshot,n,d]
            if idx == 0:
                feature_slice = feature_slice.view(x.shape[0] * self.args.snpsts_len * x.shape[2],
                                                   x.shape[-1]).contiguous()  # [b*snapshot*n,d]
                residual = feature_slice  # [b*snapshot*n,d]
                conv_res = self.GCN(self.virtual_g, feature_slice)  # [b*snapshot*n,d]
                conv_res = conv_res + residual  # [b*snapshot*n,d]
                conv_res = self.layer_norms(conv_res)  # [b*snapshot*n,d]
                conv_res = conv_res.view(self.args.batch_size, self.args.snpsts_len, self.args.num_node,
                                         self.args.hid_dim)  # [b,snapshot,n,d]
                last_res = conv_res[:, -1, :, :].unsqueeze(1)  # [b,1,n,d]
                left = right
            else:
                feature_slice = torch.cat([last_res, feature_slice], dim=1)  # [b,snapshot,n,d]
                feature_slice = feature_slice.view(x.shape[0] * self.args.snpsts_len * x.shape[2],
                                                   x.shape[-1]).contiguous()  # [b*snapshot*n,d]
                residual = feature_slice  # [b*snapshot*n,d]
                conv_res = self.GCN(self.virtual_g, feature_slice)  # [b*snapshot*n,d]
                conv_res = conv_res + residual  # [b*snapshot*n,d]
                conv_res = self.layer_norms(conv_res)  # [b*snapshot*n,d]
                conv_res = conv_res.view(self.args.batch_size, self.args.snpsts_len, self.args.num_node,
                                         self.args.hid_dim)  # [b,snapshot,n,d]
                last_res = conv_res[:, -1, :, :].unsqueeze(1)  # [b,1,n,d]
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

    def _combine_graph(self):
        rst = torch.zeros(self.args.adj_mx.shape[0] * self.args.snpsts_len,
                          self.args.adj_mx.shape[0] * self.args.snpsts_len)
        offset = 0
        for v_idx in range(self.args.snpsts_len):
            if v_idx == self.args.snpsts_len - 1:
                rst[-self.args.adj_mx.shape[0]:, -self.args.adj_mx.shape[1]:] = self.args.adj_mx
            else:
                hstack_unit = torch.cat((self.args.adj_mx, self.args.adj_mx), dim=-1)
                rst[offset:offset + hstack_unit.shape[0], offset:offset + hstack_unit.shape[1]] = hstack_unit
        return torch.Tensor(rst)

    def _generate_dgl_graphs(self):
        adj_mx = self._combine_graph()
        adj_mx_unwei = np.where(adj_mx > 0, 1, 0)
        adj_mx_unwei = sp.csr_matrix(adj_mx_unwei)
        g = dgl.DGLGraph(adj_mx_unwei)
        g.edata['w'] = torch.randn(g.number_of_edges(), 1).to(self.args.device)
        for u_idx in range(adj_mx.shape[0]):
            for v_idx in range(adj_mx.shape[1]):
                if adj_mx[u_idx][v_idx] != 0:
                    g.edata['w'][g.edge_id(u_idx, v_idx)] = adj_mx[u_idx][v_idx]
        batch_g = dgl.batch([g for _ in range(self.args.batch_size)])
        return batch_g
