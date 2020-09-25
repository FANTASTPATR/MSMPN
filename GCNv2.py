import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl.function as fn


class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats, norm="both", weight=True, bias=True, activation=None):
        super(GCNLayer, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._norm = norm

        if weight:
            self.weight = nn.Parameter(torch.Tensor(in_feats, out_feats))
        else:
            self.register_parameter('weight', None)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_feats))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()
        self._activation = activation

    def reset_parameters(self):
        if self.weight is not None:
            nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, graph, feat, weight=None):
        with graph.local_scope():
            if self._norm == "both":
                degs = graph.out_degrees().to(feat.device).float().clamp(min=1)
                norm = torch.pow(degs, -0.5)
                shp = norm.shape + (1,) * (feat.dim() - 1)
                norm = torch.reshape(norm, shp)
                feat = feat * norm

            weight = self.weight
            if self._in_feats > self._out_feats:
                if weight is not None:
                    feat = torch.matmul(feat, weight)  # WX
                graph.srcdata['h'] = feat
                graph.update_all(fn.u_mul_e(lhs_field='h', rhs_field="w", out='m'), fn.sum(msg='m', out='h'))
                rst = graph.dstdata['h']
            else:
                graph.srcdata['h'] = feat
                graph.update_all(fn.u_mul_e(lhs_field='h', rhs_field="w", out='m'), fn.sum(msg='m', out='h'))
                rst = graph.dstdata['h']
                if weight is not None:
                    rst = torch.matmul(rst, weight)  # WX

            if self._norm != "none":
                degs = graph.in_degrees().to(feat.device).float().clamp(min=1)
                if self._norm == "both":
                    norm = torch.pow(degs, -0.5)
                else:
                    norm = 1.0 / degs
                shp = norm.shape + (1,) * (feat.dim() - 1)
                norm = torch.reshape(norm, shp)
                rst = rst * norm

            if self.bias is not None:
                rst = rst + self.bias
            if self._activation is not None:
                rst = self._activation(rst)

            return rst


class GCN(nn.Module):
    def __init__(self, in_feats, hid_feats, channels, layers, dropout, supports):
        super(GCN, self).__init__()
        self.gcn_back = GCNLayer(in_feats=in_feats, out_feats=hid_feats)
        self.gcn_forward = GCNLayer(in_feats=in_feats, out_feats=hid_feats)
        self.layers = layers
        self.channels = channels
        self.dropout = dropout
        self.supports = supports
        self.out_linear = nn.Linear(hid_feats * channels, hid_feats)

    def forward(self, features):
        channel_out = []
        for _ in range(self.channels):
            x = features
            for _ in range(self.layers):
                x = self.gcn_back(self.supports[0], x)
                x = self.gcn_forward(self.supports[1], x)
            channel_out.append(x)
        gcn_out = torch.cat(channel_out, dim=-1)
        gcn_out = self.out_linear(gcn_out)
        gcn_out = F.dropout(gcn_out, self.dropout, training=self.training)
        return gcn_out
