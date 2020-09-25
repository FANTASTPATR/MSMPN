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
    def __init__(self, in_feats, hid_feats, out_feats, channels, layers, dropout):
        super(GCN, self).__init__()
        self.in_gcns = nn.ModuleList()
        self.out_gcns = nn.ModuleList()
        self.hid_gcns = nn.ModuleList()
        for _ in range(channels):
            self.in_gcns.append(GCNLayer(in_feats=in_feats, out_feats=int(hid_feats / channels)))
            self.out_gcns.append(GCNLayer(in_feats=hid_feats, out_feats=int(out_feats / channels)))
            self.hid_gcns.append(GCNLayer(in_feats=hid_feats, out_feats=int(hid_feats / channels)))
        self.layers = layers
        self.channels = channels
        self.dropout = dropout

    def forward(self, graph, features):
        for layer_idx in range(self.layers):
            channel_rst = []
            for channel_idx in range(self.channels):
                if layer_idx == 0:
                    cn_out = self.in_gcns[channel_idx](graph, features)  # [b*snapshot*n,d]
                elif layer_idx == self.layers - 1:
                    cn_out = self.out_gcns[channel_idx](graph, features)
                else:
                    cn_out = self.hid_gcns[channel_idx](graph, features)
                channel_rst.append(cn_out)
            features = torch.cat(channel_rst, dim=-1)
        features = F.dropout(features, self.dropout, training=self.training)
        return features
