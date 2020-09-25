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
    def __init__(self, in_feats, hid_feats, layers, dropout, num_node, support_len, snpsht_len, batch_size):
        super(GCN, self).__init__()
        self.gcnlayer = GCNLayer(in_feats=in_feats, out_feats=hid_feats)
        self.layers = layers
        self.hid_dim = hid_feats
        self.batch_size = batch_size
        self.out_conv = nn.Conv1d(in_channels=(layers * support_len + 1) * num_node * snpsht_len,
                                  out_channels=num_node, kernel_size=1, stride=1, padding=0)
        self.dropout = dropout

    def forward(self, supports, features):
        # features:[b*s*n,d]
        out = [features]
        for key, graph in supports.items():
            features1 = self.gcnlayer(graph, features)  # [b*s*n,d]
            out.append(features1)
            for layer_num in range(2, self.layers + 1):
                features2 = self.gcnlayer(graph, features1)  # [b*s*n,d]
                out.append(features2)
                features1 = features2

        h = torch.cat(out, dim=0)
        h = h.view(self.batch_size, -1, self.hid_dim)
        h = self.out_conv(h)
        h = h.view(-1, self.hid_dim)
        h = F.dropout(h, self.dropout, training=self.training)
        return h
