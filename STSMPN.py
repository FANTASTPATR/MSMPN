import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl

from GCNv4 import GCN


class STSMPN(nn.Module):
    def __init__(self, args, ckps, supports):
        super(STSMPN, self).__init__()
        self._args = args
        self._ckps = ckps  # [(0,1),(1,2),...]
        self._gcns = self._generate_gcn()
        self._residual_conv = nn.Conv1d(in_channels=len(ckps) * 2 * self._args.num_node,
                                        out_channels=len(ckps) * self._args.num_node, kernel_size=1)
        parral_backward_graph = dgl.batch([supports[0] for _ in range(len(self._ckps))])
        parral_forward_graph = dgl.batch([supports[1] for _ in range(len(self._ckps))])
        self._supports = [dgl.batch([parral_backward_graph for _ in range(self._args.batch_size)]),
                          dgl.batch([parral_forward_graph for _ in range(self._args.batch_size)])]
        self._out_conv = nn.Conv2d(in_channels=2 * self._args.num_node, out_channels=self._args.num_node,
                                   kernel_size=(1, 1), stride=(1, 1), padding=0)
        self._out_linear = nn.Linear(self._args.channels * self._args.hid_dim, self._args.hid_dim)

    def forward(self, inputs):
        '''

        :param inputs: [batch_size,seq_in_len,num_node,hid_dim]
        :return:
        '''
        features_conv = []
        for idx, group in enumerate(self._ckps):
            group_features = [inputs[:, group[0], :, :], inputs[:, group[1], :, :]]  # [b,n,d]
            group_features = torch.cat(group_features, dim=1)  # [b,2*n,d]
            features_conv.append(group_features)
        features_conv = torch.stack(features_conv).transpose(0, 1).contiguous()  # [b,parallel,n*2,d]
        conv_res = []
        for conv_idx, conv_channel in enumerate(self._gcns):
            channel_res = conv_channel(supports=self._supports, features=features_conv)
            conv_res.append(channel_res)
        conv_res = torch.cat(conv_res, dim=-1).transpose(1, 2)  # [b,2*num,parallel,d*channel]
        conv_res = self._out_conv(conv_res).transpose(1, 2)  # [b,parallel,num,d*channel]
        conv_res = self._out_linear(conv_res)  # [b,parallel,num,d*channel]

        return conv_res

    def _generate_gcn(self):
        gcns = nn.ModuleList()
        for _ in range(self._args.channels):
            gcns.append(
                GCN(in_feats=self._args.hid_dim, hid_feats=self._args.hid_dim, layers=self._args.layers,
                    dropout=self._args.dropout, num_node=self._args.num_node, support_len=2,
                    parrallel_num=len(self._ckps), batch_size=self._args.batch_size))
        return gcns
