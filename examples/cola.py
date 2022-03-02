# -*- coding: utf-8 -*-
""" Example code for CoLA model.
"""
# Author: Canyu Chen <>,
# License: BSD 2 clause

import torch
import argparse
import os.path as osp
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor
from torch_geometric.nn import GATConv
from sklearn.metrics import roc_auc_score
from torch_geometric.datasets import AttributedGraphDataset
from sklearn.utils.validation import check_is_fitted

from pygod.models import BaseDetector

class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, act, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU() if act == 'prelu' else act
        
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq, adj, sparse=False):
        seq_fts = self.fc(seq)
        if sparse:
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)
        else:
            out = torch.bmm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias
        
        return self.act(out)

class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq):
        return torch.mean(seq, 1)

class MaxReadout(nn.Module):
    def __init__(self):
        super(MaxReadout, self).__init__()

    def forward(self, seq):
        return torch.max(seq,1).values

class MinReadout(nn.Module):
    def __init__(self):
        super(MinReadout, self).__init__()

    def forward(self, seq):
        return torch.min(seq, 1).values

class WSReadout(nn.Module):
    def __init__(self):
        super(WSReadout, self).__init__()

    def forward(self, seq, query):
        query = query.permute(0,2,1)
        sim = torch.matmul(seq,query)
        sim = F.softmax(sim,dim=1)
        sim = sim.repeat(1, 1, 64)
        out = torch.mul(seq,sim)
        out = torch.sum(out,1)
        return out

class Discriminator(nn.Module):
    def __init__(self, n_h, negsamp_round):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

        self.negsamp_round = negsamp_round

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl):
        scs = []
        # positive
        scs.append(self.f_k(h_pl, c))

        # negative
        c_mi = c
        for _ in range(self.negsamp_round):
            c_mi = torch.cat((c_mi[-2:-1,:], c_mi[:-1,:]),0)
            scs.append(self.f_k(h_pl, c_mi))

        logits = torch.cat(tuple(scs))

        return logits

class Model(nn.Module):
    def __init__(self, n_in, n_h, activation, negsamp_round, readout):
        super(Model, self).__init__()
        self.read_mode = readout
        self.gcn = GCN(n_in, n_h, activation)
        if readout == 'max':
            self.read = MaxReadout()
        elif readout == 'min':
            self.read = MinReadout()
        elif readout == 'avg':
            self.read = AvgReadout()
        elif readout == 'weighted_sum':
            self.read = WSReadout()

        self.disc = Discriminator(n_h, negsamp_round)

    def forward(self, seq1, adj, sparse=False):
        h_1 = self.gcn(seq1, adj, sparse)

        if self.read_mode != 'weighted_sum':
            c = self.read(h_1[:,: -1,:])
            h_mv = h_1[:,-1,:]
        else:
            h_mv = h_1[:, -1, :]
            c = self.read(h_1[:,: -1,:], h_1[:,-2: -1, :])

        ret = self.disc(c, h_mv)

        return ret


class CoLA(BaseDetector):
    def __init__(self,
                 feat_size,
                 n_size,
                 embed_size,
                 hidden_size,
                 dropout=0.2,
                 weight_decay=1e-5,
                 preprocessing=True,
                 loss_fn=None,
                 contamination=0.1,
                 device=None):
        super(CoLA, self).__init__(contamination=contamination)

        
    def fit(self, adj, adj_label, attrs, args):
        pass

    def decision_function(self, attrs, adj, args):
        pass

    def predict(self, attrs, adj, args):
        pass


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='BlogCatalog',
                    help='dataset name: Flickr/BlogCatalog')
parser.add_argument('--hidden_dim', type=int, default=4,
                help='dimension of hidden embedding (default: 64)')
parser.add_argument('--embed_dim', type=int, default=128,
                help='dimension of hidden embedding (default: 128)')
parser.add_argument('--epoch', type=int, default=3, help='Training epoch')
parser.add_argument('--lr', type=float, default=5e-3, help='learning rate')
parser.add_argument('--dropout', type=float, default=0.3,
                    help='Dropout rate')
parser.add_argument('--alpha', type=float, default=0.8,
                    help='balance parameter')
parser.add_argument('--theta', type = float, default = 0.2, help= 'structure penalty')
parser.add_argument('--eta', type = float, default = 0.2, help = 'attribute penalty')
parser.add_argument('--device', default='cpu', type=str, help='cuda/cpu')

args = parser.parse_args()

# data loading
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data',
                args.dataset)
data = AttributedGraphDataset(path, 'BlogCatalog')
adj = data[0].edge_index
attrs = data[0].x[:, :4]
label = data[0].y % 2
dense_adj = SparseTensor(row=adj[0], col=adj[1]).to_dense()
rowsum = dense_adj.sum(1)
d_inv_sqrt = torch.pow(rowsum, -0.5).flatten()
d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
adj_label = (dense_adj * d_mat_inv_sqrt).T * d_mat_inv_sqrt

# train(args)
# todo need to make args part of initialization
clf = CoLA(feat_size=attrs.size(1), n_size = attrs.size(0),
             hidden_size=args.hidden_dim, embed_size = args.embed_dim,
               dropout=args.dropout)

print('training it')
clf.fit(adj, adj_label, attrs, args)

print('predict on self')
outlier_scores = clf.decision_function(attrs, adj, args)
print('Raw scores', outlier_scores)

labels = clf.predict(attrs, adj, args)
print('Labels', labels)
