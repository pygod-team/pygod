import torch
import torch.nn as nn
from torch_geometric.nn import GCN
from torch_geometric.utils import to_dense_adj

from .functional import double_recon_loss


class OCGNNBase(nn.Module):

    def __init__(self,
                 in_dim,
                 hid_dim,
                 num_layers=2,
                 dropout=0.,
                 act=torch.nn.functional.relu,
                 beta=0.5,
                 warmup=0,
                 eps=0.001,
                 backbone=GCN,
                 **kwargs):
        super(OCGNNBase, self).__init__()

        self.beta = beta
        self.warmup = warmup
        self.eps = eps

        self.gnn = backbone(in_channels=in_dim,
                            hidden_channels=hid_dim,
                            num_layers=num_layers,
                            out_channels=hid_dim,
                            dropout=dropout,
                            act=act,
                            **kwargs)

        self.r = 0
        self.c = torch.zeros(hid_dim)

        self.emb = None

    def forward(self, x, edge_index):

        emb = self.gnn(x, edge_index)
        return emb

    def loss_func(self, emb):

        dist = torch.sum(torch.pow(emb - self.c, 2), 1)
        score = dist - self.r ** 2
        loss = self.r ** 2 + 1 / self.beta * torch.mean(torch.relu(score))

        if self.warmup > 0:
            with torch.no_grad():
                self.warmup -= 1
                self.r = torch.quantile(torch.sqrt(dist), 1 - self.beta)
                self.c = torch.mean(emb, 0)
                self.c[(abs(self.c) < self.eps) & (self.c < 0)] = -self.eps
                self.c[(abs(self.c) < self.eps) & (self.c > 0)] = self.eps

        return loss, score
