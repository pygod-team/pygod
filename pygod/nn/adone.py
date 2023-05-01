import torch
from torch_geometric.nn import MLP
from torch_geometric.utils import to_dense_adj

from .done import DONEBase


class AdONEBase(torch.nn.Module):
    def __init__(self,
                 x_dim,
                 s_dim,
                 hid_dim=64,
                 num_layers=4,
                 dropout=0.,
                 act=torch.nn.functional.relu,
                 w1=0.2,
                 w2=0.2,
                 w3=0.2,
                 w4=0.2,
                 w5=0.2,
                 **kwargs):
        super(AdONEBase, self).__init__()

        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.w4 = w4
        self.w5 = w5

        self.done = DONEBase(x_dim=x_dim,
                             s_dim=s_dim,
                             hid_dim=hid_dim,
                             num_layers=num_layers,
                             dropout=dropout,
                             act=act,
                             w1=self.w1,
                             w2=self.w2,
                             w3=self.w3,
                             w4=self.w4,
                             w5=self.w5,
                             **kwargs)

        self.discriminator = MLP(in_channels=hid_dim,
                                 hidden_channels=int(hid_dim / 2),
                                 out_channels=1,
                                 num_layers=2,
                                 dropout=dropout,
                                 act=torch.tanh)
        self.emb = None

    def forward(self, x, s, edge_index):
        x_, s_, h_a, h_s, dna, dns = self.done(x, s, edge_index)
        dis_a = torch.sigmoid(self.discriminator(h_a))
        dis_s = torch.sigmoid(self.discriminator(h_s))
        self.emb = (h_a, h_s)

        return x_, s_, h_a, h_s, dna, dns, dis_a, dis_s

    def loss_func(self, x, x_, s, s_, h_a, h_s, dna, dns, dis_a, dis_s):
        # equation 9 is based on the official implementation, and it
        # is slightly different from the paper
        dx = torch.sum(torch.pow(x - x_, 2), 1)
        tmp = self.w3 * dx + self.w4 * dna
        oa = tmp / torch.sum(tmp)

        # equation 8 is based on the official implementation, and it
        # is slightly different from the paper
        ds = torch.sum(torch.pow(s - s_, 2), 1)
        tmp = self.w1 * ds + self.w2 * dns
        os = tmp / torch.sum(tmp)

        # equation 10
        dc = torch.sum(torch.pow(h_a - h_s, 2), 1)
        oc = dc / torch.sum(dc)

        # equation 4
        loss_prox_a = torch.mean(torch.log(torch.pow(oa, -1)) * dx)

        # equation 5
        loss_hom_a = torch.mean(torch.log(torch.pow(oa, -1)) * dna)

        # equation 2
        loss_prox_s = torch.mean(torch.log(torch.pow(os, -1)) * ds)

        # equation 3
        loss_hom_s = torch.mean(torch.log(torch.pow(os, -1)) * dns)

        # equation 12
        loss_alg = torch.mean(torch.log(torch.pow(oc, -1))
                              * (-torch.log(1 - dis_a) - torch.log(dis_s)))

        # equation 13
        loss = self.w3 * loss_prox_a + \
               self.w4 * loss_hom_a + \
               self.w1 * loss_prox_s + \
               self.w2 * loss_hom_s + \
               self.w5 * loss_alg

        return loss, oa, os, oc

    @staticmethod
    def process_graph(data):
        data.s = to_dense_adj(data.edge_index)[0]
