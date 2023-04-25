import torch
import torch.nn.functional as F
from torch_geometric.nn import MLP
from torch_geometric.utils import to_dense_adj

from ..nn.functional import double_recon_loss


class GAANBase(torch.nn.Module):
    def __init__(self,
                 in_dim,
                 noise_dim,
                 hid_dim,
                 generator_layers=2,
                 encoder_layers=2,
                 dropout=0.,
                 act=torch.nn.functional.relu,
                 **kwargs):
        super(GAANBase, self).__init__()

        self.generator = MLP(in_channels=noise_dim,
                             hidden_channels=hid_dim,
                             out_channels=in_dim,
                             num_layers=generator_layers,
                             dropout=dropout,
                             act=act,
                             **kwargs)

        self.discriminator = MLP(in_channels=in_dim,
                                 hidden_channels=hid_dim,
                                 out_channels=hid_dim,
                                 num_layers=encoder_layers,
                                 dropout=dropout,
                                 act=act,
                                 **kwargs)
        self.emb = None
        self.score_func = double_recon_loss

    def forward(self, x, noise):
        x_ = self.generator(noise)

        self.emb = self.discriminator(x)
        z_ = self.discriminator(x_)

        a = torch.sigmoid((self.emb @ self.emb.T))
        a_ = torch.sigmoid((z_ @ z_.T))

        return x_, a, a_

    @staticmethod
    def loss_func_g(a_):
        loss_g = F.binary_cross_entropy(a_, torch.ones_like(a_))
        return loss_g

    @staticmethod
    def loss_func_ed(a, a_):
        loss_r = F.binary_cross_entropy(a, torch.ones_like(a))
        loss_f = F.binary_cross_entropy(a_, torch.zeros_like(a_))
        return (loss_f + loss_r) / 2

    @staticmethod
    def process_graph(data):
        data.s = to_dense_adj(data.edge_index)[0]
