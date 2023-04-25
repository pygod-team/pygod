import math
import torch
from torch import nn
from torch_geometric.nn import MLP

from .conv import NeighDiff


class DONEBase(nn.Module):
    """
    DONE (Deep Outlier Aware Attributed Network Embedding) consists of
    an attribute autoencoder and a structure autoencoder. It estimates
    five losses to optimize the model, including an attribute proximity
    loss, an attribute homophily loss, a structure proximity loss, a
    structure homophily loss, and a combination loss. It calculates
    three outlier scores, and averages them as an overall scores.

    See :cite:`bandyopadhyay2020outlier` for details.

    Parameters
    ----------
    x_dim : int
        Input dimension of node features.
    s_dim : int
        Input dimension of node structures, i.e., number of nodes.
    hid_dim :  int, optional
        Hidden dimension of model. Default: ``64``.
    num_layers : int, optional
        Total number of layers in model. Default: ``4``.
    dropout : float, optional
        Dropout rate. Default: ``0.``.
    act : str or Callable, optional
        Activation function if not None.
        Default: ``torch.nn.functional.relu``.
    w1 : float, optional
        Loss balancing weight for structure proximity.
        Default: ``0.2``.
    w2 : float, optional
        Loss balancing weight for structure homophily.
        Default: ``0.2``.
    w3 : float, optional
        Loss balancing weight for attribute proximity.
        Default: ``0.2``.
    w4 : float, optional
        Loss balancing weight for attribute proximity.
        Default: ``0.2``.
    w5 : float, optional
        Loss balancing weight for combination.
        Default: ``0.2``.
    **kwargs (optional):
        Additional arguments of the underlying
        :class:`torch_geometric.nn.MLP`.
    """

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
        super(DONEBase, self).__init__()

        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.w4 = w4
        self.w5 = w5

        # split the number of layers for the encoder and decoders
        assert num_layers >= 2, \
            "Number of layers must be greater than or equal to 2."
        encoder_layers = math.floor(num_layers / 2)
        decoder_layers = math.ceil(num_layers / 2)

        self.attr_encoder = MLP(in_channels=x_dim,
                                hidden_channels=hid_dim,
                                out_channels=hid_dim,
                                num_layers=encoder_layers,
                                dropout=dropout,
                                act=act,
                                **kwargs)

        self.attr_decoder = MLP(in_channels=hid_dim,
                                hidden_channels=hid_dim,
                                out_channels=x_dim,
                                num_layers=decoder_layers,
                                dropout=dropout,
                                act=act,
                                **kwargs)

        self.struct_encoder = MLP(in_channels=s_dim,
                                  hidden_channels=hid_dim,
                                  out_channels=hid_dim,
                                  num_layers=encoder_layers,
                                  dropout=dropout,
                                  act=act,
                                  **kwargs)

        self.struct_decoder = MLP(in_channels=hid_dim,
                                  hidden_channels=hid_dim,
                                  out_channels=s_dim,
                                  num_layers=decoder_layers,
                                  dropout=dropout,
                                  act=act,
                                  **kwargs)

        self.neigh_diff = NeighDiff()

    def forward(self, x, s, edge_index):
        h_a = self.attr_encoder(x)
        x_ = self.attr_decoder(h_a)
        dna = self.neigh_diff(h_a, edge_index).squeeze()
        h_s = self.struct_encoder(s)
        s_ = self.struct_decoder(h_s)
        dns = self.neigh_diff(h_s, edge_index).squeeze()

        return x_, s_, h_a, h_s, dna, dns

    def loss_func(self, x, x_, s, s_, h_a, h_s, dna, dns):
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

        # equation 6
        loss_c = torch.mean(torch.log(torch.pow(oc, -1)) * dc)

        # equation 7
        loss = self.w3 * loss_prox_a + \
               self.w4 * loss_hom_a + \
               self.w1 * loss_prox_s + \
               self.w2 * loss_hom_s + \
               self.w5 * loss_c

        return loss, oa, os, oc
