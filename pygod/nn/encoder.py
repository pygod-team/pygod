# -*- coding: utf-8 -*-
"""Personalized Neural Network Encoders"""
# Author: Kay Liu <zliu234@uic.edu>, Yingtong Dou <ytongdou@gmail.com>
# License: BSD 2 clause

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import GNAConv


class GNA(nn.Module):
    """
    Graph Node Attention Network (GNA). See :cite:`yuan2021higher` for
    more details.

    Parameters
    ----------
    in_channels : int
        Input dimension of node features.
    hidden_channels :  int
        Hidden dimension of the model.
    num_layers : int
        Number of layers in the model.
    out_channels : int
        Output dimension of the model.
    dropout : float, optional
        Dropout rate. Default: ``0.``.
    act : callable activation function or None, optional
        Activation function if not None.
        Default: ``torch.nn.functional.relu``.
    """
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 num_layers,
                 out_channels,
                 dropout=0.,
                 act=torch.nn.functional.relu):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GNAConv(in_channels, hidden_channels))
        for layer in range(num_layers - 2):
            self.layers.append(GNAConv(hidden_channels,
                                       hidden_channels))
        self.layers.append(GNAConv(hidden_channels, out_channels))

        self.dropout = dropout
        self.act = act

    def forward(self, s, edge_index):
        """
        Forward computation.

        Parameters
        ----------
        s : torch.Tensor
            Input node embeddings.
        edge_index : torch.Tensor
            Edge index.

        Returns
        -------
        s : torch.Tensor
            Updated node embeddings.
        """
        for layer in self.layers:
            s = layer(s, edge_index)
            s = F.dropout(s, self.dropout, training=self.training)
            if self.act is not None:
               s = self.act(s)
        return s
