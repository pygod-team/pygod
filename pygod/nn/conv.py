# -*- coding: utf-8 -*-
"""Convolutional Layers for Graph Neural Networks."""
# Author: Kay Liu <zliu234@uic.edu>
# License: BSD 2 clause

import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax, add_self_loops


class NeighDiff(MessagePassing):
    """
    Calculate the Euclidean distance between the node features of the
    central node and its neighbors, reducing by mean.
    """
    def __init__(self):
        super().__init__(aggr='mean')

    def forward(self, h, edge_index):
        """
        Forward computation.

        Parameters
        ----------
        h : torch.Tensor
            Input node embeddings.
        edge_index : torch.Tensor
            Edge index.

        Returns
        -------
        h : torch.Tensor
            Updated node embeddings.
        """
        return self.propagate(edge_index, h=h)

    def message(self, h_i, h_j, edge_index):
        return torch.sum(torch.pow(h_i - h_j, 2), dim=1, keepdim=True)


class GNAConv(MessagePassing):
    """
    Graph Node Attention Network (GNA) layer. See
    :cite:`yuan2021higher` for more details.
    """
    def __init__(self,
                 in_channels,
                 out_channels):
        super().__init__(aggr='add')
        self.w1 = torch.nn.Linear(in_channels, out_channels)
        self.w2 = torch.nn.Linear(in_channels, out_channels)
        self.a = torch.nn.Parameter(torch.randn(out_channels, 1))

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
        edge_index, _ = add_self_loops(edge_index, num_nodes=s.size(0))
        out = self.propagate(edge_index, s=self.w2(s))
        return self.w1(s) + out

    def message(self, s_i, s_j, edge_index):
        alpha = (s_i - s_j) @ self.a
        alpha = softmax(alpha, edge_index[1], num_nodes=s_i.shape[0])
        return alpha * s_j
