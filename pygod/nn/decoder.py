# -*- coding: utf-8 -*-
"""Graph Decoders"""
# Author: Kay Liu <zliu234@uic.edu>, Yingtong Dou <ytongdou@gmail.com>
# License: BSD 2 clause

import torch
import torch.nn as nn
from torch_geometric.nn import GCN


class DotProductDecoder(nn.Module):
    r"""
    Dot product decoder for the structure reconstruction, which is
    defined as :math:`\symbf{A}' = \sigma(\symbf{Z}
    \symbf{Z}^\intercal)`, where :math:`\sigma` is the optional sigmoid
    function, :math:`\symbf{Z}` is the input hidden embedding, and the
    :math:`\symbf{A}'` is the reconstructed adjacency matrix.

    Parameters
    ----------
    in_dim : int
        Input dimension of node features.
    hid_dim :  int, optional
        Hidden dimension of model. Default: ``64``.
    num_layers : int, optional
        Number of layers in the decoder. Default: ``1``.
    dropout : float, optional
        Dropout rate. Default: ``0.``.
    act : callable activation function or None, optional
        Activation function if not None.
        Default: ``torch.nn.functional.relu``.
    sigmoid_s : bool, optional
        Whether to apply sigmoid to the structure reconstruction.
        Default: ``False``.
    backbone : torch.nn.Module, optional
        The backbone of the deep decoder implemented in PyG.
        Default: ``torch_geometric.nn.GCN``.
    **kwargs : optional
        Additional arguments for the backbone.
    """

    def __init__(self,
                 in_dim,
                 hid_dim=64,
                 num_layers=1,
                 dropout=0.,
                 act=torch.nn.functional.relu,
                 sigmoid_s=False,
                 backbone=GCN,
                 **kwargs):
        super(DotProductDecoder, self).__init__()

        self.sigmoid_s = sigmoid_s
        self.nn = backbone(in_channels=in_dim,
                           hidden_channels=hid_dim,
                           num_layers=num_layers,
                           out_channels=hid_dim,
                           dropout=dropout,
                           act=act,
                           **kwargs)

    def forward(self, x, edge_index):
        r"""
        Forward computation.

        Parameters
        ----------
        x : torch.Tensor
            Input node embeddings.
        edge_index : torch.Tensor
            Edge index.

        Returns
        -------
        s_ : torch.Tensor
            Reconstructed adjacency matrix.
        """
        h = self.nn(x, edge_index)
        s_ = h @ h.T
        if self.sigmoid_s:
            s_ = torch.sigmoid(s_)
        return s_
