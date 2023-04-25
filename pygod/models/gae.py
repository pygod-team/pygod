# -*- coding: utf-8 -*-
""" Multilayer Perceptron Autoencoder
"""
# Author: Kay Liu <zliu234@uic.edu>
# License: BSD 2 clause

import torch
import warnings
import torch.nn.functional as F
from torch_geometric.nn import MLP, GCN
from torch_geometric.utils import to_dense_adj

from . import DeepDetector

from ..nn.decoder import DotProductDecoder


class GAE(DeepDetector):
    """
    Graph Autoencoder.

    See :cite:`kipf2016variational` for details.

    Parameters
    ----------
    hid_dim :  int, optional
        Hidden dimension of model. Default: ``64``.
    num_layers : int, optional
        Total number of layers in model. Default: ``4``.
    dropout : float, optional
        Dropout rate. Default: ``0.``.
    weight_decay : float, optional
        Weight decay (L2 penalty). Default: ``0.``.
    act : callable activation function or None, optional
        Activation function if not None.
        Default: ``torch.relu``.
    backbone : torch.nn.Module, optional
        The backbone of the deep detector implemented in PyG.
        Default: ``torch_geometric.nn.GCN``.
    recon_s : bool, optional
        Reconstruct the structure instead of node feature .
        Default: ``False``.
    sigmoid_s : bool, optional
        Whether to use sigmoid function to scale the reconstructed
        structure. Default: ``False``.
    contamination : float, optional
        The amount of contamination of the dataset in (0., 0.5], i.e.,
        the proportion of outliers in the dataset. Used when fitting to
        define the threshold on the decision function. Default: ``0.1``.
    lr : float, optional
        Learning rate. Default: ``0.004``.
    epoch : int, optional
        Maximum number of training epoch. Default: ``100``.
    gpu : int
        GPU Index, -1 for using CPU. Default: ``-1``.
    batch_size : int, optional
        Minibatch size, 0 for full batch training. Default: ``0``.
    num_neigh : int, optional
        Number of neighbors in sampling, -1 for all neighbors.
        Default: ``-1``.
    verbose : int, optional
        Verbosity mode. Range in [0, 3]. Larger value for printing out
        more log information. Default: ``0``.
    **kwargs : optional
        Additional keyword arguments for model initialization.

    Examples
    --------
    >>> from pygod.models import GAE
    >>> model = GAE()
    >>> model.fit(data)
    >>> prediction = model.predict(data)
    """

    def __init__(self,
                 hid_dim=64,
                 num_layers=4,
                 dropout=0.,
                 weight_decay=0.,
                 act=F.relu,
                 backbone=GCN,
                 recon_s=False,
                 sigmoid_s=False,
                 contamination=0.1,
                 lr=4e-3,
                 epoch=100,
                 gpu=-1,
                 batch_size=0,
                 num_neigh=-1,
                 verbose=False,
                 **kwargs):

        if num_neigh != 0 and backbone == MLP:
            warnings.warn('MLP does not use neighbor information.')
            num_neigh = 0

        self.recon_s = recon_s
        self.sigmoid_s = sigmoid_s

        super(GAE, self).__init__(hid_dim=hid_dim,
                                  num_layers=num_layers,
                                  dropout=dropout,
                                  weight_decay=weight_decay,
                                  act=act,
                                  backbone=backbone,
                                  contamination=contamination,
                                  lr=lr,
                                  epoch=epoch,
                                  gpu=gpu,
                                  batch_size=batch_size,
                                  num_neigh=num_neigh,
                                  verbose=verbose,
                                  **kwargs)

    def process_graph(self, data):

        if self.recon_s:
            data.s = to_dense_adj(data.edge_index)[0]

    def init_model(self, **kwargs):

        if self.recon_s:
            model = DotProductDecoder(in_dim=self.in_dim,
                                      hid_dim=self.hid_dim,
                                      num_layers=self.num_layers,
                                      dropout=self.dropout,
                                      act=self.act,
                                      sigmoid_s=self.sigmoid_s,
                                      backbone=self.backbone,
                                      **kwargs).to(self.device)
        else:
            model = self.backbone(in_channels=self.in_dim,
                                  hidden_channels=self.hid_dim,
                                  out_channels=self.in_dim,
                                  num_layers=self.num_layers,
                                  dropout=self.dropout,
                                  act=self.act,
                                  **kwargs).to(self.device)
        return model

    def forward_model(self, data):

        batch_size = data.batch_size
        node_idx = data.node_idx

        x = data.x.to(self.device)
        edge_index = data.edge_index.to(self.device)

        if self.recon_s:
            s = data.s.to(self.device)[:, node_idx]

        if self.backbone == MLP:
            h = self.model(x, None)
        else:
            h = self.model(x, edge_index)

        target = s[:, node_idx] if self.recon_s else x
        score = torch.mean(F.mse_loss(target[:batch_size],
                                      h[:batch_size],
                                      reduction='none'), dim=1)

        loss = torch.mean(score)

        return loss, score.detach().cpu()