# -*- coding: utf-8 -*-
"""Deep Anomaly Detection on Attributed Networks (DOMINANT)"""
# Author: Kay Liu <zliu234@uic.edu>
# License: BSD 2 clause

import torch
from torch_geometric.nn import GCN
from torch_geometric.utils import to_dense_adj

from .base import DeepDetector
from ..nn import DOMINANTBase


class DOMINANT(DeepDetector):
    """
    DOMINANT (Deep Anomaly Detection on Attributed Networks) is an
    anomaly detector consisting of a shared graph convolutional
    encoder, a structure reconstruction decoder, and an attribute
    reconstruction decoder. The reconstruction mean squared error of the
    decoders are defined as structure anomaly score and attribute
    anomaly score, respectively.

    See :cite:`ding2019deep` for details.

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
    act : str or Callable, optional
        Activation function if not None.
        Default: ``torch.nn.functional.relu``.
    sigmoid_s : bool, optional
        Whether to use sigmoid function to scale the reconstructed
        structure. Default: ``False``.
    backbone : torch.nn.Module, optional
        The backbone of the deep detector implemented in PyG.
        Default: ``torch_geometric.nn.GIN``.
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
    weight : float, optional
        Weight between reconstruction of node feature and structure.
        Default: ``0.5``.
    verbose : int, optional
        Verbosity mode. Range in [0, 3]. Larger value for printing out
        more log information. Default: ``0``.
    **kwargs : optional
        Additional arguments for the backbone.

    Examples
    --------
    >>> import pygod
    >>> model = pygod.models.DOMINANT()
    >>> data = pygod.utils.load_data('inj_cora')
    >>> model.fit(data)
    >>> pred = model.predict(data)
    """

    def __init__(self,
                 hid_dim=64,
                 num_layers=4,
                 dropout=0.,
                 weight_decay=0.,
                 act=torch.nn.functional.relu,
                 sigmoid_s=False,
                 backbone=GCN,
                 contamination=0.1,
                 lr=4e-3,
                 epoch=100,
                 gpu=-1,
                 batch_size=0,
                 num_neigh=-1,
                 weight=0.5,
                 verbose=0,
                 **kwargs):

        self.weight = weight
        self.sigmoid_s = sigmoid_s
        super(DOMINANT, self).__init__(hid_dim=hid_dim,
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

        data.s = to_dense_adj(data.edge_index)[0]

    def init_nn(self, **kwargs):

        return DOMINANTBase(in_dim=self.in_dim,
                            hid_dim=self.hid_dim,
                            num_layers=self.num_layers,
                            dropout=self.dropout,
                            act=self.act,
                            sigmoid_s=self.sigmoid_s,
                            backbone=self.backbone,
                            **kwargs).to(self.device)

    def forward_nn(self, data):

        batch_size = data.batch_size
        node_idx = data.node_idx

        x = data.x.to(self.device)
        s = data.s.to(self.device)
        edge_index = data.edge_index.to(self.device)

        x_, s_ = self.model(x, edge_index)

        score = self.model.loss_func(x[:batch_size],
                                     x_[:batch_size],
                                     s[:batch_size, node_idx],
                                     s_[:batch_size],
                                     self.weight)

        loss = torch.mean(score)

        return loss, score.detach().cpu()
