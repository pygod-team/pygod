# -*- coding: utf-8 -*-
"""Deep Outlier Aware Attributed Network Embedding (DONE)"""
# Author: Kay Liu <zliu234@uic.edu>
# License: BSD 2 clause

import torch
from torch_geometric.utils import to_dense_adj

from . import DeepDetector
from ..nn import DONEBase


class DONE(DeepDetector):
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
    contamination : float, optional
        Valid in (0., 0.5). The proportion of outliers in the dataset.
        Used when fitting to define the threshold on the decision
        function. Default: ``0.1``.
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

    Examples
    --------
    >>> from pygod.models import DONE
    >>> model = DONE()
    >>> model.fit(data)
    >>> prediction = model.predict(data)
    """

    def __init__(self,
                 hid_dim=64,
                 num_layers=4,
                 dropout=0.,
                 weight_decay=0.,
                 act=torch.nn.functional.relu,
                 w1=0.2,
                 w2=0.2,
                 w3=0.2,
                 w4=0.2,
                 w5=0.2,
                 contamination=0.1,
                 lr=4e-3,
                 epoch=100,
                 gpu=-1,
                 batch_size=0,
                 num_neigh=-1,
                 verbose=0,
                 **kwargs):
        super(DONE, self).__init__(hid_dim=hid_dim,
                                   num_layers=1,
                                   dropout=dropout,
                                   weight_decay=weight_decay,
                                   act=act,
                                   contamination=contamination,
                                   lr=lr,
                                   epoch=epoch,
                                   gpu=gpu,
                                   batch_size=batch_size,
                                   num_neigh=num_neigh,
                                   verbose=verbose,
                                   **kwargs)

        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.w4 = w4
        self.w5 = w5
        self.num_layers = num_layers

        self.attribute_score_ = None
        self.structural_score_ = None
        self.combined_score_ = None

    def process_graph(self, data):
        data.s = to_dense_adj(data.edge_index)[0]

    def init_model(self, **kwargs):
        self.attribute_score_ = torch.zeros(self.num_nodes)
        self.structural_score_ = torch.zeros(self.num_nodes)
        self.combined_score_ = torch.zeros(self.num_nodes)

        return DONEBase(x_dim=self.in_dim,
                        s_dim=self.num_nodes,
                        hid_dim=self.hid_dim,
                        num_layers=self.num_layers,
                        dropout=self.dropout,
                        act=self.act,
                        w1=self.w1,
                        w2=self.w2,
                        w3=self.w3,
                        w4=self.w4,
                        w5=self.w5,
                        **kwargs).to(self.device)

    def forward_model(self, data):
        batch_size = data.batch_size
        node_idx = data.node_idx

        x = data.x.to(self.device)
        s = data.s.to(self.device)
        edge_index = data.edge_index.to(self.device)

        x_, s_, h_a, h_s, dna, dns = self.model(x, s, edge_index)
        loss, oa, os, oc = self.model.loss_func(x[:batch_size],
                                                x_[:batch_size],
                                                s[:batch_size],
                                                s_[:batch_size],
                                                h_a[:batch_size],
                                                h_s[:batch_size],
                                                dna[:batch_size],
                                                dns[:batch_size])

        self.attribute_score_[node_idx[:batch_size]] = oa.detach().cpu()
        self.structural_score_[node_idx[:batch_size]] = os.detach().cpu()
        self.combined_score_[node_idx[:batch_size]] = oc.detach().cpu()

        return loss, ((oa + os + oc) / 3).detach().cpu()
