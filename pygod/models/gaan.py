# -*- coding: utf-8 -*-
"""Generative Adversarial Attributed Network Anomaly Detection (GAAN)"""
# Author: Ruitong Zhang <rtzhang@buaa.edu.cn>, Kay Liu <zliu234@uic.edu>
# License: BSD 2 clause

import torch
import warnings
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import MLP
from torch_geometric.utils import to_dense_adj
from torch_geometric.loader import NeighborLoader
from sklearn.utils.validation import check_is_fitted

from ..nn import GAANBase
from . import DeepDetector


class GAAN(DeepDetector):
    """
    GAAN (Generative Adversarial Attributed Network Anomaly
    Detection) is a generative adversarial attribute network anomaly
    detection framework, including a generator module, an encoder
    module, a discriminator module, and uses anomaly evaluation
    measures that consider sample reconstruction error and real sample
    recognition confidence to make predictions. This model is
    transductive only.

    See :cite:`chen2020generative` for details.

    Parameters
    ----------
    noise_dim :  int, optional
        Dimension of the Gaussian random noise. Defaults: ``16``.
    hid_dim :  int, optional
        Hidden dimension of MLP later 3. Defaults: ``64``.
    generator_layers : int, optional
        Number of layers in generator. Defaults: ``2``.
    encoder_layers : int, optional
        Number of layers in encoder. Defaults: ``2``.
    dropout : float, optional
        Dropout rate. Defaults: ``0.``.
    weight_decay : float, optional
        Weight decay (L2 penalty). Defaults: ``0.``.
    act : callable activation function or None, optional
        Activation function if not None.
        Defaults: ``torch.nn.functional.relu``.
    alpha : float, optional
        Loss balance weight for attribute and structure. ``None`` for
        balancing by standard deviation. Default: ``None``.
    contamination : float, optional
        Valid in (0., 0.5). The proportion of outliers in the data set.
        Used when fitting to define the threshold on the decision
        function. Defaults: ``0.05``.
    lr : float, optional
        Learning rate. Defaults: ``0.005``.
    epoch : int, optional
        Maximum number of training epoch. Defaults: ``10``.
    gpu : int
        GPU Index, -1 for using CPU. Defaults: ``-1``.
    batch_size : int, optional
        Minibatch size, 0 for full batch training. Default: ``0``.
    num_neigh : int, optional
        Number of neighbors in sampling, -1 for all neighbors.
        Default: ``-1``.
    verbose : bool
        Verbosity mode. Turn on to print out log information.
        Defaults: ``False``.

    Examples
    --------
    >>> from pygod.models import GAAN
    >>> model = GAAN()
    >>> model.fit(data) # PyG graph data object
    >>> prediction = model.predict(None)
    """

    def __init__(self,
                 noise_dim=16,
                 hid_dim=64,
                 generator_layers=2,
                 encoder_layers=2,
                 dropout=0.,
                 weight_decay=0.,
                 act=F.relu,
                 backbone=MLP,
                 contamination=0.1,
                 lr=4e-3,
                 epoch=100,
                 gpu=-1,
                 batch_size=0,
                 num_neigh=0,
                 weight=0.5,
                 verbose=0,
                 **kwargs):

        self.noise_dim = noise_dim
        self.weight = weight
        self.generator_layers = generator_layers
        self.encoder_layers = encoder_layers

        if num_neigh != 0:
            warnings.warn('MLP in GAAN does not use neighbor information.')
            num_neigh = 0
        if backbone != MLP:
            warnings.warn('GAAN can only use MLP as the backbone.')
            backbone = MLP

        super(GAAN, self).__init__(
            hid_dim=hid_dim,
            num_layers=generator_layers + encoder_layers,
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
            gan=True,
            **kwargs)

    def process_graph(self, data):
        GAANBase.process_graph(data)

    def init_model(self, **kwargs):
        return GAANBase(in_dim=self.in_dim,
                        noise_dim=self.noise_dim,
                        hid_dim=self.hid_dim,
                        generator_layers=self.generator_layers,
                        encoder_layers=self.encoder_layers,
                        dropout=self.dropout,
                        act=self.act,
                        **kwargs).to(self.device)

    def forward_model(self, data):
        batch_size = data.batch_size

        x = data.x.to(self.device)
        s = data.s.to(self.device)
        edge_index = data.edge_index.to(self.device)

        noise = torch.randn(x.shape[0], self.noise_dim).to(self.device)

        x_, a, a_ = self.model(x, noise)

        loss_g = self.model.loss_func_g(a_[edge_index[0], edge_index[1]])

        self.opt_g.zero_grad()
        loss_g.backward()
        self.opt_g.step()

        self.epoch_loss_g += loss_g.item() * batch_size

        loss = self.model.loss_func_ed(a[edge_index[0], edge_index[1]],
                                       a_[edge_index[0], edge_index[
                                           1]].detach())

        score = self.model.score_func(x=x,
                                      x_=x_,
                                      s=s,
                                      s_=a,
                                      weight=self.weight,
                                      pos_weight_s=1,
                                      bce_s=True)

        return loss, score.detach().cpu()
