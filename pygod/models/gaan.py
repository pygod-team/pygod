# -*- coding: utf-8 -*-
"""Generative Adversarial Attributed Network Anomaly Detection (GAAN)"""
# Author: Ruitong Zhang <rtzhang@buaa.edu.cn>, Kay Liu <zliu234@uic.edu>
# License: BSD 2 clause

import torch
import warnings
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj
from torch_geometric.loader import NeighborLoader
from sklearn.utils.validation import check_is_fitted

from .basic_nn import MLP
from . import BaseDetector
from ..utils import validate_device
from ..metrics import eval_roc_auc


class GAAN(BaseDetector):
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
                 dropout=0.1,
                 weight_decay=0.01,
                 act=F.relu,
                 alpha=None,
                 contamination=0.1,
                 lr=0.01,
                 epoch=5,
                 gpu=0,
                 batch_size=0,
                 num_neigh=-1,
                 verbose=False):
        super(GAAN, self).__init__(contamination=contamination)

        # model param
        self.noise_dim = noise_dim
        self.hid_dim = hid_dim
        self.generator_layers = generator_layers
        self.encoder_layers = encoder_layers
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.act = act
        self.alpha = alpha

        # training param
        self.lr = lr
        self.epoch = epoch
        self.device = validate_device(gpu)
        self.batch_size = batch_size
        self.num_neigh = num_neigh

        # other param
        self.verbose = verbose
        self.model = None

    def fit(self, G, y_true=None):
        """
        Fit detector with input data.

        Parameters
        ----------
        G : torch_geometric.data.Data
            The input data.
        y_true : numpy.ndarray, optional
            The optional outlier ground truth labels used to monitor
            the training progress. They are not used to optimize the
            unsupervised model. Default: ``None``.

        Returns
        -------
        self : object
            Fitted estimator.
        """

        G.node_idx = torch.arange(G.x.shape[0])
        adj = to_dense_adj(G.edge_index)[0]

        # automated balancing by std
        if self.alpha is None:
            self.alpha = torch.std(adj).detach() / \
                         (torch.std(G.x).detach() + torch.std(adj).detach())

        if self.batch_size == 0:
            self.batch_size = G.x.shape[0]
        loader = NeighborLoader(G,
                                [self.num_neigh],
                                batch_size=self.batch_size)

        self.model = GAAN_Base(in_dim=G.x.shape[1],
                               noise_dim=self.noise_dim,
                               hid_dim=self.hid_dim,
                               generator_layers=self.generator_layers,
                               encoder_layers=self.encoder_layers,
                               dropout=self.dropout,
                               act=self.act).to(self.device)

        optimizer_ed = torch.optim.Adam(self.model.encoder.parameters(),
                                        lr=self.lr,
                                        weight_decay=self.weight_decay)

        optimizer_g = torch.optim.Adam(self.model.generator.parameters(),
                                       lr=self.lr,
                                       weight_decay=self.weight_decay)

        self.model.train()
        decision_scores = np.zeros(G.x.shape[0])
        for epoch in range(self.epoch):
            epoch_loss_g = 0
            epoch_loss_ed = 0
            for sampled_data in loader:
                batch_size = sampled_data.batch_size
                node_idx = sampled_data.node_idx
                x, edge_index = self.process_graph(sampled_data)

                # Generate noise for constructing fake attribute
                gaussian_noise = torch.randn(x.shape[0], self.noise_dim).to(
                    self.device)

                # train the model
                x_, a, a_ = self.model(x, gaussian_noise, edge_index)

                loss_g = self._loss_func_g(a_[edge_index[0], edge_index[1]])
                optimizer_g.zero_grad()
                loss_g.backward()
                optimizer_g.step()

                loss_ed = self._loss_func_ed(a[edge_index[0], edge_index[1]],
                                             a_[edge_index[0], edge_index[1]]
                                             .detach())
                optimizer_ed.zero_grad()
                loss_ed.backward()
                optimizer_ed.step()

                score = self._score_func(x,
                                         x_,
                                         a,
                                         edge_index,
                                         batch_size)
                epoch_loss_g += loss_g.item() * batch_size
                epoch_loss_ed += loss_ed.item() * batch_size
                decision_scores[node_idx[:batch_size]] = score.detach() \
                    .cpu().numpy()

            if self.verbose:
                print("Epoch {:04d}: Loss G {:.4f} | Loss ED {:4f}"
                      .format(epoch, epoch_loss_g / G.x.shape[0],
                              epoch_loss_ed / G.x.shape[0]), end='')
                if y_true is not None:
                    auc = eval_roc_auc(y_true, decision_scores)
                    print(" | AUC {:.4f}".format(auc), end='')
                print()

        self.decision_scores_ = decision_scores
        self._process_decision_scores()

        return self

    def decision_function(self, G):
        """
        Predict raw anomaly score using the fitted detector.
        Outliers are assigned with larger anomaly scores.

        Parameters
        ----------
        G : PyTorch Geometric Data instance (torch_geometric.data.Data)
            The input data.

        Returns
        -------
        outlier_scores : numpy.ndarray
            The anomaly score of shape :math:`N`.
        """
        check_is_fitted(self, ['model'])

        if G is not None:
            warnings.warn('The model is transductive only. '
                          'Training data is used to predict')

        outlier_scores = self.decision_scores_

        return outlier_scores

    def _loss_func_g(self, a_):
        loss_g = F.binary_cross_entropy(a_, torch.ones_like(a_))
        return loss_g

    def _loss_func_ed(self, a, a_):
        loss_r = F.binary_cross_entropy(a, torch.ones_like(a))
        loss_f = F.binary_cross_entropy(a_, torch.zeros_like(a_))
        return (loss_f + loss_r) / 2

    def _score_func(self, x, x_, a, edge_index, batch_size):
        # attribute reconstruction loss
        diff_attribute = torch.pow(x[:batch_size] - x_[:batch_size], 2)
        attribute_errors = torch.sqrt(torch.sum(diff_attribute, 1))

        adj = to_dense_adj(edge_index)[0]
        # structure reconstruction loss
        structure_errors = torch.sum(adj *
            F.binary_cross_entropy(a, torch.ones_like(a), reduction='none')
            , 1)[:batch_size]

        score = self.alpha * attribute_errors + (
                1 - self.alpha) * structure_errors

        return score

    def process_graph(self, G):
        """
        Process the raw PyG data object into a tuple of sub data
        objects needed for the model.

        Parameters
        ----------
        G : PyTorch Geometric Data instance (torch_geometric.data.Data)
            The input data.

        Returns
        -------
        X : torch.Tensor
            Attribute (feature) of nodes.
        edge_index : torch.Tensor
            Edge list of the graph.
        """
        # data objects needed for the network
        edge_index = G.edge_index.to(self.device)
        x = G.x.to(self.device)

        return x, edge_index


class GAAN_Base(nn.Module):
    def __init__(self,
                 in_dim,
                 noise_dim,
                 hid_dim,
                 generator_layers,
                 encoder_layers,
                 dropout,
                 act):
        super(GAAN_Base, self).__init__()

        self.generator = MLP(in_channels=noise_dim,
                             hidden_channels=hid_dim,
                             out_channels=in_dim,
                             num_layers=generator_layers,
                             dropout=dropout,
                             act=act)

        self.encoder = MLP(in_channels=in_dim,
                           hidden_channels=hid_dim,
                           out_channels=hid_dim,
                           num_layers=encoder_layers,
                           dropout=dropout,
                           act=act)

    def forward(self, x, noise, edge_index):
        x_ = self.generator(noise)

        z = self.encoder(x)
        z_ = self.encoder(x_)

        a = torch.sigmoid((z @ z.T))
        a_ = torch.sigmoid((z_ @ z_.T))

        return x_, a, a_
