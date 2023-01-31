# -*- coding: utf-8 -*-
"""Deep Anomaly Detection on Attributed Networks (DOMINANT)"""
# Author: Kay Liu <zliu234@uic.edu>
# License: BSD 2 clause

import time
import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj
from torch_geometric.loader import NeighborLoader
from sklearn.utils.validation import check_is_fitted

from .base import DeepDetector
from ..nn import DOMINANTBase
from ..utils import validate_device, logger


class DOMINANT(DeepDetector):
    """
    DOMINANT (Deep Anomaly Detection on Attributed Networks) is an
    anomaly detector consisting of a shared graph convolutional
    encoder, a structure reconstruction decoder, and an attribute
    reconstruction decoder. The reconstruction mean square error of the
    decoders are defined as structure anomaly score and attribute
    anomaly score, respectively.

    See :cite:`ding2019deep` for details.

    Parameters
    ----------
    hid_dim :  int, optional
        Hidden dimension of model. Default: ``0``.
    num_layers : int, optional
        Total number of layers in model. A half (floor) of the layers
        are for the encoder, the other half (ceil) of the layers are
        for decoders. Default: ``4``.
    dropout : float, optional
        Dropout rate. Default: ``0.``.
    weight_decay : float, optional
        Weight decay (L2 penalty). Default: ``0.``.
    act : callable activation function or None, optional
        Activation function if not None.
        Default: ``torch.nn.functional.relu``.
    alpha : float, optional
        Loss balance weight for attribute and structure. ``None`` for
        balancing by standard deviation. Default: ``None``.
    contamination : float, optional
        Valid in (0., 0.5). The proportion of outliers in the data set.
        Used when fitting to define the threshold on the decision
        function. Default: ``0.1``.
    lr : float, optional
        Learning rate. Default: ``0.004``.
    epoch : int, optional
        Maximum number of training epoch. Default: ``5``.
    gpu : int
        GPU Index, -1 for using CPU. Default: ``0``.
    batch_size : int, optional
        Minibatch size, 0 for full batch training. Default: ``0``.
    num_neigh : int, optional
        Number of neighbors in sampling, -1 for all neighbors.
        Default: ``-1``.
    verbose : bool
        Verbosity mode. Turn on to print out log information.
        Default: ``False``.

    Examples
    --------
    >>> from pygod.models import DOMINANT
    >>> model = DOMINANT()
    >>> model.fit(data) # PyG graph data object
    >>> prediction = model.predict(data)
    """

    # def __init__(self,
    #              hid_dim=64,
    #              num_layers=4,
    #              dropout=0.3,
    #              weight_decay=0.,
    #              act=F.relu,
    #              alpha=None,
    #              contamination=0.1,
    #              lr=5e-3,
    #              epoch=5,
    #              gpu=0,
    #              batch_size=0,
    #              num_neigh=-1,
    #              verbose=False,
    #              **kwargs):
    #     super(DOMINANT, self).__init__(contamination=contamination)
    #
    #     # model param
    #     self.hid_dim = hid_dim
    #     self.num_layers = num_layers
    #     self.dropout = dropout
    #     self.weight_decay = weight_decay
    #     self.act = act
    #     self.alpha = alpha
    #
    #     # training param
    #     self.lr = lr
    #     self.epoch = epoch
    #     self.device = validate_device(gpu)
    #     self.batch_size = batch_size
    #     self.num_neigh = num_neigh
    #
    #     # other param
    #     self.verbose = verbose
    #     self.model = None
    #
    # def fit(self, G, y_true=None):
    #     """
    #     Fit detector with input data.
    #
    #     Parameters
    #     ----------
    #     G : torch_geometric.data.Data
    #         The input data.
    #     y_true : numpy.ndarray, optional
    #         The optional outlier ground truth labels used to monitor
    #         the training progress. They are not used to optimize the
    #         unsupervised model. Default: ``None``.
    #
    #     Returns
    #     -------
    #     self : object
    #         Fitted estimator.
    #     """
    #     G.node_idx = torch.arange(G.x.shape[0])
    #     G.s = to_dense_adj(G.edge_index)[0]
    #
    #     # automated balancing by std
    #     if self.alpha is None:
    #         self.alpha = torch.std(G.s).detach() / \
    #                      (torch.std(G.x).detach() + torch.std(G.s).detach())
    #
    #     if self.batch_size == 0:
    #         self.batch_size = G.x.shape[0]
    #     loader = NeighborLoader(G,
    #                             [self.num_neigh] * self.num_layers,
    #                             batch_size=self.batch_size)
    #
    #     self.model = DOMINANTBase(in_dim=G.x.shape[1],
    #                               hid_dim=self.hid_dim,
    #                               num_layers=self.num_layers,
    #                               dropout=self.dropout,
    #                               act=self.act).to(self.device)
    #
    #     optimizer = torch.optim.Adam(self.model.parameters(),
    #                                  lr=self.lr,
    #                                  weight_decay=self.weight_decay)
    #
    #     self.model.train()
    #     decision_scores = np.zeros(G.x.shape[0])
    #     for epoch in range(self.epoch):
    #         if self.verbose > 1:
    #             start_time = time.time()
    #         epoch_loss = 0
    #         for sampled_data in loader:
    #             batch_size = sampled_data.batch_size
    #             node_idx = sampled_data.node_idx
    #             x, s, edge_index = self.process_graph(sampled_data)
    #
    #             x_, s_ = self.model(x, edge_index)
    #             score = self.loss_func(x[:batch_size],
    #                                    x_[:batch_size],
    #                                    s[:batch_size, node_idx],
    #                                    s_[:batch_size])
    #             decision_scores[node_idx[:batch_size]] = score.detach() \
    #                 .cpu().numpy()
    #             loss = torch.mean(score)
    #             epoch_loss += loss.item() * batch_size
    #
    #             optimizer.zero_grad()
    #             loss.backward()
    #             optimizer.step()
    #
    #         logger(epoch=epoch,
    #                loss=epoch_loss / G.x.shape[0],
    #                pred=decision_scores,
    #                target=y_true,
    #                time=time.time() - start_time,
    #                verbose=self.verbose)
    #
    #     self.decision_scores_ = decision_scores
    #     self._process_decision_scores()
    #     return self
    #
    # def decision_function(self, G):
    #     """
    #     Predict raw outlier scores of PyG Graph G using the fitted detector.
    #     The outlier score of an input sample is computed based on the fitted
    #     detector. For consistency, outliers are assigned with
    #     higher anomaly scores.
    #
    #     Parameters
    #     ----------
    #     G : PyTorch Geometric Data instance (torch_geometric.data.Data)
    #         The input data.
    #
    #     Returns
    #     -------
    #     outlier_scores : numpy.ndarray
    #         The outlier score of shape :math:`N`.
    #     """
    #     check_is_fitted(self, ['model'])
    #     G.node_idx = torch.arange(G.x.shape[0])
    #     G.s = to_dense_adj(G.edge_index)[0]
    #
    #     loader = NeighborLoader(G,
    #                             [self.num_neigh] * self.num_layers,
    #                             batch_size=self.batch_size)
    #
    #     self.model.eval()
    #     outlier_scores = np.zeros(G.x.shape[0])
    #     for sampled_data in loader:
    #         batch_size = sampled_data.batch_size
    #         node_idx = sampled_data.node_idx
    #
    #         x, s, edge_index = self.process_graph(sampled_data)
    #
    #         x_, s_ = self.model(x, edge_index)
    #         score = self.loss_func(x[:batch_size],
    #                                x_[:batch_size],
    #                                s[:batch_size, node_idx],
    #                                s_[:batch_size])
    #
    #         outlier_scores[node_idx[:batch_size]] = score.detach() \
    #             .cpu().numpy()
    #     return outlier_scores

    def __init__(self,
                 in_dim=None,
                 hid_dim=64,
                 num_layers=4,
                 dropout=0.3,
                 weight_decay=0.,
                 act=torch.relu,
                 contamination=0.1,
                 lr=5e-3,
                 epoch=5,
                 gpu=-1,
                 batch_size=0,
                 num_neigh=-1,
                 verbose=False,
                 weight=0.5,
                 **kwargs):
        self.weight = weight
        super(DOMINANT, self).__init__(in_dim=in_dim,
                                       hid_dim=hid_dim,
                                       num_layers=num_layers,
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

    def _process_graph(self, G):
        """
        Process the raw PyG data object into a tuple of sub data
        objects needed for the model.

        Parameters
        ----------
        G : PyTorch Geometric Data instance (torch_geometric.data.Data)
            The input data.
        """
        G.s = to_dense_adj(G.edge_index)[0]

    def _init_nn(self, **kwargs):
        self.model = DOMINANTBase(in_dim=self.in_dim,
                                  hid_dim=self.hid_dim,
                                  num_layers=self.num_layers,
                                  dropout=self.dropout,
                                  act=self.act,
                                  **kwargs).to(self.device)

    def _forward_nn(self, data):
        batch_size = data.batch_size

        x = data.x.to(self.device)
        s = data.s.to(self.device)
        edge_index = data.edge_index.to(self.device)

        x_, s_ = self.model(x, edge_index)
        scores = self.model.loss_func(x[:batch_size],
                                      x_[:batch_size],
                                      s[:batch_size],
                                      s_[:batch_size],
                                      self.weight)

        loss = torch.mean(scores)

        return loss, scores.detach().cpu().numpy()
