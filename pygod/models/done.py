# -*- coding: utf-8 -*-
"""Deep Outlier Aware Attributed Network Embedding (DONE)"""
# Author: Kay Liu <zliu234@uic.edu>
# License: BSD 2 clause

import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj
from torch_geometric.loader import NeighborLoader
from sklearn.utils.validation import check_is_fitted

from . import DeepDetector
from ..nn import DONEBase
from ..metrics import eval_roc_auc


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
        Hidden dimension for both attribute autoencoder and structure
        autoencoder. Default: ``0``.
    num_layers : int, optional
        Total number of layers in model. A half (ceil) of the layers
        are for the encoder, the other half (floor) of the layers are
        for decoders. Default: ``4``.
    dropout : float, optional
        Dropout rate. Default: ``0.``.
    weight_decay : float, optional
        Weight decay (L2 penalty). Default: ``0.``.
    act : callable activation function or None, optional
        Activation function if not None.
        Default: ``torch.nn.functional.relu``.
    a1 : float, optional
        Loss balance weight for structure proximity.
        Default: ``0.2``.
    a2 : float, optional
        Loss balance weight for structure homophily.
        Default: ``0.2``.
    a3 : float, optional
        Loss balance weight for attribute proximity.
        Default: ``0.2``.
    a4 : float, optional
        Loss balance weight for attribute proximity.
        Default: ``0.2``.
    a5 : float, optional
        Loss balance weight for combination.
        Default: ``0.2``.
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
    >>> from pygod.models import DONE
    >>> model = DONE()
    >>> model.fit(data)
    >>> prediction = model.predict(data)
    """

    def __init__(self,
                 in_dim=None,
                 hid_dim=32,
                 num_layers=4,
                 dropout=0.,
                 weight_decay=0.,
                 act=F.leaky_relu,
                 w1=0.2,
                 w2=0.2,
                 w3=0.2,
                 w4=0.2,
                 w5=0.2,
                 contamination=0.1,
                 lr=5e-3,
                 epoch=5,
                 gpu=0,
                 batch_size=0,
                 num_neigh=-1,
                 verbose=False,
                 **kwargs):
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.w4 = w4
        self.w5 = w5
        super(DONE, self).__init__(in_dim=in_dim,
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
    #     if self.batch_size == 0:
    #         self.batch_size = G.x.shape[0]
    #     loader = NeighborLoader(G,
    #                             [self.num_neigh],
    #                             batch_size=self.batch_size)
    #
    #     self.model = DONE_Base(x_dim=G.x.shape[1],
    #                            s_dim=G.s.shape[1],
    #                            hid_dim=self.hid_dim,
    #                            num_layers=self.num_layers,
    #                            dropout=self.dropout,
    #                            act=self.act).to(self.device)
    #
    #     optimizer = torch.optim.Adam(self.model.parameters(),
    #                                  lr=self.lr,
    #                                  weight_decay=self.weight_decay)
    #
    #     self.model.train()
    #     decision_scores = np.zeros(G.x.shape[0])
    #     for epoch in range(self.epoch):
    #         epoch_loss = 0
    #         for sampled_data in loader:
    #             batch_size = sampled_data.batch_size
    #             node_idx = sampled_data.node_idx
    #             x, s, edge_index = self.process_graph(sampled_data)
    #
    #             x_, s_, h_a, h_s, dna, dns = self.model(x, s, edge_index)
    #             score, loss = self.loss_func(x[:batch_size],
    #                                          x_[:batch_size],
    #                                          s[:batch_size],
    #                                          s_[:batch_size],
    #                                          h_a[:batch_size],
    #                                          h_s[:batch_size],
    #                                          dna[:batch_size],
    #                                          dns[:batch_size])
    #             epoch_loss += loss.item() * batch_size
    #             decision_scores[node_idx[:batch_size]] = score.detach() \
    #                 .cpu().numpy()
    #
    #             optimizer.zero_grad()
    #             loss.backward()
    #             optimizer.step()
    #
    #         if self.verbose:
    #             print("Epoch {:04d}: Loss {:.4f}"
    #                   .format(epoch, epoch_loss / G.x.shape[0]), end='')
    #             if y_true is not None:
    #                 auc = eval_roc_auc(y_true, decision_scores)
    #                 print(" | AUC {:.4f}".format(auc), end='')
    #             print()
    #
    #     self.decision_scores_ = decision_scores
    #     self._process_decision_scores()
    #     return self

    def decision_function(self, G):
        """

        Predict raw anomaly score using the fitted detector. Outliers
        are assigned with larger anomaly scores.

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
        G.node_idx = torch.arange(G.x.shape[0])
        G.s = to_dense_adj(G.edge_index)[0]
        if self.batch_size == 0:
            self.batch_size = G.x.shape[0]
        loader = NeighborLoader(G,
                                [self.num_neigh],
                                batch_size=self.batch_size)

        self.model.eval()
        outlier_scores = np.zeros(G.x.shape[0])
        for sampled_data in loader:
            batch_size = sampled_data.batch_size
            node_idx = sampled_data.node_idx

            x, s, edge_index = self.process_graph(G)

            x_, s_, h_a, h_s, dna, dns = self.model(x, s, edge_index)
            score, _ = self.loss_func(x[:batch_size],
                                      x_[:batch_size],
                                      s[:batch_size],
                                      s_[:batch_size],
                                      h_a[:batch_size],
                                      h_s[:batch_size],
                                      dna[:batch_size],
                                      dns[:batch_size])

            outlier_scores[node_idx[:batch_size]] = score.detach() \
                .cpu().numpy()
        return outlier_scores

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
        x : torch.Tensor
            Attribute (feature) of nodes.
        s : torch.Tensor
            Adjacency matrix of the graph.
        edge_index : torch.Tensor
            Edge list of the graph.
        """
        s = G.s.to(self.device)
        edge_index = G.edge_index.to(self.device)
        x = G.x.to(self.device)

        return x, s, edge_index

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
        self.model = DONEBase(in_dim=self.in_dim,
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
        scores = self.model.loss(x[:batch_size],
                                 x_[:batch_size],
                                 s[:batch_size],
                                 s_[:batch_size],
                                 self.weight)

        loss = torch.mean(scores)

        return loss, scores.detach().cpu().numpy()
