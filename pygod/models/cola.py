# -*- coding: utf-8 -*-
"""Anomaly Detection on Attributed Networks via Contrastive
Self-Supervised Learning (CoLA)
The code is adapted from the source code from paper authors:
https://github.com/GRAND-Lab/CoLA"""
# Author: Canyu Chen <cchen151@hawk.iit.edu>
# License: BSD 2 clause

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
import random
import os

from torch_geometric.utils import to_dense_adj
from torch_cluster import random_walk

from . import BaseDetector
from .basic_nn import Vanilla_GCN as GCN
from ..utils.utility import validate_device
from ..utils.metric import eval_roc_auc

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def generate_rw_subgraph(pyg_graph, nb_nodes, subgraph_size):
    """Generate subgraph with random walk algorithm."""
    row, col = pyg_graph.edge_index
    all_idx = torch.tensor(list(range(nb_nodes)))
    traces = random_walk(row, col, all_idx, walk_length=3)
    subv = traces.tolist()
    return subv


class CoLA(BaseDetector):
    """
    CoLA (Anomaly Detection on Attributed Networks via Contrastive
    Self-Supervised Learning) is a contrastive self-supervised learning
    based method for graph anomaly detection. (beta)

    See :cite:`liu2021anomaly` for details.

    Parameters
    ----------
    lr : float, optional
        Learning rate. Default: ``1e-3``.
    epoch : int, optional
        Maximum number of training epoch. Default: ``10``.
    embedding_dim : int, optional
        The node embedding dimension obtained by the GCN module of CoLA.
        Default: ``64``.
    negsamp_ratio : int, optional
        Number of negative samples for each instance used by the contrastive
        learning module. Default: ``1``.
    readout : str, optional
        The readout layer type used by CoLA model. Default: ``avg`` .
    weight_decay : float, optional
        Weight decay (L2 penalty). Default: ``0.``.
    batch_size : int, optional
        Minibatch size, 0 for full batch training. Default: ``0``.
    subgraph_size : int, optional
        Number of nodes in the subgraph sampled by random walk. Default: ``4``.
    contamination : float, optional
        Valid in (0., 0.5). The proportion of outliers in the data set.
        Used when fitting to define the threshold on the decision
        function. Default: ``0.1``.
    gpu : int, optional
        GPU Index, -1 for using CPU. Default: ``0``.
    verbose : bool, optional
        Verbosity mode. Turn on to print out log information.
        Default: ``False``.

    Examples
    --------
    >>> from pygod.models import CoLA
    >>> model = CoLA()
    >>> model.fit(data)
    >>> prediction = model.predict(data)
    """

    def __init__(self,
                 lr=1e-3,
                 epoch=10,
                 embedding_dim=64,
                 negsamp_ratio=1,
                 readout='avg',
                 weight_decay=0.,
                 batch_size=0,
                 subgraph_size=4,
                 contamination=0.1,
                 gpu=0,
                 verbose=False):
        super(CoLA, self).__init__(contamination=contamination)

        self.lr = lr
        self.num_epoch = epoch
        self.embedding_dim = embedding_dim
        self.negsamp_ratio = negsamp_ratio
        self.readout = readout
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.subgraph_size = subgraph_size
        self.device = validate_device(gpu)

        self.verbose = verbose
        self.model = None

    def fit(self, G, y_true=None):
        """
        Fit detector with input data.

        Parameters
        ----------
        G : PyTorch Geometric Data instance (torch_geometric.data.Data)
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

        x, adj = self.process_graph(G)

        # Initialize model and optimiser
        self.model = CoLA_Base(self.feat_dim,
                               self.embedding_dim,
                               'prelu',
                               self.negsamp_ratio,
                               self.readout,
                               self.subgraph_size,
                               self.device).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=self.lr,
                                     weight_decay=self.weight_decay)

        if self.batch_size:
            batch_num = self.num_nodes // self.batch_size + 1
        else:  # full batch training
            batch_num = 1

        multi_epoch_ano_score = np.zeros((self.num_epoch, self.num_nodes))

        for epoch in range(self.num_epoch):

            self.model.train()
            epoch_loss = 0
            all_idx = list(range(self.num_nodes))
            random.shuffle(all_idx)
            decision_scores = np.zeros(self.num_nodes)

            subgraphs = generate_rw_subgraph(G, self.num_nodes,
                                             self.subgraph_size)

            for batch_idx in range(batch_num):

                is_final_batch = (batch_idx == (batch_num - 1))

                if not is_final_batch:
                    idx = all_idx[
                          batch_idx *
                          self.batch_size: (batch_idx + 1) * self.batch_size]
                else:
                    idx = all_idx[batch_idx * self.batch_size:]

                cur_batch_size = len(idx)

                output = self.model(x, adj, idx, subgraphs, cur_batch_size)
                loss = self.loss_function(output, cur_batch_size)

                loss = torch.mean(loss)
                epoch_loss += loss.item() * cur_batch_size

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # compute the anomaly score
                logits = torch.sigmoid(torch.squeeze(output))
                ano_score = - (logits[:cur_batch_size] - logits[
                                cur_batch_size:]).detach().cpu().numpy()
                decision_scores[idx] = ano_score

            multi_epoch_ano_score[epoch, :] = decision_scores

            if self.verbose:
                print("Epoch {:04d}: Loss {:.4f}"
                      .format(epoch, epoch_loss / G.x.shape[0]), end='')
                if y_true is not None:
                    auc = eval_roc_auc(y_true, decision_scores)
                    print(" | AUC {:.4f}".format(auc), end='')
                print()

        ano_score_final = np.mean(multi_epoch_ano_score, axis=0)

        self.decision_scores_ = ano_score_final
        self._process_decision_scores()
        return self

    def decision_function(self, G, rounds=10):
        """
        Predict raw anomaly score using the fitted detector. Outliers
        are assigned with larger anomaly scores.

        Parameters
        ----------
        G : PyTorch Geometric Data instance (torch_geometric.data.Data)
            The input data.
        rounds : int, optional
            Number of rounds to generate the decision score. Default: ``10``.

        Returns
        -------
        outlier_scores : numpy.ndarray
            The anomaly score of shape :math:`N`.
        """

        x, adj = self.process_graph(G)

        if self.batch_size:
            batch_num = self.num_nodes // self.batch_size + 1
        else:  # full batch training
            batch_num = 1

        multi_round_ano_score = np.zeros((rounds, self.num_nodes))

        # enable the evaluation mode
        self.model.eval()

        for r in range(rounds):

            all_idx = list(range(self.num_nodes))
            random.shuffle(all_idx)

            subgraphs = generate_rw_subgraph(G, self.num_nodes,
                                             self.subgraph_size)

            for batch_idx in range(batch_num):

                is_final_batch = (batch_idx == (batch_num - 1))

                if not is_final_batch:
                    idx = all_idx[
                          batch_idx *
                          self.batch_size: (batch_idx + 1) * self.batch_size]
                else:
                    idx = all_idx[batch_idx * self.batch_size:]

                cur_batch_size = len(idx)

                with torch.no_grad():
                    output = self.model(x, adj, idx, subgraphs, cur_batch_size)
                logits = torch.sigmoid(torch.squeeze(output))
                ano_score = - (logits[:cur_batch_size] - logits[
                                cur_batch_size:]).cpu().numpy()
                multi_round_ano_score[r, idx] = ano_score

        ano_score_final = np.mean(multi_round_ano_score, axis=0)

        return ano_score_final

    def process_graph(self, G):
        """
        Description
        -----------
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
        adj : torch.Tensor
            Adjacency matrix of the graph.
        """
        self.num_nodes = G.x.shape[0]
        self.feat_dim = G.x.shape[1]
        adj = to_dense_adj(G.edge_index)[0]
        adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
        adj = (adj + sp.eye(adj.shape[0])).todense()

        x = torch.FloatTensor(G.x[np.newaxis]).to(self.device)
        adj = torch.FloatTensor(adj[np.newaxis]).to(self.device)

        # return data objects needed for the network
        return x, adj

    def loss_function(self, logits, batch_size):

        b_xent = nn.BCEWithLogitsLoss(reduction='none',
                                      pos_weight=torch.tensor(
                                          [self.negsamp_ratio]))

        lbl = torch.unsqueeze(
            torch.cat((torch.ones(batch_size),
                       torch.zeros(batch_size * self.negsamp_ratio))), 1)

        score = b_xent(logits.cpu(), lbl.cpu())

        return score


class CoLA_Base(nn.Module):
    def __init__(self,
                 n_in,
                 n_h,
                 activation,
                 negsamp_round,
                 readout,
                 subgraph_size,
                 device):

        super(CoLA_Base, self).__init__()
        self.n_in = n_in
        self.subgraph_size = subgraph_size
        self.device = device
        self.readout = readout
        self.gcn = GCN(n_in, n_h, activation).to(self.device)
        self.disc = Discriminator(n_h, negsamp_round)

    def forward(self, x, adj, idx, subgraphs, batch_size, sparse=False):

        batch_adj = []
        batch_feature = []
        added_adj_zero_row = torch.zeros(
            (batch_size, 1, self.subgraph_size)).to(self.device)
        added_adj_zero_col = torch.zeros(
            (batch_size, self.subgraph_size + 1, 1)).to(self.device)
        added_adj_zero_col[:, -1, :] = 1.
        added_feat_zero_row = torch.zeros((batch_size, 1,
                                           self.n_in)).to(self.device)

        for i in idx:
            cur_adj = adj[:, subgraphs[i], :][:, :, subgraphs[i]]
            cur_feat = x[:, subgraphs[i], :]
            batch_adj.append(cur_adj)
            batch_feature.append(cur_feat)

        batch_adj = torch.cat(batch_adj)
        batch_adj = torch.cat((batch_adj, added_adj_zero_row), dim=1)
        batch_adj = torch.cat((batch_adj, added_adj_zero_col), dim=2)
        batch_feature = torch.cat(batch_feature)
        batch_feature = torch.cat(
            (batch_feature[:, :-1, :],
             added_feat_zero_row,
             batch_feature[:, -1:, :]), dim=1)

        h_1 = self.gcn(batch_feature, batch_adj, sparse)

        if self.readout == 'max':
            h_mv = h_1[:, -1, :]
            c = torch.max(h_1[:, : -1, :], 1).values
        elif self.readout == 'min':
            h_mv = h_1[:, -1, :]
            c = torch.min(h_1[:, : -1, :], 1).values
        elif self.readout == 'avg':
            h_mv = h_1[:, -1, :]
            c = torch.mean(h_1[:, : -1, :], 1)
        elif self.readout == 'weighted_sum':
            seq, query = h_1[:, : -1, :], h_1[:, -2: -1, :],
            query = query.permute(0, 2, 1)
            sim = torch.matmul(seq, query)
            sim = F.softmax(sim, dim=1)
            sim = sim.repeat(1, 1, 64)
            out = torch.mul(seq, sim)
            c = torch.sum(out, 1)
            h_mv = h_1[:, -1, :]

        ret = self.disc(c, h_mv)

        return ret


class Discriminator(nn.Module):
    def __init__(self, n_h, negsamp_ratio):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

        self.negsamp_ratio = negsamp_ratio

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl):
        scs = []
        # positive
        scs.append(self.f_k(h_pl, c))

        # negative
        c_mi = c
        for _ in range(self.negsamp_ratio):
            c_mi = torch.cat((c_mi[-2:-1, :], c_mi[:-1, :]), 0)
            scs.append(self.f_k(h_pl, c_mi))

        logits = torch.cat(tuple(scs))

        return logits
