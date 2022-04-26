# -*- coding: utf-8 -*-
"""Contrastive Attributed Network Anomaly Detection with Data Augmentatin (CONAD)"""
# Author: Zhiming Xu <zhimng.xu@gmail.com>
# License: BSD 2 clause

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from sklearn.utils.validation import check_is_fitted
from copy import deepcopy
import numpy as np

from . import BaseDetector
from .basic_nn import GCN
from ..utils.metric import eval_roc_auc


class CONAD(BaseDetector):
    """
    CONAD (Contrastive Attributed Network Anomaly Detection)
    CONAD is an anomaly detector consisting of a shared graph
    convolutional encoder, a structure reconstruction decoder, and an
    attribute reconstruction decoder. The model is trained with both
    contrastive loss and structure/attribute reconstruction loss.
    The reconstruction mean square error of the decoders are defined
    as structure anomaly score and attribute anomaly score, respectively.

    See :cite:`xu2022contrastive` for details.

    Parameters
    ----------
    hid_dim :  int, optional
        Hidden dimension of model. Default: ``0``.
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
    alpha : float, optional
        Loss balance weight for attribute and structure.
        Default: ``0.5``.
    eta : float, optional
        Loss balance weight for contrastive and reconstruction.
        Default: ``0.5``.
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
    r : float, optional
        The rate of augmented anomalies. Default: ``.2``.
    m : int, optional
        For densely connected nodes, the number of
        edges to add. Default: ``50``.
    k : int, optional
        same as ``k`` in ``utils.outlier_generator.gen_attribute_outliers``.
        Default: ``50``.
    f : int, optional
        For disproportionate nodes, the scale factor applied
        on their attribute value. Default: ``10``.
    verbose : bool
        Verbosity mode. Turn on to print out log information.
        Default: ``False``.

    Examples
    --------
    >>> from pygod.models import CONAD
    >>> model = CONAD()
    >>> model.fit(data) # PyG graph data object
    >>> prediction = model.predict(data)
    """
    def __init__(self,
                 hid_dim=64,
                 num_layers=4,
                 dropout=0.3,
                 weight_decay=0.,
                 act=F.relu,
                 alpha=0.8,
                 eta=.5,
                 contamination=0.1,
                 lr=5e-3,
                 epoch=5,
                 gpu=0,
                 margin=.5,
                 r=.2,
                 m=50,
                 k=50,
                 f=10,
                 verbose=False):
        super(CONAD, self).__init__(contamination=contamination)

        # model param
        self.hid_dim = hid_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.act = act
        self.alpha = alpha
        self.eta = eta

        # training param
        self.lr = lr
        self.epoch = epoch
        if gpu >= 0 and torch.cuda.is_available():
            self.device = 'cuda:{}'.format(gpu)
        else:
            self.device = 'cpu'
        self.margin_loss_func = torch.nn.MarginRankingLoss(margin=margin)
        # other param
        self.verbose = verbose
        self.r = r
        self.m = m
        self.k = k
        self.f = f

    def fit(self, G, y_true=None):
        """
        Description
        -----------
        Fit detector with input data.

        Parameters
        ----------
        G : PyTorch Geometric Data instance (torch_geometric.data.Data)
            The input data.
        y_true : numpy.array, optional (default=None)
            The optional outlier ground truth labels used to monitor the
            training progress. They are not used to optimize the
            unsupervised model.
        kwargs :  parameters used to generate pseudo anomalies in
            `_data_augmentation`

        Returns
        -------
        self : object
            Fitted estimator.
        """
        
        x, adj, edge_index = self.process_graph(G)

        self.model = CONAD_Base(in_dim=x.shape[1],
                                hid_dim=self.hid_dim,
                                num_layers=self.num_layers,
                                dropout=self.dropout,
                                act=self.act).to(self.device)
        
        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=self.lr,
                                     weight_decay=self.weight_decay)
        # generate augmented graph
        x_aug, edge_index_aug, label_aug = self._data_augmentation(x, adj)
        
        score = None
        for epoch in range(self.epoch):
            self.model.train()
            h_aug = self.model.embed(x_aug, edge_index_aug)
            h = self.model.embed(x, edge_index)
            
            margin_loss = self.margin_loss_func(h, h, h_aug) * label_aug
            margin_loss = torch.mean(margin_loss)

            x_, adj_ = self.model.reconstruct(h, edge_index)
            score = self.loss_func(x, x_, adj, adj_)
            loss = self.eta * torch.mean(score) + (1 - self.eta) * margin_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if self.verbose:
                print("Epoch {:04d}: Loss {:.4f}"
                      .format(epoch, loss.item()), end='')
                if y_true is not None:
                    auc = eval_roc_auc(y_true, score.detach().cpu().numpy())
                    print(" | AUC {:.4f}".format(auc), end='')
                print()

        self.decision_scores_ = score.detach().cpu().numpy()
        self._process_decision_scores()
        return self

    def decision_function(self, G):
        """
        Description
        -----------
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

        # get needed data object from the input data
        x, adj, edge_index = self.process_graph(G)

        # enable the evaluation mode
        self.model.eval()

        # construct the vector for holding the reconstruction error
        x_, adj_ = self.model(x, edge_index)
        outlier_scores = self.loss_func(x, x_, adj, adj_)
        return outlier_scores.detach().cpu().numpy()

    def _data_augmentation(self, x, adj):
        """
        Description
        -----------
        Data augmentation on the input graph. Four types of
        pseudo anomalies will be injected:
            Attribute, deviated
            Attribute, disproportionate
            Structure, high-degree
            Structure, outlying
        
        Parameters
        -----------
        x : note attribute matrix
        adj : dense adjacency matrix

        Returns
        -------
        feat_aug, adj_aug, label_aug : augmented
            attribute matrix, adjacency matrix, and
            pseudo anomaly label to train contrastive
            graph representations
        """
        rate = self.r
        num_added_edge = self.m
        surround = self.k
        scale_factor = self.f
        
        adj_aug, feat_aug  = deepcopy(adj), deepcopy(x)
        num_nodes = adj_aug.shape[0]
        label_aug = torch.zeros(num_nodes, dtype=torch.int32)
        
        for i in range(num_nodes):
            prob = np.random.uniform()
            if prob > rate: continue
            label_aug[i] = 1
            one_fourth = np.random.randint(0, 4)
            if one_fourth == 0:
                # add clique
                new_neighbors = np.random.choice(np.arange(num_nodes), num_added_edge, replace=False)
                adj_aug[i, new_neighbors] = 1
                adj_aug[new_neighbors, i] = 1
            elif one_fourth == 1:
                # drop all connection
                neighbors = torch.nonzero(adj[i]).view(-1)
                if not neighbors.any():
                    continue
                adj_aug[i, neighbors] = 0
                adj_aug[neighbors, i] = 0
            elif one_fourth == 2:
                # attrs
                candidates = np.random.choice(np.arange(num_nodes), surround, replace=False)
                max_dev, max_idx = 0, i
                for c in candidates:
                    dev = torch.square(feat_aug[i]-feat_aug[c]).sum()
                    if dev > max_dev:
                        max_dev = dev
                        max_idx = c
                feat_aug[i] = feat_aug[max_idx]
            else:
                # scale attr
                prob = np.random.uniform(0, 1)
                if prob > 0.5:
                    feat_aug[i] *= scale_factor
                else:
                    feat_aug[i] /= scale_factor
        
        edge_index_aug = dense_to_sparse(adj_aug)[0].to(self.device)
        feat_aug = feat_aug.to(self.device)
        label_aug = label_aug.to(device=self.device)
        return feat_aug, edge_index_aug, label_aug

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
        edge_index : torch.Tensor
            Edge list of the graph.
        """
        edge_index = G.edge_index

        adj = to_dense_adj(edge_index)[0].to(self.device)

        edge_index = edge_index.to(self.device)
        adj = adj.to(self.device)
        x = G.x.to(self.device)

        # return data objects needed for the network
        return x, adj, edge_index

    def loss_func(self, x, x_, adj, adj_):
        # attribute reconstruction loss
        diff_attribute = torch.pow(x - x_, 2)
        attribute_errors = torch.sqrt(torch.sum(diff_attribute, 1))

        # structure reconstruction loss
        diff_structure = torch.pow(adj - adj_, 2)
        structure_errors = torch.sqrt(torch.sum(diff_structure, 1))

        score = self.alpha * attribute_errors \
                + (1 - self.alpha) * structure_errors
        return score


class CONAD_Base(nn.Module):
    def __init__(self,
                 in_dim,
                 hid_dim,
                 num_layers,
                 dropout,
                 act):

        super(CONAD_Base, self).__init__()

        decoder_layers = int(num_layers / 2)
        encoder_layers = num_layers - decoder_layers

        self.shared_encoder = GCN(in_channels=in_dim,
                                  hidden_channels=hid_dim,
                                  num_layers=encoder_layers,
                                  out_channels=hid_dim,
                                  dropout=dropout,
                                  act=act)

        self.attr_decoder = GCN(in_channels=hid_dim,
                                hidden_channels=hid_dim,
                                num_layers=decoder_layers,
                                out_channels=in_dim,
                                dropout=dropout,
                                act=act)

        self.struct_decoder = GCN(in_channels=hid_dim,
                                  hidden_channels=hid_dim,
                                  num_layers=decoder_layers-1,
                                  out_channels=in_dim,
                                  dropout=dropout,
                                  act=act)
    
    def embed(self, x, edge_index):
        h = self.shared_encoder(x, edge_index)
        return h

    def reconstruct(self, h, edge_index):
        # decode attribute matrix
        x_ = self.attr_decoder(h, edge_index)
        # decode structure matrix
        h_ = self.struct_decoder(h, edge_index)

        adj_ = h_ @ h_.T
        return x_, adj_

    def forward(self, x, edge_index):
        # encode
        h = self.shared_encoder(x, edge_index)
        # reconstruct
        x_, adj_ = self.reconstruct(h, edge_index)
        return x_, adj_