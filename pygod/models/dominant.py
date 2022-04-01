# -*- coding: utf-8 -*-
"""Deep Anomaly Detection on Attributed Networks (DOMINANT)"""
# Author: Kay Liu <zliu234@uic.edu>
# License: BSD 2 clause

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj
from sklearn.utils.validation import check_is_fitted

from . import BaseDetector
from .basic_nn import GCN
from ..utils.metric import eval_roc_auc


class DOMINANT(BaseDetector):
    """
    DOMINANT (Deep Anomaly Detection on Attributed Networks)
    DOMINANT is an anomaly detector consisting of a shared graph
    convolutional encoder, a structure reconstruction decoder, and an
    attribute reconstruction decoder. The reconstruction mean square
    error of the decoders are defined as structure anomaly score and
    attribute anomaly score, respectively.

    See :cite:`ding2019deep` for details.

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
    def __init__(self,
                 hid_dim=64,
                 num_layers=4,
                 dropout=0.3,
                 weight_decay=0.,
                 act=F.relu,
                 alpha=0.8,
                 contamination=0.1,
                 lr=5e-3,
                 epoch=5,
                 gpu=0,
                 verbose=False):
        super(DOMINANT, self).__init__(contamination=contamination)

        # model param
        self.hid_dim = hid_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.act = act
        self.alpha = alpha

        # training param
        self.lr = lr
        self.epoch = epoch
        if gpu >= 0 and torch.cuda.is_available():
            self.device = 'cuda:{}'.format(gpu)
        else:
            self.device = 'cpu'

        # other param
        self.verbose = verbose
        self.model = None

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

        Returns
        -------
        self : object
            Fitted estimator.
        """

        x, adj, edge_index = self.process_graph(G)

        self.model = DOMINANT_Base(in_dim=x.shape[1],
                                   hid_dim=self.hid_dim,
                                   num_layers=self.num_layers,
                                   dropout=self.dropout,
                                   act=self.act).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=self.lr,
                                     weight_decay=self.weight_decay)
        score = None
        for epoch in range(self.epoch):
            self.model.train()
            x_, adj_ = self.model(x, edge_index)
            score = self.loss_func(x, x_, adj, adj_)
            loss = torch.mean(score)

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


class DOMINANT_Base(nn.Module):
    def __init__(self,
                 in_dim,
                 hid_dim,
                 num_layers,
                 dropout,
                 act):

        super(DOMINANT_Base, self).__init__()

        # split the number of layers for the encoder and decoders
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
                                  num_layers=decoder_layers - 1,
                                  out_channels=in_dim,
                                  dropout=dropout,
                                  act=act)

    def forward(self, x, edge_index):
        # encode
        h = self.shared_encoder(x, edge_index)
        # decode feature matrix
        x_ = self.attr_decoder(h, edge_index)
        # decode adjacency matrix
        h_ = self.struct_decoder(h, edge_index)
        adj_ = h_ @ h_.T

        # return reconstructed matrices
        return x_, adj_
