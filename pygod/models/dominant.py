# -*- coding: utf-8 -*-
"""Deep Anomaly Detection on Attributed Networks (Dominant)
"""
# Author: Zekuan Liu <zliu234@uic.edu>
# License: BSD 2 clause

import torch
import argparse
import os.path as osp
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor
from torch_geometric.nn import GCNConv
from sklearn.metrics import roc_auc_score
from torch_geometric.datasets import AttributedGraphDataset
from sklearn.utils.validation import check_is_fitted

from .base import BaseDetector


class Dominant(BaseDetector):
    """Let us decide the documentation later
    Dominant(Deep Anomaly Detection on Attributed Networks)
    Dominant is an anomaly detector consisting of a shared graph
    convolutional encoder, a structure reconstruction decoder, and an attribute
    reconstruction decoder. The reconstruction mean sqare error of the decoders
    are defined as structure anomaly score and attribute anomaly score,
    respectively.

    See :cite:`ding2019deep` for details.


    Parameters
    ----------
    contamination : float in (0., 0.5), optional (default=0.1)
        The amount of contamination of the data set,
        i.e. the proportion of outliers in the data set. Used when fitting to
        define the threshold on the decision function.
    """

    def __init__(self, contamination=0.1):
        super(Dominant, self).__init__(contamination=contamination)

    def fit(self, G, args):
        """Fit detector. y is ignored in unsupervised methods.

        Parameters
        ----------
        G : PyTorch Geometric Data instance (torch_geometric.data.Data)
            The input graph.

        args : argparse object.
            Corresponding hyperparameters

        Returns
        -------
        self : object
            Fitted estimator.
        """
        # todo: need to decide which parameters are needed

        # 1. first call the data process
        adj, adj_label, attrs, labels = self.process_graph(G, args)

        # 2. set the parameters needed for the network from args.
        self.hidden_size = args.hidden_size
        self.dropout = args.dropout

        # 3. initialize the detection model
        self.model = Dominant_Base(feat_size=self.feat_size,
                                   hidden_size=self.hidden_size,
                                   dropout=self.dropout)

        # 4. to see if use GPU
        if args.device == 'cuda':
            device = torch.device(args.device)
            adj = adj.to(device)
            adj_label = adj_label.to(device)
            attrs = attrs.to(device)
            self.model = self.model.cuda()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)

        for epoch in range(args.epoch):
            # TODO: keep the best epoch only
            self.model.train()
            optimizer.zero_grad()
            A_hat, X_hat = self.model(attrs, adj)
            loss, struct_loss, feat_loss = loss_func(adj_label, A_hat, attrs,
                                                     X_hat, args.alpha)
            l = torch.mean(loss)
            l.backward()
            optimizer.step()
            print("Epoch:", '%04d' % (epoch), "train_loss=",
                  "{:.5f}".format(l.item()), "train/struct_loss=",
                  "{:.5f}".format(struct_loss.item()), "train/feat_loss=",
                  "{:.5f}".format(feat_loss.item()))

            self.model.eval()
            A_hat, X_hat = self.model(attrs, adj)
            loss, struct_loss, feat_loss = loss_func(adj_label, A_hat, attrs,
                                                     X_hat, args.alpha)
            score = loss.detach().cpu().numpy()
            # print("Epoch:", '%04d' % (epoch), 'Auc',
            #       roc_auc_score(labels, score))

        self.decision_scores_ = score
        self._process_decision_scores()
        return self

    def decision_function(self, G, args):
        """Predict raw anomaly score of X using the fitted detector.
        The anomaly score of an input sample is computed based on different
        detector algorithms. For consistency, outliers are assigned with
        larger anomaly scores.

        Parameters
        ----------
        G : PyTorch Geometric Data instance (torch_geometric.data.Data)
            The input graph.

        Returns
        -------
        anomaly_scores : numpy array of shape (n_samples,)
            The anomaly score of the input samples.
        """
        check_is_fitted(self, ['model'])

        # get needed data object from the input data
        adj, adj_label, attrs, _ = self.process_graph(G, args)

        # enable the evaluation mode
        self.model.eval()

        # construct the vector for holding the reconstruction error
        # outlier_scores = torch.zeros([attrs.shape[0], ])
        if args.device == 'cuda':
            device = torch.device(args.device)
            adj = adj.to(device)
            adj_label = adj_label.to(device)
            attrs = attrs.to(device)

        A_hat, X_hat = self.model(attrs, adj)
        outlier_scores, _, _ = loss_func(adj_label, A_hat, attrs,
                                         X_hat, args.alpha)
        return outlier_scores.detach().cpu().numpy()

    def process_graph(self, G, args):
        """Process the raw PyG data object into a tuple of sub data objects
        needed for the underlying model. For instance, if the training of the
        model need the node feature and edge index, return (G.x, G.edge_index).

        Parameters
        ----------
        G : PyTorch Geometric Data instance (torch_geometric.data.Data)
            The input graph.

        args : argparse object.
            Corresponding hyperparameters

        Returns
        -------
        processed_data : tuple of data object
            The necessary information from the raw PyG Data object.
        """
        # todo: need some assert or try/catch to make sure certain attributes
        # are presented.

        adj = G.edge_index
        attrs = G.x[:, :4]
        labels = G.y % 2
        dense_adj = SparseTensor(row=adj[0], col=adj[1]).to_dense()
        rowsum = dense_adj.sum(1)
        d_inv_sqrt = torch.pow(rowsum, -0.5).flatten()
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        adj_label = (dense_adj * d_mat_inv_sqrt).T * d_mat_inv_sqrt

        # define any network initialization parameter here
        self.feat_size = attrs.size(1)

        # return data objects needed for the network
        return adj, adj_label, attrs, labels


class Encoder(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(Encoder, self).__init__()

        self.gc1 = GCNConv(nfeat, nhid)
        self.gc2 = GCNConv(nhid, nhid)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))

        return x


class Attribute_Decoder(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(Attribute_Decoder, self).__init__()

        self.gc1 = GCNConv(nhid, nhid)
        self.gc2 = GCNConv(nhid, nfeat)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))

        return x


class Structure_Decoder(nn.Module):
    def __init__(self, nhid, dropout):
        super(Structure_Decoder, self).__init__()

        self.gc1 = GCNConv(nhid, nhid)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = x @ x.T

        return x


class Dominant_Base(nn.Module):
    """Dominant_Base (Deep Anomaly Detection on Attributed Networks)
    Dominant_Base is an anomaly detector consisting of a shared graph
    convolutional encoder, a structure reconstruction decoder, and an attribute
    reconstruction decoder. The reconstruction mean sqare error of the
    decoders are defined as structure anamoly
    score and attribute anomaly score, respectively.

    Parameters
    ----------
    in_dim : int
        Dimension of input feature
    hid_dim : int
        Dimension of  hidden layer
    dropout : float, optional
        Dropout rate of the model
        Default: 0

    Examples
    --------
    >>> model = Dominant_Base(feat_size=attrs.size(1),
                              hidden_size=args.hidden_dim, dropout=args.dropout)
    >>> A_hat, X_hat = model(attrs, adj)
    >>>
    >>> feat = dataset.feature
    >>> labels = dataset.labels
    """

    def __init__(self, feat_size, hidden_size, dropout):
        super(Dominant_Base, self).__init__()

        self.shared_encoder = Encoder(feat_size, hidden_size, dropout)
        self.attr_decoder = Attribute_Decoder(feat_size, hidden_size, dropout)
        self.struct_decoder = Structure_Decoder(hidden_size, dropout)

    def forward(self, x, adj):
        # encode
        x = self.shared_encoder(x, adj)
        # decode feature matrix
        x_hat = self.attr_decoder(x, adj)
        # decode adjacency matrix
        struct_reconstructed = self.struct_decoder(x, adj)
        # return reconstructed matrices
        return struct_reconstructed, x_hat


def loss_func(adj, A_hat, attrs, X_hat, alpha):
    # Attribute reconstruction loss
    diff_attribute = torch.pow(X_hat - attrs, 2)
    attribute_reconstruction_errors = torch.sqrt(torch.sum(diff_attribute, 1))
    attribute_cost = torch.mean(attribute_reconstruction_errors)

    # structure reconstruction loss
    diff_structure = torch.pow(A_hat - adj, 2)
    structure_reconstruction_errors = torch.sqrt(torch.sum(diff_structure, 1))
    structure_cost = torch.mean(structure_reconstruction_errors)

    cost = alpha * attribute_reconstruction_errors + (
            1 - alpha) * structure_reconstruction_errors

    return cost, structure_cost, attribute_cost
