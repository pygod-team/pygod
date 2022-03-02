# -*- coding: utf-8 -*-
"""Example code for the Dominant model
https://www.public.asu.edu/~kding9/pdf/SDM2019_Deep.pdf
"""
# Author: Yingtong Dou <ytongdou@gmail.com>, Kay Liu <zliu234@uic.edu>
# License: BSD 2 clause

import argparse
import os.path as osp

import torch
from math import ceil
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor
from torch_geometric.nn import GCN
from sklearn.utils.validation import check_is_fitted
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid

from pygod.models import BaseDetector
from pygod.utils import gen_attribute_outliers, gen_structure_outliers
from pygod.evaluator.metric import roc_auc_score


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

        self.hidden_size = None
        self.num_layers = None
        self.dropout = None
        self.weight_decay = None
        self.act = None
        self.model = None
        self.device = None
        self.decision_scores_ = None

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
        edge_index, adj_label, attrs, labels = self.process_graph(G, args)

        # 2. set the parameters needed for the network from args.
        self.hidden_size = args.hidden_size
        self.num_layers = args.num_layers
        self.dropout = args.dropout
        self.weight_decay = args.weight_decay

        # TODO: support other activation function
        if args.act:
            self.act = F.relu

        # 3. initialize the detection model
        self.model = Dominant_Base(in_channels=attrs.shape[1],
                                   hidden_channels=self.hidden_size,
                                   num_layers=self.num_layers,
                                   dropout=self.dropout,
                                   act=self.act)

        # 4. check cuda
        if args.gpu >= 0 and torch.cuda.is_available():
            self.device = 'cuda:{}'.format(args.gpu)
        else:
            self.device = 'cpu'

        edge_index = edge_index.to(self.device)
        adj_label = adj_label.to(self.device)
        attrs = attrs.to(self.device)
        self.model = self.model.to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr,
                                     weight_decay=self.weight_decay)

        # if args.patience > 0:
        #     es = EarlyStopping(args.patience, args.verbose)

        for epoch in range(args.epoch):
            self.model.train()
            A_hat, X_hat = self.model(attrs, edge_index)
            loss = loss_func(adj_label, A_hat, attrs, X_hat, args.alpha)
            score = loss.detach().cpu().numpy()
            l = torch.mean(loss)

            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            # TODO: support more metrics
            # auc = roc_auc_score(labels, score)
            if args.verbose:
                print("Epoch {:04d}: Loss {:.4f}".format(epoch, l.item()))
            # if args.patience > 0 and es.step(auc, self.model):
            #     break

        # if args.patience > 0:
        #     self.model.load_state_dict(torch.load('es_checkpoint.pt'))

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

        args : argparse object.
            Corresponding hyperparameters

        Returns
        -------
        anomaly_scores : numpy array of shape (n_samples,)
            The anomaly score of the input samples.
        """
        check_is_fitted(self, ['model'])

        # get needed data object from the input data
        edge_index, adj_label, attrs, _ = self.process_graph(G, args)

        # enable the evaluation mode
        self.model.eval()

        # construct the vector for holding the reconstruction error

        edge_index = edge_index.to(self.device)
        adj_label = adj_label.to(self.device)
        attrs = attrs.to(self.device)

        A_hat, X_hat = self.model(attrs, edge_index)
        outlier_scores = loss_func(adj_label, A_hat, attrs, X_hat, args.alpha)
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
        edge_index = G.edge_index
        # TODO: potential memory efficient improvement
        #       via sparse matrix operation
        dense_adj = SparseTensor(row=edge_index[0],
                                 col=edge_index[1]).to_dense()

        # adjacency matrix normalization
        rowsum = dense_adj.sum(1)
        d_inv_sqrt = torch.pow(rowsum, -0.5).flatten()
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        adj_label = (dense_adj * d_mat_inv_sqrt).T * d_mat_inv_sqrt

        # return data objects needed for the network
        return edge_index, adj_label, G.x, G.y


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
    """

    def __init__(self, in_channels, hidden_channels, num_layers, dropout, act):
        super(Dominant_Base, self).__init__()

        encoder_layers = ceil(num_layers / 2)
        decoder_layers = num_layers - encoder_layers

        self.shared_encoder = GCN(in_channels=in_channels,
                                  hidden_channels=hidden_channels,
                                  num_layers=encoder_layers,
                                  out_channels=hidden_channels,
                                  dropout=dropout,
                                  act=act)

        self.attr_decoder = GCN(in_channels=hidden_channels,
                                hidden_channels=hidden_channels,
                                num_layers=decoder_layers,
                                out_channels=in_channels,
                                dropout=dropout,
                                act=act)

        self.struct_decoder = GCN(in_channels=hidden_channels,
                                  hidden_channels=hidden_channels,
                                  num_layers=decoder_layers - 1,
                                  out_channels=in_channels,
                                  dropout=dropout,
                                  act=act)

    def forward(self, x, adj):
        # encode
        h = self.shared_encoder(x, adj)
        # decode feature matrix
        X_hat = self.attr_decoder(h, adj)
        # decode adjacency matrix
        h = self.struct_decoder(h, adj)
        A_hat = h @ h.T

        # return reconstructed matrices
        return A_hat, X_hat


def loss_func(adj, A_hat, attrs, X_hat, alpha):
    # Attribute reconstruction loss
    diff_attribute = torch.pow(X_hat - attrs, 2)
    attribute_reconstruction_errors = torch.sqrt(torch.sum(diff_attribute, 1))

    # structure reconstruction loss
    diff_structure = torch.pow(A_hat - adj, 2)
    structure_reconstruction_errors = torch.sqrt(torch.sum(diff_structure, 1))

    cost = alpha * attribute_reconstruction_errors + (
            1 - alpha) * structure_reconstruction_errors
    return cost


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Cora',
                    help='graph dataset to be evaluated, [Cora, Pubmed, '
                         'Citeseer] ')
parser.add_argument('--hidden_size', type=int, default=64,
                    help='dimension of hidden embedding (default: 64)')
parser.add_argument('--num_layers', type=int, default=4,
                    help='number of hidden layers,'
                         'must be greater than 2 (default: 4)')
parser.add_argument('--epoch', type=int, default=5,
                    help='maximum training epoch')
parser.add_argument('--lr', type=float, default=5e-3,
                    help='learning rate')
parser.add_argument('--dropout', type=float, default=0.3,
                    help='Dropout rate')
parser.add_argument('--alpha', type=float, default=0.8,
                    help='balance parameter')
parser.add_argument('--weight_decay', type=float, default=1e-3,
                    help='weight decay')
parser.add_argument("--act", type=bool, default=True,
                    help="using activation function or not")
parser.add_argument("--gpu", type=int, default=-1,
                    help="GPU Index, -1 for using CPU (default: 0)")
parser.add_argument("--verbose", type=bool, default=True,
                    help="print log information")
parser.add_argument("--patience", type=int, default=10,
                    help="early stopping patience,"
                         "0 for disabling early stopping (default: 10)")

args = parser.parse_args()

# data loading
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data',
                args.dataset)

# this gives us a PyG data object
data = Planetoid(path, args.dataset,
                 transform=T.NormalizeFeatures())[0]

data, ys = gen_structure_outliers(data, m=10, n=10)
data, yf = gen_attribute_outliers(data, n=100, k=50)
data.y = torch.logical_or(torch.tensor(ys), torch.tensor(yf))

# model initialization
clf = Dominant()

print('training...')
clf.fit(data, args)
print()

print('predicting for probability')
prob = clf.predict_proba(data, args)
print('Probability', prob)
print()

print('predicting for raw scores')
outlier_scores = clf.decision_function(data, args)
print('Raw scores', outlier_scores)
print()

print('predicting for labels')
labels = clf.predict(data, args)
print('Labels', labels)
print()

print('predicting for labels with confidence')
labels, confidence = clf.predict(data, args, return_confidence=True)
print('Labels', labels)
print('Confidence', confidence)

print('evaluating outlier detection performance')
auc_score = roc_auc_score(data.y, outlier_scores)
print('AUC Score', auc_score)
print()
