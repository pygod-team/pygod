# -*- coding: utf-8 -*-
"""Generative Adversarial Attributed Network Anomaly Detection (GAAN)"""
# Author: Ruitong Zhang <rtzhang@buaa.edu.cn>
# License: BSD 2 clause

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj
from torch_geometric.loader import NeighborLoader
from sklearn.utils.validation import check_is_fitted

from . import BaseDetector
from ..utils.utility import validate_device
from ..utils.metric import eval_roc_auc


class GAAN(BaseDetector):
    """
    GAAN (Generative Adversarial Attributed Network Anomaly Detection)
    is a generative adversarial attribute network anomaly detection
    framework, including a generator module, an encoder module, a
    discriminator module, and uses anomaly evaluation measures that
    consider sample reconstruction error and real sample recognition
    confidence to make predictions.

    See :cite:`chen2020generative` for details.

    Parameters
    ----------
    noise_dim :  int, optional
        Dimension of the Gaussian random noise. Defaults: ``32``.
    latent_dim :  int, optional
        Dimension of the latent space. Defaults: ``32``.
    hid_dim1 :  int, optional
        Hidden dimension of MLP later 1. Defaults: ``32``.
    hid_dim2 :  int, optional
        Hidden dimension of MLP later 2. Defaults: ``64``.
    hid_dim3 :  int, optional
        Hidden dimension of MLP later 3. Defaults: ``128``.
    num_layers : int, optional
        Total number of layers in model. Defaults: ``3``.
    dropout : float, optional
        Dropout rate. Defaults: ``0.``.
    weight_decay : float, optional
        Weight decay (L2 penalty). Defaults: ``0.``.
    act : callable activation function or None, optional
        Activation function if not None.
        Defaults: ``torch.nn.functional.relu``.
    alpha : float, optional
        loss balance weight for attribute and structure.
        Defaults: ``0.2``.
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
    >>> prediction = model.predict(data)
    """

    def __init__(self,
                 noise_dim=32,
                 latent_dim=32,
                 hid_dim1=32,
                 hid_dim2=64,
                 hid_dim3=128,
                 num_layers=2,
                 dropout=0.3,
                 weight_decay=0.,
                 act=F.relu,
                 alpha=0.2,
                 contamination=0.1,
                 lr=5e-3,
                 epoch=10,
                 gpu=0,
                 batch_size=0,
                 num_neigh=-1,
                 verbose=False):
        super(GAAN, self).__init__(contamination=contamination)

        # model param
        self.noise_dim = noise_dim
        self.latent_dim = latent_dim
        self.hid_dim1 = hid_dim1
        self.hid_dim2 = hid_dim2
        self.hid_dim3 = hid_dim3
        self.num_layers = num_layers
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

        # model
        self.generator = None
        self.encoder = None
        self.discriminator = None

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
        #X, edge_index = self.process_graph(G)
        G.node_idx = torch.arange(G.x.shape[0])
        G.s = to_dense_adj(G.edge_index)[0]
        if self.batch_size == 0:
            self.batch_size = G.x.shape[0]
        loader = NeighborLoader(G,
                                [self.num_neigh] * self.num_layers,
                                batch_size=self.batch_size)

        # initialize the model
        self.generator = Generator(in_dim=self.noise_dim,
                                   hid_dim1=self.hid_dim1,
                                   hid_dim2=self.hid_dim2,
                                   hid_dim3=self.hid_dim3,
                                   out_dim=G.x.shape[1],
                                   act=self.act).to(self.device)

        self.encoder = Encoder(in_dim=G.x.shape[1],
                               hid_dim1=self.hid_dim1,
                               hid_dim2=self.hid_dim2,
                               hid_dim3=self.hid_dim3,
                               out_dim=self.latent_dim,
                               act=self.act).to(self.device)

        self.discriminator = Discriminator().to(self.device)

        # initialize the optimizer
        optimizer_GE = torch.optim.Adam(
            params=list(self.generator.parameters()) + list(
                self.encoder.parameters()),
            lr=self.lr,
            weight_decay=self.weight_decay)

        optimizer_D = torch.optim.Adam(params=self.discriminator.parameters(),
                                       weight_decay=self.weight_decay)

        # enable the training mode
        self.generator.train()
        self.encoder.train()
        self.discriminator.train()

        # initialize the  criterion
        criterion = torch.nn.BCELoss()

        scores = np.zeros(G.x.shape[0])
        for epoch in range(self.epoch):
            epoch_loss_D, epoch_loss_GE = 0, 0
            for sampled_data in loader:
                batch_size = sampled_data.batch_size
                node_idx = sampled_data.node_idx
                X, edge_index = self.process_graph(sampled_data)

                # Generate noise for constructing fake attribute
                gaussian_noise = torch.randn(X.shape[0], self.noise_dim).to(
                    self.device)

                optimizer_D.zero_grad()
                optimizer_GE.zero_grad()

                # train the model
                X_, Y_true_pre, Y_fake_pre = self.train_model(
                    X, gaussian_noise, edge_index)

                edge_index_i = (
                    edge_index[0] < batch_size).nonzero().to(
                    self.device)
                X, X_ = X[:batch_size], X_[:batch_size]
                Y_true_pre = torch.reshape(Y_true_pre,
                                           [Y_true_pre.shape[0]])[edge_index_i]
                Y_fake_pre = torch.reshape(Y_fake_pre,
                                           [Y_fake_pre.shape[0]])[edge_index_i]
                edge_index = torch.index_select(
                    edge_index, 1, torch.reshape(
                        edge_index_i, [
                            edge_index_i.shape[0]]))

                # get loss
                loss_D, loss_GE = self.loss_function(X, X_,
                                                     Y_true_pre, Y_fake_pre,
                                                     edge_index, criterion)
                epoch_loss_D += loss_D.item() * batch_size
                epoch_loss_GE += loss_GE.item() * batch_size

                loss_D.backward(retain_graph=True)
                loss_GE.backward()

                optimizer_D.step()
                optimizer_GE.step()

                score = self.score_function(X, X_,
                                            Y_true_pre, Y_fake_pre,
                                            edge_index, criterion)
                scores[node_idx[:batch_size]] = score.detach().cpu().numpy()

                # print out log information
                if self.verbose:
                    print(
                        "Epoch {:04d}: Loss GE {:.4f} | Loss D {:.4f}"
                        .format(epoch,
                                epoch_loss_GE / G.x.shape[0],
                                epoch_loss_D / G.x.shape[0]), end='')
                    if y_true is not None:
                        auc = eval_roc_auc(y_true, scores)
                        print(" | AUC {:.4f}".format(auc))

        self.decision_scores_ = scores
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
        check_is_fitted(self, ['generator', 'encoder', 'discriminator'])

        # get needed data object from the input data
        G.node_idx = torch.arange(G.x.shape[0])
        G.s = to_dense_adj(G.edge_index)[0]
        loader = NeighborLoader(G,
                                [self.num_neigh] * self.num_layers,
                                batch_size=self.batch_size)

        # enable the evaluation mode
        self.generator.eval()
        self.encoder.eval()
        self.discriminator.eval()

        criterion = torch.nn.BCELoss()

        scores = np.zeros(G.x.shape[0])
        for sampled_data in loader:
            batch_size = sampled_data.batch_size
            node_idx = sampled_data.node_idx

            X, edge_index = self.process_graph(sampled_data)

            # construct the vector for holding the reconstruction error
            gaussian_noise = torch.randn(X.shape[0], self.noise_dim).to(
                self.device)
            X_, Y_true_pre, Y_fake_pre = self.train_model(X, gaussian_noise,
                                                          edge_index)

            edge_index_i = (
                edge_index[0] < batch_size).nonzero().to(
                self.device)
            X, X_ = X[:batch_size], X_[:batch_size]
            Y_true_pre = torch.reshape(Y_true_pre,
                                       [Y_true_pre.shape[0]])[edge_index_i]
            Y_fake_pre = torch.reshape(Y_fake_pre,
                                       [Y_fake_pre.shape[0]])[edge_index_i]
            edge_index = torch.index_select(
                edge_index, 1, torch.reshape(
                    edge_index_i, [
                        edge_index_i.shape[0]]))

            score = self.score_function(X, X_,
                                        Y_true_pre, Y_fake_pre,
                                        edge_index, criterion)

            scores[node_idx[:batch_size]] = score.detach().cpu().numpy()

        return scores

    def train_model(self, X, gaussian_noise, edge_index):
        """
        Complete the entire process from noise to generator,
        to encoder, and finally to discriminator.

        Parameters
        ----------
        X : torch.Tensor
            Attribute (feature) of nodes.
        gaussian_noise : torch.Tensor
            Gaussian noise for generator.
        edge_index : torch.Tensor
            Edge list of the graph.

        Returns
        -------
        X_ : torch.Tensor
            Fake attribute (feature) of nodes.
        Y_true_pre : torch.Tensor
            Labels predicted from the ture attribute.
        Y_fake_pre_ : torch.Tensor
            Labels predicted from the fake attribute.

        """
        # get fake attribute matrix
        X_ = self.generator(gaussian_noise)

        # get latent embedding matrix
        Z = self.encoder(X)
        Z_ = self.encoder(X_)

        # get link probability matrix
        Y_true_pre = self.discriminator(Z, edge_index)
        Y_fake_pre = self.discriminator(Z_, edge_index)

        return X_, Y_true_pre, Y_fake_pre

    def loss_function(self, X, X_, Y_true_pre, Y_fake_pre, edge_index,
                      criterion):
        """
        Obtain the generator and discriminator losses separately.

        Parameters
        ----------
        X : torch.Tensor
            Attribute (feature) of nodes.
        X_ : torch.Tensor
            Fake attribute (feature) of nodes.
        Y_true_pre : torch.Tensor
            Labels predicted from the ture attribute.
        Y_fake_pre : torch.Tensor
            Labels predicted from the fake attribute.
        edge_index : torch.Tensor
            Edge list of the graph.
        criterion : torch.nn.modules.loss.BCELoss
            Edge list of the graph.

        Returns
        -------
        loss_D : torch.Tensor
            Generator loss.
        loss_GE : torch.Tensor
            Discriminator loss.
        """

        # attribute reconstruction loss
        diff_attribute = torch.pow(X - X_, 2)
        attribute_errors = torch.sqrt(torch.sum(diff_attribute, 1))

        # structure reconstruction loss
        Y_true = torch.ones((edge_index.shape[1]), 1).to(self.device)
        Y_fake = torch.zeros((edge_index.shape[1]), 1).to(self.device)
        structure_errors = criterion(Y_true_pre, Y_true) + criterion(
            Y_fake_pre, Y_fake)

        loss_D = structure_errors
        loss_GE = torch.mean(attribute_errors)

        return loss_D, loss_GE

    def score_function(self, X, X_, Y_true_pre, Y_fake_pre, edge_index,
                       criterion):
        """
        Get anomaly score after the model training by weighted context
        reconstruction loss and structure discriminator loss.

        Parameters
        ----------
        X : torch.Tensor
            Attribute (feature) of nodes.
        X_ : torch.Tensor
            Fake attribute (feature) of nodes.
        Y_true_pre : torch.Tensor
            Labels predicted from the ture attribute.
        Y_fake_pre : torch.Tensor
            Labels predicted from the fake attribute.
        edge_index : torch.Tensor
            Edge list of the graph.
        criterion : torch.nn.modules.loss.BCELoss
            Edge list of the graph.

        Returns
        -------
        score : torch.Tensor
            Anomaly score.
        """
        # attribute reconstruction score
        diff_attribute = torch.pow(X - X_, 2)
        attribute_errors = torch.sqrt(torch.sum(diff_attribute, 1))

        # structure reconstruction score
        structure_errors = torch.zeros((X.shape[0])).to(self.device)
        for i in range(X.shape[0]):
            edge_index_i = (edge_index[0] == i).nonzero().to(self.device)

            Y_true = torch.ones((edge_index_i.shape[0]), 1).to(self.device)
            Y_fake = torch.zeros((edge_index_i.shape[0]), 1).to(self.device)

            Y_true_pre_i = torch.reshape(Y_true_pre, [Y_true_pre.shape[0]])[
                edge_index_i]
            Y_fake_pre_i = torch.reshape(Y_fake_pre, [Y_fake_pre.shape[0]])[
                edge_index_i]

            structure_errors[i] = criterion(Y_true_pre_i, Y_true) + criterion(
                Y_fake_pre_i, Y_fake)

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
        X = G.x.to(self.device)

        return X, edge_index


class Generator(nn.Module):
    def __init__(self,
                 in_dim,
                 hid_dim1,
                 hid_dim2,
                 hid_dim3,
                 out_dim,
                 act):
        super(Generator, self).__init__()

        # three layer MLP
        self.fc1 = nn.Linear(in_dim, hid_dim1)
        self.fc2 = nn.Linear(hid_dim1, hid_dim2)
        self.fc3 = nn.Linear(hid_dim2, hid_dim3)
        self.fc4 = nn.Linear(hid_dim3, out_dim)
        self.act = act

    def forward(self, noise):
        # input the low_dimensional prior Gaussian noise
        hidden1 = self.act(self.fc1(noise))

        hidden2 = self.act(self.fc2(hidden1))
        hidden3 = self.act(self.fc3(hidden2))

        # output the fake attribute matrix
        X_ = self.act(self.fc4(hidden3))

        return X_


class Encoder(nn.Module):
    def __init__(self,
                 in_dim,
                 hid_dim1,
                 hid_dim2,
                 hid_dim3,
                 out_dim,
                 act):
        super(Encoder, self).__init__()

        # three layer MLP
        self.fc1 = nn.Linear(in_dim, hid_dim1)
        self.fc2 = nn.Linear(hid_dim1, hid_dim2)
        self.fc3 = nn.Linear(hid_dim2, hid_dim3)
        self.fc4 = nn.Linear(hid_dim3, out_dim)
        self.act = act

    def forward(self, X):
        # input the original attribute matrix or the fake attribute matrix
        hidden1 = self.act(self.fc1(X))
        hidden2 = self.act(self.fc2(hidden1))
        hidden3 = self.act(self.fc3(hidden2))

        # output the low_dimensional latent embedding matrix of attribute
        # matrix
        Z = self.act(self.fc4(hidden3))

        return Z


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # the activation function of binary classifier
        self.fc1 = nn.Linear(1, 1)
        self.act = torch.sigmoid

    def forward(self, Z, edge_index):
        # dot product of the embedding output
        dot_product = Z.mm(Z.t())
        edge_prob = torch.reshape(dot_product[edge_index[0], edge_index[1]],
                                  [edge_index.shape[1], 1])
        Y = self.act(self.fc1(edge_prob))

        return Y
