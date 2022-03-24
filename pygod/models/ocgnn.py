# -*- coding: utf-8 -*-
""" One-Class Graph Neural Networks for Anomaly Detection in Attributed Networks
"""
# Author: Xueying Ding <xding2@andrew.cmu.edu>
# License: BSD 2 clause

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv
from sklearn.metrics import roc_auc_score
from sklearn.utils.validation import check_is_fitted

from . import BaseDetector


class GCN_base(nn.Module):
    """
    Describe: Backbone GCN module.
    """

    def __init__(self, in_feats, n_hidden, n_layers, dropout, act):
        super(GCN_base, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GCNConv(in_feats, n_hidden, bias=False))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GCNConv(n_hidden, n_hidden, bias=False))
        # output layer
        self.layers.append(GCNConv(n_hidden, n_hidden, bias=False))
        self.dropout = nn.Dropout(p=dropout)
        self.act = act

    def forward(self, x, edge_index):
        for i, layer in enumerate(self.layers[0:-1]):
            if i != 0:
                x = self.dropout(x)
            x = layer(x, edge_index)
            x = self.act(x)
        x = self.dropout(x)
        x = self.layers[-1](x, edge_index)
        return x


class OCGNN(BaseDetector):
    """
    OCGNN (One-Class Graph Neural Networks for Anomaly Detection in Attributed
    Networks): OCGNN is an anomaly detector that measures the distance of
    anomaly to the centroid, in the similar fashion to the support vector
    machine, but in the embedding space after feeding towards several layers
     of GCN.

    See :cite:`wang2021one` for details.

    Parameters
    ----------
    n_hidden :  int, optional
        Hidden dimension of model. Defaults: `32``.
    n_layers : int, optional
        Dimensions of underlying GCN. Defaults: ``3``.
    dropout : float, optional
        Dropout rate. Defaults: ``0.3``.
    weight_decay : float, optional
        Weight decay (L2 penalty). Defaults: ``0.``.
    act : callable activation function or None, optional
        Activation function if not None.
        Defaults: ``torch.nn.functional.relu``.
    eps : float, optional
        A small valid number for determining the center and make
        sure it does not collapse to 0. Defaults: ``0.001``.
    nu: float, optional
        Regularization parameter. Defaults: ``0.01`` 
    lr : float, optional
        Learning rate. Defaults: ``0.004``.
    epoch : int, optional
        Maximum number of training epoch. Defaults: ``100``.
    gpu : int
        GPU Index, -1 for using CPU. Defaults: ``0``.
    verbose : bool
        Verbosity mode. Turn on to print out log information.
        Defaults: ``False``.

    Examples
    --------
    >>> from pygod.models import AnomalyDAE
    >>> model = OCGNN()
    >>> model.fit(data)
    >>> prediction = model.predict(data)
    """

    def __init__(self,
                 n_hidden=32,
                 n_layers=3,
                 dropout=0.3,
                 lr=0.004,
                 weight_decay=0,
                 eps=0.001,
                 nu=0.01,
                 gpu=0,
                 epoch=100,
                 verbose=False,
                 act=F.relu):
        super(OCGNN, self).__init__()
        self.dropout = dropout
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.lr = lr
        self.weight_decay = weight_decay
        self.eps = eps
        self.nu = nu
        self.data_center = 0
        self.radius = 0.0
        self.epoch = epoch
        self.act = act

        if gpu >= 0 and torch.cuda.is_available():
            self.device = 'cuda:{}'.format(gpu)
        else:
            self.device = 'cpu'

        # other param
        self.verbose = verbose
        self.model = None

    def init_center(self, x, edge_index):
        """
        Descriptions
        ----------
        Initialize hypersphere center c as the mean from
        an initial forward pass on the data.
  
        Parameters
        ----------
        x: torch Tensor object: node features
        edge_index: torch Tensor object: edge indices for the graph data

        Returns
        ----------
        c: torh Tensor object, the new centroid 
           """
        n_samples = 0
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(x, edge_index)
            # get the inputs of the batch
            n_samples = outputs.shape[0]
            c = torch.sum(outputs, dim=0).to(self.device)
        # print(outputs)
        c /= n_samples
        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        c[(abs(c) < self.eps) & (c < 0)] = -self.eps
        c[(abs(c) < self.eps) & (c > 0)] = self.eps
        return c

    def get_radius(self, dist):
        """
        Description
        ----------
        Optimally solve for radius R via the (1-nu)-quantile of distances.
        
        Parameters
        ----------
        dist: torch Tensor. distance of the data points,
              calculated by the loss function
       
        Returns
        ----------
        r: numpy array: new radius
        """
        radius = np.quantile(np.sqrt(dist.clone().data.cpu().numpy()),
                             1 - self.nu)
        return radius

    def anomaly_scores(self, outputs):
        """
        Description
        ----------
        Calculate the nomaly score given by Euclidean distance to the center.
        
        Parameters
        ----------
        outputs: torch Tensor. The output in the reduced space by GCN.

        Returns
        ----------
        dist: torch Tensor, average distance
        scores: torch Tensor, anomaly scores
        """
        dist = torch.sum((outputs - self.data_center) ** 2, dim=1)
        scores = dist - self.radius ** 2
        return dist, scores

    def loss_function(self, outputs):
        """
        Description
        ----------
        Calculate the loss in paper Equation (4)
        
        Parameters
        ----------
        outputs: torch Tensor. The output in the reduced space by GCN.

        Returns
        ----------
        dist: torch Tensor, average distance
        scores: torch Tensor, anomaly scores
        loss: torch Tensor, a combined loss of radius and average scores
        """

        dist, scores = self.anomaly_scores(outputs)
        loss = self.radius ** 2 + (1 / self.nu) * torch.mean(
            torch.max(torch.zeros_like(scores), scores))
        return loss, dist, scores

    def fit(self, G):
        """
        Description
        -----------
        Fit detector with input data.

        Parameters
        ----------
        G : PyTorch Geometric Data instance (torch_geometric.data.Data)
            The input data.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        x, edge_index, labels = self.process_graph(G)
        self.in_feats = x.shape[1]

        #print(self.in_feats, self.n_hidden, self.n_layers, self.dropout)

        # initialize the model and optimizer
        self.model = GCN_base(self.in_feats,
                              self.n_hidden,
                              self.n_layers,
                              self.dropout,
                              self.act)

        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.lr,
                                          weight_decay=self.weight_decay)

        self.data_center = torch.zeros(self.n_hidden, device=self.device)
        self.radius = torch.tensor(0, device=self.device)

        self.model = self.model.to(self.device)
        # training the model
        self.model.train()

        for epoch in range(self.epoch):
            self.model.zero_grad()
            outputs = self.model(x, edge_index)
            loss, dist, score = self.loss_function(outputs)
            #if epoch == 0:
            #    self.data_center = self.init_center(x, edge_index)
            #    self.radius = self.get_radius(dist)
            #else:
            loss.backward()
            self.optimizer.step()
            if self.verbose:
                print('RRR', self.radius.data)
                
            print("Epoch {:05d},loss {:.4f}".format(epoch, loss.item()))

            if self.verbose:
                # TODO: support more metrics
                auc = roc_auc_score(labels, score.detach().cpu().numpy())

                print("Epoch {:04d}: Loss {:.4f} | AUC {:.4f}"
                      .format(epoch, loss.item(), auc))

        self.decision_scores_ = score.detach().cpu().numpy()
        self._process_decision_scores()
        return self

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
        edge_index : torch.Tensor
            Edge list of the graph.
        y : torch.Tensor
            Labels of nodes. (currently I use %2 to obtain binary label)
        """
        edge_index = G.edge_index

        # TODO: potential memory efficient improvement
        #  via sparse matrix operation

        edge_index = edge_index.to(self.device)
        x = G.x.to(self.device)
        y = G.y % 2

        # return data objects needed for the network
        return x, edge_index, y

    def decision_function(self, G):
        """Predict raw anomaly score of X using the fitted detector.
        The anomaly score of an input sample is computed based on distance 
        to the centroid and measurement within the radius
        Parameters
        ----------
        G : torch geometric graph objects
        
        Returns
        -------
        anomaly_scores : numpy array of shape (n_samples,)
            The anomaly score of the input samples.
        """
        check_is_fitted(self, ['model'])

        # get needed data object from the input data
        x, edge_index, _ = self.process_graph(G)

        # enable the evaluation mode
        self.model.eval()

        # construct the vector for holding the reconstruction error
        # outlier_scores = torch.zeros([attrs.shape[0], ])
        outputs = self.model(x, edge_index)
        loss, dist, score = self.loss_function(outputs)

        return score.detach().cpu().numpy()
