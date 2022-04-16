# -*- coding: utf-8 -*-
""" Multilayer Perceptron Autoencoder
"""
# Author: Kay Liu <zliu234@uic.edu>
# License: BSD 2 clause

import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.utils.validation import check_is_fitted

from . import BaseDetector
from .basic_nn import MLP
from ..utils.metric import eval_roc_auc
from ..utils.dataset import PlainDataset


class MLPAE(BaseDetector):
    """
    Vanila Multilayer Perceptron Autoencoder

    See :cite:`yuan2021higher` for details.

    Parameters
    ----------
    hid_dim :  int, optional
        Hidden dimension of model. Default: ``0``.
    num_layers : int, optional
        Total number of layers in autoencoders. Default: ``4``.
    dropout : float, optional
        Dropout rate. Default: ``0.``.
    weight_decay : float, optional
        Weight decay (L2 penalty). Default: ``0.``.
    act : callable activation function or None, optional
        Activation function if not None.
        Default: ``torch.nn.functional.relu``.
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
    >>> from pygod.models import MLPAE
    >>> model = MLPAE()
    >>> model.fit(data) # PyG graph data object
    >>> prediction = model.predict(data)
    """
    def __init__(self,
                 hid_dim=64,
                 num_layers=4,
                 dropout=0.3,
                 weight_decay=0.,
                 act=F.relu,
                 contamination=0.1,
                 lr=5e-3,
                 epoch=5,
                 gpu=0,
                 batch_size=1024,
                 verbose=False):
        super(MLPAE, self).__init__(contamination=contamination)

        # model param
        self.hid_dim = hid_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.act = act

        # training param
        self.lr = lr
        self.epoch = epoch
        if gpu >= 0 and torch.cuda.is_available():
            self.device = 'cuda:{}'.format(gpu)
        else:
            self.device = 'cpu'
        self.batch_size = batch_size

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
        full_x = self.process_graph(G)
        dataset = PlainDataset(full_x)
        loader = DataLoader(dataset, batch_size=self.batch_size)

        self.model = MLP(in_channels=G.x.shape[1],
                         hidden_channels=self.hid_dim,
                         out_channels=G.x.shape[1],
                         num_layers=self.num_layers,
                         dropout=self.dropout,
                         act=self.act).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=self.lr,
                                     weight_decay=self.weight_decay)

        self.model.train()
        decision_scores = np.zeros(full_x.shape[0])
        for epoch in range(self.epoch):
            epoch_loss = 0
            for x, node_idx in loader:
                x_ = self.model(x)
                score = torch.mean(F.mse_loss(x_, x, reduction='none'), dim=1)
                decision_scores[node_idx] = score.detach().cpu().numpy()
                loss = torch.mean(score)
                epoch_loss += loss.item() * x.shape[0]

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if self.verbose:
                print("Epoch {:04d}: Loss {:.4f}"
                      .format(epoch, epoch_loss / G.x.shape[0]), end='')
                if y_true is not None:
                    auc = eval_roc_auc(y_true, decision_scores)
                    print(" | AUC {:.4f}".format(auc), end='')
                print()

        self.decision_scores_ = decision_scores
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
        full_x = self.process_graph(G)
        dataset = PlainDataset(full_x)
        loader = DataLoader(dataset, batch_size=self.batch_size)

        self.model.eval()
        outlier_scores = np.zeros(full_x.shape[0])
        for x, node_idx in loader:
            x_ = self.model(x)
            score = torch.mean(F.mse_loss(x_, x, reduction='none'), dim=1)
            outlier_scores[node_idx] = score.detach().cpu().numpy()
        return outlier_scores

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
        """
        x = G.x.to(self.device)

        return x
