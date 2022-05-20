# -*- coding: utf-8 -*-
""" A Joint Modeling Approach for Anomaly Detection on
    Attributed Networks
"""
# Author: Kay Liu <zliu234@uic.edu>
# License: BSD 2 clause

import torch
from torch import nn
from pygod.metrics import *
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj
from sklearn.utils.validation import check_is_fitted

from . import BaseDetector
from ..utils import validate_device


class ANOMALOUS(BaseDetector):
    """
    ANOMALOUS (A Joint Modeling Approach for Anomaly Detection on
    Attributed Networks) is an anomaly detector with CUR decomposition
    and residual analysis. This model is transductive only.

    See :cite:`peng2018anomalous` for details.

    Parameters
    ----------
    gamma : float, optional
        Loss balance weight for attribute and structure.
        Default: ``1.``.
    weight_decay : float, optional
        Weight decay (alpha and beta in the original paper).
        Default: ``0.01``.
    contamination : float, optional
        Valid in (0., 0.5). The proportion of outliers in the data set.
        Used when fitting to define the threshold on the decision
        function. Default: ``0.1``.
    epoch : int, optional
        Maximum number of training epoch. Default: ``5``.
    verbose : bool
        Verbosity mode. Turn on to print out log information.
        Default: ``False``.

    Examples
    --------
    >>> from pygod.models import ANOMALOUS
    >>> model = ANOMALOUS()
    >>> model.fit(data) # PyG graph data object
    >>> prediction = model.predict(None)
    """

    def __init__(self,
                 gamma=1.,
                 weight_decay=0.01,
                 lr=0.004,
                 epoch=100,
                 gpu=0,
                 contamination=0.1,
                 verbose=False):
        super(ANOMALOUS, self).__init__(contamination=contamination)

        # model param
        self.gamma = gamma
        self.weight_decay = weight_decay

        # training param
        self.lr = lr
        self.epoch = epoch
        self.device = validate_device(gpu)

        # other param
        self.verbose = verbose
        self.model = None

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
        G.s = to_dense_adj(G.edge_index)[0]
        x, s = self.process_graph(G)

        s = torch.max(s, s.T)
        l = self._comp_laplacian(s)

        n, d = x.shape
        w_init = torch.inverse(x)
        r_init = torch.inverse((1 + self.weight_decay) * torch.eye(n)
            + self.gamma * l) @ x
        self.model = Radar_Base(w_init, r_init)
        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=self.lr,
                                     weight_decay=self.weight_decay)

        for epoch in range(self.epoch):
            x_, r = self.model(x)
            loss = self._loss(x, x_, r, l)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            decision_scores = torch.sum(torch.pow(r, 2), dim=1).detach() \
                .cpu().numpy()

            if self.verbose:
                print("Epoch {:04d}: Loss {:.4f}"
                      .format(epoch, loss.item()), end='')
                if y_true is not None:
                    auc = eval_roc_auc(y_true, decision_scores)
                    print(" | AUC {:.4f}".format(auc), end='')
                print()

        self.decision_scores_ = decision_scores
        self._process_decision_scores()
        return self

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

        if G is not None:
            warnings.warn('The model is transductive only. '
                          'Training data is used to predict')

        outlier_scores = self.decision_scores_

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
        """
        x = G.x.to(self.device)
        s = G.s.to(self.device)

        return x, s

    def _loss(self, x, x_, r, l):
        return torch.norm(x - x_ - r, 2) + \
               self.gamma * torch.trace(r.T @ l @ r)

    def _comp_laplacian(self, adj):
        d = torch.diag(torch.sum(adj, dim=1))
        return d - adj


class ANOMALOUS_Base(nn.Module):
    def __init__(self, w, r):
        super(ANOMALOUS_Base, self).__init__()
        self.w = nn.Parameter(w)
        self.r = nn.Parameter(r)

    def forward(self, x):
        return x @ self.w @ x, self.r
