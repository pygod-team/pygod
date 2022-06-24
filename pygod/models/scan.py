# -*- coding: utf-8 -*-
""" Structural Clustering Algorithm for Networks
"""
# Author: Kay Liu <zliu234@uic.edu>
# License: BSD 2 clause

import math
import torch
import warnings
from torch import nn
import networkx as nx
from pygod.metrics import *
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj
from sklearn.utils.validation import check_is_fitted

from . import BaseDetector
from ..utils import validate_device


class SCAN(BaseDetector):
    """
    SCAN (Structural Clustering Algorithm for Networks) is a clustering
    algorithm, which only takes the graph structure without the node
    features as the input. Note: This model will output detected
    clusters instead of "outliers" descibed in the original paper.

    See :cite:`xu2007scan` for details.

    Parameters
    ----------
    eps : float, optional
        Neighborhood threshold. Default: ``.5``.
    mu : int, optional
        Minimal size of clusters. Default: ``2``.
    contamination : float, optional
        Valid in (0., 0.5). The proportion of outliers in the data set.
        Used when fitting to define the threshold on the decision
        function. Default: ``0.1``.
    verbose : bool
        Verbosity mode. Turn on to print out log information.
        Default: ``False``.

    Examples
    --------
    >>> from pygod.models import SCAN
    >>> model = SCAN()
    >>> model.fit(data) # PyG graph data object
    >>> prediction = model.predict(None)
    """

    def __init__(self,
                 eps=.5,
                 mu=2,
                 contamination=0.1,
                 verbose=False):
        super(SCAN, self).__init__(contamination=contamination)

        # model param
        self.eps = eps
        self.mu = mu

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
        decision_scores = np.zeros(G.num_nodes)
        G = self.process_graph(G)

        c = 0
        clusters = {}
        nomembers = []
        for n in G.nodes():
            if self.hasLabel(clusters, n):
                continue
            else:
                N = self.neighborhood(G, n)
                if len(N) > self.mu:
                    c = c + 1
                    Q = self.neighborhood(G, n)
                    clusters[c] = []
                    # append core vertex itself
                    clusters[c].append(n)
                    while len(Q) != 0:
                        w = Q.pop(0)
                        R = self.neighborhood(G, w)
                        # include current vertex itself
                        R.append(w)
                        for s in R:
                            if not (self.hasLabel(clusters, s)) or \
                                    s in nomembers:
                                clusters[c].append(s)
                            if not (self.hasLabel(clusters, s)):
                                Q.append(s)
                else:
                    nomembers.append(n)

        for k, v in clusters.items():
            decision_scores[v] = 1

        if self.verbose and y_true is not None:
            auc = eval_roc_auc(y_true, decision_scores)
            print("AUC {:.4f}".format(auc))

        self.decision_scores_ = decision_scores
        self._process_decision_scores()
        return self

    def similarity(self, G, v, u):
        v_set = set(G.neighbors(v))
        u_set = set(G.neighbors(u))
        inter = v_set.intersection(u_set)
        if inter == 0:
            return 0
        # need to account for vertex itself, add 2(1 for each vertex)
        sim = (len(inter) + 2) / (
            math.sqrt((len(v_set) + 1) * (len(u_set) + 1)))
        return sim

    def neighborhood(self, G, v):
        eps_neighbors = []
        v_list = G.neighbors(v)
        for u in v_list:
            if (self.similarity(G, u, v)) > self.eps:
                eps_neighbors.append(u)
        return eps_neighbors

    def hasLabel(self, cliques, vertex):
        for k, v in cliques.items():
            if vertex in v:
                return True
        return False

    def sameClusters(self, G, clusters, u):
        n = G.neighbors(u)
        b = []
        i = 0
        while i < len(n):
            for k, v in clusters.items():
                if n[i] in v:
                    if k in b:
                        continue
                    else:
                        b.append(k)
            i = i + 1
        if len(b) > 1:
            return False
        return True

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
        G : networkx.classes.graph.Graph
            NetworkX Graph
        """

        G = nx.from_edgelist(G.edge_index.T.tolist())
        return G
