# -*- coding: utf-8 -*-
"""Higher-order Structure based Anomaly Detection on Attributed
    Networks (GUIDE)"""
# Author: Kay Liu <zliu234@uic.edu>
# License: BSD 2 clause

import os
import torch
import hashlib
import numpy as np
import networkx as nx
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import softmax
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from torch_geometric.loader import NeighborLoader
from networkx.generators.atlas import graph_atlas_g
from sklearn.utils.validation import check_is_fitted

from . import BaseDetector
from .basic_nn import GCN
from ..utils.utility import validate_device
from ..utils.metric import eval_roc_auc


class GUIDE(BaseDetector):
    """
    GUIDE (Higher-order Structure based Anomaly Detection on Attributed
    Networks)
    GUIDE is an anomaly detector consisting of an attribute graph
    convolutional autoencoder, and a structure graph attentive
    autoencoder (not same as the graph attention networks). Instead of
    adjacency matrix, node motif degree (graphlet degree is used in
    this implementation by default) is used as input of
    structure autoencoder. The reconstruction mean square error of the
    autoencoders are defined as structure anomaly score and attribute
    anomaly score, respectively.

    Note: The calculation of node motif degree in preprocesing has high
    time complexity. It may take longer than you expect.

    See :cite:`yuan2021higher` for details.

    Parameters
    ----------
    a_hid :  int, optional
        Hidden dimension for attribute autoencoder. Default: ``32``.
    s_hid :  int, optional
        Hidden dimension for structure autoencoder. Default: ``4``.
    num_layers : int, optional
        Total number of layers in autoencoders. Default: ``4``.
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
        Maximum number of training epoch. Default: ``10``.
    gpu : int
        GPU Index, -1 for using CPU. Default: ``0``.
    batch_size : int, optional
        Minibatch size, 0 for full batch training. Default: ``0``.
    num_neigh : int, optional
        Number of neighbors in sampling, -1 for all neighbors.
        Default: ``-1``.
    graphlet_size : int
        The maximum graphlet size used to compute structure input.
        Default: ``4``.
    selected_motif : bool
        Use selected motifs which are defined in the original paper.
        Default: ``True``.
    cache_dir : str
        The directory for the node motif degree caching.
        Default: ``None``.
    verbose : bool
        Verbosity mode. Turn on to print out log information.
        Default: ``False``.

    Examples
    --------
    >>> from pygod.models import GUIDE
    >>> model = GUIDE()
    >>> model.fit(data) # PyG graph data object
    >>> prediction = model.predict(data)
    """

    def __init__(self,
                 a_hid=32,
                 s_hid=4,
                 num_layers=4,
                 dropout=0.3,
                 weight_decay=0.,
                 act=F.relu,
                 alpha=0.1,
                 contamination=0.1,
                 lr=0.001,
                 epoch=10,
                 gpu=0,
                 batch_size=0,
                 num_neigh=-1,
                 graphlet_size=4,
                 selected_motif=True,
                 cache_dir=None,
                 verbose=False):
        super(GUIDE, self).__init__(contamination=contamination)

        # model param
        self.a_hid = a_hid
        self.s_hid = s_hid
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
        self.graphlet_size = graphlet_size
        if selected_motif:
            assert self.graphlet_size == 4, \
                "Graphlet size is fixed when using selected motif"
        self.selected_motif = selected_motif
        self.verbose = verbose
        self.model = None

        if cache_dir is None:
            cache_dir = os.path.join(os.path.expanduser('~'), '.pygod')
        self.cache_dir = cache_dir

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
        G.node_idx = torch.arange(G.x.shape[0])
        G.s = self._get_nmf(G, self.cache_dir)
        if self.batch_size == 0:
            self.batch_size = G.x.shape[0]
        loader = NeighborLoader(G,
                                [self.num_neigh] * self.num_layers,
                                batch_size=self.batch_size)

        self.model = GUIDE_Base(a_dim=G.x.shape[1],
                                s_dim=G.s.shape[1],
                                a_hid=self.a_hid,
                                s_hid=self.s_hid,
                                num_layers=self.num_layers,
                                dropout=self.dropout,
                                act=self.act).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=self.lr,
                                     weight_decay=self.weight_decay)

        self.model.train()
        decision_scores = np.zeros(G.x.shape[0])
        for epoch in range(self.epoch):
            epoch_loss = 0
            for sampled_data in loader:
                batch_size = sampled_data.batch_size
                node_idx = sampled_data.node_idx
                x, s, edge_index = self.process_graph(G)

                x_, s_ = self.model(x, s, edge_index)
                score = self.loss_func(x[:batch_size],
                                       x_[:batch_size],
                                       s[:batch_size],
                                       s_[:batch_size])
                decision_scores[node_idx[:batch_size]] = score.detach() \
                                                              .cpu().numpy()
                loss = torch.mean(score)
                epoch_loss += loss.item() * batch_size

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
        G.node_idx = torch.arange(G.x.shape[0])
        G.s = self._get_nmf(G, self.cache_dir)
        loader = NeighborLoader(G,
                                [self.num_neigh] * self.num_layers,
                                batch_size=self.batch_size)

        self.model.eval()
        outlier_scores = np.zeros(G.x.shape[0])
        for sampled_data in loader:
            batch_size = sampled_data.batch_size
            node_idx = sampled_data.node_idx

            x, s, edge_index = self.process_graph(sampled_data)

            x_, s_ = self.model(x, s, edge_index)
            score = self.loss_func(x[:batch_size],
                                   x_[:batch_size],
                                   s[:batch_size],
                                   s_[:batch_size])

            outlier_scores[node_idx[:batch_size]] = score.detach()\
                                                         .cpu().numpy()

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
        s : torch.Tensor
            Structure matrix (node motif degree/graphlet degree)
        edge_index : torch.Tensor
            Edge list of the graph.
        """
        edge_index = G.edge_index.to(self.device)
        s = G.s.to(self.device)
        x = G.x.to(self.device)

        return x, s, edge_index

    def loss_func(self, x, x_, s, s_):
        # attribute reconstruction loss
        diff_attribute = torch.pow(x - x_, 2)
        attribute_errors = torch.sqrt(torch.sum(diff_attribute, 1))

        # structure reconstruction loss
        diff_structure = torch.pow(s - s_, 2)
        structure_errors = torch.sqrt(torch.sum(diff_structure, 1))

        score = self.alpha * attribute_errors + (1 - self.alpha) * \
                structure_errors
        return score

    def _get_nmf(self, G, cache_dir):
        """
        Calculation of Node Motif Degree / Graphlet Degree
        Distribution. Part of this function is adapted
        from https://github.com/benedekrozemberczki/OrbitalFeatures.

        Parameters
        ----------
        G : PyTorch Geometric Data instance (torch_geometric.data.Data)
            The input data.
        cache_dir : str
            The directory for the node motif degree caching

        Returns
        -------
        s : torch.Tensor
            Structure matrix (node motif degree/graphlet degree)
        """
        hash_func = hashlib.sha1()
        hash_func.update(str(G).encode('utf-8'))
        file_name = 'nmd_' + str(hash_func.hexdigest()[:8]) + \
                    str(self.graphlet_size) + \
                    str(self.selected_motif)[0] + '.pt'
        file_path = os.path.join(cache_dir, file_name)
        if os.path.exists(file_path):
            s = torch.load(file_path)
        else:
            edge_index = G.edge_index
            g = nx.from_edgelist(edge_index.T.tolist())

            # create edge subsets
            edge_subsets = dict()
            subsets = [[edge[0], edge[1]] for edge in g.edges()]
            edge_subsets[2] = subsets
            unique_subsets = dict()
            for i in range(3, self.graphlet_size + 1):
                for subset in subsets:
                    for node in subset:
                        for neb in g.neighbors(node):
                            new_subset = subset + [neb]
                            if len(set(new_subset)) == i:
                                new_subset.sort()
                                unique_subsets[tuple(new_subset)] = 1
                subsets = [list(k) for k, v in unique_subsets.items()]
                edge_subsets[i] = subsets
                unique_subsets = dict()

            # enumerate graphs
            graphs = graph_atlas_g()
            interesting_graphs = {i: [] for i in
                                  range(2, self.graphlet_size + 1)}
            for graph in graphs:
                if 1 < graph.number_of_nodes() < self.graphlet_size + 1:
                    if nx.is_connected(graph):
                        interesting_graphs[graph.number_of_nodes()].append(
                            graph)

            # enumerate categories
            main_index = 0
            categories = dict()
            for size, graphs in interesting_graphs.items():
                categories[size] = dict()
                for index, graph in enumerate(graphs):
                    categories[size][index] = dict()
                    degrees = list(
                        set([graph.degree(node) for node in graph.nodes()]))
                    for degree in degrees:
                        categories[size][index][degree] = main_index
                        main_index += 1
            unique_motif_count = main_index

            # setup feature
            features = {node: {i: 0 for i in range(unique_motif_count)}
                        for node in g.nodes()}
            for size, node_lists in edge_subsets.items():
                graphs = interesting_graphs[size]
                for nodes in node_lists:
                    sub_gr = g.subgraph(nodes)
                    for index, graph in enumerate(graphs):
                        if nx.is_isomorphic(sub_gr, graph):
                            for node in sub_gr.nodes():
                                features[node][categories[size][index][
                                    sub_gr.degree(node)]] += 1
                            break

            motifs = [[n] + [features[n][i] for i in range(
                unique_motif_count)] for n in g.nodes()]
            motifs = torch.Tensor(motifs)
            motifs = motifs[torch.sort(motifs[:, 0]).indices, 1:]

            if self.selected_motif:
                # use motif selected in the original paper only
                s = torch.zeros((G.x.shape[0], 6))
                # m31
                s[:, 0] = motifs[:, 3]
                # m32
                s[:, 1] = motifs[:, 1] + motifs[:, 2]
                # m41
                s[:, 2] = motifs[:, 14]
                # m42
                s[:, 3] = motifs[:, 12] + motifs[:, 13]
                # m43
                s[:, 4] = motifs[:, 11]
                # node degree
                s[:, 5] = motifs[:, 0]
            else:
                # use graphlet degree
                s = motifs

            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            torch.save(s, file_path)

        return s


class GUIDE_Base(nn.Module):
    def __init__(self,
                 a_dim,
                 s_dim,
                 a_hid,
                 s_hid,
                 num_layers,
                 dropout,
                 act):
        super(GUIDE_Base, self).__init__()

        self.attr_ae = GCN(in_channels=a_dim,
                           hidden_channels=a_hid,
                           num_layers=num_layers,
                           out_channels=a_dim,
                           dropout=dropout,
                           act=act)

        self.struct_ae = GNA(in_channels=s_dim,
                             hidden_channels=s_hid,
                             num_layers=num_layers,
                             out_channels=s_dim,
                             dropout=dropout,
                             act=act)

    def forward(self, x, s, edge_index):
        x_ = self.attr_ae(x, edge_index)
        s_ = self.struct_ae(s, edge_index)
        return x_, s_


class GNA(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 num_layers,
                 out_channels,
                 dropout,
                 act):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        self.layers.append(GNAConv(in_channels, hidden_channels))
        for layer in range(num_layers - 2):
            self.layers.append(GNAConv(hidden_channels,
                                       hidden_channels))
        self.layers.append(GNAConv(hidden_channels, out_channels))

        self.dropout = dropout
        self.act = act

    def forward(self, s, edge_index):
        for layer in self.layers:
            s = layer(s, edge_index)
            s = F.dropout(s, self.dropout, training=self.training)
            s = self.act(s)
        return s


class GNAConv(MessagePassing):
    def __init__(self,
                 in_channels,
                 out_channels):
        super().__init__(aggr='add')
        self.w1 = torch.nn.Linear(in_channels, out_channels)
        self.w2 = torch.nn.Linear(in_channels, out_channels)
        self.a = nn.Parameter(torch.randn(out_channels, 1))

    def forward(self, s, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=s.size(0))
        out = self.propagate(edge_index, s=self.w2(s))
        return self.w1(s) + out

    def message(self, s_i, s_j, edge_index):
        alpha = (s_i - s_j) @ self.a
        alpha = softmax(alpha, edge_index[1], num_nodes=s_i.shape[0])
        return alpha * s_j
