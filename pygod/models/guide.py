# -*- coding: utf-8 -*-
"""Higher-order Structure based Anomaly Detection on Attributed
    Networks (GUIDE)"""
# Author: Kay Liu <zliu234@uic.edu>
# License: BSD 2 clause

import os
import warnings

import torch
import hashlib
import networkx as nx
import torch.nn.functional as F
from networkx.generators.atlas import graph_atlas_g

from . import DeepDetector
from ..nn import GUIDEBase


class GUIDE(DeepDetector):
    """
    GUIDE (Higher-order Structure based Anomaly Detection on Attributed
    Networks) is an anomaly detector consisting of an attribute graph
    convolutional autoencoder, and a structure graph attentive
    autoencoder (not the same as the graph attention networks). Instead
    of the adjacency matrix, node motif degree is used as input of
    structure autoencoder. The reconstruction mean square error of the
    autoencoders are defined as structure anomaly score and attribute
    anomaly score, respectively.

    Note: The calculation of node motif degree in preprocessing has
    high time complexity. It may take longer than you expect.

    See :cite:`yuan2021higher` for details.

    Parameters
    ----------
    hid_x :  int, optional
        Hidden dimension for attribute autoencoder. Default: ``32``.
    hid_s :  int, optional
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
        Loss balance weight for attribute and structure. ``None`` for
        balancing by standard deviation. Default: ``None``.
    contamination : float, optional
        Valid in (0., 0.5). The proportion of outliers in the data set.
        Used when fitting to define the threshold on the decision
        function. Default: ``0.1``.
    lr : float, optional
        Learning rate. Default: ``0.004``.
    epoch : int, optional
        Maximum number of training epoch. Default: ``10``.
    gpu : int, optional
        GPU Index, -1 for using CPU. Default: ``0``.
    batch_size : int, optional
        Minibatch size, 0 for full batch training. Default: ``0``.
    num_neigh : int, optional
        Number of neighbors in sampling, -1 for all neighbors.
        Default: ``-1``.
    graphlet_size : int, optional
        The maximum graphlet size used to compute structure input.
        Default: ``4``.
    selected_motif : bool, optional
        Use selected motifs which are defined in the original paper.
        Default: ``True``.
    cache_dir : str, option
        The directory for the node motif degree caching.
        Default: ``None``.
    verbose : bool, optional
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
                 hid_x=64,
                 hid_s=4,
                 num_layers=4,
                 dropout=0.,
                 weight_decay=0.,
                 act=F.relu,
                 backbone=None,
                 alpha=0.5,
                 contamination=0.1,
                 lr=0.004,
                 epoch=100,
                 gpu=-1,
                 batch_size=0,
                 num_neigh=-1,
                 graphlet_size=4,
                 selected_motif=True,
                 cache_dir=None,
                 verbose=0,
                 **kwargs):

        if backbone is not None:
            warnings.warn("Backbone and num_layers are not used in AnomalyDAE")

        super(GUIDE, self).__init__(hid_dim=(hid_x, hid_s),
                                    num_layers=num_layers,
                                    dropout=dropout,
                                    weight_decay=weight_decay,
                                    act=act,
                                    backbone=backbone,
                                    contamination=contamination,
                                    lr=lr,
                                    epoch=epoch,
                                    gpu=gpu,
                                    batch_size=batch_size,
                                    num_neigh=num_neigh,
                                    verbose=verbose,
                                    **kwargs)

        self.dim_s = None
        self.alpha = alpha
        self.graphlet_size = graphlet_size
        if selected_motif:
            assert self.graphlet_size == 4, \
                "Graphlet size is fixed when using selected motif"
        self.selected_motif = selected_motif
        self.verbose = verbose

        if cache_dir is None:
            cache_dir = os.path.join(os.path.expanduser('~'), '.pygod')
        self.cache_dir = cache_dir

    def process_graph(self, data):

        data.s = self._get_nmf(data, self.cache_dir)
        self.dim_s = data.s.shape[1]

    def init_model(self, **kwargs):

        return GUIDEBase(dim_x=self.in_dim,
                         dim_s=self.dim_s,
                         hid_x=self.hid_dim[0],
                         hid_s=self.hid_dim[1],
                         num_layers=self.num_layers,
                         dropout=self.dropout,
                         act=self.act,
                         **kwargs).to(self.device)

    def forward_model(self, data):

        batch_size = data.batch_size

        x = data.x.to(self.device)
        s = data.s.to(self.device)
        edge_index = data.edge_index.to(self.device)

        x_, s_ = self.model(x, s, edge_index)

        score = self.model.loss_func(x[:batch_size],
                                     x_[:batch_size],
                                     s[:batch_size],
                                     s_[:batch_size],
                                     self.alpha)

        loss = torch.mean(score)

        return loss, score.detach().cpu()

    def _get_nmf(self, data, cache_dir):
        """
        Calculation of Node Motif Degree / Graphlet Degree
        Distribution. Part of this function is adapted
        from https://github.com/benedekrozemberczki/OrbitalFeatures.

        Parameters
        ----------
        data : PyTorch Geometric Data instance (torch_geometric.data.Data)
            The input data.
        cache_dir : str
            The directory for the node motif degree caching

        Returns
        -------
        s : torch.Tensor
            Structure matrix (node motif degree/graphlet degree)
        """
        hash_func = hashlib.sha1()
        hash_func.update(str(data).encode('utf-8'))
        file_name = 'nmd_' + str(hash_func.hexdigest()[:8]) + \
                    str(self.graphlet_size) + \
                    str(self.selected_motif)[0] + '.pt'
        file_path = os.path.join(cache_dir, file_name)
        if os.path.exists(file_path):
            s = torch.load(file_path)
        else:
            edge_index = data.edge_index
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
                s = torch.zeros((data.x.shape[0], 6))
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
