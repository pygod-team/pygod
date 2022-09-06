# -*- coding: utf-8 -*-
"""
This file including functions to generate different types of outliers given
the input dataset for benchmarking
"""
# Author: Yingtong Dou <ytongdou@gmail.com>
# License: BSD 2 clause
import math
import random

import numpy as np
import torch
from torch_geometric.data import Data

from pygod.utils.utility import check_parameter


def gen_structural_outliers(data, m, n, p=0, random_state=None):
    """Generating structural outliers according to paper
    "Deep Anomaly Detection on Attributed Networks"
    <https://epubs.siam.org/doi/abs/10.1137/1.9781611975673.67>.
    We randomly select m nodes from the network and then make those nodes
    fully connected, and then all the m nodes in the clique are regarded as
    outliers. We iteratively repeat this process until a number of n
    cliques are generated and the total number of structural outliers is m×n.

    Parameters
    ----------
    data : PyTorch Geometric Data instance (torch_geometric.data.Data)
        The input data.
    m : int
        Number nodes in the outlier cliques.
    n : int
        Number of outlier cliques.
    p : int, optional
        Probability of edge drop in cliques. Default: ``0``.
    random_state : int, optional
        The seed to control the randomness, Default: ``None``.

    Returns
    -------
    data : PyTorch Geometric Data instance (torch_geometric.data.Data)
        The structural outlier graph with injected edges.
    y_outlier : torch.Tensor
        The outlier label tensor where 1 represents outliers and 0 represents
        regular nodes.
    """

    if not isinstance(data, Data):
        raise TypeError("data should be torch_geometric.data.Data")

    if isinstance(m, int):
        check_parameter(m, low=0, high=data.num_nodes, param_name='m')
    else:
        raise ValueError("m should be int, got %s" % m)

    if isinstance(n, int):
        check_parameter(n, low=0, high=data.num_nodes, param_name='n')
    else:
        raise ValueError("n should be int, got %s" % n)

    check_parameter(m * n, low=0, high=data.num_nodes, param_name='m*n')

    if random_state:
        np.random.seed(random_state)

    new_edges = []

    outlier_idx = np.random.choice(data.num_nodes, size=m * n, replace=False)

    # connect all m nodes in each clique
    for i in range(0, n):
        for j in range(m * i, m * (i + 1)):
            for k in range(m * i, m * (i + 1)):
                if j != k:
                    node1, node2 = outlier_idx[j], outlier_idx[k]
                    new_edges.append(
                        torch.tensor([[node1, node2]], dtype=torch.long))

    new_edges = torch.cat(new_edges)

    # drop edges with probability p
    if p != 0:
        indices = torch.randperm(len(new_edges))[:int((1-p) * len(new_edges))]
        new_edges = new_edges[indices]

    y_outlier = torch.zeros(data.x.shape[0], dtype=torch.long)
    y_outlier[outlier_idx] = 1

    data.edge_index = torch.cat([data.edge_index, new_edges.T], dim=1)

    return data, y_outlier


def gen_contextual_outliers(data, n, k, random_state=None):
    """Generating contextual outliers according to paper
    "Deep Anomaly Detection on Attributed Networks"
    <https://epubs.siam.org/doi/abs/10.1137/1.9781611975673.67>.
    We randomly select n nodes as the attribute perturbation candidates.
    For each selected node i, we randomly pick another k nodes from the data
    and select the node j whose attributes deviate the most from node i
    among the k nodes by maximizing the Euclidean distance ||xi − xj ||2.
    Afterwards, we then change the attributes xi of node i to xj.

    Parameters
    ----------
    data : PyTorch Geometric Data instance (torch_geometric.data.Data)
        The input data.
    n : int
        Number of nodes converting to outliers.
    k : int
        Number of candidate nodes for each outlier node.
    random_state : int, optional
        The seed to control the randomness, Default: ``None``.

    Returns
    -------
    data : PyTorch Geometric Data instance (torch_geometric.data.Data)
        The contextual outlier graph with modified node attributes.
    y_outlier : torch.Tensor
        The outlier label tensor where 1 represents outliers and 0 represents
        regular nodes.
    """

    if not isinstance(data, Data):
        raise TypeError("data should be torch_geometric.data.Data")

    if isinstance(n, int):
        check_parameter(n, low=0, high=data.num_nodes, param_name='n')
    else:
        raise ValueError("n should be int, got %s" % n)

    if isinstance(k, int):
        check_parameter(k, low=0, high=data.num_nodes - n, param_name='k')
    else:
        raise ValueError("k should be int, got %s" % k)

    if random_state:
        np.random.seed(random_state)

    node_set = set(range(data.num_nodes))

    outlier_idx = np.random.choice(list(node_set), size=n, replace=False)
    candidate_set = node_set.difference(set(outlier_idx))
    candidate_idx = np.random.choice(list(candidate_set), size=n * k)

    for i, idx in enumerate(outlier_idx):
        cur_candidates = candidate_idx[k * i: k * (i + 1)]

        euclidean_dist = torch.cdist(data.x[idx].unsqueeze(0), data.x[list(
            cur_candidates)])
        max_dist_idx = torch.argmax(euclidean_dist)
        max_dist_node = list(cur_candidates)[max_dist_idx]

        data.x[idx] = data.x[max_dist_node]

    y_outlier = torch.zeros(data.x.shape[0], dtype=torch.long)
    y_outlier[outlier_idx] = 1

    return data, y_outlier
