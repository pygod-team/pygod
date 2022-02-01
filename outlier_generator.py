# -*- coding: utf-8 -*-
"""
This file including functions to generate different types of outliers given
the input dataset for benchmarking
"""

from typing import Tuple
import numpy as np

import torch
import torch_geometric

# set the random seed
np.random.seed(999)


def gen_structural_outliers(data: torch_geometric.datasets.data, m: int,
                            n: int) -> Tuple[torch_geometric.datasets.data,
                                             torch.tensor]:
    """Generating structural outliers according to paper
    "Deep Anomaly Detection on Attributed Networks"
    <https://epubs.siam.org/doi/abs/10.1137/1.9781611975673.67>.

    We randomly select m nodes from the network and then make those nodes
    fully connected, and then all the m nodes in the clique are regarded as
    outliers. We iteratively repeat this process until a number of n
    cliques are generated and the total number of structural outliers is m×n.

    Parameters
    ----------
    data : torch_geometric.datasets.data
        PyG dataset object.
    m : int
        Number nodes in the outlier cliques.
    n : int
        Number of outlier cliques.
    Returns
    -------
    data : torch_geometric.datasets.data
        The structural outlier graph with injected edges.
    y_outlier : torch.tensor
        The outlier label tensor where 1 represents outliers and 0 represents
        regular nodes.
    """

    node_set = set(range(data.x.shape[0]))
    new_edges = []
    sample_indices = []
    for i in range(0, n):

        sampled_idx = np.random.choice(list(node_set), size=m, replace=False)

        sample_indices += sampled_idx.tolist()

        for j, id1 in enumerate(sampled_idx):
            for k, id2 in enumerate(sampled_idx):
                if j != k:
                    new_edges.append(
                        torch.tensor([[id1, id2]], dtype=torch.long))

        node_set = node_set.difference(set(sampled_idx))

    y_outlier = torch.zeros(data.x.shape[0], dtype=torch.long)
    y_outlier[sample_indices] = 1

    data.edge_index = torch.cat([data.edge_index, torch.cat(new_edges).T],
                                dim=1)

    return data, y_outlier


def gen_attribute_outliers(data: torch_geometric.datasets.data, n: int,
                           k: int) -> Tuple[torch_geometric.datasets.data,
                                            torch.tensor]:
    """Generating attribute outliers according to paper
    "Deep Anomaly Detection on Attributed Networks"
    <https://epubs.siam.org/doi/abs/10.1137/1.9781611975673.67>.

    We randomly select n nodes as the attribute perturbation candidates.
    For each selected node i, we randomly pick another k nodes from the data
    and select the node j whose attributes deviate the most from node i
    among the k nodes by maximizing the Euclidean distance ||xi − xj ||2.
    Afterwards, we then change the attributes xi of node i to xj.

    Parameters
    ----------
    data : torch_geometric.datasets.data
        PyG dataset object.
    n : int
        Number of nodes converting to outliers.
    k : int
        Number of candidate nodes for each outlier node.
    Returns
    -------
    data : torch_geometric.datasets.data
        The attribute outlier graph with modified node attributes.
    y_outlier : torch.tensor
        The outlier label tensor where 1 represents outliers and 0 represents
        regular nodes.
    """

    node_set = set(range(data.x.shape[0]))
    target_nodes = np.random.choice(list(node_set), size=n, replace=False)
    candidate_set = node_set.difference(set(target_nodes))

    for u in target_nodes:
        temp_candidates = np.random.choice(list(candidate_set), size=k,
                                           replace=False)

        euclidean_dist = torch.cdist(data.x[u].unsqueeze(0), data.x[list(
            temp_candidates)])
        max_dist_idx = torch.argmax(euclidean_dist)
        max_dist_node = list(temp_candidates)[max_dist_idx]

        data.x[u] = data.x[max_dist_node]

    y_outlier = torch.zeros(data.x.shape[0], dtype=torch.long)
    y_outlier[target_nodes] = 1

    return data, y_outlier


def gen_structural_outliers_II(data):
    """Generating structural outliers according to paper
    "Outlier Resistant Unsupervised Deep Architectures for Attributed
    Network Embedding"
    <http://eprints.iisc.ac.in/64704/1/WSDM%2020.pdf>.

    (1) Computing the probability distribution of number of nodes in each
    class, (2) selecting a class using these probabilities. For a
    structural outlier: (3) plant an outlier node in the selected class
    such that the node has (m + /−10%) of edges connecting nodes from the
    remaining (unselected) classes where m is the average degree of a node
    in the selected class and (4) the attributes of the structural outlier
    node are made semantically consistent with the keywords sampled from
    the selected class.

    """
    # TODO Implement the other approach to generate the outliers


def gen_attribute_outliers_II(data):
    """Generating attribute outliers according to paper
    "Outlier Resistant Unsupervised Deep Architectures for Attributed
    Network Embedding"
    <http://eprints.iisc.ac.in/64704/1/WSDM%2020.pdf>.

    (1) Computing the probability distribution of number of nodes in each
    class, (2) selecting a class using these probabilities. For an
    attribute outlier: (3) plant a node in the selected class where its
    attribute is randomly sampled from nodes in different classes (4)
    the edges of the attribute outliers are sampled from edges of nodes in
    the selected class and the planted node degree is (m + /−10%) where m
    is the average degree of a node in the selected class.


    """
    # TODO Implement the other approach to generate the outliers


def gen_combined_outliers(data):
    """Generating combined outliers according to paper
    "Outlier Resistant Unsupervised Deep Architectures for Attributed
    Network Embedding"
    <http://eprints.iisc.ac.in/64704/1/WSDM%2020.pdf>.

    (1) Computing the probability distribution of number of nodes in each
    class, (2) selecting a class using these probabilities. For an
    attribute outlier: (3) plant a node in the selected class where its
    attribute is randomly sampled from nodes in different classes (4)
    the edges of the attribute outliers are sampled from edges of nodes in
    the selected class and the planted node degree is (m + /−10%) where m
    is the average degree of a node in the selected class.

    """
    # TODO Implement the other approach to generate the outliers
