# -*- coding: utf-8 -*-
"""Outlier Score Converters
"""
# Author: Kay Liu <zliu234@uic.edu>
# License: BSD 2 clause


def to_edge_score(score, edge_index):
    """Convert outlier node score to outlier edge score by averaging the
    scores of two nodes connected by an edge.

    Parameters
    ----------
    score : torch.Tensor
        The node score.
    edge_index : torch.Tensor
        The edge index.

    Returns
    -------
    score : torch.Tensor
        The edge score.
    """
    score = (score[edge_index[0]] + score[edge_index[1]]) / 2
    return score


def to_graph_score(score):
    """Convert outlier node score to outlier graph score by averaging
    the scores of all nodes in a graph.

    Parameters
    ----------
    score : torch.Tensor
        The node score.

    Returns
    -------
    score : torch.Tensor
        The graph score.
    """

    return score.mean(dim=-1)
