# -*- coding: utf-8 -*-
"""
Metrics used to evaluate the anomaly detection performance
"""
# Author: Yingtong Dou <ytongdou@gmail.com>, Kay Liu <zliu234@uic.edu>
# License: BSD 2 clause

from sklearn.metrics import roc_auc_score, average_precision_score, ndcg_score


def eval_roc_auc(labels, pred):
    """
    ROC-AUC score for binary classification.

    Parameters
    ----------
    labels : numpy.array
        Labels in shape of ``(N, )``, where 1 represents outliers,
        0 represents normal instances.
    pred : numpy.array
        Outlier scores in shape of ``(N, )``.

    Returns
    -------
    roc_auc : float
        Average ROC-AUC score across different labels.
    """

    # outlier detection is a binary classification problem
    roc_auc = roc_auc_score(y_true=labels, y_score=pred)
    return roc_auc


def eval_recall_at_k(labels, pred, k):
    """
    Recall score for top k instances with the highest outlier scores.

    Parameters
    ----------
    labels : numpy.ndarray
        Labels in shape of ``(N, )``, where 1 represents outliers,
        0 represents normal instances.
    pred : numpy.ndarray
        Outlier scores in shape of ``(N, )``.
    k : int
        The number of instances to evaluate.

    Returns
    -------
    recall_at_k : float
        Recall for top k instances with the highest outlier scores.
    """

    N = len(pred)
    recall_at_k = sum(labels[pred.argpartition(N - k)[-k:]]) / sum(labels)

    return recall_at_k


def eval_precision_at_k(labels, pred, k):
    """
    Precision score for top k instances with the highest outlier scores.

    Parameters
    ----------
    labels : numpy.ndarray
        Labels in shape of ``(N, )``, where 1 represents outliers,
        0 represents normal instances.
    pred : numpy.ndarray
        Outlier scores in shape of ``(N, )``.
    k : int
        The number of instances to evaluate.

    Returns
    -------
    precision_at_k : float
        Precision for top k instances with the highest outlier scores.
    """

    N = len(pred)
    precision_at_k = sum(labels[pred.argpartition(N - k)[-k:]]) / k

    return precision_at_k


def eval_average_precision(labels, pred):
    """
    Average precision score for binary classification.

    Parameters
    ----------
    labels : numpy.ndarray
        Labels in shape of ``(N, )``, where 1 represents outliers,
        0 represents normal instances.
    pred : numpy.ndarray
        Outlier scores in shape of ``(N, )``.

    Returns
    -------
    ap : float
        Average precision score.
    """

    # outlier detection is a binary classification problem
    ap = average_precision_score(y_true=labels, y_score=pred)
    return ap

def eval_ndcg(labels, pred):
    """
    Normalized discounted cumulative gain for ranking.

    Parameters
    ----------
    labels : numpy.ndarray
        Labels in shape of ``(N, )``, where 1 represents outliers,
        0 represents normal instances.
    pred : numpy.ndarray
        Outlier scores in shape of ``(N, )``.

    Returns
    -------
    ndcg : float
        Average precision score.
    """

    # outlier detection is a binary classification problem
    ndcg = ndcg_score(y_true=labels, y_score=pred)
    return ndcg
