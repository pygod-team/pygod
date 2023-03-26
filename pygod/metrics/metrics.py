# -*- coding: utf-8 -*-
"""
Metrics used to evaluate the outlier detection performance
"""
# Author: Yingtong Dou <ytongdou@gmail.com>, Kay Liu <zliu234@uic.edu>
# License: BSD 2 clause

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    recall_score,
    precision_score,
    f1_score
)


def eval_roc_auc(labels, pred):
    """
    ROC-AUC score for binary classification.

    Parameters
    ----------
    labels : torch.Tensor
        Labels in shape of ``(N, )``, where 1 represents outliers,
        0 represents normal instances.
    pred : torch.Tensor
        Outlier scores in shape of ``(N, )``.

    Returns
    -------
    roc_auc : float
        Average ROC-AUC score across different labels.
    """

    roc_auc = roc_auc_score(y_true=labels, y_score=pred)
    return roc_auc


def eval_recall_at_k(labels, pred, k=None):
    """
    Recall score for top k instances with the highest outlier scores.

    Parameters
    ----------
    labels : torch.Tensor
        Labels in shape of ``(N, )``, where 1 represents outliers,
        0 represents normal instances.
    pred : torch.Tensor
        Outlier scores in shape of ``(N, )``.
    k : int, optional
        The number of instances to evaluate. ``None`` for
        recall. Default: ``None``.

    Returns
    -------
    recall_at_k : float
        Recall for top k instances with the highest outlier scores.
    """

    if k is None:
        recall_at_k = recall_score(y_true=labels, y_score=pred)
    else:
        recall_at_k = sum(labels[pred.topk(k).indices]) / sum(labels)
    return recall_at_k


def eval_precision_at_k(labels, pred, k=None):
    """
    Precision score for top k instances with the highest outlier scores.

    Parameters
    ----------
    labels : torch.Tensor
        Labels in shape of ``(N, )``, where 1 represents outliers,
        0 represents normal instances.
    pred : torch.Tensor
        Outlier scores in shape of ``(N, )``.
    k : int, optional
        The number of instances to evaluate. ``None`` for
        precision. Default: ``None``.

    Returns
    -------
    precision_at_k : float
        Precision for top k instances with the highest outlier scores.
    """

    if k is None:
        precision_at_k = precision_score(y_true=labels, y_score=pred)
    else:
        precision_at_k = sum(labels[pred.topk(k).indices]) / k
    return precision_at_k


def eval_average_precision(labels, pred):
    """
    Average precision score for binary classification.

    Parameters
    ----------
    labels : torch.Tensor
        Labels in shape of ``(N, )``, where 1 represents outliers,
        0 represents normal instances.
    pred : torch.Tensor
        Outlier scores in shape of ``(N, )``.

    Returns
    -------
    ap : float
        Average precision score.
    """

    ap = average_precision_score(y_true=labels, y_score=pred)
    return ap


def eval_f1(labels, pred):
    """
    F1 score for binary classification.

    Parameters
    ----------
    labels : torch.Tensor
        Labels in shape of ``(N, )``, where 1 represents outliers,
        0 represents normal instances.
    pred : torch.Tensor
        Outlier scores in shape of ``(N, )``.

    Returns
    -------
    f1 : float
        F1 score.
    """

    f1 = f1_score(y_true=[labels], y_score=[pred])
    return f1
