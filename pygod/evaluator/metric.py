# -*- coding: utf-8 -*-
"""
Metrics used to evaluate the anomaly detection performance
"""
# Author: Yingtong Dou <ytongdou@gmail.com>
# License: BSD 2 clause

import numpy as np
from sklearn.metrics import roc_auc_score


def eval_rocauc(pred, labels):
    """
    Description
    -----------
    ROC-AUC score for multi-label node classification.
    Parameters
    ----------
    pred : numpy.array
        Output logits of model in form of ``N * 2``.
    labels : numpy.array
        Labels in form of ``1 * N``.

    Returns
    -------
    rocauc : float
        Average ROC-AUC score across different labels.
    """

    # anomaly detection is a binary classification problem
    return roc_auc_score(y_true=labels, y_score=pred[:, 1])


def eval_recall_at_k(pred, labels, k):
    pass


def eval_precision_at_k(pred, labels, k):
    pass
