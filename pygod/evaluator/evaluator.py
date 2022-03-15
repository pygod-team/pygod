# -*- coding: utf-8 -*-
"""
Evaluating given model using benchmark data and metrics
"""
# Author: Yingtong Dou <ytongdou@gmail.com>
# License: BSD 2 clause

import torch
import argparse
import os.path as osp
from torch_geometric.datasets import AttributedGraphDataset

from ..utils import outlier_generator
from .metric import eval_rocauc, eval_recall_at_k, eval_precision_at_k


class Evaluator(object):
    """The base class for evaluating anomaly detection model performance on
    the benchmark anomaly data
    """
    def __init__(self, dataset, outlier_type, **kwargs):

        # the clean graph used for training
        self.old_G = dataset

        # generate outliers in the graph
        if outlier_type == 'structure':
            self.new_G, self.y_outlier = \
                outlier_generator.gen_structure_outliers(dataset, kwargs[
                    'm'], kwargs['n'])
        if outlier_type == 'attribute':
            self.new_G, self.y_outlier = \
                outlier_generator.gen_attribute_outliers(dataset, kwargs[
                    'n'], kwargs['k'])

    def eval(self, clf, metric, args, k=0, verbose=False):
        """Evaluate the anomaly detection performance of a given model.

        Parameters
        ----------
        clf : BaseDetector
            Model implemented based on ``models.base.BaseDetector``.
        metric : str
            The evaluation metric name.
        k: int
            The K used to computed recall@K and precision@K
        args : argparse.ArgumentParser().parse_args()
            Arguments of the input anomaly detector
        verbose : bool, optional
            Whether to display logs. Default: ``False``.

        Returns
        -------
        score : float
            The evaluation result of corresponding metric.
        """
        if verbose:
            print('training...')
        clf.fit(self.old_G, args)

        if verbose:
            print('predicting for probability')
        prob = clf.predict_proba(self.new_G, args)

        if verbose:
            print('Probability', prob)

        if metric == 'rocauc':
            score = eval_rocauc(prob, self.y_outlier)
        if metric == 'recall@k':
            score = eval_recall_at_k(prob, self.y_outlier, k)
        if metric == 'precision@k':
            score = eval_precision_at_k(prob, self.y_outlier, k)

        return score
