"""
Evaluating all models under the same settings
"""

import torch
import argparse
import os.path as osp
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor
from torch_geometric.nn import GATConv
from sklearn.metrics import roc_auc_score

from ..utils import outlier_generator
from .metric import eval_rocauc, eval_recall_at_k, eval_precision_at_k


class Evaluator(object):
    """The base class for evaluating anomaly detection model performance on
    the benchmark dataset
    """
    def __init__(self, dataset, outlier_type, **kwargs):

        if outlier_type == 'structure':
            self.G, self.y_outlier = \
                outlier_generator.gen_structure_outliers(dataset, m, n)
        if outlier_type == 'attribute':
            self.G, self.y_outlier = \
                outlier_generator.gen_attribute_outliers(dataset, n, k)

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
        clf.fit(self.G, args)

        if verbose:
            print('predicting for probability')
        prob = clf.predict_proba(self.G, args)

        if verbose:
            print('Probability', prob)

        if metric == 'rocauc':
            score = eval_rocauc(prob, self.y_outlier)
        if metric == 'recall@k':
            score = eval_recall_at_k(prob, self.y_outlier, k)
        if metric == 'recall@k':
            score = eval_recall_at_k(prob, self.y_outlier, k)

        return score


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='BlogCatalog',
                        help='dataset name: Flickr/BlogCatalog')
    parser.add_argument('--hidden_size', type=int, default=64,
                        help='dimension of hidden embedding (default: 64)')
    parser.add_argument('--epoch', type=int, default=3, help='Training epoch')
    parser.add_argument('--lr', type=float, default=5e-3, help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate')
    parser.add_argument('--alpha', type=float, default=0.8,
                        help='balance parameter')
    parser.add_argument('--device', default='cpu', type=str, help='cuda/cpu')

    args = parser.parse_args()

    # data loading
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data',
                    args.dataset)

    # this gives us a PyG data object
    G = AttributedGraphDataset(root='./data', name='cora')[0]

    # model initialization
    clf = Dominant()

    # evaluator initialization
    eval = Evaluator(dataset=G, outlier_type='attribute')

    # evaluate under different metrics
    score = eval.eval(clf=clf, metric='rocauc', args=args)
    print('ROC AUC score:', score)
