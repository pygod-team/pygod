# -*- coding: utf-8 -*-
import unittest

# noinspection PyProtectedMember
from numpy.testing import assert_allclose
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import ndcg_score

from pygod.metrics import eval_roc_auc
from pygod.metrics import eval_recall_at_k
from pygod.metrics import eval_precision_at_k
from pygod.metrics import eval_average_precision
from pygod.metrics import eval_ndcg


class TestMetric(unittest.TestCase):

    def setUp(self):
        self.y = [0, 0, 1, 1, 1, 0, 0, 0, 1, 0]
        self.decision_scores_ = [0.1, 0.2, 0.2, 0.8, 0.2, 0.5, 0.7, 0.9, 1.,
                                 0.3]
        self.manual_labels = [0, 0, 0, 1, 0, 0, 1, 1, 1, 0]

    def test_eval_roc_auc(self):
        assert_allclose(roc_auc_score(self.y, self.manual_labels),
                        eval_roc_auc(self.y, self.decision_scores_))

    def test_eval_recall_at_k(self):
        assert_allclose(recall_score(self.y, self.manual_labels),
                        eval_recall_at_k(self.y, self.decision_scores_, k=4))

    def test_eval_precision_at_k(self):
        assert_allclose(precision_score(self.y, self.manual_labels),
                        eval_precision_at_k(self.y,
                                            self.decision_scores_,
                                            k=4))

    def test_eval_average_precision(self):
        assert_allclose(average_precision_score(self.y, self.decision_scores_),
                        eval_average_precision(self.y, self.decision_scores_))

    def test_eval_ndcg(self):
        assert_allclose(ndcg_score([self.y], [self.decision_scores_]),
                        eval_ndcg(self.y, self.decision_scores_))
