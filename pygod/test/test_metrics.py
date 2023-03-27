# -*- coding: utf-8 -*-
import unittest
from numpy.testing import assert_allclose

import torch
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score

from pygod.metrics import eval_roc_auc
from pygod.metrics import eval_recall_at_k
from pygod.metrics import eval_precision_at_k
from pygod.metrics import eval_average_precision
from pygod.metrics import eval_f1


class TestMetric(unittest.TestCase):

    def setUp(self):
        self.y = torch.tensor([0, 0, 1, 1, 1, 0, 0, 0, 1])
        self.score = torch.tensor([0.1, 0.2, 0.2, 0.8, 0.2, 0.5, 0.7, 0.9, 1.])
        self.pred = torch.tensor([0, 0, 0, 1, 0, 0, 1, 1, 1])

    def test_eval_roc_auc(self):
        assert_allclose(roc_auc_score(self.y, self.score),
                        eval_roc_auc(self.y, self.score))

    def test_eval_recall_at_k(self):
        assert_allclose(recall_score(self.y, self.pred),
                        eval_recall_at_k(self.y, self.score))
        assert_allclose(recall_score(self.y, self.pred),
                        eval_recall_at_k(self.y, self.score, k=4))

    def test_eval_precision_at_k(self):
        assert_allclose(precision_score(self.y, self.pred),
                        eval_precision_at_k(self.y, self.score))
        assert_allclose(precision_score(self.y, self.pred),
                        eval_precision_at_k(self.y, self.score, k=4))

    def test_eval_average_precision(self):
        assert_allclose(average_precision_score(self.y, self.score),
                        eval_average_precision(self.y, self.score))

    def test_eval_f1(self):
        assert_allclose(f1_score(self.y, self.pred),
                        eval_f1(self.y, self.pred))
