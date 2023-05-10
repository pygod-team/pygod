# -*- coding: utf-8 -*-
import os
import unittest
from numpy.testing import assert_equal
from numpy.testing import assert_raises

import torch
from torch_geometric.seed import seed_everything
from pygod.detector import DOMINANT
from pygod.metric import eval_roc_auc

seed_everything(42)


class TestDOMINANT(unittest.TestCase):
    def setUp(self):
        # use the pre-defined fake graph with injected outliers
        # for testing purpose

        # the roc should be higher than this; it is model dependent
        self.roc_floor = 0.60

        self.data = torch.load(os.path.join('pygod/test/train_graph.pt'))

        self.detector = DOMINANT(epoch=5)
        self.detector.fit(self.data)

    def test_parameters(self):
        assert (hasattr(self.detector, 'decision_score_') and
                self.detector.decision_score_ is not None)
        assert (hasattr(self.detector, 'label_') and
                self.detector.label_ is not None)
        assert (hasattr(self.detector, 'threshold_') and
                self.detector.threshold_ is not None)
        assert (hasattr(self.detector, 'model') and
                self.detector.model is not None)

    def test_train_score(self):
        assert_equal(len(self.detector.decision_score_), len(self.data.y))

    def test_prediction_score(self):
        score = self.detector.decision_function(self.data)

        # check score shapes
        assert_equal(score.shape[0], self.data.y.shape[0])

        # check performance
        assert (eval_roc_auc(self.data.y, score) >= self.roc_floor)

    def test_prediction_label(self):
        pred = self.detector.predict(self.data)
        assert_equal(pred.shape[0], self.data.y.shape[0])

    def test_prediction_prob_linear(self):
        _, prob = self.detector.predict(self.data,
                                        return_prob=True,
                                        prob_method='linear')
        assert (prob.min() >= 0)
        assert (prob.max() <= 1)

    def test_prediction_prob_unify(self):
        _, prob = self.detector.predict(self.data,
                                        return_prob=True,
                                        prob_method='unify')
        assert (prob.min() >= 0)
        assert (prob.max() <= 1)

    def test_prediction_prob_parameter(self):
        with assert_raises(ValueError):
            self.detector.predict(self.data,
                                  return_prob=True,
                                  prob_method='something')

    def test_prediction_conf(self):
        pred, conf = self.detector.predict(self.data, return_conf=True)
        assert_equal(pred.shape[0], self.data.y.shape[0])
        assert_equal(conf.shape[0], self.data.y.shape[0])
        assert (conf.min() >= 0)
        assert (conf.max() <= 1)
