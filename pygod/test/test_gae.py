# -*- coding: utf-8 -*-
import os
import unittest
from numpy.testing import assert_equal
from numpy.testing import assert_raises

import torch
from torch_geometric.seed import seed_everything
from pygod.detector import GAE
from pygod.metric import eval_roc_auc

seed_everything(42)


class TestGAE(unittest.TestCase):
    def setUp(self):
        # use the pre-defined fake graph with injected outliers
        # for testing purpose

        # the roc should be higher than this; it is model dependent
        self.roc_floor = 0.55

        test_graph = torch.load(os.path.join('pygod/test/train_graph.pt'))
        self.data = test_graph

        self.model = GAE(epoch=5)
        self.model.fit(self.data)

    def test_parameters(self):
        assert (hasattr(self.model, 'decision_score_') and
                self.model.decision_score_ is not None)
        assert (hasattr(self.model, 'label_') and
                self.model.label_ is not None)
        assert (hasattr(self.model, 'threshold_') and
                self.model.threshold_ is not None)
        assert (hasattr(self.model, 'model') and
                self.model.model is not None)

    def test_train_score(self):
        assert_equal(len(self.model.decision_score_), len(self.data.y))

    def test_prediction_score(self):
        score = self.model.decision_function(self.data)

        # check score shapes
        assert_equal(score.shape[0], self.data.y.shape[0])

        # check performance
        assert (eval_roc_auc(self.data.y, score) >= self.roc_floor)

    def test_prediction_label(self):
        pred = self.model.predict(self.data)
        assert_equal(pred.shape[0], self.data.y.shape[0])

    def test_prediction_prob_linear(self):
        _, prob = self.model.predict(self.data,
                                     return_prob=True,
                                     prob_method='linear')
        assert (prob.min() >= 0)
        assert (prob.max() <= 1)

    def test_prediction_prob_unify(self):
        _, prob = self.model.predict(self.data,
                                     return_prob=True,
                                     prob_method='unify')
        assert (prob.min() >= 0)
        assert (prob.max() <= 1)

    def test_prediction_prob_parameter(self):
        with assert_raises(ValueError):
            self.model.predict(self.data,
                               return_prob=True,
                               prob_method='something')

    def test_prediction_conf(self):
        pred, conf = self.model.predict(self.data, return_conf=True)
        assert_equal(pred.shape[0], self.data.y.shape[0])
        assert_equal(conf.shape[0], self.data.y.shape[0])
        assert (conf.min() >= 0)
        assert (conf.max() <= 1)

    def test_recon_s(self):
        self.model = GAE(epoch=5, recon_s=True)
        self.model.fit(self.data)
        _, score = self.model.predict(self.data, return_score=True)

        # check score shapes
        assert_equal(score.shape[0], self.data.y.shape[0])

        # check performance
        assert (eval_roc_auc(self.data.y, score) >= self.roc_floor)
