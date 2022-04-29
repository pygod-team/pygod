# -*- coding: utf-8 -*-
import os
import unittest
# noinspection PyProtectedMember
from numpy.testing import assert_equal
from numpy.testing import assert_raises

import torch
from torch_geometric.seed import seed_everything
from pygod.models import ANEMONE
from pygod.utils.metric import eval_roc_auc

seed_everything(42)


class TestAnemone(unittest.TestCase):
    def setUp(self):
        # use the pre-defined fake graph with injected outliers
        # for testing purpose

        # the roc should be higher than this; it is model dependent
        self.roc_floor = 0.5

        test_graph = torch.load(os.path.join('pygod', 'test', 'test_graph.pt'))
        self.data = test_graph

        self.model = ANEMONE()
        self.model.fit(self.data)

    def test_parameters(self):
        assert (hasattr(self.model, 'decision_scores_') and
                self.model.decision_scores_ is not None)
        assert (hasattr(self.model, 'labels_') and
                self.model.labels_ is not None)
        assert (hasattr(self.model, 'threshold_') and
                self.model.threshold_ is not None)
        assert (hasattr(self.model, '_mu') and
                self.model._mu is not None)
        assert (hasattr(self.model, '_sigma') and
                self.model._sigma is not None)
        assert (hasattr(self.model, 'model') and
                self.model.model is not None)

    def test_train_scores(self):
        assert_equal(len(self.model.decision_scores_), len(self.data.y))

    def test_prediction_scores(self):
        pred_scores = self.model.decision_function(self.data)

        # check score shapes
        assert_equal(pred_scores.shape[0], self.data.y.shape[0])

        # check performance
        assert (eval_roc_auc(self.data.y, pred_scores) >= self.roc_floor)

    def test_prediction_labels(self):
        pred_labels = self.model.predict(self.data)
        assert_equal(pred_labels.shape[0], self.data.y.shape[0])

    def test_prediction_proba(self):
        pred_proba = self.model.predict_proba(self.data)
        assert (pred_proba.min() >= 0)
        assert (pred_proba.max() <= 1)

    def test_prediction_proba_linear(self):
        pred_proba = self.model.predict_proba(self.data, method='linear')
        assert (pred_proba.min() >= 0)
        assert (pred_proba.max() <= 1)

    def test_prediction_proba_unify(self):
        pred_proba = self.model.predict_proba(self.data, method='unify')
        assert (pred_proba.min() >= 0)
        assert (pred_proba.max() <= 1)

    def test_prediction_proba_parameter(self):
        with assert_raises(ValueError):
            self.model.predict_proba(self.data, method='something')

    def test_prediction_labels_confidence(self):
        pred_labels, confidence = self.model.predict(self.data,
                                                     return_confidence=True)
        assert_equal(pred_labels.shape[0], self.data.y.shape[0])
        assert_equal(confidence.shape[0], self.data.y.shape[0])
        assert (confidence.min() >= 0)
        assert (confidence.max() <= 1)

    def test_prediction_proba_linear_confidence(self):
        pred_proba, confidence = self.model.predict_proba(self.data,
                                                          method='linear',
                                                          return_confidence=True)
        assert (pred_proba.min() >= 0)
        assert (pred_proba.max() <= 1)

        assert_equal(confidence.shape[0], self.data.y.shape[0])
        assert (confidence.min() >= 0)
        assert (confidence.max() <= 1)

    def test_model_clone(self):
        pass
        # clone_clf = clone(self.model)

    def tearDown(self):
        pass
        # remove the data folder
        # rmtree(self.path)


if __name__ == '__main__':
    unittest.main()
