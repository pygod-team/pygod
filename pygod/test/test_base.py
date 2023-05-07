# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function

import os

import unittest
from numpy.testing import assert_equal
from numpy.testing import assert_raises

import torch
import numpy as np

from ..detectors import Detector


class MyEstimator(Detector):

    def __init__(self, l1=0, empty=None):
        self.l1 = l1
        self.empty = empty

    def fit(self, X, y=None):
        pass

    def decision_function(self, X):
        pass


# noinspection PyMissingConstructor
class K(Detector):
    def __init__(self, c=None, d=None):
        self.c = c
        self.d = d

    def fit(self, X, y=None):
        pass

    def decision_function(self, X):
        pass


# noinspection PyMissingConstructor
class T(Detector):
    def __init__(self, a=None, b=None):
        self.a = a
        self.b = b

    def fit(self, X, y=None):
        pass

    def decision_function(self, X):
        pass


# noinspection PyMissingConstructor
class ModifyInitParams(Detector):
    """Deprecated behavior.
    Equal parameters but with a type cast.
    Doesn't fulfill a is a
    """

    def __init__(self, a=np.array([0])):
        self.a = a.copy()

    def fit(self, X, y=None):
        pass

    def decision_function(self, X):
        pass


# noinspection PyMissingConstructor
class VargEstimator(Detector):
    """scikit-learn estimators shouldn't have vargs."""

    def __init__(self, *vargs):
        pass

    def fit(self, X, y=None):
        pass

    def decision_function(self, X):
        pass


class Dummy1(Detector):
    def __init__(self, contamination=0.1):
        super(Dummy1, self).__init__(contamination=contamination)

    def decision_function(self, X):
        pass

    def fit(self, X, y=None):
        pass


class Dummy2(Detector):
    def __init__(self, contamination=0.1):
        super(Dummy2, self).__init__(contamination=contamination)

    def decision_function(self, X):
        pass

    def fit(self, X, y=None):
        return X


class Dummy3(Detector):
    def __init__(self, contamination=0.1):
        super(Dummy3, self).__init__(contamination=contamination)

    def decision_function(self, X):
        pass

    def fit(self, X, y=None):
        self.labels_ = X


class TestBase(unittest.TestCase):
    def setUp(self):
        test_graph = torch.load(os.path.join('test_graph.pt'))
        self.data = test_graph

    def test_init(self):
        """
        Test base class initialization

        :return:
        """
        self.dummy_clf = Dummy1()
        assert_equal(self.dummy_clf.contamination, 0.1)

        self.dummy_clf = Dummy1(contamination=0.2)
        assert_equal(self.dummy_clf.contamination, 0.2)

        with assert_raises(ValueError):
            Dummy1(contamination=0.51)

        with assert_raises(ValueError):
            Dummy1(contamination=0)

        with assert_raises(ValueError):
            Dummy1(contamination=-0.5)

    def test_fit(self):
        self.dummy_clf = Dummy2()
        assert_equal(self.dummy_clf.fit(0), 0)

    def test_predict_proba(self):
        # TODO: create uniform testcases
        pass

    def test_predict_confidence(self):
        # TODO: create uniform testcases
        pass

    def test_rank(self):
        # TODO: create uniform testcases
        pass

    def test_repr(self):
        # Smoke test the repr of the base estimator.
        my_estimator = MyEstimator()
        repr(my_estimator)
        test = T(K(), K())
        assert_equal(
            repr(test),
            "T(a=K(c=None, d=None), b=K(c=None, d=None))"
        )

        some_est = T(a=["long_params"] * 1000)
        assert_equal(len(repr(some_est)), 415)

    def test_str(self):
        # Smoke test the str of the base estimator
        my_estimator = MyEstimator()
        str(my_estimator)


if __name__ == '__main__':
    unittest.main()
