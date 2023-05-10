# -*- coding: utf-8 -*-

import torch
import pygod
import unittest

from numpy.testing import assert_equal
from numpy.testing import assert_raises
from numpy.testing import assert_warns

from pygod.utils import (
    validate_device,
    check_parameter,
    load_data,
    logger,
    init_detector,
    init_nn,
    is_fitted
)


class TestUtils(unittest.TestCase):

    def test_validate_device(self):
        assert (validate_device(-1) == 'cpu')

        if torch.cuda.is_available():
            with assert_raises(ValueError):
                validate_device(9999)
            assert (validate_device(0) == 'cuda:0')
            assert (validate_device(
                torch.cuda.device_count() - 1) == 'cuda:{}'.format(
                torch.cuda.device_count() - 1))
        else:
            with assert_warns(Warning):
                assert (validate_device(0) == 'cpu')

    def test_check_parameter(self):
        # verify parameter type correction
        with assert_raises(TypeError):
            check_parameter('f', 0, 100)

        with assert_raises(TypeError):
            check_parameter(1, 'f', 100)

        with assert_raises(TypeError):
            check_parameter(1, 0, 'f')

        # if low and high are both unset
        with assert_raises(ValueError):
            check_parameter(50)

        # if low <= high
        with assert_raises(ValueError):
            check_parameter(50, 100, 99)

        with assert_raises(ValueError):
            check_parameter(50, 100, 100)

        # check one side
        with assert_raises(ValueError):
            check_parameter(50, low=100)
        with assert_raises(ValueError):
            check_parameter(50, high=0)

        assert_equal(True, check_parameter(50, low=10))
        assert_equal(True, check_parameter(50, high=100))

        # if check fails
        with assert_raises(ValueError):
            check_parameter(-1, 0, 100)

        with assert_raises(ValueError):
            check_parameter(101, 0, 100)

        with assert_raises(ValueError):
            check_parameter(0.5, 0.2, 0.3)

        # if check passes
        assert_equal(True, check_parameter(50, 0, 100))

        assert_equal(True, check_parameter(0.5, 0.1, 0.8))

        # if includes left or right bounds
        with assert_raises(ValueError):
            check_parameter(100, 0, 100, include_left=False,
                            include_right=False)
        with assert_raises(ValueError):
            check_parameter(100, 0, 100, include_left=True,
                            include_right=False)
        with assert_raises(ValueError):
            check_parameter(0, 0, 100, include_left=False,
                            include_right=True)
        with assert_raises(ValueError):
            check_parameter(-1, 0, 100, include_left=True,
                            include_right=True)
        assert_equal(True, check_parameter(0, 0, 100, include_left=True,
                                           include_right=False))
        assert_equal(True, check_parameter(0, 0, 100, include_left=True,
                                           include_right=True))
        assert_equal(True, check_parameter(100, 0, 100, include_left=False,
                                           include_right=True))
        assert_equal(True, check_parameter(100, 0, 100, include_left=True,
                                           include_right=True))

    def test_load_data(self):
        data1 = load_data('disney')
        assert data1.edge_index.shape[1] == 335

        data2 = load_data('disney')
        assert data2.x.shape[0] == 124

    def test_logger(self):
        logger(epoch=1,
               loss=0.1,
               score=torch.Tensor([1, 0.1]),
               target=torch.tensor([1, 0]),
               time=0.1,
               verbose=3,
               train=True,
               deep=True)

        logger(epoch=1,
               loss=(0.1, 0.2),
               score=torch.Tensor([1, 0.1]),
               target=torch.tensor([1, 0]),
               time=0.1,
               verbose=3,
               train=True,
               deep=True)

    def test_init_detector(self):
        detector = init_detector('DOMINANT')
        assert type(detector) is pygod.detector.DOMINANT

    def test_init_nn(self):
        nn = init_nn('DONEBase', x_dim=4, s_dim=4)
        assert type(nn) is pygod.nn.DONEBase

    def test_is_fitted(self):
        detector = pygod.detector.DOMINANT()
        with assert_raises(Exception):
            is_fitted(detector)

        detector.fit(torch.load('test_graph.pt'))
        is_fitted(detector)
