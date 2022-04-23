# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import os
import sys

import unittest

# noinspection PyProtectedMember
import torch

from numpy.testing import assert_equal
from numpy.testing import assert_raises

# temporary solution for relative imports in case pyod is not installed
# if pyod is installed, no need to use the following line
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pygod.utils.utility import check_parameter
from pygod.utils.utility import validate_device


class TestValidateDevice(unittest.TestCase):
    def setUp(self):
        pass

    def test_check_parameter_range(self):
        assert (validate_device('cpu') == 'cpu')

        if torch.cuda.is_available():
            with assert_raises(ValueError):
                validate_device(9999)

        if torch.cuda.is_available():
            assert (validate_device(0) == 'cuda:0')
            assert (validate_device(torch.cuda.device_count() - 1) == 'cuda:{}'.format(torch.cuda.device_count() - 1))


class TestParameters(unittest.TestCase):
    def setUp(self):
        pass

    def test_check_parameter_range(self):
        # verify parameter type correction
        with assert_raises(TypeError):
            check_parameter('f', 0, 100)

        with assert_raises(TypeError):
            check_parameter(1, 'f', 100)

        with assert_raises(TypeError):
            check_parameter(1, 0, 'f')

        # with assert_raises(TypeError):
        #     check_parameter(argmaxn(value_list=[1, 2, 3], n=1), 0, 100)

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
        assert_equal(True, check_parameter(0, 0, 100, include_left=True,
                                           include_right=False))
        assert_equal(True, check_parameter(0, 0, 100, include_left=True,
                                           include_right=True))
        assert_equal(True, check_parameter(100, 0, 100, include_left=False,
                                           include_right=True))
        assert_equal(True, check_parameter(100, 0, 100, include_left=True,
                                           include_right=True))

    def tearDown(self):
        pass
