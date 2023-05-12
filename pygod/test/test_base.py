# -*- coding: utf-8 -*-
import unittest
from pygod.detector import DOMINANT


class TestBase(unittest.TestCase):

    def test_contamination(self):
        with self.assertRaises(ValueError):
            DOMINANT(epoch=5, contamination=10)

    def test_repr(self):
        self.assertEqual(repr(DOMINANT())[:8], 'DOMINANT')

    def test_params(self):
        with self.assertRaises(ValueError):
            DOMINANT(num_neigh=[1, 2], num_layers=3)

        with self.assertRaises(ValueError):
            DOMINANT(num_neigh='1, 2, 3')

