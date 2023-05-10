# -*- coding: utf-8 -*-
import unittest
from pygod.detector import DOMINANT


class TestBase(unittest.TestCase):

    def test_contamination(self):
        with self.assertRaises(ValueError):
            DOMINANT(epoch=5, contamination=10)

    def test_repr(self):
        self.assertEqual(repr(DOMINANT())[:8], 'DOMINANT')
