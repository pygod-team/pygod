# -*- coding: utf-8 -*-
import torch
import unittest
from torch.testing import assert_close
from pygod.utils import to_edge_score, to_graph_score


class TestConverter(unittest.TestCase):
    def setUp(self):
        self.data = torch.load('pygod/test/test_graph.pt')

    def test_to_edge_score1(self):
        edge_score = to_edge_score(torch.Tensor([1, 0.4, 0.5, 0.2, 0.3, 0]),
                                   self.data.edge_index)

        assert_close(edge_score, torch.Tensor([0.7, 0.75, 0.6, 0.65, 0.5]))

    def test_to_edge_score2(self):
        edge_score = to_edge_score(torch.Tensor([3, 2, 1, 4, -1, 0.6]),
                                   self.data.edge_index)

        assert_close(edge_score, torch.Tensor([2.5, 2, 3.5, 1, 1.8]))

    def test_to_graph_score1(self):
        score = to_graph_score(torch.Tensor([1, 0.4, 0.5, 0.2, 0.3, 0]))
        assert_close(score, torch.tensor(0.4))

    def test_to_graph_score2(self):
        score = to_graph_score(torch.Tensor([3, 2, 1, 4, -1, 0.6]))
        assert_close(score, torch.tensor(1.6))
