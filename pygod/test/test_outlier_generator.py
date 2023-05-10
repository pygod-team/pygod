# -*- coding: utf-8 -*-
import unittest
from numpy.testing import assert_allclose
from numpy.testing import assert_equal
from numpy.testing import assert_raises

import torch
import copy as cp
from pygod.generator import gen_contextual_outlier
from pygod.generator import gen_structural_outlier


class TestData(unittest.TestCase):
    def setUp(self):
        self.m_structure = 5
        self.n_structure = 5
        self.n_context = 5
        self.k_context = 5
        self.seed = 42

        self.data = torch.load('pygod/test/train_graph.pt')

    def test_structural_outliers(self):
        data = cp.deepcopy(self.data)
        data, y_outlier = gen_structural_outlier(data=data,
                                                 m=self.m_structure,
                                                 n=self.n_structure)

        assert_equal(self.data.x.numpy(), data.x.numpy())
        assert (self.data.num_nodes == y_outlier.shape[0])
        assert (data.num_edges ==
                self.m_structure * (self.m_structure - 1) * self.n_structure
                + self.data.num_edges)
        assert_equal(self.data.edge_index.numpy(),
                     data.edge_index[:, :self.data.edge_index.shape[1]])
        assert (self.m_structure * self.n_structure == y_outlier.sum().item())

    def test_contextual_outliers(self):
        data = cp.deepcopy(self.data)
        data, y_outlier = gen_contextual_outlier(data=data,
                                                 n=self.n_context,
                                                 k=self.k_context)

        assert_equal(self.data.edge_index.numpy(),
                     data.edge_index.numpy())
        assert_allclose(self.data.x.numpy().shape, data.x.numpy().shape)
        assert (self.data.num_nodes == y_outlier.shape[0])
        assert (self.n_context == y_outlier.sum().item())

    def test_structural_outliers2(self):
        data, y_outlier = \
            gen_structural_outlier(data=cp.deepcopy(self.data),
                                   m=self.m_structure,
                                   n=self.n_structure,
                                   seed=self.seed)

        data2, y_outlier2 = \
            gen_structural_outlier(data=cp.deepcopy(self.data),
                                   m=self.m_structure,
                                   n=self.n_structure,
                                   seed=self.seed)

        assert_equal(data.x.numpy(), data2.x.numpy())
        assert_equal(data.edge_index.numpy(),
                     data2.edge_index.numpy())
        assert_equal(y_outlier.numpy(), y_outlier2.numpy())

    def test_contextual_outliers2(self):
        data, y_outlier = \
            gen_contextual_outlier(data=cp.deepcopy(self.data),
                                   n=self.n_context,
                                   k=self.k_context,
                                   seed=self.seed)

        data2, y_outlier2 = \
            gen_contextual_outlier(data=cp.deepcopy(self.data),
                                   n=self.n_context,
                                   k=self.k_context,
                                   seed=self.seed)

        assert_equal(data.x.numpy(), data2.x.numpy())
        assert_equal(data.edge_index.numpy(),
                     data2.edge_index.numpy())
        assert_equal(y_outlier.numpy(), y_outlier2.numpy())

    def test_structural_outliers3(self):
        data = cp.deepcopy(self.data)

        with assert_raises(ValueError):
            gen_structural_outlier(data=data,
                                   m='not int',
                                   n=self.n_structure)

        with assert_raises(ValueError):
            gen_structural_outlier(data=data,
                                   m=self.m_structure,
                                   n='not int')

    def test_contextual_outliers3(self):
        data = cp.deepcopy(self.data)

        with assert_raises(ValueError):
            gen_contextual_outlier(data=data,
                                   n='not int',
                                   k=self.k_context)

        with assert_raises(ValueError):
            gen_contextual_outlier(data=data,
                                   n=self.n_context,
                                   k='not int')

    def test_data_type(self):
        with assert_raises(TypeError):
            gen_structural_outlier('not Data', 1, 1)
        with assert_raises(TypeError):
            gen_contextual_outlier('not Data', 1, 1)

    def test_structural_outliers4(self):
        data = cp.deepcopy(self.data)
        data, y_outlier = gen_structural_outlier(data=data,
                                                 m=self.m_structure,
                                                 n=self.n_structure,
                                                 p=0.2)

        assert (data.num_edges <
                self.m_structure * (self.m_structure - 1) * self.n_structure
                + self.data.num_edges)
