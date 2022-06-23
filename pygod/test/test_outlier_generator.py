# -*- coding: utf-8 -*-
import unittest
# noinspection PyProtectedMember
from numpy.testing import assert_allclose
from numpy.testing import assert_equal
from numpy.testing import assert_raises

import os
import copy as cp

import torch
from torch_geometric.seed import seed_everything
from pygod.generator import gen_contextual_outliers
from pygod.generator import gen_structural_outliers

seed_everything(42)


class TestData(unittest.TestCase):
    def setUp(self):
        self.m_structure = 5
        self.n_structure = 5
        self.n_attribute = 25
        self.k_attribute = 25
        self.random_state = 42

        test_graph = torch.load(os.path.join('pygod', 'test', 'test_graph.pt'))
        self.data = test_graph

    def test_structure_outliers(self):
        test_data = cp.deepcopy(self.data)
        test_data, y_outlier = gen_structural_outliers(data=test_data,
                                                       m=self.m_structure,
                                                       n=self.n_structure)

        assert_equal(self.data.x.numpy(), test_data.x.numpy())
        assert (self.data.num_nodes == y_outlier.shape[0])
        assert (test_data.num_edges ==
                self.m_structure * (self.m_structure - 1) * self.n_structure
                + self.data.num_edges)
        assert_equal(self.data.edge_index.numpy(),
                     test_data.edge_index[:, :self.data.edge_index.shape[1]])
        assert (self.m_structure * self.n_structure == y_outlier.sum().item())

    def test_attribute_outliers(self):
        test_data = cp.deepcopy(self.data)
        test_data, y_outlier = gen_contextual_outliers(data=test_data,
                                                       n=self.n_attribute,
                                                       k=self.k_attribute)

        assert_equal(self.data.edge_index.numpy(),
                     test_data.edge_index.numpy())
        assert_allclose(self.data.x.numpy().shape, test_data.x.numpy().shape)
        assert (self.data.num_nodes == y_outlier.shape[0])
        assert (self.n_attribute == y_outlier.sum().item())

    def test_structure_outliers2(self):
        test_data, y_outlier = \
            gen_structural_outliers(data=cp.deepcopy(self.data),
                                    m=self.m_structure,
                                    n=self.n_structure,
                                    random_state=self.random_state)

        test_data2, y_outlier2 = \
            gen_structural_outliers(data=cp.deepcopy(self.data),
                                    m=self.m_structure,
                                    n=self.n_structure,
                                    random_state=self.random_state)

        assert_equal(test_data.x.numpy(), test_data2.x.numpy())
        assert_equal(test_data.edge_index.numpy(),
                     test_data2.edge_index.numpy())
        assert_equal(y_outlier.numpy(), y_outlier2.numpy())

    def test_attribute_outliers2(self):
        test_data, y_outlier = \
            gen_contextual_outliers(data=cp.deepcopy(self.data),
                                    n=self.n_attribute,
                                    k=self.k_attribute,
                                    random_state=self.random_state)

        test_data2, y_outlier2 = \
            gen_contextual_outliers(data=cp.deepcopy(self.data),
                                    n=self.n_attribute,
                                    k=self.k_attribute,
                                    random_state=self.random_state)

        assert_equal(test_data.x.numpy(), test_data2.x.numpy())
        assert_equal(test_data.edge_index.numpy(),
                     test_data2.edge_index.numpy())
        assert_equal(y_outlier.numpy(), y_outlier2.numpy())

    def test_structure_outliers3(self):
        test_data = cp.deepcopy(self.data)

        with assert_raises(ValueError):
            gen_structural_outliers(data=test_data,
                                    m='not int',
                                    n=self.n_structure)

        with assert_raises(ValueError):
            gen_structural_outliers(data=test_data,
                                    m=self.m_structure,
                                    n='not int')

    def test_attribute_outliers3(self):
        test_data = cp.deepcopy(self.data)

        with assert_raises(ValueError):
            gen_contextual_outliers(data=test_data,
                                    n='not int',
                                    k=self.k_attribute)

        with assert_raises(ValueError):
            gen_contextual_outliers(data=test_data,
                                    n=self.n_attribute,
                                    k='not int')

    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
