# -*- coding: utf-8 -*-
import torch
import unittest
from torch_geometric.seed import seed_everything
from pygod.models.basic_nn import MLP, GCN
from torch_geometric.testing import is_full_test

seed_everything(42)


class TestBasicNN(unittest.TestCase):
    def test_gcn(self):
        x = torch.randn(3, 8)
        edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
        out_dim = 16

        model = GCN(8, 16, num_layers=2, out_channels=out_dim)
        assert str(model) == f'GCN(8, {out_dim}, num_layers=2)'
        assert model(x, edge_index).size() == (3, out_dim)

    def test_mlp(self):
        x = torch.randn(4, 16)

        torch.manual_seed(12345)
        mlp = MLP([16, 32, 32, 64])
        assert str(mlp) == 'MLP(16, 32, 32, 64)'
        out = mlp(x)
        assert out.size() == (4, 64)

        if is_full_test():
            jit = torch.jit.script(mlp)
            assert torch.allclose(jit(x), out)

        torch.manual_seed(12345)
        mlp = MLP(16, hidden_channels=32, out_channels=64, num_layers=3)
        assert torch.allclose(mlp(x), out)


if __name__ == '__main__':
    unittest.main()
