import torch
from torch import Tensor
from torch_sparse import SparseTensor
from torch_geometric.nn import MessagePassing


class NeighDiff(MessagePassing):
    def __init__(self):
        super().__init__(aggr='mean')

    def forward(self, h, edge_index):
        return self.propagate(edge_index, h=h)

    def message(self, h_i, h_j, edge_index):
        return torch.sum(torch.pow(h_i - h_j, 2), dim=1, keepdim=True)

    def edge_update(self) -> Tensor:
        pass

    def message_and_aggregate(self, adj_t: SparseTensor) -> Tensor:
        pass
