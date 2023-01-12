# -*- coding: utf-8 -*-
""" Basic Models
"""
# Author: Kay Liu <zliu234@uic.edu>
# License: BSD 2 clause

import copy
from typing import Any, Dict, List, Optional, Union, Callable

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Identity, Linear, ModuleList

from torch_geometric.typing import Adj
from torch_geometric.nn.conv import MessagePassing, GCNConv, SAGEConv
from torch_geometric.nn.models.jumping_knowledge import JumpingKnowledge


class MLP(torch.nn.Module):
    r"""Multilayer Perceptron (MLP) model.
    Adapted from PyG for upward compatibility
    There exists two ways to instantiate an :class:`MLP`:
    1. By specifying explicit channel sizes, *e.g.*,
       .. code-block:: python
          mlp = MLP([16, 32, 64, 128])
       creates a three-layer MLP with **differently** sized hidden layers.
    1. By specifying fixed hidden channel sizes over a number of layers,
       *e.g.*,
       .. code-block:: python
          mlp = MLP(in_channels=16, hidden_channels=32,
                    out_channels=128, num_layers=3)
       creates a three-layer MLP with **equally** sized hidden layers.
    Args:
        channel_list (List[int] or int, optional): List of input, intermediate
            and output channels such that :obj:`len(channel_list) - 1` denotes
            the number of layers of the MLP (default: :obj:`None`)
        in_channels (int, optional): Size of each input sample.
            Will override :attr:`channel_list`. (default: :obj:`None`)
        hidden_channels (int, optional): Size of each hidden sample.
            Will override :attr:`channel_list`. (default: :obj:`None`)
        out_channels (int, optional): Size of each output sample.
            Will override :attr:`channel_list`. (default: :obj:`None`)
        num_layers (int, optional): The number of layers.
            Will override :attr:`channel_list`. (default: :obj:`None`)
        dropout (float, optional): Dropout probability of each hidden
            embedding. (default: :obj:`0.`)
        act (str or Callable, optional): The non-linear activation function to
            use. (default: :obj:`"relu"`)
        batch_norm (bool, optional): If set to :obj:`False`, will not make use
            of batch normalization. (default: :obj:`True`)
        act_first (bool, optional): If set to :obj:`True`, activation is
            applied before normalization. (default: :obj:`False`)
        batch_norm_kwargs (Dict[str, Any], optional): Arguments passed to
            :class:`torch.nn.BatchNorm1d` in case :obj:`batch_norm == True`.
            (default: :obj:`None`)
        bias (bool, optional): If set to :obj:`False`, the module will not
            learn additive biases. (default: :obj:`True`)
        relu_first (bool, optional): Deprecated in favor of :obj:`act_first`.
            (default: :obj:`False`)
    """
    def __init__(
        self,
        channel_list: Optional[Union[List[int], int]] = None,
        *,
        in_channels: Optional[int] = None,
        hidden_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        num_layers: Optional[int] = None,
        dropout: float = 0.,
        act: Callable = F.relu,
        batch_norm: bool = True,
        act_first: bool = False,
        batch_norm_kwargs: Optional[Dict[str, Any]] = None,
        bias: bool = True,
        relu_first: bool = False,
    ):
        super().__init__()

        act_first = act_first or relu_first  # Backward compatibility.
        batch_norm_kwargs = batch_norm_kwargs or {}

        if isinstance(channel_list, int):
            in_channels = channel_list

        if in_channels is not None:
            assert num_layers >= 1
            channel_list = [hidden_channels] * (num_layers - 1)
            channel_list = [in_channels] + channel_list + [out_channels]

        assert isinstance(channel_list, (tuple, list))
        assert len(channel_list) >= 2
        self.channel_list = channel_list

        self.dropout = dropout
        self.act = act
        self.act_first = act_first

        self.lins = torch.nn.ModuleList()
        pairwise = zip(channel_list[:-1], channel_list[1:])
        for in_channels, out_channels in pairwise:
            self.lins.append(Linear(in_channels, out_channels, bias=bias))

        self.norms = torch.nn.ModuleList()
        for hidden_channels in channel_list[1:-1]:
            if batch_norm:
                norm = BatchNorm1d(hidden_channels, **batch_norm_kwargs)
            else:
                norm = Identity()
            self.norms.append(norm)

        self.reset_parameters()

    @property
    def in_channels(self) -> int:
        r"""Size of each input sample."""
        return self.channel_list[0]

    @property
    def out_channels(self) -> int:
        r"""Size of each output sample."""
        return self.channel_list[-1]

    @property
    def num_layers(self) -> int:
        r"""The number of layers."""
        return len(self.channel_list) - 1

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for norm in self.norms:
            if hasattr(norm, 'reset_parameters'):
                norm.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        """"""
        x = self.lins[0](x)
        for lin, norm in zip(self.lins[1:], self.norms):
            if self.act_first:
                x = self.act(x)
            x = norm(x)
            if not self.act_first:
                x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = lin.forward(x)
        return x

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({str(self.channel_list)[1:-1]})'


class GCN(torch.nn.Module):
    r"""The Graph Neural Network from the `"Semi-supervised
    Classification with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper, using the
    :class:`~torch_geometric.nn.conv.GCNConv` operator for message passing.
    Adapted from PyG for upward compatibility
    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Size of each hidden sample.
        num_layers (int): Number of message passing layers.
        out_channels (int, optional): If not set to :obj:`None`, will apply a
            final linear transformation to convert hidden node embeddings to
            output size :obj:`out_channels`. (default: :obj:`None`)
        dropout (float, optional): Dropout probability. (default: :obj:`0.`)
        act (Callable, optional): The non-linear activation function to
            use. (default: :obj:`"relu"`)
        norm (torch.nn.Module, optional): The normalization operator to use.
            (default: :obj:`None`)
        jk (str, optional): The Jumping Knowledge mode
            (:obj:`"last"`, :obj:`"cat"`, :obj:`"max"`, :obj:`"lstm"`).
            (default: :obj:`"last"`)
        act_first (bool, optional): If set to :obj:`True`, activation is
            applied before normalization. (default: :obj:`False`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.GCNConv`.
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        out_channels: Optional[int] = None,
        dropout: float = 0.0,
        act: Union[Callable, None] = F.relu,
        norm: Optional[torch.nn.Module] = None,
        jk: Optional[str] = None,
        act_first: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers

        self.dropout = dropout
        self.act = act
        self.jk_mode = jk
        self.act_first = act_first

        if out_channels is not None:
            self.out_channels = out_channels
        else:
            self.out_channels = hidden_channels

        self.convs = ModuleList()
        self.convs.append(
            self.init_conv(in_channels, hidden_channels, **kwargs))
        for _ in range(num_layers - 2):
            self.convs.append(
                self.init_conv(hidden_channels, hidden_channels, **kwargs))
        if out_channels is not None and jk is None:
            self._is_conv_to_out = True
            self.convs.append(
                self.init_conv(hidden_channels, out_channels, **kwargs))
        else:
            self.convs.append(
                self.init_conv(hidden_channels, hidden_channels, **kwargs))

        self.norms = None
        if norm is not None:
            self.norms = ModuleList()
            for _ in range(num_layers - 1):
                self.norms.append(copy.deepcopy(norm))
            if jk is not None:
                self.norms.append(copy.deepcopy(norm))

        if jk is not None and jk != 'last':
            self.jk = JumpingKnowledge(jk, hidden_channels, num_layers)

        if jk is not None:
            if jk == 'cat':
                in_channels = num_layers * hidden_channels
            else:
                in_channels = hidden_channels
            self.lin = Linear(in_channels, self.out_channels)

    def init_conv(self, in_channels: int, out_channels: int,
                  **kwargs) -> MessagePassing:
        return GCNConv(in_channels, out_channels, **kwargs)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for norm in self.norms or []:
            norm.reset_parameters()
        if hasattr(self, 'jk'):
            self.jk.reset_parameters()
        if hasattr(self, 'lin'):
            self.lin.reset_parameters()

    def forward(self, x: Tensor, edge_index: Adj, *args, **kwargs) -> Tensor:
        """"""
        xs: List[Tensor] = []
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, *args, **kwargs)
            if i == self.num_layers - 1 and self.jk_mode is None:
                break
            if self.act_first:
                x = self.act(x)
            if self.norms is not None:
                x = self.norms[i](x)
            if not self.act_first:
                x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if hasattr(self, 'jk'):
                xs.append(x)

        x = self.jk(xs) if hasattr(self, 'jk') else x
        x = self.lin(x) if hasattr(self, 'lin') else x
        return x

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, num_layers={self.num_layers})')


class Vanilla_GCN(torch.nn.Module):
    def __init__(self, in_ft, out_ft, act, bias=True):
        super(Vanilla_GCN, self).__init__()
        self.fc = torch.nn.Linear(in_ft, out_ft, bias=False)
        self.act = torch.nn.PReLU() if act == 'prelu' else act

        if bias:
            self.bias = torch.nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq, adj, sparse=False):
        seq_fts = self.fc(seq)
        if sparse:
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)),
                                  0)
        else:
            out = torch.bmm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias

        return self.act(out)

class SAGE(torch.nn.Module):
    """"
    Part of the code from the SAGE example in PyG
    Source: https://github.com/pyg-team/pytorch_geometric/blob/master/examples/ogbn_products_sage.py
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super().__init__()

        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adjs):
        """
        `train_loader` computes the k-hop neighborhood of a batch of nodes,
        and returns, for each layer, a bipartite graph object, holding the
        bipartite edges `edge_index`, the index `e_id` of the original edges,
        and the size/shape `size` of the bipartite graph.
        Target nodes are also included in the source nodes so that one can
        easily apply skip-connections or add self-loops.
        """
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        return x.log_softmax(dim=-1)

    def inference(self, x_all):
        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.
        total_edges = 0
        for i in range(self.num_layers):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                total_edges += edge_index.size(1)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                x = self.convs[i]((x, x_target), edge_index)
                if i != self.num_layers - 1:
                    x = F.relu(x)
                xs.append(x.cpu())

            x_all = torch.cat(xs, dim=0)


        return x_all