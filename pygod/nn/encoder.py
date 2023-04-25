import torch
import torch.nn.functional as F

from .conv import GNAConv


class GNA(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 num_layers,
                 out_channels,
                 dropout,
                 act):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        self.layers.append(GNAConv(in_channels, hidden_channels))
        for layer in range(num_layers - 2):
            self.layers.append(GNAConv(hidden_channels,
                                       hidden_channels))
        self.layers.append(GNAConv(hidden_channels, out_channels))

        self.dropout = dropout
        self.act = act

    def forward(self, s, edge_index):
        for layer in self.layers:
            s = layer(s, edge_index)
            s = F.dropout(s, self.dropout, training=self.training)
            s = self.act(s)
        return s

