import torch
from torch_geometric.nn import GCN

from ..nn.encoder import GNA
from ..nn.functional import double_recon_loss


class GUIDEBase(torch.nn.Module):
    def __init__(self,
                 dim_x,
                 dim_s,
                 hid_x,
                 hid_s,
                 num_layers,
                 dropout,
                 act,
                 **kwargs):
        super(GUIDEBase, self).__init__()

        self.attr_ae = GCN(in_channels=dim_x,
                           hidden_channels=hid_x,
                           num_layers=num_layers,
                           out_channels=dim_x,
                           dropout=dropout,
                           act=act,
                           **kwargs)

        self.stru_ae = GNA(in_channels=dim_s,
                           hidden_channels=hid_s,
                           num_layers=num_layers,
                           out_channels=dim_s,
                           dropout=dropout,
                           act=act)

        self.loss_func = double_recon_loss

    def forward(self, x, s, edge_index):
        x_ = self.attr_ae(x, edge_index)
        s_ = self.stru_ae(s, edge_index)
        return x_, s_
