import math
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import MLP, GCN

from ..nn.decoder import DotProductDecoder


class GAEBase(nn.Module):
    """

    """

    def __init__(self,
                 in_dim,
                 hid_dim=64,
                 num_layers=4,
                 dropout=0.,
                 act=F.relu,
                 recon_s=False,
                 sigmoid_s=False,
                 backbone=GCN,
                 **kwargs):
        super(GAEBase, self).__init__()

        self.backbone = backbone
        # split the number of layers for the encoder and decoders
        assert num_layers >= 2, \
            "Number of layers must be greater than or equal to 2."
        encoder_layers = math.floor(num_layers / 2)
        decoder_layers = math.ceil(num_layers / 2)

        self.encoder = self.backbone(in_channels=in_dim,
                                     hidden_channels=hid_dim,
                                     out_channels=hid_dim,
                                     num_layers=encoder_layers,
                                     dropout=dropout,
                                     act=act,
                                     **kwargs)

        if recon_s:
            self.decoder = DotProductDecoder(in_dim=hid_dim,
                                             hid_dim=hid_dim,
                                             num_layers=decoder_layers,
                                             dropout=dropout,
                                             act=act,
                                             sigmoid_s=sigmoid_s,
                                             backbone=self.backbone,
                                             **kwargs)
        else:
            self.decoder = self.backbone(in_channels=hid_dim,
                                         hidden_channels=hid_dim,
                                         out_channels=in_dim,
                                         num_layers=decoder_layers,
                                         dropout=dropout,
                                         act=act,
                                         **kwargs)

        self.loss_func = F.mse_loss
        self.emb = None

    def forward(self, x, edge_index):

        if self.backbone == MLP:
            self.emb = self.encoder(x, None)
            x_ = self.decoder(self.emb, None)
        else:
            self.emb = self.encoder(x, edge_index)
            x_ = self.decoder(self.emb, edge_index)
        return x_
