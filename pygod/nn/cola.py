import torch
import torch.nn as nn
from torch_geometric.nn import GCN
from torch.nn.functional import binary_cross_entropy_with_logits


class CoLABase(nn.Module):
    """
    Anomaly Detection on Attributed Networks via Contrastive
    Self-Supervised Learning

    CoLA is a contrastive self-supervised learning based method for
    graph anomaly detection. This implementation is base on random
    neighbor sampling instead of random walk sampling in the original
    paper.

    See :cite:`liu2021anomaly` for details.

    Parameters
    ----------
    in_dim : int
        Input dimension of model.
    hid_dim :  int, optional
        Hidden dimension of model. Default: ``64``.
    num_layers : int, optional
        Total number of layers in model. Default: ``4``.
    dropout : float, optional
        Dropout rate. Default: ``0.``.
    act : callable activation function or None, optional
        Activation function if not None.
        Default: ``torch.nn.functional.relu``.
    backbone : torch.nn.Module
        The backbone of the deep detector implemented in PyG.
        Default: ``torch_geometric.nn.GCN``.
    **kwargs
        Other parameters for the backbone.
    """

    def __init__(self,
                 in_dim,
                 hid_dim=64,
                 num_layers=4,
                 dropout=0.,
                 act=torch.nn.functional.relu,
                 backbone=GCN,
                 **kwargs):
        super(CoLABase, self).__init__()

        self.encoder = backbone(in_channels=in_dim,
                                hidden_channels=hid_dim,
                                num_layers=num_layers,
                                out_channels=hid_dim,
                                dropout=dropout,
                                act=act,
                                **kwargs)

        self.discriminator = nn.Bilinear(in_dim, hid_dim, 1)

        self.loss_func = binary_cross_entropy_with_logits
        self.emb = None

    def forward(self, x, edge_index):

        self.emb = self.encoder(x, edge_index)
        logits = self.discriminator(x, self.emb)

        perm_idx = torch.randperm(x.shape[0]).to(x.device)
        neg_logits = self.discriminator(x[perm_idx], self.emb)
        return logits.squeeze(), neg_logits.squeeze()
