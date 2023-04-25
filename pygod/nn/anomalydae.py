import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GATConv

from .functional import double_recon_loss


class AnomalyDAEBase(nn.Module):
    """
    AnomalyDAE (Dual autoencoder for anomaly detection on attributed
    networks) is an anomaly detector that consists of a structure
    autoencoder and an attribute autoencoder to learn both node
    embedding and attribute embedding jointly in latent space. The
    structural autoencoder uses Graph Attention layers. The
    reconstruction mean square error of the decoders are defined as
    structure anomaly score and attribute anomaly score, respectively,
    with two additional penalties on the reconstructed adj matrix and
    node attributes (force entries to be nonzero).

    See :cite:`fan2020anomalydae` for details.

    Parameters
    ----------
    in_dim : int
         Dimension of input feature
    num_nodes: int
         Dimension of the input number of nodes
    emb_dim:: int
         Dimension of the embedding after the first reduced linear
         layer (D1)
    hid_dim : int
         Dimension of final representation
    dropout : float, optional
        Dropout rate of the model
        Default: 0
    act : Callable, optional
        Activation function if not None.
        Default: ``torch.nn.functional.relu``.
    **kwargs : optional
        Additional arguments of ``torch_geometric.nn.GATConv``.
    """

    def __init__(self,
                 in_dim,
                 num_nodes,
                 emb_dim,
                 hid_dim,
                 dropout,
                 act,
                 **kwargs):
        super(AnomalyDAEBase, self).__init__()

        self.num_nodes = num_nodes

        self.dense_stru = nn.Linear(in_dim, emb_dim)
        self.gat_layer = GATConv(emb_dim, hid_dim, **kwargs)

        self.dense_attr_1 = nn.Linear(self.num_nodes, emb_dim)
        self.dense_attr_2 = nn.Linear(emb_dim, hid_dim)

        self.dropout = dropout
        self.act = act

        self.loss_func = double_recon_loss

    def forward(self, x, edge_index, batch_size):
        h = F.dropout(self.act(self.dense_stru(x)), self.dropout)
        h = self.gat_layer(h, edge_index)

        s_ = torch.sigmoid(h @ h.T)

        if batch_size < self.num_nodes:
            x = F.pad(x, (0, 0, 0, self.num_nodes - batch_size))

        x = self.act(self.dense_attr_1(x[:self.num_nodes].T))
        x = F.dropout(x, self.dropout)
        x = self.dense_attr_2(x)
        x = F.dropout(x, self.dropout)
        x_ = h @ x.T

        return x_, s_
