import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.utils import to_dense_adj

from .functional import double_recon_loss


class AnomalyDAEBase(nn.Module):
    """
    Dual Autoencoder for Anomaly Detection on Attributed Networks

    AnomalyDAE is an anomaly detector that consists of a structure
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
         Input dimension of model.
    num_nodes: int
         Number of input nodes or batch size in minibatch training.
    emb_dim:: int
         Embedding dimension of model. Default: ``64``.
    hid_dim : int
         Hidden dimension of model. Default: ``64``.
    dropout : float, optional
        Dropout rate. Default: ``0.``.
    act : callable activation function or None, optional
        Activation function if not None.
        Default: ``torch.nn.functional.relu``.
    **kwargs : optional
        Other parameters of ``torch_geometric.nn.GATConv``.
    """

    def __init__(self,
                 in_dim,
                 num_nodes,
                 emb_dim=64,
                 hid_dim=64,
                 dropout=0.,
                 act=F.relu,
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
        self.emb = None

    def forward(self, x, edge_index, batch_size):
        """
        Forward computation.

        Parameters
        ----------
        x : torch.Tensor
            Input attribute embeddings.
        edge_index : torch.Tensor
            Edge index.
        batch_size : int
            Batch size.

        Returns
        -------
        x_ : torch.Tensor
            Reconstructed attribute embeddings.
        s_ : torch.Tensor
            Reconstructed adjacency matrix.
        """
        h = F.dropout(self.act(self.dense_stru(x)), self.dropout)
        self.emb = self.gat_layer(h, edge_index)

        s_ = torch.sigmoid(self.emb @ self.emb.T)

        if batch_size < self.num_nodes:
            x = F.pad(x, (0, 0, 0, self.num_nodes - batch_size))

        x = self.act(self.dense_attr_1(x[:self.num_nodes].T))
        x = F.dropout(x, self.dropout)
        x = self.dense_attr_2(x)
        x = F.dropout(x, self.dropout)
        x_ = self.emb @ x.T

        return x_, s_

    @staticmethod
    def process_graph(data):
        """
        Obtain the dense adjacency matrix of the graph.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input graph.
        """
        data.s = to_dense_adj(data.edge_index)[0]
