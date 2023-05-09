import math
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import MLP, GCN
from torch_geometric.utils import to_dense_adj

from ..nn.decoder import DotProductDecoder


class GAEBase(nn.Module):

    """
    Graph Autoencoder

    See :cite:`kipf2016variational` for details.

    Parameters
    ----------
    in_dim : int
        Input dimension of model.
    hid_dim : int
        Hidden dimension of model. Default: ``64``.
    num_layers : int, optional
        Total number of layers in model. Default: ``4``.
    dropout : float, optional
        Dropout rate. Default: ``0.``.
    act : callable activation function or None, optional
        Activation function if not None.
        Default: ``torch.nn.functional.relu``.
    backbone : torch.nn.Module, optional
        The backbone of the deep detector implemented in PyG.
        Default: ``torch_geometric.nn.GCN``.
    recon_s : bool, optional
        Reconstruct the structure instead of node feature .
        Default: ``False``.
    sigmoid_s : bool, optional
        Whether to use sigmoid function to scale the reconstructed
        structure. Default: ``False``.
    **kwargs : optional
        Other parameters for the backbone.
    """

    def __init__(self,
                 in_dim,
                 hid_dim=64,
                 num_layers=4,
                 dropout=0.,
                 act=F.relu,
                 backbone=GCN,
                 recon_s=False,
                 sigmoid_s=False,
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
        self.recon_s = recon_s
        if self.recon_s:
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
        """
        Forward computation.

        Parameters
        ----------
        x : torch.Tensor
            Input attribute embeddings.
        edge_index : torch.Tensor
            Edge index.

        Returns
        -------
        x_ : torch.Tensor
            Reconstructed embeddings.
        """

        if self.backbone == MLP:
            self.emb = self.encoder(x, None)
            x_ = self.decoder(self.emb, None)
        else:
            self.emb = self.encoder(x, edge_index)
            x_ = self.decoder(self.emb, edge_index)
        return x_

    def process_graph(self, data):
        """
        Obtain the dense adjacency matrix of the graph.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input graph.
        """
        if self.recon_s:
            data.s = to_dense_adj(data.edge_index)[0]
