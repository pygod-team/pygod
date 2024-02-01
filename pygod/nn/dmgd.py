import math
import warnings

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import MLP
from torch_geometric.utils import to_dense_adj
from sklearn.cluster import KMeans

from .conv import NeighDiff


class DMGDBase(nn.Module):
    """
    Deep Multiclass Graph Description

    DMGD is a support vector based multiclass outlier detector. Its
    backbone is an autoencoder that reconstructs the adjacency matrix
    of the graph with MSE loss and homophily loss. It applies k-means
    to cluster the nodes embedding and then uses support vector to
    detect outliers.

    See :cite:`bandyopadhyay2020integrating` for details.

    Parameters
    ----------
    in_dim : int
        Input dimension.
    hid_dim :  int, optional
        Hidden dimension of model. Default: ``64``.
    num_layers : int, optional
        Total number of layers in model. A half (floor) of the layers
        are for the encoder, the other half (ceil) of the layers are for
        decoders. Default: ``4``.
    dropout : float, optional
        Dropout rate. Default: ``0.``.
    weight_decay : float, optional
        Weight decay (L2 penalty). Default: ``0.``.
    act : callable activation function or None, optional
        Activation function if not None.
        Default: ``torch.nn.functional.relu``.
    backbone : torch.nn.Module, optional
        The backbone of the deep detector implemented in PyG.
        Default: ``torch_geometric.nn.MLP``.
    alpha : float, optional
        Weight of the radius loss. Default: ``1``.
    beta : float, optional
        Weight of the reconstruction loss. Default: ``1``.
    gamma : float, optional
        Weight of the homophily loss. Default: ``1``.
    k : int, optional
        The number of clusters. Default: ``2``.
    **kwargs
        Other parameters for the backbone.
    """

    def __init__(self,
                 in_dim,
                 hid_dim=64,
                 num_layers=2,
                 dropout=0.,
                 act=torch.nn.functional.relu,
                 backbone=MLP,
                 alpha=1,
                 beta=1,
                 gamma=1,
                 warmup=2,
                 k=2,
                 **kwargs):
        super(DMGDBase, self).__init__()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.backbone = backbone
        self.warmup = warmup
        self.k = k

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

        self.decoder = self.backbone(in_channels=hid_dim,
                                     hidden_channels=hid_dim,
                                     out_channels=in_dim,
                                     num_layers=decoder_layers,
                                     dropout=dropout,
                                     act=act,
                                     **kwargs)

        self.neigh_diff = NeighDiff()
        self.emb = None
        self.clustered = False
        self.cluster = None
        self.centers = None
        self.r = torch.nn.Parameter(torch.zeros(self.k))

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
            Reconstructed attribute embeddings.
        nd : torch.Tensor
            Neighbor distance.
        """

        if self.backbone == MLP:
            self.emb = self.encoder(x, None)
            x_ = self.decoder(self.emb, None)
        else:
            self.emb = self.encoder(x, edge_index)
            x_ = self.decoder(self.emb, edge_index)

        nd = self.neigh_diff(self.emb, edge_index).squeeze()
        return x_, nd, self.emb

    def loss_func(self, x, x_, nd, emb):
        """
        Loss function for DMGD.

        Parameters
        ----------
        x : torch.Tensor
            Input attribute embeddings.
        x_ : torch.Tensor
            Reconstructed attribute embeddings.
        nd : torch.Tensor
            Neighbor distance.
        emb : torch.Tensor
            Embeddings.

        Returns
        -------
        loss : torch.Tensor
            Loss value.
        """
        dx = torch.sum(torch.pow(x - x_, 2))
        loss = self.beta * dx + self.gamma * nd.sum()

        if self.warmup > 0:
            self.warmup -= 1
            score = torch.zeros(x.shape[0])
        else:
            if not self.clustered:
                self.clustered = True
                kmeans = KMeans(n_clusters=self.k,
                                n_init='auto').fit(emb.detach())
                self.cluster = torch.tensor(kmeans.labels_).long()
                self.centers = torch.Tensor(kmeans.cluster_centers_)
            else:
                distances = torch.cdist(emb, self.centers, p=2)
                self.cluster = torch.argmin(distances, dim=1)

                one_hot = F.one_hot(self.cluster,  num_classes=self.k).float()
                sums = torch.matmul(one_hot.T, emb)
                counts = one_hot.sum(dim=0).view(self.k, 1)
                counts = counts + (counts == 0).type(torch.float32)
                self.centers = sums / counts

            loss += torch.pow(torch.relu(self.r), 2).sum()
            score = torch.relu(torch.sum(torch.pow(emb -
                                                   self.centers[self.cluster],
                                                   2),
                                         1) -
                               torch.pow(self.r[self.cluster], 2))
            loss += self.alpha * torch.sum(score)

        return loss, score

    @staticmethod
    def process_graph(data):
        """
        Obtain the dense adjacency matrix of the graph.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input graph.
        """
        if data.x is not None:
            warnings.warn('DMGD overwrites x with adjacency matrx.')
        data.x = to_dense_adj(data.edge_index)[0]
