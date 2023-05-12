import math
import torch
import torch.nn.functional as F
from torch_geometric.nn import MLP
from torch_geometric.utils import to_dense_adj

from ..nn.functional import double_recon_loss


class GAANBase(torch.nn.Module):
    """
    Generative Adversarial Attributed Network Anomaly Detection

    GAAN is a generative adversarial attribute network anomaly
    detection framework, including a generator module, an encoder
    module, a discriminator module, and uses anomaly evaluation
    measures that consider sample reconstruction error and real sample
    recognition confidence to make predictions. This model is
    transductive only.

    See :cite:`chen2020generative` for details.

    Parameters
    ----------
    in_dim : int
        Input dimension of the node features.
    noise_dim :  int, optional
        Input dimension of the Gaussian random noise. Defaults: ``16``.
    hid_dim :  int, optional
        Hidden dimension of model. Default: ``64``.
    num_layers : int, optional
       Total number of layers in model. A half (floor) of the layers
       are for the generator, the other half (ceil) of the layers are
       for encoder. Default: ``4``.
    dropout : float, optional
        Dropout rate. Default: ``0.``.
    act : callable activation function or None, optional
        Activation function if not None.
        Default: ``torch.nn.functional.relu``.
    **kwargs
        Other parameters for the backbone.
    """
    def __init__(self,
                 in_dim,
                 noise_dim,
                 hid_dim=64,
                 num_layers=4,
                 dropout=0.,
                 act=torch.nn.functional.relu,
                 **kwargs):
        super(GAANBase, self).__init__()

        # split the number of layers for the encoder and decoders
        assert num_layers >= 2, \
            "Number of layers must be greater than or equal to 2."
        generator_layers = math.floor(num_layers / 2)
        encoder_layers = math.ceil(num_layers / 2)

        self.generator = MLP(in_channels=noise_dim,
                             hidden_channels=hid_dim,
                             out_channels=in_dim,
                             num_layers=generator_layers,
                             dropout=dropout,
                             act=act,
                             **kwargs)

        self.discriminator = MLP(in_channels=in_dim,
                                 hidden_channels=hid_dim,
                                 out_channels=hid_dim,
                                 num_layers=encoder_layers,
                                 dropout=dropout,
                                 act=act,
                                 **kwargs)
        self.emb = None
        self.score_func = double_recon_loss

    def forward(self, x, noise):
        """
        Forward computation.

        Parameters
        ----------
        x : torch.Tensor
            Input attribute embeddings.
        noise : torch.Tensor
            Input noise.

        Returns
        -------
        x_ : torch.Tensor
            Reconstructed node features.
        a : torch.Tensor
            Reconstructed adjacency matrix from real samples.
        a_ : torch.Tensor
            Reconstructed adjacency matrix from fake samples.
        """
        x_ = self.generator(noise)

        self.emb = self.discriminator(x)
        z_ = self.discriminator(x_)

        a = torch.sigmoid((self.emb @ self.emb.T))
        a_ = torch.sigmoid((z_ @ z_.T))

        return x_, a, a_

    @staticmethod
    def loss_func_g(a_):

        loss_g = F.binary_cross_entropy(a_, torch.ones_like(a_))
        return loss_g

    @staticmethod
    def loss_func_ed(a, a_):
        loss_r = F.binary_cross_entropy(a, torch.ones_like(a))
        loss_f = F.binary_cross_entropy(a_, torch.zeros_like(a_))
        return (loss_f + loss_r) / 2

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
