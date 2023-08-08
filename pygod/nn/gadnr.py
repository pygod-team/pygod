import math
import torch
import torch.nn as nn
from torch_geometric.nn import GIN
from torch_geometric.utils import to_dense_adj

from .encoder import MLP_GAD_NR
from .decoder import DotProductDecoder
from .functional import double_recon_loss


class GADNRBase(nn.Module):
    """
    Deep Anomaly Detection on Attributed Networks

    DOMINANT is an anomaly detector consisting of a shared graph
    convolutional encoder, a structure reconstruction decoder, and an
    attribute reconstruction decoder. The reconstruction mean squared
    error of the decoders are defined as structure anomaly score and
    attribute anomaly score, respectively.

    See :cite:`ding2019deep` for details.

    Parameters
    ----------
    in_dim : int
        Input dimension of model.
    hid_dim :  int
       Hidden dimension of model. Default: ``64``.
    num_layers : int, optional
       Total number of layers in model. A half (floor) of the layers
       are for the encoder, the other half (ceil) of the layers are
       for decoders. Default: ``4``.
    dropout : float, optional
       Dropout rate. Default: ``0.``.
    act : callable activation function or None, optional
       Activation function if not None.
       Default: ``torch.nn.functional.relu``.
    sigmoid_s : bool, optional
        Whether to apply sigmoid to the structure reconstruction.
        Default: ``False``.
    backbone : torch.nn.Module, optional
        The backbone of the deep detector implemented in PyG.
        Default: ``torch_geometric.nn.GCN``.
    **kwargs : optional
        Additional arguments for the backbone.
    """

    def __init__(self,
                 in_dim,
                 hid_dim=64,
                 num_layers=2,
                 sample_size=2,
                 neibor_num_list=None,
                 lambda_loss1=1e-2,
                 lambda_loss2=1e-3,
                 lambda_loss3=1e-4,
                 dropout=0.,
                 act=torch.nn.functional.relu,
                 extra_linear_proj=True,
                 backbone=GIN,
                 **kwargs):
        super(DOMINANTBase, self).__init__()

        # split the number of layers for the encoder and decoders
        assert num_layers >= 2, \
            "Number of layers must be greater than or equal to 2."
        encoder_layers = math.floor(num_layers / 2)
        decoder_layers = math.ceil(num_layers / 2)
        
        self.linear = nn.Linear(in_dim, hid_dim)
        self.out_dim = hid_dim
        self.lambda_loss1 = lambda_loss1
        self.lambda_loss2 = lambda_loss2
        self.lambda_loss3 = lambda_loss3

        self.neighbor_num_list = neighbor_num_list
        self.tot_node = len(neighbor_num_list)

        self.gaussian_mean = nn.Parameter(
            torch.FloatTensor(sample_size, hid_dim).uniform_(-0.5 / hid_dim,
                                                                                     0.5 / hid_dim)).to(device)
        self.gaussian_log_sigma = nn.Parameter(
            torch.FloatTensor(sample_size, hid_dim).uniform_(-0.5 / hid_dim,
                                                                                     0.5 / hid_dim)).to(device)
        self.m = torch.distributions.Normal(torch.zeros(sample_size, hid_dim),
                                            torch.ones(sample_size, hid_dim))
        
        self.m_batched = torch.distributions.Normal(torch.zeros(sample_size, self.tot_node, hid_dim),
                                            torch.ones(sample_size, self.tot_node, hid_dim))

        self.m_h = torch.distributions.Normal(torch.zeros(sample_size, hid_dim),
                                            50* torch.ones(sample_size, hid_dim))

        # Before MLP Gaussian Means, and std

        self.mlp_gaussian_mean = nn.Parameter(
            torch.FloatTensor(hid_dim).uniform_(-0.5 / hid_dim, 0.5 / hid_dim)).to(device)
        self.mlp_gaussian_log_sigma = nn.Parameter(
            torch.FloatTensor(hid_dim).uniform_(-0.5 / hid_dim, 0.5 / hid_dim)).to(device)
        self.mlp_m = torch.distributions.Normal(torch.zeros(hid_dim), torch.ones(hid_dim))

        self.mlp_mean = FNN(hid_dim, hid_dim, hid_dim, 3)
        self.mlp_sigma = FNN(hid_dim, hid_dim, hid_dim, 3)
        self.softplus = nn.Softplus()

        self.mean_agg = SAGEConv(hid_dim, hid_dim, aggr='mean', normalize = False)
        # self.mean_agg = GraphSAGE(hid_dim, hid_dim, aggr='mean', num_layers=1)
        self.std_agg = PNAConv(hid_dim, hid_dim, aggregators=["std"],scalers=["identity"], deg=neighbor_num_list)        
        self.layer1_generator = MLP_generator(hid_dim, hid_dim)

        # GNN Encoder
        self.shared_encoder = backbone(in_channels=hid_dim,
                                       hidden_channels=hid_dim,
                                       num_layers=encoder_layers,
                                       out_channels=hid_dim,
                                       dropout=dropout,
                                       act=act,
                                       **kwargs)


        self.loss_func = double_recon_loss
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
            Reconstructed attribute embeddings.
        s_ : torch.Tensor
            Reconstructed adjacency matrix.
        """
        
        # feature projection
        x = self.linear(x)

        # encode feature matrix
        self.emb = self.shared_encoder(x, edge_index)

        # reconstruct feature matrix
        x_ = self.attr_decoder(self.emb, edge_index)

        # decode adjacency matrix
        s_ = self.struct_decoder(self.emb, edge_index)

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
