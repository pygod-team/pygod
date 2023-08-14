import math
import random

import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch_geometric.nn import GIN, SAGEConv, PNAConv
from torch_geometric.utils import to_dense_adj

from .nn import MLP_GAD_NR, MLP_generator, FNN_GAD_NR
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
                 neighbor_num_list=None,
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

        self.mlp_mean = FNN_GAD_NR(hid_dim, hid_dim, hid_dim, 3)
        self.mlp_sigma = FNN_GAD_NR(hid_dim, hid_dim, hid_dim, 3)
        self.softplus = nn.Softplus()

        self.mean_agg = SAGEConv(hid_dim, hid_dim, aggr='mean', normalize = False)
        self.std_agg = PNAConv(hid_dim, hid_dim, aggregators=["std"],scalers=["identity"], deg=neighbor_num_list)        
        self.layer1_generator = MLP_generator(hid_dim, hid_dim)

        # Encoder
        self.shared_encoder = backbone(in_channels=hid_dim,
                                       hidden_channels=hid_dim,
                                       num_layers=encoder_layers,
                                       out_channels=hid_dim,
                                       dropout=dropout,
                                       act=act,
                                       **kwargs)

        # Decoder
        self.degree_decoder = FNN_GAD_NR(hid_dim, hid_dim, 1, 4)
        self.feature_decoder = FNN_GAD_NR(hid_dim, hid_dim, in_dim, 3)
        self.degree_loss_func = nn.MSELoss()
        self.feature_loss_func = nn.MSELoss()
        self.pool = mp.Pool(4)
        self.in_dim = in_dim
        self.sample_size = sample_size 
        self.init_projection = FNN_GAD_NR(in_dim, hid_dim, hid_dim, 1)
        self.emb = None


    # Sample neighbors from neighbor set, if the length of neighbor set less than sample size, then do the padding.
    def sample_neighbors(self, indexes, neighbor_dict, gt_embeddings):
        sampled_embeddings_list = []
        mark_len_list = []
        for index in indexes:
            sampled_embeddings = []
            neighbor_indexes = neighbor_dict[index]
            if len(neighbor_indexes) < self.sample_size:
                mask_len = len(neighbor_indexes)
                sample_indexes = neighbor_indexes
            else:
                sample_indexes = random.sample(neighbor_indexes, self.sample_size)
                mask_len = self.sample_size
            for index in sample_indexes:
                sampled_embeddings.append(gt_embeddings[index].tolist())
            if len(sampled_embeddings) < self.sample_size:
                for _ in range(self.sample_size - len(sampled_embeddings)):
                    sampled_embeddings.append(torch.zeros(self.out_dim).tolist())
            sampled_embeddings_list.append(sampled_embeddings)
            mark_len_list.append(mask_len)
        
        return sampled_embeddings_list, mark_len_list

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
        # TODO add extra projection for GIN model

        # encode feature matrix
        self.emb = self.shared_encoder(x, edge_index)

        # reconstruct feature matrix
        x_ = self.attr_decoder(self.emb, edge_index)

        # decode adjacency matrix
        s_ = self.struct_decoder(self.emb, edge_index)

        return x_, s_

    def loss_func(self,
                  gij,
                  ground_truth_degree_matrix,
                  h0,
                  neighbor_dict,
                  device,
                  h,
                  edge_index):
        """
        Obtain the dense adjacency matrix of the graph.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input graph.
        """

        # TODO dissecting the decoders and put it into the forward function
        
        # Degree decoder below:
        tot_nodes = gij.shape[0]
        degree_logits = self.degree_decoding(gij)
        ground_truth_degree_matrix = torch.unsqueeze(ground_truth_degree_matrix, dim=1)
        degree_loss = self.degree_loss_func(degree_logits, ground_truth_degree_matrix.float())
        degree_loss_per_node = (degree_logits-ground_truth_degree_matrix).pow(2)
        _, degree_masks = torch.max(degree_logits.data, dim=1)
        h_loss = 0
        feature_loss = 0
        
        # layer 1
        loss_list = []
        loss_list_per_node = []
        feature_loss_list = []
        # Sample multiple times to remove noise
        for _ in range(3):
            local_index_loss_sum = 0
            local_index_loss_sum_per_node = []
            indexes = []
            h0_prime = self.feature_decoder(gij)
            feature_losses = self.feature_loss_func(h0, h0_prime)
            feature_losses_per_node = (h0-h0_prime).pow(2).mean(1)
            feature_loss_list.append(feature_losses_per_node)
            
            local_index_loss, local_index_loss_per_node = self.reconstruction_neighbors2(gij,h0,edge_index)
            
            loss_list.append(local_index_loss)
            loss_list_per_node.append(local_index_loss_per_node)
            
        loss_list = torch.stack(loss_list)
        h_loss += torch.mean(loss_list)
        
        loss_list_per_node = torch.stack(loss_list_per_node)
        h_loss_per_node = torch.mean(loss_list_per_node,dim=0)
        
        feature_loss_per_node = torch.mean(torch.stack(feature_loss_list),dim=0)
        feature_loss += torch.mean(torch.stack(feature_loss_list))
                
        h_loss_per_node = h_loss_per_node.reshape(tot_nodes,1)
        degree_loss_per_node = degree_loss_per_node.reshape(tot_nodes,1)
        feature_loss_per_node = feature_loss_per_node.reshape(tot_nodes,1)
        
        loss = self.lambda_loss1 * h_loss + degree_loss * self.lambda_loss3 + self.lambda_loss2 * feature_loss
        loss_per_node = self.lambda_loss1 * h_loss_per_node + degree_loss_per_node * self.lambda_loss3 + self.lambda_loss2 * feature_loss_per_node
        
        return loss,loss_per_node,h_loss_per_node,degree_loss_per_node,feature_loss_per_node


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
