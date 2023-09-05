import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch_geometric.nn import GIN, SAGEConv, PNAConv
from torch_geometric.utils import to_dense_adj

from .nn import MLP_GAD_NR, MLP_generator, FNN_GAD_NR
from .functional import double_recon_loss


# TODO update the argument documentation
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
                 encoder_layers=2,
                 deg_dec_layers=4,
                 fea_dec_layers=3,
                 sample_size=2,
                 sample_time=3,
                 neighbor_num_list=None,
                 lambda_loss1=1e-2,
                 lambda_loss2=1e-3,
                 lambda_loss3=1e-4,
                 dropout=0.,
                 act=torch.nn.functional.relu,
                 backbone=GIN,
                 device='cpu',
                 **kwargs):
        super(GADNRBase, self).__init__()
        
        self.linear = nn.Linear(in_dim, hid_dim)
        self.out_dim = hid_dim
        self.sample_time = sample_time
        self.lambda_loss1 = lambda_loss1
        self.lambda_loss2 = lambda_loss2
        self.lambda_loss3 = lambda_loss3
        self.device = device

        self.neighbor_num_list = neighbor_num_list
        self.tot_node = len(neighbor_num_list)

        self.gaussian_mean = nn.Parameter(
            torch.FloatTensor(sample_size, hid_dim).uniform_(-0.5 / hid_dim,
                                                             0.5 / hid_dim))
        self.gaussian_log_sigma = nn.Parameter(
            torch.FloatTensor(sample_size, hid_dim).uniform_(-0.5 / hid_dim,
                                                             0.5 / hid_dim))
        self.m = torch.distributions.Normal(torch.zeros(sample_size, hid_dim),
                                            torch.ones(sample_size, hid_dim))
        
        self.m_batched = torch.distributions.Normal(torch.zeros(sample_size,
                                                                self.tot_node,
                                                                hid_dim),
                                            torch.ones(sample_size,
                                                       self.tot_node,
                                                       hid_dim))

        self.m_h = torch.distributions.Normal(torch.zeros(sample_size,
                                                          hid_dim),
                                            50* torch.ones(sample_size,
                                                           hid_dim))

        # Before MLP Gaussian Means, and std

        self.mlp_gaussian_mean = nn.Parameter(
            torch.FloatTensor(hid_dim).uniform_(-0.5 / hid_dim,
                                                0.5 / hid_dim))
        self.mlp_gaussian_log_sigma = nn.Parameter(
            torch.FloatTensor(hid_dim).uniform_(-0.5 / hid_dim,
                                                0.5 / hid_dim))
        self.mlp_m = torch.distributions.Normal(torch.zeros(hid_dim),
                                                torch.ones(hid_dim))

        self.mlp_mean = FNN_GAD_NR(hid_dim, hid_dim, hid_dim, 3)
        self.mlp_sigma = FNN_GAD_NR(hid_dim, hid_dim, hid_dim, 3)
        self.softplus = nn.Softplus()

        self.mean_agg = SAGEConv(hid_dim, hid_dim,
                                 aggr='mean', normalize = False)
        self.std_agg = PNAConv(hid_dim, hid_dim, aggregators=["std"],
                               scalers=["identity"], deg=neighbor_num_list)        
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
        self.degree_decoder = FNN_GAD_NR(hid_dim, hid_dim, 1, deg_dec_layers)
        self.feature_decoder = FNN_GAD_NR(hid_dim, hid_dim,
                                          in_dim, fea_dec_layers)
        self.degree_loss_func = nn.MSELoss()
        self.feature_loss_func = nn.MSELoss()
        self.pool = mp.Pool(4)
        self.in_dim = in_dim
        self.sample_size = sample_size 
        self.emb = None

    def sample_neighbors(self, indexes, neighbor_dict, gt_embeddings):
        """ Sample neighbors from neighbor set, if the length of neighbor set
            less than sample size, then do the padding.
        """
        sampled_embeddings_list = []
        mark_len_list = []
        for index in indexes:
            sampled_embeddings = []
            neighbor_indexes = neighbor_dict[index]
            if len(neighbor_indexes) < self.sample_size:
                mask_len = len(neighbor_indexes)
                sample_indexes = neighbor_indexes
            else:
                sample_indexes = random.sample(neighbor_indexes,
                                               self.sample_size)
                mask_len = self.sample_size
            for index in sample_indexes:
                sampled_embeddings.append(gt_embeddings[index].tolist())
            if len(sampled_embeddings) < self.sample_size:
                for _ in range(self.sample_size - len(sampled_embeddings)):
                    sampled_embeddings.append(torch.zeros(self.out_dim
                                                          ).tolist())
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
        h0 : torch.Tensor
            Node feature initial embeddings.
        l1 : torch.Tensor
            Node embedding after encoder.
        degree_logits : torch.Tensor
            Reconstructed node degree logits.
        feat_recon_list : List[torch.Tensor]
            Reconstructed node features.
        neigh_recon_list :  List[torch.Tensor]
            Reconstructed neighbor distributions.
        """
        
        # feature projection
        h0 = self.linear(x)

        # encode feature matrix
        l1 = self.shared_encoder(h0, edge_index)
        
        # save embeddings
        self.emb = l1

        # decode node degree
        degree_logits = F.relu(self.degree_decoder(l1))

        # decode the node feature and neighbor distribution
        feat_recon_list = []
        neigh_recon_list = []
        # sample multiple times to remove noises
        for _ in range(self.sample_time):
            h0_prime = self.feature_decoder(l1)
            feat_recon_list.append(h0_prime)
            
            neigh_recon_info = self.neigh_distr_recon(l1,h0,edge_index)
            neigh_recon_list.append(neigh_recon_info)

        return h0, l1, degree_logits, feat_recon_list, neigh_recon_list

    def neigh_distr_recon(self, l1, h0, edge_index, device):
        """Computing the target neighbor distribution and 
        reconstructed neighbor distribution
        """
                
        mean_neigh = self.mean_agg(h0, edge_index).detach()
        std_neigh = self.std_agg(h0, edge_index).detach()
        
        cov_neigh = torch.bmm(std_neigh.unsqueeze(dim=-1),
                              std_neigh.unsqueeze(dim=1))
        
        target_mean = mean_neigh
        target_cov = cov_neigh
        
        self_embedding = l1
        self_embedding = self_embedding.unsqueeze(0)
        self_embedding = self_embedding.repeat(self.sample_size, 1, 1)
        generated_mean = self.mlp_mean(self_embedding)
        generated_sigma = self.mlp_sigma(self_embedding)

        
        std_z = self.m_batched.sample().to(device)
        var = generated_mean + generated_sigma.exp() * std_z
        nhij = self.layer1_generator(var)
        
        generated_mean = torch.mean(nhij,dim=0)
        generated_std = torch.std(nhij,dim=0)
        generated_cov = torch.bmm(generated_std.unsqueeze(dim=-1),
                                  generated_std.unsqueeze(dim=1))/ \
                                  self.sample_size
           
        tot_nodes = l1.shape[0]
        h_dim = l1.shape[1]
        
        single_eye = torch.eye(h_dim).to(device)
        single_eye = single_eye.unsqueeze(dim=0)
        batch_eye = single_eye.repeat(tot_nodes,1,1)
        
        target_cov = target_cov + batch_eye
        generated_cov = generated_cov + batch_eye

        det_target_cov = torch.linalg.det(target_cov) 
        det_generated_cov = torch.linalg.det(generated_cov) 
        trace_mat = torch.matmul(torch.inverse(generated_cov), target_cov)
             
        x = torch.bmm(torch.unsqueeze(generated_mean - target_mean,dim=1),
                      torch.inverse(generated_cov))
        y = torch.unsqueeze(generated_mean - target_mean,dim=-1)
        z = torch.bmm(x,y).squeeze()

        # the information needed for loss computation
        recon_info = [det_target_cov, det_generated_cov, h_dim, trace_mat, z]
    
        return recon_info  

    def loss_func(self,
                  h0,
                  l1,
                  degree_logits,
                  feat_recon_list,
                  neigh_recon_list,
                  ground_truth_degree_matrix,
                  neighbor_dict):
        """
        Obtain the dense adjacency matrix of the graph.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input graph.
        """
        
        tot_nodes = l1.shape[0]

        # degree reconstruction loss
        ground_truth_degree_matrix = \
            torch.unsqueeze(ground_truth_degree_matrix, dim=1)
        degree_loss = self.degree_loss_func(degree_logits,
                                            ground_truth_degree_matrix.float())
        degree_loss_per_node = \
            (degree_logits-ground_truth_degree_matrix).pow(2)
        
        h_loss = 0
        feature_loss = 0
        loss_list = []
        loss_list_per_node = []
        feature_loss_list = []
        # Sample multiple times to remove noise
        for t in range(self.sample_time):
            # feature reconstrcution loss 
            h0_prime = feat_recon_list[t]
            feature_losses_per_node = (h0-h0_prime).pow(2).mean(1)
            feature_loss_list.append(feature_losses_per_node)
            
            # neigbor distribution reconstruction loss
            det_target_cov, det_generated_cov, h_dim, trace_mat, z = \
                                                            neigh_recon_list[t]
            KL_loss = 0.5 * (torch.log(det_target_cov / det_generated_cov) - \
                         h_dim + trace_mat.diagonal(offset=0, dim1=-1, 
                                                    dim2=-2).sum(-1) + z)
            local_index_loss = torch.mean(KL_loss)
            local_index_loss_per_node = KL_loss
    
            loss_list.append(local_index_loss)
            loss_list_per_node.append(local_index_loss_per_node)
            
        loss_list = torch.stack(loss_list)
        h_loss += torch.mean(loss_list)
        
        loss_list_per_node = torch.stack(loss_list_per_node)
        h_loss_per_node = torch.mean(loss_list_per_node,dim=0)
        
        feature_loss_per_node = torch.mean(torch.stack(feature_loss_list),
                                           dim=0)
        feature_loss += torch.mean(torch.stack(feature_loss_list))
                
        h_loss_per_node = h_loss_per_node.reshape(tot_nodes,1)
        degree_loss_per_node = degree_loss_per_node.reshape(tot_nodes,1)
        feature_loss_per_node = feature_loss_per_node.reshape(tot_nodes,1)
        
        loss = self.lambda_loss1 * h_loss \
            + degree_loss * self.lambda_loss3 \
            + self.lambda_loss2 * feature_loss
        loss_per_node = self.lambda_loss1 * h_loss_per_node \
            + degree_loss_per_node * self.lambda_loss3 \
                + self.lambda_loss2 * feature_loss_per_node
        
        return loss, loss_per_node, h_loss_per_node, \
            degree_loss_per_node, feature_loss_per_node

    @staticmethod
    def process_graph(data):
        """
        Obtain the neighbor dictornary and number of neighbors per node list

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input graph.
        """
        in_nodes = data.edge_index[0,:]
        out_nodes = data.edge_index[1,:]

        neighbor_dict = {}
        for in_node, out_node in zip(in_nodes, out_nodes):
            if in_node.item() not in neighbor_dict:
                neighbor_dict[in_node.item()] = []
            neighbor_dict[in_node.item()].append(out_node.item())

        neighbor_num_list = []
        for i in neighbor_dict:
            neighbor_num_list.append(len(neighbor_dict[i]))
        
        neighbor_num_list = torch.tensor(neighbor_num_list)

        return neighbor_dict, neighbor_num_list
