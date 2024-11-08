import torch
import torch.nn as nn
from torch.nn.functional import binary_cross_entropy_with_logits, normalize

from torch_geometric.nn import GCN
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import to_dense_adj, to_undirected
from torch_geometric.transforms import GDC
from .functional import double_recon_loss


class CARDBase(nn.Module):
    '''
    Community-Guided Contrastive Learning with Anomaly-Aware Reconstruction for 
    Anomaly Detection on Attributed Networks.

    CARD is a contrastive learning based method and utilizes mask reconstruction and community
    information to make anomalies more distinct. This model is train with contrastive loss and 
    local and global attribute reconstruction loss. Random neighbor sampling instead of random walk 
    sampling is used to sample the subgraph corresponding to each node. Since random neighbor sampling 
    cannot accurately control the number of neighbors for each sampling, it may run slower compared to 
    the method implementation in the original paper.

    See:cite:`Wang2024Card` for details.

    Parameters
    ----------
    in_dim : int
        Input dimension of model.
    subgraph_num_neigh: int, optional
        Number of neighbors in subgraph sampling for each node, Values not exceeding 4 are recommended for efficiency.
        Default: ``4``.
    fp: float, optional
        The balance parameter between the mask autoencoder module and contrastive learning.
        Default: ``0.6``
    gama: float, optional
        The proportion of the local reconstruction in contrastive learning module.
        Default: ``0.5``
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
    '''

    def __init__(self,
                 in_dim,
                 fp=0.6,
                 gama=0.4,
                 subgraph_num_neigh=4,
                 alpha=0.1,
                 hid_dim=64,
                 num_layers=4,
                 dropout=0.,
                 act=torch.nn.functional.relu,
                 backbone=GCN,
                 **kwargs):
        super(CARDBase, self).__init__()
        self.alpha = alpha
        self.hid_dim = hid_dim
        self.num_layers = num_layers
        self.subgraph_num_neigh = subgraph_num_neigh
        self.fp = fp
        self.gama = gama

        # subgraph encoder
        self.encoder = backbone(in_channels=in_dim,
                                hidden_channels=hid_dim,
                                num_layers=num_layers,
                                out_channels=hid_dim,
                                dropout=dropout,
                                act=act,
                                **kwargs)

        self.global_feat_encoder = backbone(in_channels=in_dim,
                                            hidden_channels=hid_dim,
                                            num_layers=num_layers,
                                            out_channels=hid_dim,
                                            dropout=dropout,
                                            act=act,
                                            **kwargs)

        self.global_feat_decoder = backbone(in_channels=hid_dim,
                                            hidden_channels=hid_dim,
                                            num_layers=num_layers,
                                            out_channels=in_dim,
                                            dropout=dropout,
                                            act=act,
                                            **kwargs)

        self.local_feat_decoder = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.PReLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.PReLU(),
            nn.Linear(hid_dim, in_dim),
            nn.PReLU()
        )

        self.community_encoder = nn.Sequential(
            nn.Linear(self.subgraph_num_neigh, int(hid_dim / 2)),
            nn.PReLU(),
            nn.Linear(int(hid_dim / 2), hid_dim),
            nn.PReLU(),
        )

        self.combine_encoder = backbone(in_channels=hid_dim,
                                        hidden_channels=hid_dim,
                                        num_layers=num_layers,
                                        out_channels=hid_dim,
                                        dropout=dropout,
                                        act=act,
                                        **kwargs)

        self.discriminator = nn.Bilinear(hid_dim, hid_dim, 1)

        self.disc_loss_func = binary_cross_entropy_with_logits
        self.emb = None
        self.disc_emb = None

    def forward(self, data):
        """
        Forward computation.

        Parameters
        ----------
        Data: torch_geometric.data.Data
              Input graph.

        Returns
        -------
        logits : torch.Tensor
            Discriminator logits of positive examples.  
        neg_logits : torch.Tensor
            Discriminator logits of negative examples.
        x_: torch.Tensor
            feature reconstract matrix
        local_x_: torch.Tensor
            subgraph feature reconstract matrix
        """

        self.emb, self.disc_emb = self._train_subgraph_network(data)
        x = data.x
        logits = self.discriminator(self.disc_emb, self.emb)

        perm_idx = torch.randperm(
            self.disc_emb.shape[0]).to(self.disc_emb.device)
        neg_logits = self.discriminator(self.disc_emb[perm_idx], self.emb)

        local_x_ = self.local_feat_decoder(self.emb)

        attr_emb = self.global_feat_encoder(x, data.edge_index)
        x_ = self.global_feat_decoder(attr_emb, data.edge_index)

        return logits.squeeze(), neg_logits.squeeze(), x_, local_x_

    def loss_func(self, logits, diff_logits, x_, local_x_, x, con_label):
        """
        The loss function proposed in the CARD paper.
        This implementation ignores the KL-loss as it contributes little to the accuracy.

        Parameters
        ----------
        logits : _type_
            _description_
        diff_logits : _type_
            _description_
        x_ : _type_
            _description_
        local_x_ : _type_
            _description_
        x : _type_
            _description_
        con_label : _type_
            _description_

        Returns
        -------
        final_loss: torch.Tensor
            The total loss value used to backpropagate and update
            the model parameters.
        score: torch.Tensor
            The anomaly score for each node.
        """
        
        ori_loss = self.disc_loss_func(logits, con_label)
        diff_loss = self.disc_loss_func(diff_logits, con_label)
        logit_loss = (ori_loss + diff_loss) / 2
        batch_size = int(logits.shape[0] / 2)
        h_1 = normalize(logits[:batch_size], dim=0, p=2)
        h_2 = normalize(diff_logits[:batch_size], dim=0, p=2)
        inter_logit_loss = 2 - 2 * (h_1 * h_2).sum(dim=-1).mean()

        rec_loss = double_recon_loss(x, x_, x, x, 1)
        local_rec_loss = double_recon_loss(x, local_x_, x, x, 1)
        constra_loss = torch.mean(logit_loss) + \
            inter_logit_loss + self.gama * torch.mean(local_rec_loss)

        final_loss = (1 - self.fp) * constra_loss + \
            self.fp * torch.mean(rec_loss)  # + 0.5 * kl

        constra_score = ((logits[batch_size:] - logits[:batch_size]) +
                         (diff_logits[batch_size:] - diff_logits[:batch_size])) / 2

        score = (1 - self.fp) * (constra_score + self.gama *
                                 local_rec_loss[:batch_size]) + self.fp * rec_loss[:batch_size]

        return final_loss, score

    def _train_subgraph_network(self, data):
        """
        Train the model subgraph encoder and community-guided module
        with each node and its corresponding subgraph as input. 

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input graph.

        Returns
        -------
        res_emb: torch.Tensor
            Subgraph embedding readout.
        disc_emb: torch.Tensor
            Target node embedding.
        """
        
        res_emb = []
        disc_emb = []

        for index in range(data.num_nodes):
            subgraphs = NeighborLoader(
                data, num_neighbors=[self.subgraph_num_neigh] * self.num_layers)
            subgraph = subgraphs([index])
            community_idx = []
            i = 0
            while len(community_idx) < self.subgraph_num_neigh:
                community_idx.append(subgraph.n_id[i])
                i = (i + 1) % len(subgraph.n_id)

            community_adj = subgraph.community_adj[:, community_idx]

            subgraph.x[0, :] = 0
            x = subgraph.x
            edge_index = subgraph.edge_index

            # diff_subgraphs = NeighborLoader(
            #      self.diff, num_neighbors=[-1] * self.num_layers)
            # diff_subgraph = diff_subgraphs([index])
            # diff_subgraph.x[0, :] = 0
            # diff_x = diff_subgraph.x.to(self.device)
            # diff_edge_index = diff_subgraph.edge_index.to(self.device)

            ori_emb = self.encoder(x, edge_index)
            community_emb = self.community_encoder(community_adj)
            combine_emb = self.combine_encoder(
                ori_emb + self.alpha * community_emb, edge_index)
            # avoid nan problem
            if combine_emb.shape[0] > 1:
                res_emb.append(torch.mean(combine_emb[1:, :], 0))
            else:
                res_emb.append(combine_emb[0, :])

            disc_emb.append(combine_emb[0, :])

        return torch.stack(res_emb), torch.stack(disc_emb)

    @staticmethod
    def process_graph(data):
        """
        Obtain the community structure matrix and the diffusion graph data.

        Parameters
        ----------
        data: torch_geometric.data.Data
            Input graph.

        Returns
        -------
        community_adj: torch.Tensor
                       Community structure matrix, corresponding to the B matrix in the paper.

        diff_data: torch_geometric.data.Data
                  Diffusion graph Data
        """

        # only support undirected graph
        if not data.is_undirected():
            data.edge_index = to_undirected(data.edge_index)

        data.s = to_dense_adj(data.edge_index)[0]
        k1 = torch.sum(data.s, axis=1)
        k2 = k1.reshape(data.num_nodes, 1)
        e = k1 * k2 / (2 * data.num_edges)
        community_adj = (data.s - e).clone().detach()

        transform = GDC(
            self_loop_weight=1,
            normalization_in='sym',
            normalization_out='col',
            diffusion_kwargs=dict(method='ppr', alpha=0.01, eps=0.0001),
            sparsification_kwargs=dict(method='topk', k=128, dim=0),
            exact=True)
        diff_data = transform(data)

        return community_adj, diff_data
