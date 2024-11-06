import torch
import torch.nn as nn
from torch.nn.functional import binary_cross_entropy_with_logits, normalize

from torch_geometric.nn import GCN
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import to_dense_adj
from torch_geometric.transforms import GDC
from .functional import double_recon_loss

HYPER_PARAM_THRE = 78000


class CARDBase(nn.Module):

    def __init__(self,
                 in_dim,
                 community_adj,
                 alpha=0.1,
                 hid_dim=64,
                 num_layers=4,
                 dropout=0.,
                 act=torch.nn.functional.relu,
                 backbone=GCN,

                 **kwargs):
        super(CARDBase, self).__init__()
        self.community_adj = community_adj
        self.alpha = alpha
        self.hid_dim = hid_dim
        self.num_layers = num_layers

        self.fp = 0.6
        self.gama = 0.5

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
            nn.Linear(hid_dim, int(hid_dim / 2)),
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

    def forward(self, data):
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
        logits : torch.Tensor
            Discriminator logits of positive examples.  
        neg_logits : torch.Tensor
            Discriminator logits of negative examples.
        x_: torch.Tensor
            feature reconstract matrix
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
        ori_loss = self.disc_loss_func(logits, con_label)
        diff_loss = self.disc_loss_func(diff_logits, con_label)
        logit_loss = (ori_loss + diff_loss) / 2
        batch_size = logits.shape[0] / 2
        h_1 = normalize(logits[:batch_size], dim=0, p=2)
        h_2 = normalize(diff_logits[:batch_size], dim=0, p=2)
        inter_logit_loss = 2 - 2 * (h_1 * h_2).sum(dim=-1).mean()

        rec_loss = double_recon_loss(x, x_, 0, 0, 1)
        local_rec_loss = double_recon_loss(x, local_x_, 0, 0, 1)
        constra_loss = torch.mean(logit_loss) + \
            inter_logit_loss + self.gama * torch.mean(local_rec_loss)

        final_loss = (1 - self.fp) * constra_loss + \
            self.fp * torch.mean(rec_loss)  # + 0.5 * kl

        constra_score = ((logits[1, :] - logits[0, :]) +
                         (diff_logits[1, :] - diff_logits[0, :])) / 2

        score = (1 - self.fp) * (constra_score + self.gama *
                                 local_rec_loss) + self.fp * rec_loss

        return final_loss, score

    def _train_subgraph_network(self, data):
        res_emb = []
        disc_emb = []

        for index in range(data.num_nodes):
            subgraphs = NeighborLoader(
                data, num_neighbors=[-1] * self.num_layers)
            subgraph = subgraphs([index])
            community_idx = []
            i = 0
            while len(community_idx) < self.hid_dim:
                community_idx.append(subgraph.n_id[i])
                i = (i + 1) % len(community_idx)

            community_adj = (
                self.community_adj[subgraph.n_id][:, community_idx])

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

            res_emb.append(torch.mean(combine_emb[1:, :], 0))
            disc_emb.append(combine_emb[0, :])

        return torch.stack(res_emb), torch.stack(disc_emb)

    @staticmethod
    def process_graph(data):
        """
        Obtain the community structure matrix and the diffusion struture matrix.

        Parameters
        ----------
        data: torch_geometric.data.Data
            Input graph.

        Returns
        -------
        community_adj: torch.Tensor
                       Community structure matrix, corresponding to the B matrix in the paper.

        diff_adj: torch.Tensor
                  diffusion struture matrix using GDC, corresponding to the S maxtrix in the paper.
        """
        data.s = to_dense_adj(data.edge_index)[0]
        k1 = torch.sum(data.s, axis=1)
        k2 = k1.reshape(data.num_nodes, 1)
        e = k1 * k2 / (2 * data.num_edges)
        community_adj = torch.tensor(data.s - e)

        transform = GDC(
            self_loop_weight=1,
            normalization_in='sym',
            normalization_out='col',
            diffusion_kwargs=dict(method='ppr', alpha=0.01, eps=0.0001),
            sparsification_kwargs=dict(method='topk', k=128, dim=0),
            exact=True)
        diff_data = transform(data)

        return community_adj, diff_data
