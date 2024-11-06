import torch
from torch_geometric.nn import GCN

from .base import DeepDetector
from ..nn import CARDBase


class CARD(DeepDetector):

    def __init__(self,
                 hid_dim=64,
                 num_layers=4,
                 dropout=0.,
                 weight_decay=0.,
                 act=torch.nn.functional.relu,
                 backbone=GCN,
                 contamination=0.1,
                 lr=4e-3,
                 epoch=100,
                 gpu=-1,
                 batch_size=0,
                 num_neigh=-1,
                 verbose=0,
                 save_emb=False,
                 compile_model=False,
                 **kwargs):
        super(CARD, self).__init__(hid_dim=hid_dim,
                                   num_layers=num_layers,
                                   dropout=dropout,
                                   weight_decay=weight_decay,
                                   act=act,
                                   backbone=backbone,
                                   contamination=contamination,
                                   lr=lr,
                                   epoch=epoch,
                                   gpu=gpu,
                                   batch_size=batch_size,
                                   num_neigh=num_neigh,
                                   verbose=verbose,
                                   save_emb=save_emb,
                                   compile_model=compile_model,
                                   **kwargs)

    def process_graph(self, data):
        self.community_adj, self.diff_data = CARDBase.process_graph(data)
        self.community_adj = self.community_adj.to(self.device)
        self.diff_data = self.diff_data.to(self.device)

    def init_model(self, **kwargs):
        if self.save_emb:
            self.emb = torch.zeros(self.num_nodes,
                                   self.hid_dim)
        return CARDBase(in_dim=self.in_dim,
                        community_adj=self.community_adj,
                        hid_dim=self.hid_dim,
                        num_layers=self.num_layers,
                        dropout=self.dropout,
                        act=self.act,
                        backbone=self.backbone,
                        **kwargs).to(self.device)

    def forward_model(self, data):
        batch_size = data.batch_size
        # node_idx = data.n_id
        # warren comment test
        # for index in len(node_idx):
        #     subgraphs = NeighborLoader(data, num_neighbors=[-1] * self.num_layers, input_nodes=[index])
        #     for subgraph in subgraphs:
        #         subgraph.x[0, :] = 0
        #         x = subgraph.x.to(self.device)
        #         edge_index = subgraph.edge_index.to(self.device)
        #         pos_logits, neg_logits = self.model(x, edge_index)

        data.x.to(self.device)
        data.edge_index.to(self.device)

        pos_logits, neg_logits, x_, local_x_ = self.model(data)
        diff_pos_logits, diff_neg_logits, x_, local_x_ = self.model(
            self.diff_data)

        logits = torch.cat([pos_logits[:batch_size],
                            neg_logits[:batch_size]])
        diff_logits = torch.cat([diff_pos_logits[:batch_size],
                                 diff_neg_logits[:batch_size]])

        con_label = torch.cat([torch.ones(batch_size),
                               torch.zeros(batch_size)]).to(self.device)

        loss, score = self.model.loss_func(
            logits, diff_logits, x_, local_x_, data.x, con_label)

        return loss, score.detach().cpu()
