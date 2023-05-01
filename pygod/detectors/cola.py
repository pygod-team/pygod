# -*- coding: utf-8 -*-
"""Anomaly Detection on Attributed Networks via Contrastive
Self-Supervised Learning (CoLA)"""
# Author: Canyu Chen <cchen151@hawk.iit.edu>, Kay Liu <zliu234@uic.edu>
# License: BSD 2 clause

import torch
from torch_geometric.nn import GCN

from .base import DeepDetector
from ..nn import CoLABase


class CoLA(DeepDetector):
    """

    """

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
                 **kwargs):
        super(CoLA, self).__init__(hid_dim=hid_dim,
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
                                   **kwargs)

    def process_graph(self, data):
        pass

    def init_model(self, **kwargs):
        if self.save_emb:
            self.emb = torch.zeros(self.num_nodes,
                                   self.hid_dim).to(self.device)
        return CoLABase(in_dim=self.in_dim,
                        hid_dim=self.hid_dim,
                        num_layers=self.num_layers,
                        dropout=self.dropout,
                        act=self.act,
                        backbone=self.backbone,
                        **kwargs).to(self.device)

    def forward_model(self, data):
        batch_size = data.batch_size

        x = data.x.to(self.device)
        edge_index = data.edge_index.to(self.device)

        pos_logits, neg_logits = self.model(x, edge_index)
        logits = torch.cat([pos_logits[:batch_size],
                            neg_logits[:batch_size]])
        con_label = torch.cat([torch.ones(batch_size),
                               torch.zeros(batch_size)]).to(self.device)
        print(logits.shape, con_label.shape)
        loss = self.model.loss_func(logits, con_label)

        score = neg_logits[:batch_size] - pos_logits[:batch_size]

        return loss, score.detach().cpu()
