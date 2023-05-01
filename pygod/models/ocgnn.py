# -*- coding: utf-8 -*-
""" One-Class Graph Neural Networks for Anomaly Detection in Attributed Networks
"""
# Author: Xueying Ding <xding2@andrew.cmu.edu>
# License: BSD 2 clause

import torch
from torch_geometric.nn import GCN

from . import DeepDetector
from ..nn import OCGNNBase


class OCGNN(DeepDetector):
    """
    OCGNN (One-Class Graph Neural Networks for Anomaly Detection in
    Attributed Networks) is an anomaly detector that measures the
    distance of anomaly to the centroid, in a similar fashion to the
    support vector machine, but in the embedding space after feeding
    towards several layers of GCN.

    See :cite:`wang2021one` for details.



    Examples
    --------
    >>> from pygod.models import AnomalyDAE
    >>> model = OCGNN()
    >>> model.fit(data) # PyG graph data object
    >>> prediction = model.predict(data)
    """

    def __init__(self,
                 hid_dim=64,
                 num_layers=2,
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
                 beta=0.5,
                 warmup=2,
                 eps=0.001,
                 verbose=0,
                 **kwargs):
        super(OCGNN, self).__init__(hid_dim=hid_dim,
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

        self.beta = beta
        self.warmup = warmup
        self.eps = eps

    def process_graph(self, data):
        pass

    def init_model(self, **kwargs):
        if self.save_emb:
            self.emb = torch.zeros(self.num_nodes,
                                   self.hid_dim).to(self.device)

        return OCGNNBase(in_dim=self.in_dim,
                         hid_dim=self.hid_dim,
                         num_layers=self.num_layers,
                         dropout=self.dropout,
                         act=self.act,
                         beta=self.beta,
                         warmup=self.warmup,
                         eps=self.eps,
                         backbone=self.backbone,
                         **kwargs).to(self.device)

    def forward_model(self, data):
        batch_size = data.batch_size

        x = data.x.to(self.device)
        edge_index = data.edge_index.to(self.device)

        emb = self.model(x, edge_index)
        loss, score = self.model.loss_func(emb[:batch_size])

        return loss, score.detach().cpu()
