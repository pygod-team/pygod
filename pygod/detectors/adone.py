# -*- coding: utf-8 -*-
"""Adversarial Outlier Aware Attributed Network Embedding (AdONE)"""
# Author: Kay Liu <zliu234@uic.edu>
# License: BSD 2 clause

import torch
from torch_geometric.utils import to_dense_adj

from . import DeepDetector
from ..nn import AdONEBase


class AdONE(DeepDetector):
    """
    AdONE (Adversarial Outlier Aware Attributed Network
    Embedding) consists of an attribute autoencoder and a structure
    autoencoder. It estimates five loss to optimize the model,
    including an attribute proximity loss, an attribute homophily loss,
    a structure proximity loss, a structure homophily loss, and an
    alignment loss. It calculates three outlier scores, and averages
    them as an overall score.

    See :cite:`bandyopadhyay2020outlier` for details.

    Parameters
    ----------

    Examples
    --------
    >>> from pygod.detectors import AdONE
    >>> model = AdONE()
    >>> model.fit(data) # PyG graph data object
    >>> prediction = model.predict(data)
    """

    def __init__(self,
                 hid_dim=64,
                 num_layers=4,
                 dropout=0.,
                 weight_decay=0.,
                 act=torch.nn.functional.relu,
                 w1=0.2,
                 w2=0.2,
                 w3=0.2,
                 w4=0.2,
                 w5=0.2,
                 contamination=0.1,
                 lr=4e-3,
                 epoch=100,
                 gpu=-1,
                 batch_size=0,
                 num_neigh=-1,
                 verbose=0,
                 **kwargs):
        super(AdONE, self).__init__(hid_dim=hid_dim,
                                    num_layers=1,
                                    dropout=dropout,
                                    weight_decay=weight_decay,
                                    act=act,
                                    contamination=contamination,
                                    lr=lr,
                                    epoch=epoch,
                                    gpu=gpu,
                                    batch_size=batch_size,
                                    num_neigh=num_neigh,
                                    verbose=verbose,
                                    **kwargs)

        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.w4 = w4
        self.w5 = w5
        self.num_layers = num_layers

        self.attribute_score_ = None
        self.structural_score_ = None
        self.combined_score_ = None

    def process_graph(self, data):
        AdONEBase.process_graph(data)

    def init_model(self, **kwargs):

        self.attribute_score_ = torch.zeros(self.num_nodes)
        self.structural_score_ = torch.zeros(self.num_nodes)
        self.combined_score_ = torch.zeros(self.num_nodes)

        if self.save_emb:
            self.emb = [torch.zeros(self.num_nodes, self.hid_dim),
                        torch.zeros(self.num_nodes, self.hid_dim)]

        return AdONEBase(x_dim=self.in_dim,
                         s_dim=self.num_nodes,
                         hid_dim=self.hid_dim,
                         num_layers=self.num_layers,
                         dropout=self.dropout,
                         act=self.act,
                         w1=self.w1,
                         w2=self.w2,
                         w3=self.w3,
                         w4=self.w4,
                         w5=self.w5,
                         **kwargs).to(self.device)

    def forward_model(self, data):
        batch_size = data.batch_size
        node_idx = data.node_idx

        x = data.x.to(self.device)
        s = data.s.to(self.device)
        edge_index = data.edge_index.to(self.device)

        x_, s_, h_a, h_s, dna, dns, dis_a, dis_s = self.model(x, s, edge_index)
        loss, oa, os, oc = self.model.loss_func(x[:batch_size],
                                                x_[:batch_size],
                                                s[:batch_size],
                                                s_[:batch_size],
                                                h_a[:batch_size],
                                                h_s[:batch_size],
                                                dna[:batch_size],
                                                dns[:batch_size],
                                                dis_a[:batch_size],
                                                dis_s[:batch_size])

        self.attribute_score_[node_idx[:batch_size]] = oa.detach().cpu()
        self.structural_score_[node_idx[:batch_size]] = os.detach().cpu()
        self.combined_score_[node_idx[:batch_size]] = oc.detach().cpu()

        return loss, ((oa + os + oc) / 3).detach().cpu()
