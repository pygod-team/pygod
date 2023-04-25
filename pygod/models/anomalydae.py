# -*- coding: utf-8 -*-
"""AnomalyDAE: Dual autoencoder for anomaly detection
on attributed networks"""
import warnings

# Author: Xueying Ding <xding2@andrew.cmu.edu>,
#         Kay Liu <zliu234@uic.edu>
# License: BSD 2 clause

import torch
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj

from . import DeepDetector
from ..nn import AnomalyDAEBase
from ..utils import validate_device


class AnomalyDAE(DeepDetector):
    """
    AnomalyDAE (Dual autoencoder for anomaly detection on attributed
    networks) is an anomaly detector that consists of a structure
    autoencoder and an attribute autoencoder to learn both node
    embedding and attribute embedding jointly in latent space. The
    structural autoencoder uses Graph Attention layers. The
    reconstruction mean square error of the decoders are defined as
    structure anomaly score and attribute anomaly score, respectively,
    with two additional penalties on the reconstructed adj matrix and 
    node attributes (force entries to be nonzero).

    See :cite:`fan2020anomalydae` for details.

    Parameters
    ----------
    hid_dim :  int, optional
        Hidden dimension of model. Defaults: `64``.
    out_dim : int, optional
        Dimension of the reduced representation after passing through the 
        structure autoencoder and attribute autoencoder. Defaults: ``4``.
    dropout : float, optional
        Dropout rate. Defaults: ``0.2``.
    weight_decay : float, optional
        Weight decay (L2 penalty). Defaults: ``1e-5``.
    act : callable activation function or None, optional
        Activation function if not None.
        Defaults: ``torch.nn.functional.relu``.
    alpha : float, optional
        Loss balance weight for attribute and structure. ``None`` for
        balancing by standard deviation. Default: ``None``.
    theta: float, optional
         greater than 1, impose penalty to the reconstruction error of
         the non-zero elements in the adjacency matrix
         Defaults: ``1.01``
    eta: float, optional
         greater than 1, imporse penalty to the reconstruction error of 
         the non-zero elements in the node attributes
         Defaults: ``1.01``
    contamination : float, optional
        Valid in (0., 0.5). The proportion of outliers in the data set.
        Used when fitting to define the threshold on the decision
        function. Defaults: ``0.1``.
    lr : float, optional
        Learning rate. Defaults: ``0.004``.
    epoch : int, optional
        Maximum number of training epoch. Defaults: ``5``.
    gpu : int
        GPU Index, -1 for using CPU. Defaults: ``0``.
    batch_size : int, optional
        Minibatch size, 0 for full batch training. Default: ``0``.
    num_neigh : int, optional
        Number of neighbors in sampling, -1 for all neighbors.
        Default: ``-1``.
    verbose : bool
        Verbosity mode. Turn on to print out log information.
        Defaults: ``False``.

    Examples
    --------

    """

    def __init__(self,
                 emb_dim=64,
                 hid_dim=64,
                 num_layers=4,
                 dropout=0.2,
                 weight_decay=1e-5,
                 act=F.relu,
                 backbone=None,
                 alpha=0.5,
                 theta=1.,
                 eta=1.,
                 contamination=0.1,
                 lr=0.004,
                 epoch=5,
                 gpu=0,
                 batch_size=0,
                 num_neigh=-1,
                 verbose=0,
                 **kwargs):
        if backbone is not None or num_layers != 4:
            warnings.warn("Backbone and num_layers are not used in AnomalyDAE")

        super(AnomalyDAE, self).__init__(hid_dim=hid_dim,
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

        self.emb_dim = emb_dim
        self.alpha = alpha
        self.theta = theta
        self.eta = eta

    def process_graph(self, data):
        data.s = to_dense_adj(data.edge_index)[0]

    def init_model(self, **kwargs):
        return AnomalyDAEBase(in_dim=self.in_dim,
                              num_nodes=self.num_nodes,
                              emb_dim=self.emb_dim,
                              hid_dim=self.hid_dim,
                              dropout=self.dropout,
                              act=self.act,
                              **kwargs).to(self.device)

    def forward_model(self, data):
        batch_size = data.batch_size
        node_idx = data.node_idx

        x = data.x.to(self.device)
        s = data.s.to(self.device)
        edge_index = data.edge_index.to(self.device)

        x_, s_ = self.model(x, edge_index, batch_size)

        # positive weight conversion
        weight = 1 - self.alpha
        pos_weight_a = self.eta / (1 + self.eta)
        pos_weight_s = self.theta / (1 + self.theta)

        score = self.model.loss_func(x[:batch_size],
                                     x_[:batch_size],
                                     s[:batch_size, node_idx],
                                     s_[:batch_size],
                                     weight,
                                     pos_weight_a,
                                     pos_weight_s)

        loss = torch.mean(score)

        return loss, score.detach().cpu()
