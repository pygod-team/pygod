# -*- coding: utf-8 -*-
"""Adversarial Outlier Aware Attributed Network Embedding (AdONE)"""
# Author: Kay Liu <zliu234@uic.edu>
# License: BSD 2 clause

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj
from torch_geometric.nn import MessagePassing
from sklearn.utils.validation import check_is_fitted

from . import BaseDetector
from .basic_nn import MLP
from ..utils.metric import eval_roc_auc


class AdONE(BaseDetector):
    """
    AdONE (Adversarial Outlier Aware Attributed Network Embedding)
    AdONE is consist of an attribute autoencoder and a structure
    autoencoder. It estimates five loss to optimize the model,
    including an attribute proximity loss, an attribute homophily loss,
    a structure proximity loss, a structure homophily loss, and an
    alignment loss. It calculates three outlier score, and averages
    them as an overall score.

    See :cite:`bandyopadhyay2020outlier` for details.

    Parameters
    ----------
    hid_dim :  int, optional
        Hidden dimension for both attribute autoencoder and structure
        autoencoder. Default: ``0``.
    num_layers : int, optional
        Total number of layers in model. A half (ceil) of the layers
        are for the encoder, the other half (floor) of the layers are
        for decoders. Default: ``4``.
    dropout : float, optional
        Dropout rate. Default: ``0.``.
    weight_decay : float, optional
        Weight decay (L2 penalty). Default: ``0.``.
    act : callable activation function or None, optional
        Activation function if not None.
        Default: ``torch.nn.functional.relu``.
    a1 : float, optional
        Loss balance weight for structure proximity.
        Default: ``0.2``.
    a2 : float, optional
        Loss balance weight for structure homophily.
        Default: ``0.2``.
    a3 : float, optional
        Loss balance weight for attribute proximity.
        Default: ``0.2``.
    a4 : float, optional
        Loss balance weight for attribute proximity.
        Default: ``0.2``.
    a5 : float, optional
        Loss balance weight for alignment.
        Default: ``0.2``.
    contamination : float, optional
        Valid in (0., 0.5). The proportion of outliers in the data set.
        Used when fitting to define the threshold on the decision
        function. Default: ``0.1``.
    lr : float, optional
        Learning rate. Default: ``0.004``.
    epoch : int, optional
        Maximum number of training epoch. Default: ``5``.
    gpu : int
        GPU Index, -1 for using CPU. Default: ``0``.
    verbose : bool
        Verbosity mode. Turn on to print out log information.
        Default: ``False``.

    Examples
    --------
    >>> from pygod.models import AdONE
    >>> model = AdONE()
    >>> model.fit(data) # PyG graph data object
    >>> prediction = model.predict(data)
    """

    def __init__(self,
                 hid_dim=32,
                 num_layers=4,
                 dropout=0.,
                 weight_decay=0.,
                 act=F.leaky_relu,
                 a1=0.2,
                 a2=0.2,
                 a3=0.2,
                 a4=0.2,
                 a5=0.2,
                 contamination=0.1,
                 lr=5e-3,
                 epoch=5,
                 gpu=0,
                 verbose=False):
        super(AdONE, self).__init__(contamination=contamination)

        # model param
        self.hid_dim = hid_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.act = act
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
        self.a4 = a4
        self.a5 = a5

        # training param
        self.lr = lr
        self.epoch = epoch
        if gpu >= 0 and torch.cuda.is_available():
            self.device = 'cuda:{}'.format(gpu)
        else:
            self.device = 'cpu'

        # other param
        self.verbose = verbose
        self.model = None

    def fit(self, G, y_true=None):
        """
        Description
        -----------
        Fit detector with input data.

        Parameters
        ----------
        G : PyTorch Geometric Data instance (torch_geometric.data.Data)
            The input data.
        y_true : numpy.array, optional (default=None)
            The optional outlier ground truth labels used to monitor the
            training progress. They are not used to optimize the
            unsupervised model.

        Returns
        -------
        self : object
            Fitted estimator.
        """

        x, s, edge_index = self.process_graph(G)

        self.model = AdONE_Base(x_dim=x.shape[1],
                                s_dim=s.shape[1],
                                hid_dim=self.hid_dim,
                                num_layers=self.num_layers,
                                dropout=self.dropout,
                                act=self.act).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=self.lr,
                                     weight_decay=self.weight_decay)

        score = None
        for epoch in range(self.epoch):
            self.model.train()
            x_, s_, h_a, h_s, dna, dns, dis_a, dis_s \
                = self.model(x, s, edge_index)
            score, loss = self.loss_func(x, x_, s, s_, h_a, h_s,
                                         dna, dns, dis_a, dis_s)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if self.verbose:
                print("Epoch {:04d}: Loss {:.4f}"
                      .format(epoch, loss.item()), end='')
                if y_true is not None:
                    auc = eval_roc_auc(y_true, score.detach().cpu().numpy())
                    print(" | AUC {:.4f}".format(auc), end='')
                print()

        self.decision_scores_ = score.detach().cpu().numpy()
        self._process_decision_scores()
        return self

    def decision_function(self, G):
        """
        Description
        -----------
        Predict raw anomaly score using the fitted detector. Outliers
        are assigned with larger anomaly scores.

        Parameters
        ----------
        G : PyTorch Geometric Data instance (torch_geometric.data.Data)
            The input data.

        Returns
        -------
        outlier_scores : numpy.ndarray
            The anomaly score of shape :math:`N`.
        """
        check_is_fitted(self, ['model'])

        # get needed data object from the input data
        x, s, edge_index = self.process_graph(G)

        # enable the evaluation mode
        self.model.eval()

        # construct the vector for holding the reconstruction error
        x_, s_, h_a, h_s, dna, dns, dis_a, dis_s = self.model(x, s, edge_index)
        outlier_scores, _ = self.loss_func(x, x_, s, s_, h_a, h_s,
                                           dna, dns, dis_a, dis_s)
        return outlier_scores.detach().cpu().numpy()

    def process_graph(self, G):
        """
        Description
        -----------
        Process the raw PyG data object into a tuple of sub data
        objects needed for the model.

        Parameters
        ----------
        G : PyTorch Geometric Data instance (torch_geometric.data.Data)
            The input data.

        Returns
        -------
        x : torch.Tensor
            Attribute (feature) of nodes.
        s : torch.Tensor
            Adjacency matrix of the graph.
        edge_index : torch.Tensor
            Edge list of the graph.
        """
        edge_index = G.edge_index

        s = to_dense_adj(edge_index)[0].to(self.device)
        edge_index = edge_index.to(self.device)
        x = G.x.to(self.device)

        return x, s, edge_index

    def loss_func(self, x, x_, s, s_, h_a, h_s, dna, dns, dis_a, dis_s):
        # equation 9 is based on the official implementation, and it
        # is slightly different from the paper
        dx = torch.sum(torch.pow(x - x_, 2), 1)
        tmp = self.a3 * dx + self.a4 * dna
        oa = tmp / torch.sum(tmp)

        # equation 8 is based on the official implementation, and it
        # is slightly different from the paper
        ds = torch.sum(torch.pow(s - s_, 2), 1)
        tmp = self.a1 * ds + self.a2 * dns
        os = tmp / torch.sum(tmp)

        # equation 10
        dc = torch.sum(torch.pow(h_a - h_s, 2), 1)
        oc = dc / torch.sum(dc)

        # equation 4
        loss_prox_a = torch.mean(torch.log(torch.pow(oa, -1)) * dx)

        # equation 5
        loss_hom_a = torch.mean(torch.log(torch.pow(oa, -1)) * dna)

        # equation 2
        loss_prox_s = torch.mean(torch.log(torch.pow(os, -1)) * ds)

        # equation 3
        loss_hom_s = torch.mean(torch.log(torch.pow(os, -1)) * dns)

        # equation 6
        loss_alg = torch.mean(torch.log(torch.pow(oc, -1))
                              * (-torch.log(1 - dis_a) - torch.log(dis_s)))

        # equation 7
        loss = self.a3 * loss_prox_a + \
               self.a4 * loss_hom_a + \
               self.a1 * loss_prox_s + \
               self.a2 * loss_hom_s + \
               self.a5 * loss_alg

        score = (oa + os + oc) / 3
        return score, loss


class AdONE_Base(nn.Module):
    def __init__(self,
                 x_dim,
                 s_dim,
                 hid_dim,
                 num_layers,
                 dropout,
                 act):
        super(AdONE_Base, self).__init__()

        # split the number of layers for the encoder and decoders
        decoder_layers = int(num_layers / 2)
        encoder_layers = num_layers - decoder_layers

        self.attr_encoder = MLP(in_channels=x_dim,
                                hidden_channels=hid_dim,
                                out_channels=hid_dim,
                                num_layers=encoder_layers,
                                dropout=dropout,
                                act=act)

        self.attr_decoder = MLP(in_channels=hid_dim,
                                hidden_channels=hid_dim,
                                out_channels=x_dim,
                                num_layers=decoder_layers,
                                dropout=dropout,
                                act=act)

        self.struct_encoder = MLP(in_channels=s_dim,
                                  hidden_channels=hid_dim,
                                  out_channels=hid_dim,
                                  num_layers=encoder_layers,
                                  dropout=dropout,
                                  act=act)

        self.struct_decoder = MLP(in_channels=hid_dim,
                                  hidden_channels=hid_dim,
                                  out_channels=s_dim,
                                  num_layers=decoder_layers,
                                  dropout=dropout,
                                  act=act)

        self.neigh_diff = NeighDiff()

        self.discriminator = MLP(in_channels=hid_dim,
                                 hidden_channels=int(hid_dim / 2),
                                 out_channels=1,
                                 num_layers=2,
                                 dropout=dropout,
                                 act=torch.tanh)

    def forward(self, x, s, edge_index):
        h_a = self.attr_encoder(x)
        x_ = self.attr_decoder(h_a)
        dna = self.neigh_diff(h_a, edge_index).squeeze()
        h_s = self.struct_encoder(s)
        s_ = self.struct_decoder(h_s)
        dns = self.neigh_diff(h_s, edge_index).squeeze()
        dis_a = torch.sigmoid(self.discriminator(h_a))
        dis_s = torch.sigmoid(self.discriminator(h_s))

        return x_, s_, h_a, h_s, dna, dns, dis_a, dis_s


class NeighDiff(MessagePassing):
    def __init__(self):
        super().__init__(aggr='mean')

    def forward(self, h, edge_index):
        return self.propagate(edge_index, h=h)

    def message(self, h_i, h_j, edge_index):
        return torch.sum(torch.pow(h_i - h_j, 2), dim=1, keepdim=True)
