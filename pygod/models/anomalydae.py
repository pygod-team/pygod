# -*- coding: utf-8 -*-
"""AnomalyDAE: Dual autoencoder for anomaly detection on attributed networks"""
# Author: Xueying Ding <xding2@andrew.cmu.edu>
# License: BSD 2 clause


import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.utils import to_dense_adj
from torch_geometric.loader import NeighborLoader
from sklearn.utils.validation import check_is_fitted

from . import BaseDetector
from ..utils import validate_device
from ..metrics import eval_roc_auc


class AnomalyDAE(BaseDetector):
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
    embed_dim :  int, optional
        Hidden dimension of model. Defaults: `8``.
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
        loss balance weight for attribute and structure.
        Defaults: ``0.5``.
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
    >>> from pygod.models import AnomalyDAE
    >>> model = AnomalyDAE()
    >>> model.fit(data) # PyG graph data object
    >>> prediction = model.predict(data)
    """

    def __init__(self,
                 embed_dim=8,
                 out_dim=4,
                 dropout=0.2,
                 weight_decay=1e-5,
                 act=F.relu,
                 alpha=0.5,
                 theta=1.01,
                 eta=1.01,
                 contamination=0.1,
                 lr=0.004,
                 epoch=5,
                 gpu=0,
                 batch_size=0,
                 num_neigh=-1,
                 verbose=False):
        super(AnomalyDAE, self).__init__(contamination=contamination)

        # model param
        self.embed_dim = embed_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.act = act
        self.alpha = alpha
        self.theta = theta
        self.eta = eta

        # training param
        self.lr = lr
        self.epoch = epoch
        self.device = validate_device(gpu)
        self.batch_size = batch_size
        self.num_neigh = num_neigh

        # other param
        self.verbose = verbose
        self.model = None

    def fit(self, G, y_true=None):
        """
        Fit detector with input data.

        Parameters
        ----------
        G : torch_geometric.data.Data
            The input data.
        y_true : numpy.ndarray, optional
            The optional outlier ground truth labels used to monitor
            the training progress. They are not used to optimize the
            unsupervised model. Default: ``None``.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        G.node_idx = torch.arange(G.x.shape[0])
        G.s = to_dense_adj(G.edge_index)[0]
        if self.batch_size == 0:
            self.batch_size = G.x.shape[0]
        loader = NeighborLoader(G,
                                [self.num_neigh],
                                batch_size=self.batch_size)

        self.model = AnomalyDAE_Base(in_node_dim=G.x.shape[1],
                                     in_num_dim=self.batch_size,
                                     embed_dim=self.embed_dim,
                                     out_dim=self.out_dim,
                                     dropout=self.dropout,
                                     act=self.act).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=self.lr,
                                     weight_decay=self.weight_decay)

        self.model.train()
        decision_scores = np.zeros(G.x.shape[0])
        for epoch in range(self.epoch):
            epoch_loss = 0
            for sampled_data in loader:
                batch_size = sampled_data.batch_size
                node_idx = sampled_data.node_idx
                x, s, edge_index = self.process_graph(sampled_data)

                x_, s_ = self.model(x, edge_index, batch_size)
                score = self.loss_func(x[:batch_size],
                                       x_[:batch_size],
                                       s[:batch_size, node_idx],
                                       s_[:batch_size])
                decision_scores[node_idx[:batch_size]] = score.detach() \
                    .cpu().numpy()
                loss = torch.mean(score)
                epoch_loss += loss.item() * batch_size

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if self.verbose:
                print("Epoch {:04d}: Loss {:.4f}"
                      .format(epoch, epoch_loss / G.x.shape[0]), end='')
                if y_true is not None:
                    auc = eval_roc_auc(y_true, decision_scores)
                    print(" | AUC {:.4f}".format(auc), end='')
                print()

        self.decision_scores_ = decision_scores
        self._process_decision_scores()
        return self

    def decision_function(self, G):
        """
        Predict raw anomaly score of X using the fitted detector.
        The anomaly score of an input sample is computed based on different
        detector algorithms. For consistency, outliers are assigned with
        larger anomaly scores.

        Parameters
        ----------
        G : PyTorch Geometric Data instance (torch_geometric.data.Data)
            The input data.

        Returns
        -------
        anomaly_scores : numpy.ndarray
            The anomaly score of shape :math:`N`.
        """
        check_is_fitted(self, ['model'])
        G.node_idx = torch.arange(G.x.shape[0])
        G.s = to_dense_adj(G.edge_index)[0]

        loader = NeighborLoader(G,
                                [self.num_neigh],
                                batch_size=self.batch_size)

        self.model.eval()
        outlier_scores = np.zeros(G.x.shape[0])
        for sampled_data in loader:
            batch_size = sampled_data.batch_size
            node_idx = sampled_data.node_idx

            x, s, edge_index = self.process_graph(sampled_data)

            x_, s_ = self.model(x, edge_index, batch_size)
            score = self.loss_func(x[:batch_size],
                                   x_[:batch_size],
                                   s[:batch_size, node_idx],
                                   s_[:batch_size])

            outlier_scores[node_idx[:batch_size]] = score.detach() \
                .cpu().numpy()
        return outlier_scores

    def process_graph(self, G):
        """
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
        s = G.s.to(self.device)
        edge_index = G.edge_index.to(self.device)
        x = G.x.to(self.device)

        return x, s, edge_index

    def loss_func(self, x, x_, s, s_):
        # generate hyperparameter - structure penalty
        reversed_adj = 1 - s
        thetas = torch.where(reversed_adj > 0, reversed_adj,
                             torch.full(s.shape, self.theta).to(self.device))

        # generate hyperparameter - node penalty
        reversed_attr = 1 - x
        etas = torch.where(reversed_attr == 1, reversed_attr,
                           torch.full(x.shape, self.eta).to(self.device))

        # attribute reconstruction loss
        diff_attribute = torch.pow(x_ - x, 2) * etas
        attribute_errors = torch.sqrt(torch.sum(diff_attribute, 1))

        # structure reconstruction loss
        diff_structure = torch.pow(s_ - s, 2) * thetas
        structure_errors = torch.sqrt(torch.sum(diff_structure, 1))

        score = self.alpha * attribute_errors \
                + (1 - self.alpha) * structure_errors

        return score


class AnomalyDAE_Base(nn.Module):
    """
    AnomalyDAE_Base is an anomaly detector consisting of a structure
    autoencoder and an attribute reconstruction autoencoder.

    Parameters
    ----------
    in_node_dim : int
         Dimension of input feature
    in_num_dim: int
         Dimension of the input number of nodes
    embed_dim:: int
         Dimension of the embedding after the first reduced linear
         layer (D1)
    out_dim : int
         Dimension of final representation
    dropout : float, optional
        Dropout rate of the model
        Default: 0
    act: F, optional
         Choice of activation function
    """

    def __init__(self,
                 in_node_dim,
                 in_num_dim,
                 embed_dim,
                 out_dim,
                 dropout,
                 act):
        super(AnomalyDAE_Base, self).__init__()

        self.num_center_nodes = in_num_dim
        self.structure_ae = StructureAE(in_node_dim,
                                        embed_dim,
                                        out_dim,
                                        dropout,
                                        act)
        self.attribute_ae = AttributeAE(self.num_center_nodes,
                                        embed_dim,
                                        out_dim,
                                        dropout,
                                        act)

    def forward(self, x, edge_index, batch_size):
        s_, h = self.structure_ae(x, edge_index)
        if batch_size < self.num_center_nodes:
            x = F.pad(x, (0, 0, 0, self.num_center_nodes - batch_size))
        x_ = self.attribute_ae(x[:self.num_center_nodes], h)
        return x_, s_


class StructureAE(nn.Module):
    """
    Structure Autoencoder in AnomalyDAE model: the encoder
    transforms the node attribute X into the latent
    representation with the linear layer, and a graph attention
    layer produces an embedding with weight importance of node
    neighbors. Finally, the decoder reconstructs the final embedding
    to the original.

    Parameters
    ----------
    in_dim: int
        input dimension of node data
    embed_dim: int
        the latent representation dimension of node
       (after the first linear layer)
    out_dim: int
        the output dim after the graph attention layer
    dropout: float
        dropout probability for the linear layer
    act: F, optional
         Choice of activation function

    Returns
    -------
    x : torch.Tensor
        Reconstructed attribute (feature) of nodes.
    embed_x : torch.Tensor
        Embed nodes after the attention layer
    """

    def __init__(self,
                 in_dim,
                 embed_dim,
                 out_dim,
                 dropout,
                 act):
        super(StructureAE, self).__init__()
        self.dense = nn.Linear(in_dim, embed_dim)
        self.attention_layer = GATConv(embed_dim, out_dim)
        self.dropout = dropout
        self.act = act

    def forward(self, x, edge_index):
        # encoder
        x = self.act(self.dense(x))
        x = F.dropout(x, self.dropout)
        h = self.attention_layer(x, edge_index)
        # decoder
        s_ = torch.sigmoid(h @ h.T)
        return s_, h


class AttributeAE(nn.Module):
    """
    Attribute Autoencoder in AnomalyDAE model: the encoder
    employs two non-linear feature transform to the node attribute
    x. The decoder takes both the node embeddings from the structure
    autoencoder and the reduced attribute representation to
    reconstruct the original node attribute.

    Parameters
    ----------
    in_dim:  int
        dimension of the input number of nodes
    embed_dim: int
        the latent representation dimension of node
        (after the first linear layer)
    out_dim:  int
        the output dim after two linear layers
    dropout: float
        dropout probability for the linear layer
    act: F, optional
         Choice of activation function

    Returns
    -------
    x : torch.Tensor
        Reconstructed attribute (feature) of nodes.
    """

    def __init__(self,
                 in_dim,
                 embed_dim,
                 out_dim,
                 dropout,
                 act):
        super(AttributeAE, self).__init__()
        self.dense1 = nn.Linear(in_dim, embed_dim)
        self.dense2 = nn.Linear(embed_dim, out_dim)
        self.dropout = dropout
        self.act = act

    def forward(self, x, h):
        # encoder
        x = self.act(self.dense1(x.T))
        x = F.dropout(x, self.dropout)
        x = self.dense2(x)
        x = F.dropout(x, self.dropout)
        # decoder
        x = h @ x.T
        return x
