# -*- coding: utf-8 -*-
"""AnomalyDAE: Dual autoencoder for anomaly detection on attributed networks"""
# Author: Xueying Ding <xding2@andrew.cmu.edu>
# License: BSD 2 clause


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor
from torch_geometric.nn import GATConv
from sklearn.utils.validation import check_is_fitted

from . import BaseDetector
from ..utils.metric import eval_roc_auc


class StructureAE(nn.Module):
    """
    Description
    -----------
    Structure Autoencoder in AnomalyDAE model: the encoder
    transforms the node attribute X into the latent
    representation with the linear layer, and a graph attention
    layer produces an embedding with weight importance of node 
    neighbors. Finally, the decoder reconstructs the final embedding
    to the original.

    See :cite:`fan2020anomalydae` for details.

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
              Embedd nodes after the attention layer
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

    def forward(self,
                x,
                edge_index):
        # encoder
        x = self.act(self.dense(x))
        x = F.dropout(x, self.dropout)
        # print(x.shape, adj.shape)
        embed_x = self.attention_layer(x, edge_index)
        # decoder
        x = torch.sigmoid(embed_x @ embed_x.T)
        return x, embed_x


class AttributeAE(nn.Module):
    """
    Description
    -----------
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

    def forward(self,
                x,
                struct_embed):
        # encoder
        x = self.act(self.dense1(x.T))
        x = F.dropout(x, self.dropout)
        x = self.dense2(x)
        x = F.dropout(x, self.dropout)
        # decoder
        x = struct_embed @ x.T
        return x


class AnomalyDAE_Base(nn.Module):
    """
    Description
    -----------
    AdnomalyDAE_Base is an anomaly detector consisting of a structure autoencoder,
    and an attribute reconstruction autoencoder. 

    Parameters
    ----------
    in_node_dim : int
         Dimension of input feature
    in_num_dim: int
         Dimension of the input number of nodes
    embed_dim:: int
         Dimension of the embedding after the first reduced linear layer (D1)   
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
        self.structure_AE = StructureAE(in_node_dim, embed_dim,
                                        out_dim, dropout, act)
        self.attribute_AE = AttributeAE(in_num_dim, embed_dim,
                                        out_dim, dropout, act)

    def forward(self,
                x,
                edge_index):
        A_hat, embed_x = self.structure_AE(x, edge_index)
        X_hat = self.attribute_AE(x, embed_x)
        return A_hat, X_hat


class AnomalyDAE(BaseDetector):
    """
    AnomalyDAE (Dual autoencoder for anomaly detection on attributed networks):
    AnomalyDAE is an anomaly detector that. consists of a structure autoencoder 
    and an attribute autoencoder to learn both node embedding and attribute 
    embedding jointly in latent space. The structural autoencoer uses Graph Attention
    layers. The reconstruction mean square error of the decoders are defined 
    as structure anamoly score and attribute anomaly score, respectively, 
    with two additional penalties on the reconstructed adj matrix and 
    node attributes (force entries to be nonzero).

    See: cite 'fan2020anomalydae' for details.

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
        attrs, adj, edge_index = self.process_graph(G)
        self.model = AnomalyDAE_Base(in_node_dim=attrs.shape[1],
                                     in_num_dim=attrs.shape[0],
                                     embed_dim=self.embed_dim,
                                     out_dim=self.out_dim,
                                     dropout=self.dropout,
                                     act=self.act).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=self.lr,
                                     weight_decay=self.weight_decay)

        for epoch in range(self.epoch):
            self.model.train()
            optimizer.zero_grad()
            A_hat, X_hat = self.model(attrs, edge_index)
            loss, struct_loss, feat_loss = self.loss_func(adj, A_hat,
                                                          attrs, X_hat)
            loss_mean = torch.mean(loss)
            loss_mean.backward()
            optimizer.step()

            if self.verbose:
                print("Epoch:", '%04d' % epoch, "train_loss=",
                      "{:.5f}".format(loss_mean.item()), "train/struct_loss=",
                      "{:.5f}".format(struct_loss.item()), "train/feat_loss=",
                      "{:.5f}".format(feat_loss.item()))

            self.model.eval()
            A_hat, X_hat = self.model(attrs, edge_index)
            loss, struct_loss, feat_loss = self.loss_func(adj, A_hat,
                                                          attrs, X_hat)
            score = loss.detach().cpu().numpy()
            if self.verbose:
                print('AUC', eval_roc_auc(y_true, score))

        self.decision_scores_ = score
        self._process_decision_scores()
        return self

    def decision_function(self, G):
        """
        Description
        -----------
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

        # get needed data object from the input data
        attrs, adj, edge_index = self.process_graph(G)

        # enable the evaluation mode
        self.model.eval()

        # construct the vector for holding the reconstruction error
        A_hat, X_hat = self.model(attrs, edge_index)
        outlier_scores, _, _ = self.loss_func(adj, A_hat, attrs, X_hat)
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
        adj : torch.Tensor
            Adjacency matrix of the graph.
        edge_index : torch.Tensor
            Edge list of the graph.
        """
        edge_index = G.edge_index

        #  via sparse matrix operation
        dense_adj \
            = SparseTensor(row=edge_index[0], col=edge_index[1]).to_dense()

        # adjacency matrix normalization
        rowsum = dense_adj.sum(1)
        d_inv_sqrt = torch.pow(rowsum, -0.5).flatten()
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        adj = (dense_adj * d_mat_inv_sqrt).T * d_mat_inv_sqrt

        edge_index = edge_index.to(self.device)
        adj = adj.to(self.device)
        x = G.x.to(self.device)

        # return data objects needed for the network
        return x, adj, edge_index

    def loss_func(self,
                  adj,
                  A_hat,
                  attrs,
                  X_hat):
        # generate hyperparameter - structure penalty
        reversed_adj = torch.ones(adj.shape).to(self.device) - adj
        thetas = torch.where(reversed_adj > 0, reversed_adj,
                             torch.full(adj.shape, self.theta).to(self.device))

        # generate hyperparameter - node penalty
        reversed_attr = torch.ones(attrs.shape).to(self.device) - attrs
        etas = torch.where(reversed_attr == 1, reversed_attr,
                           torch.full(attrs.shape, self.eta).to(self.device))

        # Attribute reconstruction loss
        diff_attribute = torch.pow(X_hat -
                                   attrs, 2) * etas
        attribute_reconstruction_errors = \
            torch.sqrt(torch.sum(diff_attribute, 1))
        attribute_cost = torch.mean(attribute_reconstruction_errors)

        # structure reconstruction loss
        diff_structure = torch.pow(A_hat - adj, 2) * thetas
        structure_reconstruction_errors = \
            torch.sqrt(torch.sum(diff_structure, 1))
        structure_cost = torch.mean(structure_reconstruction_errors)

        cost = self.alpha * attribute_reconstruction_errors + (
                1 - self.alpha) * structure_reconstruction_errors

        return cost, structure_cost, attribute_cost
