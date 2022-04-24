import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from sklearn.utils.validation import check_is_fitted
from copy import deepcopy
import numpy as np

from . import BaseDetector
from .basic_nn import GCN
from ..utils.metric import eval_roc_auc


class CONAD(BaseDetector):
    """
    CONAD (Contrastive Attributed Network Anomaly Detection)
    CONAD is an anomaly detector consisting of a shared graph
    convolutional encoder, a structure reconstruction decoder, and an
    attribute reconstruction decoder. The model is trained with both
    contrastive loss and structure/attribute reconstruction loss.
    The reconstruction mean square error of the decoders are defined
    as structure anomaly score and attribute anomaly score, respectively.

    See :cite:`xu2022contrastive` for details.

    Parameters
    ----------
    hid_dim :  int, optional
        Hidden dimension of model. Default: ``0``.
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
    alpha : float, optional
        Loss balance weight for attribute and structure.
        Default: ``0.5``.
    eta : float, optional
        Loss balance weight for contrastive and reconstruction.
        Default: ``0.5``.
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
    >>> from pygod.models import DOMINANT
    >>> model = DOMINANT()
    >>> model.fit(data) # PyG graph data object
    >>> prediction = model.predict(data)
    """
    def __init__(self,
                 hid_dim=64,
                 num_layers=4,
                 dropout=0.3,
                 weight_decay=0.,
                 act=F.relu,
                 alpha=0.8,
                 eta=.5,
                 contamination=0.1,
                 lr=5e-3,
                 epoch=5,
                 gpu=0,
                 margin=.5,
                 verbose=False):
        super(CONAD, self).__init__(contamination=contamination)

        # model param
        self.hid_dim = hid_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.act = act
        self.alpha = alpha
        self.eta = eta

        # training param
        self.lr = lr
        self.epoch = epoch
        if gpu >= 0 and torch.cuda.is_available():
            self.device = 'cuda:{}'.format(gpu)
        else:
            self.device = 'cpu'
        self.margin_loss_func = torch.nn.MarginRankingLoss(margin=margin)
        # other param
        self.verbose = verbose

    def fit(self, G, y_true=None, **kwargs):
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
        decoder_layers = int(self.num_layers / 2)
        encoder_layers = self.num_layers - decoder_layers

        x, adj, edge_index = self.process_graph(G)

        self.encoder = conad_encoder(in_dim=x.shape[1],
                                     hid_dim=self.hid_dim,
                                     num_layers=encoder_layers,
                                     dropout=self.dropout,
                                     act=self.act).to(self.device)
        
        self.decoder = conad_decoder(in_dim=x.shape[1],
                                     hid_dim=self.hid_dim,
                                     num_layers=decoder_layers,
                                     dropout=self.dropout,
                                     act=self.act).to(self.device)
        
        optimizer = torch.optim.Adam(list(self.encoder.parameters())+list(self.decoder.parameters()),
                                     lr=self.lr,
                                     weight_decay=self.weight_decay)
        score = None
        for epoch in range(self.epoch):
            self.encoder.train()
            self.decoder.train()
            x_aug, edge_index_aug, label_aug = self._make_pseudo_anomalies(x, adj)
            h_aug = self.encoder(x_aug, edge_index_aug)
            h = self.encoder(x, edge_index)
            
            margin_loss = self.margin_loss_func(h, h, h_aug) * label_aug
            margin_loss = torch.mean(margin_loss)

            x_, adj_ = self.decoder(h, edge_index)
            score = self.loss_func(x, x_, adj, adj_)
            loss = self.eta * torch.mean(score) + (1 - self.eta) * margin_loss

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
        check_is_fitted(self, ['encoder', 'decoder'])

        # get needed data object from the input data
        x, adj, edge_index = self.process_graph(G)

        # enable the evaluation mode
        self.encoder.eval()
        self.decoder.eval()

        # construct the vector for holding the reconstruction error
        x_, adj_ = self.decoder(self.encoder(x, edge_index), edge_index)
        outlier_scores = self.loss_func(x, x_, adj, adj_)
        return outlier_scores.detach().cpu().numpy()

    def _make_pseudo_anomalies(self, x, adj, rate=.1, num_added_edge=30, surround=50, scale_factor=10):
        """
        Description
        -----------
        Data augmentation on the input graph. Four types of
        pseudo anomalies will be injected:
            Attribute, deviated
            Attribute, disproportionate
            Structure, high-degree
            Structure, outlying
        
        Parameters
        -----------
        x : note attribute matrix
        adj : dense adjacency matrix
        rate : pseudo anomaly rate
        num_added_edge : additional edges to add for 
            "high-degree" pseudo anomalies
        surround : num of candidate node attributes for
            "deviated" pseudo anomalies
        scale_factor : scale ratio for
            "disproportionate" pseudo anomalies
        """
        adj_aug, feat_aug  = deepcopy(adj), deepcopy(x)
        label_aug = np.zeros(adj_aug.shape[0])
        assert(adj_aug.shape[0]==feat_aug.shape[0])
        num_nodes = adj_aug.shape[0]
        for i in range(num_nodes):
            prob = np.random.uniform()
            if prob > rate: continue
            label_aug[i] = 1
            one_fourth = np.random.randint(0, 4)
            if one_fourth == 0:
                # add clique
                new_neighbors = np.random.choice(np.arange(num_nodes), num_added_edge, replace=False)
                for n in new_neighbors:
                    adj_aug[n][i] = 1
                    adj_aug[i][n] = 1
            elif one_fourth == 1:
                # drop all connection
                neighbors = np.nonzero(adj[i])
                if not neighbors.any():
                    continue
                else: 
                    neighbors = neighbors[0]
                for n in neighbors:
                    adj_aug[i][n] = 0
                    adj_aug[n][i] = 0
            elif one_fourth == 2:
                # attrs
                candidates = np.random.choice(np.arange(num_nodes), surround, replace=False)
                max_dev, max_idx = 0, i
                for c in candidates:
                    dev = torch.square(feat_aug[i]-feat_aug[c]).sum()
                    if dev > max_dev:
                        max_dev = dev
                        max_idx = c
                feat_aug[i] = feat_aug[max_idx]
            else:
                # scale attr
                prob = np.random.uniform(0, 1)
                if prob > 0.5:
                    feat_aug[i] *= scale_factor
                else:
                    feat_aug[i] /= scale_factor
        edge_index_aug = dense_to_sparse(adj_aug)[0].to(self.device)
        feat_aug = feat_aug.to(self.device)
        label_aug = torch.LongTensor(label_aug).to(device=self.device)
        return feat_aug, edge_index_aug, label_aug

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

        adj = to_dense_adj(edge_index)[0].to(self.device)

        edge_index = edge_index.to(self.device)
        adj = adj.to(self.device)
        x = G.x.to(self.device)

        # return data objects needed for the network
        return x, adj, edge_index

    def loss_func(self, x, x_, adj, adj_):
        # attribute reconstruction loss
        diff_attribute = torch.pow(x - x_, 2)
        attribute_errors = torch.sqrt(torch.sum(diff_attribute, 1))

        # structure reconstruction loss
        diff_structure = torch.pow(adj - adj_, 2)
        structure_errors = torch.sqrt(torch.sum(diff_structure, 1))

        score = self.alpha * attribute_errors \
                + (1 - self.alpha) * structure_errors
        return score


class conad_encoder(nn.Module):
    def __init__(self,
                 in_dim,
                 hid_dim,
                 num_layers,
                 dropout,
                 act):

        super(conad_encoder, self).__init__()


        self.shared_encoder = GCN(in_channels=in_dim,
                                  hidden_channels=hid_dim,
                                  num_layers=num_layers,
                                  out_channels=hid_dim,
                                  dropout=dropout,
                                  act=act)
    
    def forward(self, x, edge_index):
        # encode
        h = self.shared_encoder(x, edge_index)
        # return reconstructed matrices
        return h


class conad_decoder(nn.Module):
    def __init__(self,
                 in_dim,
                 hid_dim,
                 num_layers,
                 dropout,
                 act):
        
        super(conad_decoder, self).__init__()

        self.attr_decoder = GCN(in_channels=hid_dim,
                                hidden_channels=hid_dim,
                                num_layers=num_layers,
                                out_channels=in_dim,
                                dropout=dropout,
                                act=act)

        self.struct_decoder = GCN(in_channels=hid_dim,
                                  hidden_channels=hid_dim,
                                  num_layers=num_layers-1,
                                  out_channels=in_dim,
                                  dropout=dropout,
                                  act=act)

    def forward(self, h, edge_index):
        # decode feature matrix
        x_ = self.attr_decoder(h, edge_index)
        # decode adjacency matrix
        h_ = self.struct_decoder(h, edge_index)
        adj_ = h_ @ h_.T

        # return reconstructed matrices
        return x_, adj_