# -*- coding: utf-8 -*-
""" Multilayer Perceptron Autoencoder
"""
# Author: Kay Liu <zliu234@uic.edu>
# License: BSD 2 clause

import torch
import torch.nn.functional as F

from . import DeepDetector
from torch_geometric.nn import MLP


class MLPAE(DeepDetector):
    """
    Vanila Multilayer Perceptron Autoencoder.

    See :cite:`yuan2021higher` for details.

    Parameters
    ----------
    hid_dim :  int, optional
        Hidden dimension of model. Default: ``0``.
    num_layers : int, optional
        Total number of layers in autoencoders. Default: ``4``.
    dropout : float, optional
        Dropout rate. Default: ``0.``.
    weight_decay : float, optional
        Weight decay (L2 penalty). Default: ``0.``.
    act : callable activation function or None, optional
        Activation function if not None.
        Default: ``torch.nn.functional.relu``.
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
    batch_size : int, optional
        Minibatch size, 0 for full batch training. Default: ``0``.
    verbose : bool
        Verbosity mode. Turn on to print out log information.
        Default: ``False``.

    Examples
    --------
    >>> from pygod.models import MLPAE
    >>> model = MLPAE()
    >>> model.fit(data) # PyG graph data object
    >>> prediction = model.predict(data)
    """

    def __init__(self,
                 in_dim=None,
                 hid_dim=64,
                 num_layers=4,
                 dropout=0.3,
                 weight_decay=0.,
                 act=F.relu,
                 contamination=0.1,
                 lr=5e-3,
                 epoch=5,
                 gpu=0,
                 batch_size=0,
                 num_neigh=0,
                 verbose=False,
                 **kwargs):

        super(MLPAE, self).__init__(in_dim=in_dim,
                                    hid_dim=hid_dim,
                                    num_layers=num_layers,
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

    # def fit(self, G, y_true=None):
    #     """
    #     Fit detector with input data.
    #
    #     Parameters
    #     ----------
    #     G : torch_geometric.data.Data
    #         The input data.
    #     y_true : numpy.ndarray, optional
    #         The optional outlier ground truth labels used to monitor
    #         the training progress. They are not used to optimize the
    #         unsupervised model. Default: ``None``.
    #
    #     Returns
    #     -------
    #     self : object
    #         Fitted estimator.
    #     """
    #     full_x = self.process_graph(G)
    #     dataset = PlainDataset(full_x)
    #     if self.batch_size == 0:
    #         self.batch_size = G.x.shape[0]
    #     loader = DataLoader(dataset, batch_size=self.batch_size)
    #
    #     self.model = MLP(in_channels=G.x.shape[1],
    #                      hidden_channels=self.hid_dim,
    #                      out_channels=G.x.shape[1],
    #                      num_layers=self.num_layers,
    #                      dropout=self.dropout,
    #                      act=self.act).to(self.device)
    #
    #     optimizer = torch.optim.Adam(self.model.parameters(),
    #                                  lr=self.lr,
    #                                  weight_decay=self.weight_decay)
    #
    #     self.model.train()
    #     decision_scores = np.zeros(full_x.shape[0])
    #     for epoch in range(self.epoch):
    #         epoch_loss = 0
    #         for x, node_idx in loader:
    #             x_ = self.model(x)
    #             score = torch.mean(F.mse_loss(x_, x, reduction='none'), dim=1)
    #             decision_scores[node_idx] = score.detach().cpu().numpy()
    #             loss = torch.mean(score)
    #             epoch_loss += loss.item() * x.shape[0]
    #
    #             optimizer.zero_grad()
    #             loss.backward()
    #             optimizer.step()
    #
    #         if self.verbose:
    #             print("Epoch {:04d}: Loss {:.4f}"
    #                   .format(epoch, epoch_loss / G.x.shape[0]), end='')
    #             if y_true is not None:
    #                 auc = eval_roc_auc(y_true, decision_scores)
    #                 print(" | AUC {:.4f}".format(auc), end='')
    #             print()
    #
    #     self.decision_scores_ = decision_scores
    #     self._process_decision_scores()
    #     return self
    #
    # def decision_function(self, G):
    #     """
    #     Predict raw anomaly score using the fitted detector. Outliers
    #     are assigned with larger anomaly scores.
    #
    #     Parameters
    #     ----------
    #     G : PyTorch Geometric Data instance (torch_geometric.data.Data)
    #         The input data.
    #
    #     Returns
    #     -------
    #     outlier_scores : numpy.ndarray
    #         The anomaly score of shape :math:`N`.
    #     """
    #     check_is_fitted(self, ['model'])
    #     full_x = self.process_graph(G)
    #     dataset = PlainDataset(full_x)
    #     loader = DataLoader(dataset, batch_size=self.batch_size)
    #
    #     self.model.eval()
    #     outlier_scores = np.zeros(full_x.shape[0])
    #     for x, node_idx in loader:
    #         x_ = self.model(x)
    #         score = torch.mean(F.mse_loss(x_, x, reduction='none'), dim=1)
    #         outlier_scores[node_idx] = score.detach().cpu().numpy()
    #     return outlier_scores

    def _process_graph(self, G):
        """
        Process the raw PyG data object into a tuple of sub data
        objects needed for the model.

        Parameters
        ----------
        G : PyTorch Geometric Data instance (torch_geometric.data.Data)
            The input data.
        """
        pass

    def _init_nn(self, **kwargs):
        self.model = MLP(in_channels=self.in_dim,
                         hidden_channels=self.hid_dim,
                         out_channels=self.in_dim,
                         num_layers=self.num_layers,
                         dropout=self.dropout,
                         act=self.act,
                         **kwargs).to(self.device)

    def _forward_nn(self, data):
        batch_size = data.batch_size

        x = data.x.to(self.device)

        x_ = self.model(x)
        scores = torch.mean(F.mse_loss(x[:batch_size],
                                       x_[:batch_size],
                                       reduction='none'), dim=1)

        loss = torch.mean(scores)

        return loss, scores.detach().cpu().numpy()
