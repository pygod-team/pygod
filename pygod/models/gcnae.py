# -*- coding: utf-8 -*-
""" Graph Convolutional Network Autoencoder
"""
# Author: Kay Liu <zliu234@uic.edu>
# License: BSD 2 clause

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCN
from sklearn.metrics import roc_auc_score
from sklearn.utils.validation import check_is_fitted

from . import BaseDetector
from ..utils import EarlyStopping


class GCNAE(BaseDetector):
    """Let us decide the documentation later

    """
    def __init__(self, contamination=0.1):
        super(GCNAE, self).__init__(contamination=contamination)

    def fit(self, G, args):

        # 1. first call the data process
        x, edge_index, labels = self.process_graph(G, args)

        # 2. set the parameters needed for the network from args.
        self.hidden_channels = args.hidden_size
        self.num_layers = args.num_layers
        self.dropout = args.dropout
        self.weight_decay = args.weight_decay

        # TODO: support other activation function
        if args.act:
            self.act = F.relu

        # 3. initialize the detection model
        self.model = GCN(in_channels=x.shape[1],
                         hidden_channels=self.hidden_channels,
                         num_layers=self.num_layers,
                         out_channels=x.shape[1],
                         dropout=self.dropout,
                         act=self.act)

        # 4. check cuda
        if args.gpu >= 0 and torch.cuda.is_available():
            self.device = 'cuda:{}'.format(args.gpu)
        else:
            self.device = 'cpu'

        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        self.model = self.model.to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=self.weight_decay)

        if args.patience > 0:
            es = EarlyStopping(args.patience, args.verbose)

        for epoch in range(args.epoch):
            self.model.train()
            x_ = self.model(x, edge_index)
            loss = F.mse_loss(x_, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # TODO: support more metrics
            score = torch.mean(F.mse_loss(x_, x, reduction='none'), dim=1).detach().cpu().numpy()
            auc = roc_auc_score(labels, score)
            if args.verbose:
                print("Epoch {:04d}: Loss {:.4f} | AUC {:.4f}".format(epoch, loss.item(), auc))

            if args.patience > 0 and es.step(auc, self.model):
                break

        if args.patience > 0:
            self.model.load_state_dict(torch.load('es_checkpoint.pt'))

        self.decision_scores_ = score
        self._process_decision_scores()
        return self

    def decision_function(self, G, args):
        check_is_fitted(self, ['model'])
        self.model.eval()

        x, edge_index, _ = self.process_graph(G, args)

        x = x.to(self.device)
        edge_index = edge_index.to(self.device)

        x_ = self.model(x, edge_index)
        outlier_scores = torch.mean(F.mse_loss(x_, x, reduction='none'), dim=1).detach().cpu().numpy()
        return outlier_scores

    def process_graph(self, G, args):
        # return feature only
        return G.x, G.edge_index, G.y
