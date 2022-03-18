# -*- coding: utf-8 -*-
"""Fraud Detection Dual-Resistant toGraph Inconsistency and Imbalance (FRAUDRE)
"""
# Author: Xiyang Hu <xiyanghu@cmu.edu>
# License: BSD 2 clause

import time
import random
import argparse
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable


from base import BaseDetector

import numpy as np
import scipy.sparse as sp
from sklearn.metrics import f1_score, accuracy_score, recall_score, roc_auc_score, average_precision_score, \
    precision_score
from collections import defaultdict
import math

from torch_geometric.datasets import FakeHeteroDataset
from torch_geometric.utils import to_scipy_sparse_matrix
from scipy.sparse import csc_matrix


class Fraudre(BaseDetector):
    """FRAUDRE: Fraud Detection Dual-Resistant toGraph Inconsistency and Imbalance

    See :cite:`zhang2021fraudre` for details.
    """

    def __init__(self, contamination=0.1):
        super(Fraudre, self).__init__(contamination=contamination)

    def fit(self, G, args):
        """Fit detector. y is ignored in unsupervised methods.
        Parameters
        ----------
        G : PyTorch Geometric Data instance (torch_geometric.data.Data)
            The input graph.
        args : argparse object.
            Corresponding hyperparameters
        Returns
        -------
        self : object
            Fitted estimator.
        """

        idx_train = args.idx_train
        idx_test = args.idx_test
        y_test = args.y_test
        prior = args.prior

        relation1, relation2, relation3, feat_data, labels = self.process_graph(G, args)

        # initialize model input
        features = nn.Embedding(feat_data.shape[0], feat_data.shape[1])
        feat_data = normalize(feat_data)
        features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
        if args.cuda:
            features.cuda()

        # set input graph topology
        adj_lists = [relation1, relation2, relation3]

        # build model

        # the first neural network layer (ego-feature embedding module)
        mlp = MLP_(features, feat_data.shape[1], args.embed_dim, cuda=args.cuda)

        # first convolution layer
        intra1_1 = IntraAgg(cuda=args.cuda)
        intra1_2 = IntraAgg(cuda=args.cuda)
        intra1_3 = IntraAgg(cuda=args.cuda)
        agg1 = InterAgg(lambda nodes: mlp(nodes), args.embed_dim, adj_lists, [intra1_1, intra1_2, intra1_3],
                        cuda=args.cuda)

        # second convolution layer
        intra2_1 = IntraAgg(cuda=args.cuda)
        intra2_2 = IntraAgg(cuda=args.cuda)
        intra2_3 = IntraAgg(cuda=args.cuda)

        # def __init__(self, features, embed_dim, adj_lists, intraggs, cuda = False):
        agg2 = InterAgg(lambda nodes: agg1(nodes), args.embed_dim * 2, adj_lists, [intra2_1, intra2_2, intra2_3],
                        cuda=args.cuda)
        gnn_model = Fraudre_Base(2, 2, args.embed_dim, agg2, args.lambda_1, prior, cuda=args.cuda)

        # gnn_model in one convolution layer
        # gnn_model = MODEL(1, 2, args.embed_dim, agg1, args.lambda_1, prior, cuda = args.cuda)

        if args.cuda:
            gnn_model.to('cuda:0')

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, gnn_model.parameters()), lr=args.lr,
                                     weight_decay=args.lambda_2)
        performance_log = []

        # train the model

        overall_time = 0
        for epoch in range(args.num_epochs):

            # gnn_model.train()
            # shuffle
            random.shuffle(idx_train)
            num_batches = int(len(idx_train) / args.batch_size) + 1

            loss = 0.0
            epoch_time = 0

            # mini-batch training
            for batch in range(num_batches):

                print(f'Epoch: {epoch}, batch: {batch}')

                i_start = batch * args.batch_size
                i_end = min((batch + 1) * args.batch_size, len(idx_train))

                batch_nodes = idx_train[i_start:i_end]

                batch_label = labels[np.array(batch_nodes)]

                optimizer.zero_grad()

                start_time = time.time()

                if args.cuda:
                    loss = gnn_model.loss(batch_nodes, Variable(torch.cuda.LongTensor(batch_label)))
                else:
                    loss = gnn_model.loss(batch_nodes, Variable(torch.LongTensor(batch_label)))

                end_time = time.time()

                epoch_time += end_time - start_time

                loss.backward()
                optimizer.step()
                loss += loss.item()

            print(f'Epoch: {epoch}, loss: {loss.item() / num_batches}, time: {epoch_time}s')
            overall_time += epoch_time

            # testing the model for every $test_epoch$ epoch
            if epoch % args.test_epochs == 0:
                # gnn_model.eval()
                auc, precision, a_p, recall, f1 = test_model(idx_test, y_test, gnn_model)
                performance_log.append([auc, precision, a_p, recall, f1])

        self.model = gnn_model
        print("The training time per epoch")
        print(overall_time / args.num_epochs)

        self.decision_scores_ = self.model.to_prob(args.idx_train, train_flag=False)[:,1].detach().cpu().numpy()
        self._process_decision_scores()

    def decision_function(self, G, args):
        gnn_prob = self.model.to_prob(args.idx_test, train_flag=False)[:,1].detach().cpu().numpy()
        return gnn_prob

    def process_graph(self, G, args):
        # load topology, feature, and label
        relation1 = sparse_to_adjlist(csc_matrix(to_scipy_sparse_matrix(G[0][('v0', 'e0', 'v0')]['edge_index'])))
        relation2 = sparse_to_adjlist(csc_matrix(to_scipy_sparse_matrix(G[0][('v0', 'e1', 'v0')]['edge_index'])))
        relation3 = sparse_to_adjlist(csc_matrix(to_scipy_sparse_matrix(G[0][('v0', 'e2', 'v0')]['edge_index'])))
        feat_data = G[0]['v0']['x'].numpy()
        labels = G[0]['v0']['y'].numpy()
        return relation1, relation2, relation3, feat_data, labels


def sparse_to_adjlist(sp_matrix):
    """Transfer sparse matrix to adjacency list"""

    # add self loop
    homo_adj = sp_matrix + sp.eye(sp_matrix.shape[0])
    # creat adj_list
    adj_lists = defaultdict(set)
    edges = homo_adj.nonzero()

    for index, node in enumerate(edges[0]):
        adj_lists[node].add(edges[1][index])
        adj_lists[edges[1][index]].add(node)
    adj_lists = {keya: random.sample(adj_lists[keya], 10) if len(adj_lists[keya]) >= 10 else adj_lists[keya] for i, keya
                 in enumerate(adj_lists)}

    return adj_lists


def normalize(mx):
    """Row-normalize sparse matrix"""

    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


# !
def test_model(test_cases, labels, model):
    """
    test the performance of model
    :param test_cases: a list of testing node
    :param labels: a list of testing node labels
    :param model: the GNN model
    """
    gnn_prob = model.to_prob(test_cases, train_flag=False)

    auc_gnn = roc_auc_score(labels, gnn_prob.data.cpu().numpy()[:, 1].tolist())
    precision_gnn = precision_score(labels, gnn_prob.data.cpu().numpy().argmax(axis=1), average="macro")
    a_p = average_precision_score(labels, gnn_prob.data.cpu().numpy()[:, 1].tolist())
    recall_gnn = recall_score(labels, gnn_prob.data.cpu().numpy().argmax(axis=1), average="macro")
    f1 = f1_score(labels, gnn_prob.data.cpu().numpy().argmax(axis=1), average="macro")

    # print(gnn_prob.data.cpu().numpy().argmax(axis=1))

    print(f"GNN auc: {auc_gnn:.4f}")
    print(f"GNN precision: {precision_gnn:.4f}")
    print(f"GNN a_precision: {a_p:.4f}")
    print(f"GNN Recall: {recall_gnn:.4f}")
    # print(f"GNN f1: {f1:.4f}")

    return auc_gnn, precision_gnn, a_p, recall_gnn, f1

def weight_inter_agg(num_relations, neigh_feats, embed_dim, alpha, n, cuda):
    """
    Weight inter-relation aggregator
    :param num_relations: number of relations in the graph
    :param neigh_feats: intra_relation aggregated neighbor embeddings for each aggregation
    :param embed_dim: the dimension of output embedding
    :param alpha: weight paramter for each relation
    :param n: number of nodes in a batch
    :param cuda: whether use GPU
    """

    neigh_h = neigh_feats.t()

    w = F.softmax(alpha, dim=1)

    if cuda:
        aggregated = torch.zeros(size=(embed_dim, n)).cuda()  #
    else:
        aggregated = torch.zeros(size=(embed_dim, n))

    for r in range(num_relations):
        aggregated += torch.mul(w[:, r].unsqueeze(1).repeat(1, n), neigh_h[:, r * n:(r + 1) * n])

    return aggregated.t()


class MLP_(nn.Module):
    """
    the ego-feature embedding module
    """

    def __init__(self, features, input_dim, output_dim, cuda=False):

        super(MLP_, self).__init__()

        self.features = features
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.cuda = cuda
        self.mlp_layer = nn.Linear(self.input_dim, self.output_dim)

    def forward(self, nodes):

        if self.cuda:
            batch_features = self.features(torch.cuda.LongTensor(nodes))
        else:
            batch_features = self.features(torch.LongTensor(nodes))

        if self.cuda:
            self.mlp_layer.cuda()

        result = self.mlp_layer(batch_features)

        result = F.relu(result)

        return result


class InterAgg(nn.Module):
    """
    the fraud-aware convolution module
    Inter aggregation layer
    """

    def __init__(self, features, embed_dim, adj_lists, intraggs, cuda=False):

        """
        Initialize the inter-relation aggregator
        :param features: the input embeddings for all nodes
        :param embed_dim: the dimension need to be aggregated
        :param adj_lists: a list of adjacency lists for each single-relation graph
        :param intraggs: the intra-relation aggregatore used by each single-relation graph
        :param cuda: whether to use GPU
        """

        super(InterAgg, self).__init__()

        self.features = features
        self.dropout = 0.6
        self.adj_lists = adj_lists
        self.intra_agg1 = intraggs[0]
        self.intra_agg2 = intraggs[1]
        self.intra_agg3 = intraggs[2]
        self.embed_dim = embed_dim
        self.cuda = cuda
        self.intra_agg1.cuda = cuda
        self.intra_agg2.cuda = cuda
        self.intra_agg3.cuda = cuda

        if self.cuda:
            self.alpha = nn.Parameter(torch.FloatTensor(self.embed_dim * 2, 3)).cuda()

        else:
            self.alpha = nn.Parameter(torch.FloatTensor(self.embed_dim * 2, 3))

        init.xavier_uniform_(self.alpha)

    def forward(self, nodes, train_flag=True):

        """
        nodes: a list of batch node ids
        """

        if (isinstance(nodes, list) == False):
            nodes = nodes.cpu().numpy().tolist()

        to_neighs = []

        # adj_lists = [relation1, relation2, relation3]

        for adj_list in self.adj_lists:
            to_neighs.append([set(adj_list[int(node)]) for node in nodes])

        # to_neighs: [[set, set, set], [set, set, set], [set, set, set]]

        # find unique nodes and their neighbors used in current batch   #set(nodes)
        unique_nodes = set.union(set.union(*to_neighs[0]), set.union(*to_neighs[1]),
                                 set.union(*to_neighs[2], set(nodes)))

        # id mapping
        unique_nodes_new_index = {n: i for i, n in enumerate(list(unique_nodes))}

        if self.cuda:
            batch_features = self.features(torch.cuda.LongTensor(list(unique_nodes)))
        else:
            batch_features = self.features(torch.LongTensor(list(unique_nodes)))

        # get neighbor node id list for each batch node and relation
        r1_list = [set(to_neigh) for to_neigh in to_neighs[0]]  # [[set],[set],[ser]]  //   [[list],[list],[list]]
        r2_list = [set(to_neigh) for to_neigh in to_neighs[1]]
        r3_list = [set(to_neigh) for to_neigh in to_neighs[2]]

        center_nodes_new_index = [unique_nodes_new_index[int(n)] for n in nodes]  ################
        '''
        if self.cuda and isinstance(nodes, list):
            self_feats = self.features(torch.cuda.LongTensor(nodes))
        else:
            self_feats = self.features(index)
        '''

        # center_feats = self_feats[:, -self.embed_dim:]

        self_feats = batch_features[center_nodes_new_index]

        r1_feats = self.intra_agg1.forward(batch_features[:, -self.embed_dim:], nodes, r1_list, unique_nodes_new_index,
                                           self_feats[:, -self.embed_dim:])
        r2_feats = self.intra_agg2.forward(batch_features[:, -self.embed_dim:], nodes, r2_list, unique_nodes_new_index,
                                           self_feats[:, -self.embed_dim:])
        r3_feats = self.intra_agg3.forward(batch_features[:, -self.embed_dim:], nodes, r3_list, unique_nodes_new_index,
                                           self_feats[:, -self.embed_dim:])

        neigh_feats = torch.cat((r1_feats, r2_feats, r3_feats), dim=0)

        n = len(nodes)

        attention_layer_outputs = weight_inter_agg(len(self.adj_lists), neigh_feats, self.embed_dim * 2, self.alpha, n,
                                                   self.cuda)

        result = torch.cat((self_feats, attention_layer_outputs), dim=1)

        return result


class IntraAgg(nn.Module):
    """
    the fraud-aware convolution module
    Intra Aggregation Layer
    """

    def __init__(self, cuda=False):
        super(IntraAgg, self).__init__()

        self.cuda = cuda

    def forward(self, embedding, nodes, neighbor_lists, unique_nodes_new_index, self_feats):
        """
        Code partially from https://github.com/williamleif/graphsage-simple/
        :param nodes: list of nodes in a batch
        :param embedding: embedding of all nodes in a batch
        :param neighbor_lists: neighbor node id list for each batch node in one relation # [[list],[list],[list]]
        :param unique_nodes_new_index
        """

        # find unique nodes
        unique_nodes_list = list(set.union(*neighbor_lists))

        # id mapping
        unique_nodes = {n: i for i, n in enumerate(unique_nodes_list)}

        mask = Variable(torch.zeros(len(neighbor_lists), len(unique_nodes)))

        column_indices = [unique_nodes[n] for neighbor_list in neighbor_lists for n in neighbor_list]
        row_indices = [i for i in range(len(neighbor_lists)) for _ in range(len(neighbor_lists[i]))]

        mask[row_indices, column_indices] = 1

        num_neigh = mask.sum(1, keepdim=True)
        mask = torch.true_divide(mask, num_neigh)

        neighbors_new_index = [unique_nodes_new_index[n] for n in unique_nodes_list]

        embed_matrix = embedding[neighbors_new_index]

        embed_matrix = embed_matrix.cpu()

        _feats_1 = mask.mm(embed_matrix)
        if self.cuda:
            _feats_1 = _feats_1.cuda()

        # difference
        _feats_2 = self_feats - _feats_1
        return torch.cat((_feats_1, _feats_2), dim=1)


class Fraudre_Base(nn.Module):
    """FRAUDRE: Fraud Detection Dual-Resistant toGraph Inconsistency and Imbalance
    """

    def __init__(self, K, num_classes, embed_dim, agg, lambda_1, prior, cuda):
        super(Fraudre_Base, self).__init__()

        """
        Initialize the model
        :param K: the number of CONVOLUTION layers of the model
        :param num_classes: number of classes (2 in our paper)
        :param embed_dim: the output dimension of MLP layer
        :agg: the inter-relation aggregator that output the final embedding
        :lambad 1: the weight of MLP layer (ignore it)
        :prior:prior
        """

        self.agg = agg
        self.cuda = cuda
        self.lambda_1 = lambda_1
        self.K = K  # how many layers
        self.prior = prior
        self.xent = nn.CrossEntropyLoss()
        self.embed_dim = embed_dim
        self.fun = nn.LeakyReLU(0.3)

        self.weight_mlp = nn.Parameter(torch.FloatTensor(self.embed_dim, num_classes))  # Default requires_grad = True
        self.weight_model = nn.Parameter(torch.FloatTensor((int(math.pow(2, K + 1) - 1) * self.embed_dim), 64))

        self.weight_model2 = nn.Parameter(torch.FloatTensor(64, num_classes))

        init.xavier_uniform_(self.weight_mlp)
        init.xavier_uniform_(self.weight_model)
        init.xavier_uniform_(self.weight_model2)

    def forward(self, nodes, train_flag=True):

        embedding = self.agg(nodes, train_flag)

        scores_model = embedding.mm(self.weight_model)
        scores_model = self.fun(scores_model)
        scores_model = scores_model.mm(self.weight_model2)
        # scores_model = self.fun(scores_model)

        scores_mlp = embedding[:, 0: self.embed_dim].mm(self.weight_mlp)
        scores_mlp = self.fun(scores_mlp)

        return scores_model, scores_mlp

    # dimension, the number of center nodes * 2

    def to_prob(self, nodes, train_flag=False):

        scores_model, scores_mlp = self.forward(nodes, train_flag)
        scores_model = torch.sigmoid(scores_model)
        return scores_model

    def loss(self, nodes, labels, train_flag=True):

        # the classification module

        if self.cuda:
            logits = (torch.from_numpy(self.prior + 1e-8)).cuda()
        else:
            logits = (torch.from_numpy(self.prior + 1e-8))

        scores_model, scores_mlp = self.forward(nodes, train_flag)

        scores_model = scores_model + torch.log(logits)
        scores_mlp = scores_mlp + torch.log(logits)

        loss_model = self.xent(scores_model, labels.squeeze())
        loss_mlp = self.xent(scores_mlp, labels.squeeze())
        final_loss = loss_model + self.lambda_1 * loss_mlp
        return final_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # dataset and model dependent args
    parser.add_argument('--data', type=str, default='Amazon_demo',
                        help='The dataset name. [Amazon_demo, Yelp_demo, amazon,yelp]')
    parser.add_argument('--batch-size', type=int, default=100, help='Batch size 1024 for yelp, 256 for amazon.')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='Initial learning rate. [0.1 for amazon and 0.001 for yelp]')
    parser.add_argument('--lambda_1', type=float, default=0, help='lambad_1 = 0, ignore it')
    parser.add_argument('--lambda_2', type=float, default=1e-4, help='Weight decay (L2 loss weight).')
    parser.add_argument('--embed_dim', type=int, default=64, help='Node embedding size at the first layer.')
    parser.add_argument('--num_epochs', type=int, default=71, help='Number of epochs.')
    parser.add_argument('--test_epochs', type=int, default=10, help='Epoch interval to run test set.')
    parser.add_argument('--seed', type=int, default=123, help='Random seed.')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='Disables CUDA training.')

    if (torch.cuda.is_available()):
        print("cuda is available")

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    if (args.cuda):
        print("runing with GPU")

    print(f'run on {args.data}')

    # set seed
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    #Graph = AttributedGraphDataset(path, 'CiteSeer')[0]
    Graph = FakeHeteroDataset(num_graphs=1, num_node_types=1, num_edge_types=3, avg_num_nodes=1000, num_classes=2, task="node")

    labels = Graph[0]['v0']['y'].numpy()

    index = list(range(len(labels)))
    idx_train, idx_test, y_train, y_test = train_test_split(index, labels, stratify=labels, test_size=0.80,
                                                            random_state=2, shuffle=True)

    num_1 = len(np.where(y_train == 1)[0])
    num_2 = len(np.where(y_train == 0)[0])
    p0 = (num_1 / (num_1 + num_2))
    p1 = 1 - p0
    prior = np.array([p1, p0])

    args.idx_train = idx_train
    args.idx_test = idx_test
    args.y_train = y_train
    args.y_test = y_test
    args.prior = prior

    # model initialization
    clf = Fraudre()

    print('training...')
    clf.fit(Graph, args)
    print()

    print('predicting for probability')
    prob = clf.predict_proba(Graph, args)
    print('Probability', prob)
    print()

    print('predicting for raw scores')
    outlier_scores = clf.decision_function(Graph, args)
    print('Raw scores', outlier_scores)
    print()

    print('predicting for labels')
    labels = clf.predict(Graph, args)
    print('Labels', labels)
    print()

    print('predicting for labels with confidence')
    labels, confidence = clf.predict(Graph, args, return_confidence=True)
    print('Labels', labels)
    print('Confidence', confidence)
    print()
