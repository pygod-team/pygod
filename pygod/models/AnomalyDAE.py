import torch
import argparse
import os.path as osp
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor
from torch_geometric.nn import GATConv
from sklearn.metrics import roc_auc_score
from torch_geometric.datasets import AttributedGraphDataset
from sklearn.utils.validation import check_is_fitted

from base import BaseDetector

class StructureAE(nn.Module):
    def __init__(self, in_dim, embed_dim, hid_dim, dropout):
        super(StructureAE, self).__init__()
        
        self.dense = nn.Linear(in_dim, embed_dim)
        self.attention_layer = GATConv(embed_dim, hid_dim)
        self.dropout = dropout

    def forward(self, x, adj):
        #encoder
        x = F.relu(self.dense(x))
        #x = F.dropout(x, self.dropout)
        embed_x = self.attention_layer(x, adj)
        #decoder
        x = torch.sigmoid(embed_x @ embed_x.T)
        return x, embed_x

class AttributeAE(nn.Module):
    def __init__(self, in_dim, embed_dim, out_dim, dropout):
        super(AttributeAE, self).__init__()
        self.dense1 = nn.Linear(in_dim, embed_dim)
        self.dense2 = nn.Linear(embed_dim, out_dim)
        self.dropout = dropout
    
    def forward(self, x, struct_embed):
        #encoder
        x = F.relu(self.dense1(x.T))
        x = self.dense2(x)
        #decoder
        #print(x.shape)
        #print(struct_embed.shape)
        x =  struct_embed @x.T
        #print(x.shape)
        return x 

class AnomalyDAE_Base(nn.Module):
    r"""AnomalyDAE (dual autoencoder for anomaly detection on attributed networks)
    AdnomalyDAE_Base is an anomaly detector consisting of a structure autoencoder, and an attribute
    reconstruction autoencoder. The reconstruction mean sqare error of the
    decoders are defined as structure anamoly score and attribute anomaly score, respectively, with two
    additional penalties on the reconstructed adj matrix and node attributes.
    Reference: https://haoyfan.github.io/papers/AnomalyDAE_ICASSP2020.pdf

    Parameters
    ----------
    in_dim : int
        Dimension of input feature
    in_dim2: int
        Dimension of the input number of nodes
    hid_dim : int
        Dimension of  hidden layer
    embed_dim:: int
        Dimension of the embedding (D1 in the papaer)
    dropout : float, optional
        Dropout rate of the model
        Default: 0
    """

    def __init__(self, in_dim, in_dim2, embed_dim, hid_dim, dropout):
        super(AnomalyDAE_Base, self).__init__()
        self.structure_AE = StructureAE(in_dim, embed_dim, hid_dim, dropout)
        self.attribute_AE = AttributeAE(in_dim2, embed_dim, hid_dim, dropout)  

    def forward(self, x, adj):
        A_hat, embed_x = self.structure_AE(x, adj)
        X_hat = self.attribute_AE(x, embed_x)
        return A_hat, X_hat

def loss_func(adj, A_hat, attrs, X_hat, alpha, theta, eta):
    # generate hyperparameter - structure penalty
    reversed_adj  = torch.ones(adj.shape) - adj
    thetas = torch.where(reversed_adj > 0, reversed_adj,  torch.full(adj.shape, theta))
    
    # generate hyperparameter - node penalty
    reversed_attr = torch.ones(attrs.shape) - attrs
    etas = torch.where(reversed_attr == 1, reversed_attr, torch.full(attrs.shape, eta))
    
    # Attribute reconstruction loss
    diff_attribute = torch.pow(X_hat - attrs, 2)  * etas
    attribute_reconstruction_errors = torch.sqrt(torch.sum(diff_attribute, 1))
    attribute_cost = torch.mean(attribute_reconstruction_errors)

    # structure reconstruction loss
    diff_structure = torch.pow(A_hat - adj, 2) * thetas
    structure_reconstruction_errors = torch.sqrt(torch.sum(diff_structure, 1))
    structure_cost = torch.mean(structure_reconstruction_errors)

    cost = alpha * attribute_reconstruction_errors + (
            1 - alpha) * structure_reconstruction_errors
    
    return cost, structure_cost, attribute_cost


class AnomalyDAE(BaseDetector):
    def __init__(self,
                 feat_size,
                 n_size,
                 embed_size,
                 hidden_size,
                 dropout=0.2,
                 weight_decay=1e-5,
                 preprocessing=True,
                 loss_fn=None,
                 contamination=0.1,
                 device=None):
        super(AnomalyDAE, self).__init__(contamination=contamination)
        self.feat_size = feat_size
        self.n_size = n_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.embed_size = embed_size
        self.model = AnomalyDAE_Base(in_dim = self.feat_size,
                                     in_dim2 = self.n_size,
                                     embed_dim = self.embed_size,
                                     hid_dim = self.hidden_size,
                                     dropout = self.dropout)

        
    def fit(self, adj, adj_label, attrs, args):
        if args.device == 'cuda':
            device = torch.device(args.device)
            adj = adj.to(device)
            adj_label = adj_label.to(device)
            attrs = attrs.to(device)
            self.model = self.model.cuda()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)

        for epoch in range(args.epoch):
            # TODO: keep the best epoch only
            self.model.train()
            optimizer.zero_grad()
            A_hat, X_hat = self.model(attrs, adj)
            #print(A_hat.shape, X_hat.shape, attrs.shape, adj_label.shape)
            loss, struct_loss, feat_loss = loss_func(adj_label, A_hat, attrs,
                                                     X_hat, args.alpha, args.theta, args.eta)
            l = torch.mean(loss)
            l.backward()
            optimizer.step()
            print("Epoch:", '%04d' % (epoch), "train_loss=",
                  "{:.5f}".format(l.item()), "train/struct_loss=",
                  "{:.5f}".format(struct_loss.item()), "train/feat_loss=",
                  "{:.5f}".format(feat_loss.item()))

            self.model.eval()
            A_hat, X_hat = self.model(attrs, adj)
            loss, struct_loss, feat_loss = loss_func(adj_label, A_hat, attrs,
                                                     X_hat, args.alpha, args.theta, args.eta)
            score = loss.detach().cpu().numpy()
            print("Epoch:", '%04d' % (epoch), 'Auc',
                  roc_auc_score(label, score))

        self.decision_scores_ = score
        self._process_decision_scores()
        return self

    def decision_function(self, attrs, adj, args):
        """Predict raw anomaly score of X using the fitted detector.
        The anomaly score of an input sample is computed based on different
        detector algorithms. For consistency, outliers are assigned with
        larger anomaly scores.
        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The training input samples. Sparse matrices are accepted only
            if they are supported by the base estimator.
        Returns
        -------
        anomaly_scores : numpy array of shape (n_samples,)
            The anomaly score of the input samples.
        """
        check_is_fitted(self, ['model'])
        # enable the evaluation mode
        self.model.eval()

        # construct the vector for holding the reconstruction error
        A_hat, X_hat = self.model(attrs, adj)
        outlier_scores, _, _ = loss_func(adj_label, A_hat, attrs,
                                         X_hat, args.alpha, args.theta, args.eta)
        return outlier_scores.detach().cpu().numpy()

    def predict(self, attrs, adj, args):
        check_is_fitted(self, ['decision_scores_', 'threshold_', 'labels_'])
        pred_score = self.decision_function(attrs, adj, args)
        prediction = (pred_score > self.threshold_).astype('int').ravel()
        return prediction


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='BlogCatalog',
                        help='dataset name: Flickr/BlogCatalog')
    parser.add_argument('--hidden_dim', type=int, default=4,
                    help='dimension of hidden embedding (default: 64)')
    parser.add_argument('--embed_dim', type=int, default=128,
                    help='dimension of hidden embedding (default: 128)')
    parser.add_argument('--epoch', type=int, default=3, help='Training epoch')
    parser.add_argument('--lr', type=float, default=5e-3, help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate')
    parser.add_argument('--alpha', type=float, default=0.8,
                        help='balance parameter')
    parser.add_argument('--theta', type = float, default = 0.2, help= 'structure penalty')
    parser.add_argument('--eta', type = float, default = 0.2, help = 'attribute penalty')
    parser.add_argument('--device', default='cpu', type=str, help='cuda/cpu')

    args = parser.parse_args()

    # data loading
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data',
                    args.dataset)
    data = AttributedGraphDataset(path, 'BlogCatalog')
    adj = data[0].edge_index
    attrs = data[0].x[:, :4]
    label = data[0].y % 2
    dense_adj = SparseTensor(row=adj[0], col=adj[1]).to_dense()
    rowsum = dense_adj.sum(1)
    d_inv_sqrt = torch.pow(rowsum, -0.5).flatten()
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    adj_label = (dense_adj * d_mat_inv_sqrt).T * d_mat_inv_sqrt

    # train(args)
    # todo need to make args part of initialization
    clf = AnomalyDAE(feat_size=attrs.size(1), n_size = attrs.size(0),
                 hidden_size=args.hidden_dim, embed_size = args.embed_dim,
                   dropout=args.dropout)

    print('training it')
    clf.fit(adj, adj_label, attrs, args)

    print('predict on self')
    outlier_scores = clf.decision_function(attrs, adj, args)
    print('Raw scores', outlier_scores)

    labels = clf.predict(attrs, adj, args)
    print('Labels', labels)
    
