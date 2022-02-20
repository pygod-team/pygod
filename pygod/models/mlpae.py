import torch
import argparse
import os.path as osp
import torch.nn.functional as F
from torch_geometric.nn import MLP
import torch_geometric.transforms as T
from sklearn.metrics import roc_auc_score
from torch_geometric.datasets import Planetoid
from sklearn.utils.validation import check_is_fitted

from . import BaseDetector
from ..utils import EarlyStopping
from ..utils import gen_attribute_outliers, gen_structural_outliers


class MLPAE(BaseDetector):
    """Let us decide the documentation later

    """
    def __init__(self, contamination=0.1):
        super(MLPAE, self).__init__(contamination=contamination)

    def fit(self, G, args):

        # 1. first call the data process
        x, labels = self.process_graph(G, args)

        # 2. set the parameters needed for the network from args.
        self.channel_list = [x.shape[1]] + args.channel_list + [x.shape[1]]
        self.in_channels = x.shape[1]
        self.hidden_channels = args.hidden_size
        self.out_channels = x.shape[1]
        self.num_layers = args.num_layers
        self.dropout = args.dropout
        self.weight_decay = args.weight_decay

        # TODO: support other activation function
        if args.act:
            self.act = F.relu

        # 3. initialize the detection model
        self.model = MLP(channel_list=self.channel_list,
                         dropout=self.dropout,
                         batch_norm=False)

        # TODO: support channel specification after next pyg release
        # self.model = MLP(in_channels=self.in_channels,
        #                  hidden_channels=self.hidden_channels,
        #                  out_channels=self.out_channels,
        #                  num_layers=self.num_layers,
        #                  dropout=self.dropout,
        #                  act=self.act)

        # 4. check cuda
        if args.gpu >= 0 and torch.cuda.is_available():
            device = 'cuda:{}'.format(args.gpu)
        else:
            device = 'cpu'

        x = x.to(device)
        self.model = self.model.to(device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=self.weight_decay)

        if args.patience > 0:
            es = EarlyStopping(args.patience, args.verbose)

        for epoch in range(args.epoch):
            self.model.train()
            x_ = self.model(x)
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

        x, _ = self.process_graph(G, args)
        x_ = self.model(x)
        outlier_scores = torch.mean(F.mse_loss(x_, x, reduction='none'), dim=1).detach().cpu().numpy()
        return outlier_scores

    def process_graph(self, G, args):
        # return feature only
        return G.x, G.y


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--channel_list', type=list, default=[16, 16], help='dimensions of hidden layers (default: 16)')
    parser.add_argument('--hidden_size', type=int, default=16, help='dimension of hidden embedding (default: 16)')
    parser.add_argument('--num_layers', type=int, default=2, help='number of linear layers in MLP (default: 2)')
    parser.add_argument('--epoch', type=int, default=100, help='maximum training epoch')
    parser.add_argument('--lr', type=float, default=4e-3, help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='weight decay')
    parser.add_argument("--act", type=bool, default=True, help="using activation function or not")
    parser.add_argument("--gpu", type=int, default=0, help="GPU Index, -1 for using CPU (default: 0)")
    parser.add_argument("--verbose", type=bool, default=False, help="print log information")
    parser.add_argument("--patience", type=int, default=10,
                        help="early stopping patience, 0 for disabling early stopping (default: 10)")

    args = parser.parse_args()

    # data loading
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Cora')
    data = Planetoid(path, 'Cora', transform=T.NormalizeFeatures())[0]

    data, ys = gen_structural_outliers(data, 10, 10)
    data, yf = gen_attribute_outliers(data, 100, 30)
    data.y = torch.logical_or(ys, yf)

    # model initialization
    clf = MLPAE()

    print('training...')
    clf.fit(data, args)
    print()

    print('predicting for probability')
    prob = clf.predict_proba(data, args)
    print('Probability', prob)
    print()

    print('predicting for raw scores')
    outlier_scores = clf.decision_function(data, args)
    print('Raw scores', outlier_scores)
    print()

    print('predicting for labels')
    labels = clf.predict(data, args)
    print('Labels', labels)
    print()

    print('predicting for labels with confidence')
    labels, confidence = clf.predict(data, args, return_confidence=True)
    print('Labels', labels)
    print('Confidence', confidence)
    print()
