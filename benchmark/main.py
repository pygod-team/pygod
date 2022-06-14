import tqdm
import torch
import argparse
import warnings
from random import choice
from pygod.models import *
from pygod.metrics import *
from pyod.models.lof import LOF
from sklearn.ensemble import IsolationForest
from torch_geometric import seed_everything


def init_model(args):
    model_name = args.model
    gpu = args.gpu
    if model_name == "adone":
        return AdONE(hid_dim=choice(hid_dim),
                     weight_decay=weight_decay,
                     dropout=choice(dropout),
                     lr=choice(lr),
                     epoch=epoch,
                     gpu=gpu,
                     batch_size=batch_size,
                     num_neigh=num_neigh)
    elif model_name == 'anomalydae':
        hd = choice(hid_dim)
        return AnomalyDAE(embed_dim=hd,
                          out_dim=hd,
                          weight_decay=weight_decay,
                          dropout=choice(dropout),
                          theta=choice([10., 40., 90.]),
                          eta=choice([3., 5., 8.]),
                          lr=choice(lr),
                          epoch=epoch,
                          gpu=gpu,
                          alpha=choice(alpha),
                          batch_size=batch_size,
                          num_neigh=num_neigh)
    elif model_name == 'conad':
        return CONAD(hid_dim=choice(hid_dim),
                     weight_decay=weight_decay,
                     dropout=choice(dropout),
                     lr=choice(lr),
                     epoch=epoch,
                     gpu=gpu,
                     alpha=choice(alpha),
                     batch_size=batch_size,
                     num_neigh=num_neigh)
    elif model_name == 'dominant':
        return DOMINANT(hid_dim=choice(hid_dim),
                        weight_decay=weight_decay,
                        dropout=choice(dropout),
                        lr=choice(lr),
                        epoch=epoch,
                        gpu=gpu,
                        alpha=choice(alpha),
                        batch_size=batch_size,
                        num_neigh=num_neigh)
    elif model_name == 'done':
        return DONE(hid_dim=choice(hid_dim),
                    weight_decay=weight_decay,
                    dropout=choice(dropout),
                    lr=choice(lr),
                    epoch=epoch,
                    gpu=gpu,
                    batch_size=batch_size,
                    num_neigh=num_neigh)
    elif model_name == 'gaan':
        return GAAN(noise_dim=choice([8, 16, 32]),
                    hid_dim=choice(hid_dim),
                    weight_decay=weight_decay,
                    dropout=choice(dropout),
                    lr=choice(lr),
                    epoch=epoch,
                    gpu=gpu,
                    alpha=choice(alpha),
                    batch_size=batch_size,
                    num_neigh=num_neigh)
    elif model_name == 'gcnae':
        return GCNAE(hid_dim=choice(hid_dim),
                     weight_decay=weight_decay,
                     dropout=choice(dropout),
                     lr=choice(lr),
                     epoch=epoch,
                     gpu=gpu,
                     batch_size=batch_size,
                     num_neigh=num_neigh)
    elif model_name == 'guide':
        return GUIDE(a_hid=choice(hid_dim),
                     s_hid=choice([4, 5, 6]),
                     weight_decay=weight_decay,
                     dropout=choice(dropout),
                     lr=choice(lr),
                     epoch=epoch,
                     gpu=gpu,
                     alpha=choice(alpha),
                     batch_size=batch_size,
                     num_neigh=num_neigh)
    elif model_name == "mlpae":
        return MLPAE(hid_dim=choice(hid_dim),
                     weight_decay=weight_decay,
                     dropout=choice(dropout),
                     lr=choice(lr),
                     epoch=epoch,
                     gpu=gpu,
                     batch_size=batch_size)
    elif model_name == 'lof':
        return LOF()
    elif model_name == 'iforest':
        return IsolationForest()
    elif model_name == 'radar':
        return Radar(lr=choice(lr), gpu=gpu)
    elif model_name == 'anomalous':
        return ANOMALOUS(lr=choice(lr), gpu=gpu)
    elif model_name == 'scan':
        return SCAN(eps=choice([0.3, 0.5, 0.8]), mu=choice([2, 5, 10]))


def main(args):
    auc, ap, rec = [], [], []

    for _ in tqdm.tqdm(range(num_trial)):
        model = init_model(args)
        data = torch.load('data/' + args.dataset + '.pt')

        if args.model == 'iforest' or args.model == 'lof':
            model.fit(data.x)
            score = model.decision_function(data.x)
        else:
            model.fit(data)
            score = model.decision_scores_

        y = data.y.bool()
        k = sum(y)

        if np.isnan(score).any():
            warnings.warn('contains NaN, skip one trial.')
            continue

        auc.append(eval_roc_auc(y, score))
        ap.append(eval_average_precision(y, score))
        rec.append(eval_recall_at_k(y, score, k))

    print(args.dataset + " " + model.__class__.__name__ + " " +
          "AUC: {:.4f}±{:.4f} ({:.4f})\t"
          "AP: {:.4f}±{:.4f} ({:.4f})\t"
          "Recall: {:.4f}±{:.4f} ({:.4f})".format(np.mean(auc), np.std(auc),
                                                  np.max(auc), np.mean(ap),
                                                  np.std(ap), np.max(ap),
                                                  np.mean(rec), np.std(rec),
                                                  np.max(rec)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="dominant",
                        help="supported model: [lof, if, mlpae, scan, radar, "
                             "anomalous, gcnae, dominant, done, adone, "
                             "anomalydae, gaan, guide, conad]")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU Index. Default: -1, using CPU.")
    parser.add_argument("--dataset", type=str, default='inj_cora',
                        help="supported dataset: [inj_cora, inj_amazon, "
                             "inj_flickr, weibo, reddit]")
    args = parser.parse_args()

    # global setting
    seed_everything(0)
    num_trial = 20

    dropout = [0, 0.1, 0.3]
    lr = [0.1, 0.05, 0.01]
    weight_decay = 0.01

    if args.dataset == 'inj_flickr':
        # sampling and minibatch training on large dataset flickr
        batch_size = 64
        num_neigh = 3
        epoch = 2
    else:
        batch_size = 0
        num_neigh = -1
        epoch = 300

    if args.dataset == 'reddit':
        # for the low feature dimension dataset
        hid_dim = [32, 48, 64]
    else:
        hid_dim = [32, 64, 128, 256]

    if args.dataset[:3] == 'inj':
        # auto balancing on injected dataset
        alpha = [None]
    else:
        alpha = [0.8, 0.5, 0.2]

    main(args)
