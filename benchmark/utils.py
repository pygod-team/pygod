from random import choice
from pygod.detectors import *
from pyod.models.lof import LOF
from sklearn.ensemble import IsolationForest


def init_model(args):
    dropout = [0, 0.1, 0.3]
    lr = [0.1, 0.05, 0.01]
    weight_decay = 0.01

    if args.dataset == 'inj_flickr' or args.dataset == 'dgraph':
        # sampling and minibatch training on large dataset flickr
        batch_size = 64
        num_neigh = 3
        epoch = 2
    else:
        batch_size = 0
        num_neigh = -1
        epoch = 300

    model_name = args.model
    gpu = args.gpu

    if hasattr(args, 'epoch'):
        epoch = args.epoch

    if args.dataset == 'reddit':
        # for the low feature dimension dataset
        hid_dim = [32, 48, 64]
    elif args.dataset in ['enron', 'disney', 'dgraph', 'books']:
        hid_dim = [8, 12, 16]
    else:
        hid_dim = [32, 64, 128, 256]

    if args.dataset[:3] == 'inj' or args.dataset[:3] == 'gen':
        # auto balancing on injected dataset
        alpha = [None]
    else:
        alpha = [0.8, 0.5, 0.2]

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
                     num_neigh=num_neigh,
                     cache_dir='./tmp')
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
    elif model_name == 'if':
        return IsolationForest()
    elif model_name == 'radar':
        return Radar(lr=choice(lr), gpu=gpu)
    elif model_name == 'anomalous':
        return ANOMALOUS(lr=choice(lr), gpu=gpu)
    elif model_name == 'scan':
        return SCAN(eps=choice([0.3, 0.5, 0.8]), mu=choice([2, 5, 10]))
