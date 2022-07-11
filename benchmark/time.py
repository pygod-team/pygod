import os
import time
import torch
import shutil
import argparse
import warnings
import numpy as np
from utils import init_model
from pygod.metrics import eval_roc_auc
from pygod.utils import load_data
from torch_geometric.utils import remove_isolated_nodes


def main(args):
    for epoch in [10, 100, 200, 300, 400]:
        args.epoch = epoch
        model = init_model(args)
        data = load_data(args.dataset)

        data.edge_index, _, mask = \
            remove_isolated_nodes(data.edge_index, num_nodes=data.num_nodes)
        data.x = data.x[mask]

        start_time = time.time()
        if args.model == 'if' or args.model == 'lof':
            model.fit(data.x)
            t = time.time() - start_time
            score = model.decision_function(data.x)
        else:
            model.fit(data)
            t = time.time() - start_time
            score = model.decision_scores_

        if os.path.isdir('./tmp'):
            shutil.rmtree('./tmp')

        y = data.y.bool()[mask]
        auc = eval_roc_auc(y, score)

        if np.isnan(score).any():
            warnings.warn('contains NaN, skip one trial.')
            continue

        print('Epoch {}: AUC: {:.4f}, Time: {:.4f}s.'.format(epoch, auc, t))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="dominant",
                        help="supported model: [lof, if, mlpae, scan, radar, "
                             "anomalous, gcnae, dominant, done, adone, "
                             "anomalydae, gaan, guide, conad]. "
                             "Default: dominant")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU Index. Default: -1, using CPU.")
    parser.add_argument("--dataset", type=str, default='gen_time',
                        help="Default: gen_time.")
    args = parser.parse_args()

    main(args)
