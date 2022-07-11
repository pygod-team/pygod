import tqdm
import torch
import argparse
import warnings
from pygod.metrics import *
from pygod.utils import load_data
from utils import init_model


def main(args):
    aucc, aucs = [], []
    apc, aps = [], []
    recc, recs = [], []

    for _ in tqdm.tqdm(range(num_trial)):
        model = init_model(args)
        data = load_data(args.dataset)

        if args.model == 'iforest' or args.model == 'lof':
            model.fit(data.x)
            score = model.decision_function(data.x)
        else:
            model.fit(data)
            score = model.decision_scores_

        yc = data.y >> 0 & 1
        ys = data.y >> 1 & 1
        kc, ks = sum(yc), sum(ys)

        if np.isnan(score).any():
            warnings.warn('contains NaN, skip one trial.')
            continue

        aucc.append(eval_roc_auc(yc, score))
        apc.append(eval_average_precision(yc, score))
        recc.append(eval_recall_at_k(yc, score, kc))

        aucs.append(eval_roc_auc(ys, score))
        aps.append(eval_average_precision(ys, score))
        recs.append(eval_recall_at_k(ys, score, ks))

    print(args.dataset + " " + model.__class__.__name__ + "\n" +
          "Contextual: AUC: {:.4f}±{:.4f} ({:.4f})\t"
          "AP: {:.4f}±{:.4f} ({:.4f})\tRecall: {:.4f}±{:.4f} ({:.4f})\n"
          "Structural: AUC: {:.4f}±{:.4f} ({:.4f})\t"
          "AP: {:.4f}±{:.4f} ({:.4f})\tRecall: {:.4f}±{:.4f} ({:.4f})"
          .format(np.mean(aucc), np.std(aucc), np.max(aucc),
                  np.mean(apc), np.std(apc), np.max(apc),
                  np.mean(recc), np.std(recc), np.max(recc),
                  np.mean(aucs), np.std(aucs), np.max(aucs),
                  np.mean(aps), np.std(aps), np.max(aps),
                  np.mean(recs), np.std(recs), np.max(recs)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="dominant",
                        help="supported model: [lof, if, mlpae, scan, radar, "
                             "anomalous, gcnae, dominant, done, adone, "
                             "anomalydae, gaan, guide, conad]. "
                             "Default: dominant")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU Index. Default: -1, using CPU.")
    parser.add_argument("--dataset", type=str, default='inj_cora',
                        help="supported dataset: [inj_cora, inj_amazon, "
                             "inj_flickr]. Default: inj_cora.")
    args = parser.parse_args()

    # global setting
    num_trial = 20

    main(args)
