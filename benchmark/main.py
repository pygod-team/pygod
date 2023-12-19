import tqdm
import torch
import argparse
import warnings
from pygod.metric import *
from pygod.utils import load_data
from utils import init_model


def main(args):
    auc, ap, rec = [], [], []

    for _ in tqdm.tqdm(range(num_trial)):
        model = init_model(args)
        data = load_data(args.dataset)

        if args.model == 'if' or args.model == 'lof':
            model.fit(data.x)
            score = model.decision_function(data.x)
        else:
            model.fit(data)
            score = model.decision_score_

        y = data.y.bool()
        k = sum(y)

        if torch.isnan(score).any():
            warnings.warn('contains NaN, skip one trial.')
            continue

        auc.append(eval_roc_auc(y, score))
        ap.append(eval_average_precision(y, score))
        rec.append(eval_recall_at_k(y, score, k))

    auc = torch.tensor(auc)
    ap = torch.tensor(ap)
    rec = torch.tensor(rec)

    print(args.dataset + " " + model.__class__.__name__ + " " +
          "AUC: {:.4f}±{:.4f} ({:.4f})\t"
          "AP: {:.4f}±{:.4f} ({:.4f})\t"
          "Recall: {:.4f}±{:.4f} ({:.4f})".format(torch.mean(auc),
                                                  torch.std(auc),
                                                  torch.max(auc),
                                                  torch.mean(ap),
                                                  torch.std(ap),
                                                  torch.max(ap),
                                                  torch.mean(rec),
                                                  torch.std(rec),
                                                  torch.max(rec)))


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
                             "inj_flickr, weibo, reddit, disney, books, "
                             "enron]. Default: inj_cora")
    args = parser.parse_args()

    # global setting
    num_trial = 20

    main(args)
