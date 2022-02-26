# -*- coding: utf-8 -*-
"""Example code for the Dominant model
https://www.public.asu.edu/~kding9/pdf/SDM2019_Deep.pdf
"""
# Author: Yingtong Dou <ytongdou@gmail.com>
# License: BSD 2 clause

import argparse
import os.path as osp

from torch_geometric.datasets import AttributedGraphDataset

from pygod.models.dominant import Dominant
from pygod.evaluator.evaluator import Evaluator

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='BlogCatalog',
                    help='dataset name: cora, pubmed, acm')
parser.add_argument('--hidden_size', type=int, default=64,
                    help='dimension of hidden embedding (default: 64)')
parser.add_argument('--epoch', type=int, default=10, help='Training epoch')
parser.add_argument('--lr', type=float, default=5e-3, help='learning rate')
parser.add_argument('--dropout', type=float, default=0.3,
                    help='Dropout rate')
parser.add_argument('--alpha', type=float, default=0.8,
                    help='balance parameter')
parser.add_argument('--device', default='cuda', type=str, help='cuda/cpu')

args = parser.parse_args()

# data loading
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data',
                args.dataset)

# this gives us a PyG data object
G = AttributedGraphDataset(path, args.dataset)[0]

# model initialization
clf = Dominant()

# evaluator initialization
eval = Evaluator(dataset=G, outlier_type='attribute', n=50, k=5)

# evaluate under different metrics
score = eval.eval(clf=clf, metric='rocauc', args=args)
print('ROC AUC score:', score)
