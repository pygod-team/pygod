# -*- coding: utf-8 -*-
"""Example code for the Dominant model
https://www.public.asu.edu/~kding9/pdf/SDM2019_Deep.pdf
"""
# Author: Yingtong Dou <ytongdou@gmail.com>, Kay Liu <zliu234@uic.edu>
# License: BSD 2 clause

import argparse
import os.path as osp

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid

from pygod.models.dominant import Dominant
from pygod.evaluator.evaluator import Evaluator

parser = argparse.ArgumentParser()
parser.add_argument('--hidden_size', type=int, default=64,
                    help='dimension of hidden embedding (default: 64)')
parser.add_argument('--num_layers', type=int, default=4,
                    help='number of hidden layers, must be greater than 2 (default: 4)')
parser.add_argument('--epoch', type=int, default=100,
                    help='maximum training epoch')
parser.add_argument('--lr', type=float, default=5e-3,
                    help='learning rate')
parser.add_argument('--dropout', type=float, default=0.3,
                    help='Dropout rate')
parser.add_argument('--alpha', type=float, default=0.8,
                    help='balance parameter')
parser.add_argument('--weight_decay', type=float, default=1e-3,
                    help='weight decay')
parser.add_argument("--act", type=bool, default=True,
                    help="using activation function or not")
parser.add_argument("--gpu", type=int, default=0,
                    help="GPU Index, -1 for using CPU (default: 0)")
parser.add_argument("--verbose", type=bool, default=False,
                    help="print log information")
parser.add_argument("--patience", type=int, default=10,
                    help="early stopping patience, 0 for disabling early stopping (default: 10)")


args = parser.parse_args()

# data loading
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Cora')

# this gives us a PyG data object
G = Planetoid(path, 'Cora', transform=T.NormalizeFeatures())[0]

# model initialization
clf = Dominant()

# evaluator initialization
eval = Evaluator(dataset=G, outlier_type='attribute', n=50, k=5)

# evaluate under different metrics
score = eval.eval(clf=clf, metric='rocauc', args=args)
print('ROC AUC score:', score)
