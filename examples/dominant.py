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

# Move the code from dominant to here.
# # todo: need a default args template
# parser = argparse.ArgumentParser()
# parser.add_argument('--dataset', default='BlogCatalog',
#                     help='dataset name: Flickr/BlogCatalog')
# parser.add_argument('--hidden_size', type=int, default=64,
#                     help='dimension of hidden embedding (default: 64)')
# parser.add_argument('--epoch', type=int, default=3, help='Training epoch')
# parser.add_argument('--lr', type=float, default=5e-3, help='learning rate')
# parser.add_argument('--dropout', type=float, default=0.3,
#                     help='Dropout rate')
# parser.add_argument('--alpha', type=float, default=0.8,
#                     help='balance parameter')
# parser.add_argument('--device', default='cpu', type=str, help='cuda/cpu')
#
# args = parser.parse_args()
#
# # data loading
# path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data',
#                 args.dataset)
#
# # this gives us a PyG data object
# G = AttributedGraphDataset(path, 'BlogCatalog')[0]
#
# # model initialization
# clf = Dominant()
#
# print('training...')
# clf.fit(G, args)
# print()
#
# print('predicting for probability')
# prob = clf.predict_proba(G, args)
# print('Probability', prob)
# print()
#
# print('predicting for raw scores')
# outlier_scores = clf.decision_function(G, args)
# print('Raw scores', outlier_scores)
# print()
#
# print('predicting for labels')
# labels = clf.predict(G, args)
# print('Labels', labels)
# print()
#
# print('predicting for labels with confidence')
# labels, confidence = clf.predict(G, args, return_confidence=True)
# print('Labels', labels)
# print('Confidence', confidence)
# print()

