# -*- coding: utf-8 -*-
"""Example code for the DOMINANT model
https://www.public.asu.edu/~kding9/pdf/SDM2019_Deep.pdf
"""
# Author: Yingtong Dou <ytongdou@gmail.com>, Kay Liu <zliu234@uic.edu>
# License: BSD 2 clause

import os.path as osp

import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid

from pygod.models import DOMINANT
from pygod.utils import gen_attribute_outliers, gen_structure_outliers
from pygod.utils.metric import eval_roc_auc, eval_recall_at_k, \
    eval_precision_at_k

dataset = 'Cora'

# data loading
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)

# this gives us a PyG data object
data = Planetoid(path, dataset, transform=T.NormalizeFeatures())[0]

data, ys = gen_structure_outliers(data, m=10, n=10)
data, ya = gen_attribute_outliers(data, n=100, k=50)
data.y = torch.logical_or(ys, ya).int()

# model initialization
model = DOMINANT()

print('training...')
model.fit(data)
print()

print('predicting for probability')
prob = model.predict_proba(data)
print('Probability', prob)
print()

print('predicting for raw scores')
outlier_scores = model.decision_function(data)
print('Raw scores', outlier_scores)
print()

print('predicting for labels')
labels = model.predict(data)
print('Labels', labels)
print()

print('predicting for labels with confidence')
labels, confidence = model.predict(data, return_confidence=True)
print('Labels', labels)
print('Confidence', confidence)

print('evaluating outlier detection performance')
auc_score = eval_roc_auc(data.y.numpy(), outlier_scores)
k = 200
recall_at_k = eval_recall_at_k(data.y.numpy(), outlier_scores, k=k,
                               threshold=model.threshold_)
precision_at_k = eval_precision_at_k(data.y.numpy(), outlier_scores, k=k,
                                     threshold=model.threshold_)

print('AUC Score:', auc_score)
print(f'Recall@{k}:', recall_at_k)
print(f'Precision@{k}:', precision_at_k)
print()
