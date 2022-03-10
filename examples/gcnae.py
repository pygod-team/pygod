# -*- coding: utf-8 -*-
""" Example code for Graph Convolutional Network Autoencoder
"""
# Author: Kay Liu <zliu234@uic.edu>
# License: BSD 2 clause

import torch
import os.path as osp
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid

from pygod.models import GCNAE
from pygod.utils import gen_attribute_outliers, gen_structure_outliers
from pygod.evaluator.metric import roc_auc_score


# data loading
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Cora')
data = Planetoid(path, 'Cora', transform=T.NormalizeFeatures())[0]

data, ys = gen_structure_outliers(data, 10, 10)
data, yf = gen_attribute_outliers(data, 100, 30)
data.y = torch.logical_or(torch.tensor(ys), torch.tensor(yf))

# model initialization
model = GCNAE()

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
print()

print('evaluating outlier detection performance')
auc_score = roc_auc_score(data.y, outlier_scores)
print('AUC Score', auc_score)
print()
