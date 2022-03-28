# -*- coding: utf-8 -*-
"""Outlier Aware Network Embedding for Attributed Networks (ONE)
"""
# Author: Xiyang Hu <xiyanghu@cmu.edu>
# License: BSD 2 clause

import os.path as osp

import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import AttributedGraphDataset

from pygod.models import ONE
from pygod.utils import gen_attribute_outliers, gen_structure_outliers
from pygod.evaluator.metric import roc_auc_score


dataset = 'CiteSeer'

# data loading
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data',
                dataset)

# this gives us a PyG data object
G = AttributedGraphDataset(path, 'CiteSeer')[0]

# model initialization
model = ONE()

print('training...')
model.fit(G)
print()

print('predicting for probability')
prob = model.predict_proba(G)
print('Probability', prob)
print()

print('predicting for raw scores')
outlier_scores = model.decision_function(G)
print('Raw scores', outlier_scores)
print()

print('predicting for labels')
labels = model.predict(G)
print('Labels', labels)
print()

print('predicting for labels with confidence')
labels, confidence = model.predict(G, return_confidence=True)
print('Labels', labels)
print('Confidence', confidence)
print()
