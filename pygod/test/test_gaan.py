# -*- coding: utf-8 -*-
import unittest
# noinspection PyProtectedMember
from numpy.testing import assert_equal
from numpy.testing import assert_raises

import os.path as osp
from shutil import rmtree

import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid

from pygod.models import GAAN
from pygod.utils import gen_attribute_outliers, gen_structure_outliers
from pygod.evaluator.metric import roc_auc_score
from torch_geometric.datasets import AttributedGraphDataset
from torch_sparse import SparseTensor

import torch
import argparse
import os.path as osp
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor
from torch_geometric.nn import GATConv
from sklearn.metrics import roc_auc_score
from torch_geometric.datasets import AttributedGraphDataset
from sklearn.utils.validation import check_is_fitted

from torch_geometric.datasets import AttributedGraphDataset, HGBDataset
from torch_geometric.utils import to_scipy_sparse_matrix


if __name__ == "__main__":
    # 'Flickr', 'BlogCatalog', 'Cora'
    data = 'Flickr'

    if data == 'Flickr':
        # data loading
        path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data',
                        'Flickr')

        # this gives us a PyG data object
        data = AttributedGraphDataset(path, 'Flickr')[0]
        data.x = data.x.to_dense()

        data, ys = gen_structure_outliers(data, m=15, n=15)
        data, yf = gen_attribute_outliers(data, n=450, k=50)
        data.y = torch.logical_or(torch.tensor(ys), torch.tensor(yf))

    if data == 'BlogCatalog':
        # data loading
        path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data',
                        'BlogCatalog')

        # this gives us a PyG data object
        data = AttributedGraphDataset(path, 'BlogCatalog', transform=T.NormalizeFeatures())[0]

        data, ys = gen_structure_outliers(data, m=15, n=10)
        data, yf = gen_attribute_outliers(data, n=300, k=50)
        data.y = torch.logical_or(torch.tensor(ys), torch.tensor(yf))

    if data == 'Cora':
        # data loading
        dataset = 'Cora'

        # data loading
        path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)

        # this gives us a PyG data object
        data = Planetoid(path, dataset, transform=T.NormalizeFeatures())[0]

        data, ys = gen_structure_outliers(data, m=10, n=10)
        data, yf = gen_attribute_outliers(data, n=100, k=50)
        data.y = torch.logical_or(torch.tensor(ys), torch.tensor(yf))

    # model initialization
    model = GAAN()

    print('training...')
    model.fit(data)

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
    auc_score = roc_auc_score(data.y, outlier_scores)
    print('AUC Score', auc_score)
    print()