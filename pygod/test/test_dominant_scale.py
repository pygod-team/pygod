# -*- coding: utf-8 -*-
import os
import unittest
# noinspection PyProtectedMember
from numpy.testing import assert_equal
from numpy.testing import assert_raises

import torch
from torch_geometric.seed import seed_everything
from pygod.models import DOMINANT_S
from pygod.metrics import eval_roc_auc

seed_everything(42)


if __name__ == '__main__':
    test_graph = torch.load(os.path.join('pygod', 'test', 'test_graph.pt'))
    data = test_graph

    model = DOMINANT_S()
    model.fit(data)

    pred_scores = model.decision_function(data)

    # check score shapes
    assert_equal(pred_scores.shape[0], data.y.shape[0])

    # check performance
    print(eval_roc_auc(data.y, pred_scores))