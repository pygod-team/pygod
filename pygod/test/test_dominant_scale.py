# -*- coding: utf-8 -*-
import os
import unittest
from typing import Union, Any

# noinspection PyProtectedMember
from numpy.testing import assert_equal
from numpy.testing import assert_raises


import torch
from torch.jit import RecursiveScriptModule
from torch_geometric.seed import seed_everything
from pygod.models import DOMINANT
from pygod.metrics import eval_roc_auc
from pygod.utils import load_data

seed_everything(42)


if __name__ == '__main__':

    data = data = load_data('inj_cora')

    model = DOMINANT(num_layers=3, batch_size=128, scalable=True, verbose=2,epoch=80, lr=5e-3, gpu=0)
    ys = data.y >> 1 & 1
    model.fit(data, y_true=ys)

    pred_scores = model.decision_function(data)
    #
    # # check score shapes
    # assert_equal(pred_scores.shape[0], data.y.shape[0])
    #
    # # check performance
    print(eval_roc_auc(data.y, pred_scores))