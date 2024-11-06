import os
import torch

from torch_geometric.loader import NeighborLoader, RandomNodeLoader, GraphSAINTRandomWalkSampler, NodeLoader
from torch_geometric.sampler import NumNeighbors, NeighborSampler, NodeSamplerInput
from torch_geometric.utils import to_dense_adj

from pygod.detector import CARD, DeepDetector
from pygod.metric import eval_roc_auc
from numpy.testing import assert_equal
from numpy.testing import assert_raises
from torch_geometric.nn import GCN
from torch import nn
import numpy as np


train_data = torch.load(os.path.join('./train_graph.pt'))
test_data = torch.load(os.path.join('./test_graph.pt'))
detector: DeepDetector = CARD(epoch=10, num_layers=2, hid_dim=8)
b = torch.tensor([[1, 2, 3], [2, 3, 4]])

b_sum = torch.sum(b, axis=1)
detector.fit(train_data)
pred, score, conf = detector.predict(test_data,
                                     return_pred=True,
                                     return_score=True,
                                     return_conf=True)
auc = eval_roc_auc(test_data.y, score)

print(auc)
