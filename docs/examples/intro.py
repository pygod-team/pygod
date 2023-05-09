"""
Detector Example
================

In this tutorial, you will learn the basic workflow of
PyGOD with an example of DOMINANT. This tutorial assumes that
you have basic familiarity with PyTorch and PyTorch Geometric (PyG).

(Time estimate: 5 minutes)
"""
#######################################################################
# Data Loading
# ------------
# PyGOD use ``torch_geometric.data.Data`` to handle the data. Here, we
# use Cora, a PyG built-in dataset, as an example. To load your own
# dataset into PyGOD, you can refer to `creating your own datasets
# tutorial <https://pytorch-geometric.readthedocs.io/en/latest/notes
# /create_dataset.html>`__ in PyG.


import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid

data = Planetoid('./data/Cora', 'Cora', transform=T.NormalizeFeatures())[0]

#######################################################################
# Because there is no ground truth label of outliers in Cora, we follow
# the method used by DOMINANT to inject 100 contextual outliers and 100
# structure outliers into the graph. Note: If your dataset already
# contains the outliers you want to detect, you don't have to inject
# more outliers.


import torch
from pygod.generator import gen_contextual_outlier, gen_structural_outlier

data, ya = gen_contextual_outlier(data, n=100, k=50)
data, ys = gen_structural_outlier(data, m=10, n=10)
data.y = torch.logical_or(ys, ya).long()

#######################################################################
# We also provide various type of built-in datasets. You can load them
# by passing the name of the dataset to ``load_data`` function.
# See `data repository <https://github.com/pygod-team/data>`__
# for more details.


from pygod.utils import load_data

data = load_data('inj_cora')
data.y = data.y.bool()

#######################################################################
# Initialization
# --------------
# You can use any detector by simply initializing without passing any
# arguments. Default hyperparameters are ready for you. Of course, you
# can also customize the parameters by passing arguments. Here, we use
# ``pygod.detector.DOMINANT`` as an example.


from pygod.detector import DOMINANT

detector = DOMINANT(hid_dim=64, num_layers=4, epoch=100)

#######################################################################
# Training
# --------
# To train the detector with the loaded data, simply feed the
# ``torch_geometric.data.Data`` object into the detector via ``fit``.


detector.fit(data)

#######################################################################
# Inference
# ---------
# After training, the detector is ready to use. You can use the detector
# to predict the labels, raw outlier scores, probability of the
# outlierness, and prediction confidence. Here, we use the loaded data
# as an example.


pred, score, prob, conf = detector.predict(data,
                                           return_pred=True,
                                           return_score=True,
                                           return_prob=True,
                                           return_conf=True)
print('Labels:')
print(pred)

print('Raw scores:')
print(score)

print('Probability:')
print(prob)

print('Confidence:')
print(conf)

#######################################################################
# Evaluation
# ----------
# To evaluate the performance outlier detector with AUC score, you can:


from pygod.metric import eval_roc_auc

auc_score = eval_roc_auc(data.y, score)
print('AUC Score:', auc_score)
