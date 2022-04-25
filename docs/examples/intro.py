"""
DOMINANT Example
================

In this introductory tutorial, you will learn the basic workflow of
PyGOD. This tutorial assumes that you have basic familiarity with
PyTorch and PyTorch Geometric (PyG).

(Time estimate: 5 minutes)
"""
#######################################################################
# Data Loading
# ------------
# PyGOD use `torch_geometric.data.Data` to handle the data. Here, we
# use Cora, a PyG built-in dataset, as an example. To load your own
# dataset into PyGOD, you can refer to [creating your own datasets
# tutorial](https://pytorch-geometric.readthedocs.io/en/latest/notes/create_dataset.html)
# in PyG.


import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid

data = Planetoid('./data/Cora', 'Cora', transform=T.NormalizeFeatures())[0]

#######################################################################
# Because there is no ground truth label of outliers in Cora, we follow
# the method used by DOMINANT to inject 100 attribute outliers and 100
# structure outliers into the graph. **Note**: If your dataset already
# contains the outliers you want to detect, you don't need to inject
# more outliers.


import torch
from pygod.utils import gen_attribute_outliers, gen_structure_outliers

data, ya = gen_attribute_outliers(data, n=100, k=50)
data, ys = gen_structure_outliers(data, m=10, n=10)
data.y = torch.logical_or(ys, ya).int()

#######################################################################
# Initialization
# --------------
# You can use any model by simply initializing without passing any
# arguments. Default hyperparameters are ready for you. Of course, you
# can also customize the parameters by passing arguments. Here, we use
# `pygod.models.DOMINANT` as an example.


from pygod.models import DOMINANT

model = DOMINANT()

#######################################################################
# Training
# --------
# To train the model with the loaded data, simply feed the
# `torch_geometric.data.Data` object into the model via method `fit`.


model.fit(data)

#######################################################################
# Inference
# ---------
# Then, your model is ready to use. We provide several inference methods.
#######################################################################
# To predict the labels only:


labels = model.predict(data)
print('Labels:')
print(labels)

#######################################################################
# To predict raw outlier scores:


outlier_scores = model.decision_function(data)
print('Raw scores:')
print(outlier_scores)

#######################################################################
# To predict the probability of the outlierness:


prob = model.predict_proba(data)
print('Probability:')
print(prob)

#######################################################################
# To predict the labels with confidence:


labels, confidence = model.predict(data, return_confidence=True)
print('Labels:')
print(labels)
print('Confidence:')
print(confidence)

#######################################################################
# To evaluate the performance outlier detector:


from pygod.utils.metric import \
    eval_roc_auc, \
    eval_recall_at_k, \
    eval_precision_at_k

k = 200

auc_score = eval_roc_auc(data.y.numpy(), outlier_scores)
recall_at_k = eval_recall_at_k(data.y.numpy(), outlier_scores,
                               k=k, threshold=model.threshold_)
precision_at_k = eval_precision_at_k(data.y.numpy(), outlier_scores,
                                     k=k, threshold=model.threshold_)

print('AUC Score:', auc_score)
print(f'Recall@{k}:', recall_at_k)
print(f'Precision@{k}:', precision_at_k)
