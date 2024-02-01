"""
Score Conversion
================

Currently, the majority of outlier detectors are on node level. However,
in some real-world applications, we may interest in the outlier edges or
outlier graphs. In this tutorial, we introduce score converters provided
in ``pygod.utils`` module, including ``to_edge_score`` and
``to_graph_score``.

(Time estimate: 3 minutes)
"""

#######################################################################
# Data Loading
# ------------
# We first load the data from PyGOD with ``load_data`` function.


from pygod.utils import load_data

data = load_data('weibo')
print(data)

#######################################################################
# Detector Training
# -----------------
# Initialize and train a detector in PyGOD. Here, we use
# ``pygod.detector.DOMINANT`` as an example. For faster demonstration,
# we set ``epoch`` to 3.


from pygod.detector import DOMINANT

detector = DOMINANT(epoch=3)
detector.fit(data)

#######################################################################
# Obtaining Node Score
# --------------------
# After training, we obtain raw outlier scores for each node with
# ``predict``. The shape of ``node_score`` is ``(N, )``.


node_score = detector.predict(data, return_pred=False, return_score=True)
print(node_score.shape)

#######################################################################
# Converting Score to Edge Level
# ------------------------------
# To detect outlier edges, we convert the outlier scores on node level
# to edge level. The shape of ``edge_score`` is ``(E, )``.


from pygod.utils import to_edge_score

edge_score = to_edge_score(node_score, data.edge_index)
print(edge_score.shape)

#######################################################################
# Converting Score to Graph Level
# -------------------------------
# To detect outlier graphs, we convert the outlier scores on node level
# to graph level for each graph. ``graph_score`` is a scalar for each
# ``Data`` object. Here, we give an example for scoring a list of graph.


from pygod.utils import to_graph_score

data_list = [data, data, data]
graph_scores = []
for data in data_list:
    node_score = detector.predict(data, return_pred=False, return_score=True)
    graph_score = to_graph_score(node_score)
    graph_scores.append(graph_score.item())

print(graph_scores)
