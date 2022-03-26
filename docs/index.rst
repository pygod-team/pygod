.. TDC documentation master file, created by
   sphinx-quickstart on Wed Jul  7 12:08:39 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


.. figure:: pygod_logo.png
    :scale: 30%
    :alt: logo

----

.. image:: https://badgen.net/badge/icon/github?icon=github&label
   :target: https://github.com/pygod-team/pygod
   :alt: GitHub


----

PyGOD is a comprehensive **Python library** for **detecting outlying objects**
in **graphs**. This exciting yet challenging field has many key applications
in fraud detection :cite:`liu2020alleviating`  and fake news detection :cite:`dou2021user`.

PyGOD includes more than **10** latest graph-based detection algorithms,
such as Dominant (SDM'19) and coLA (TNNLS'21).
For consistently and accessibility, PyGOD is developed on top of `PyTorch Geometric (PyG) <https://www.pyg.org/>`_
and `PyTorch <https://pytorch.org/>`_, and follows the API design of `PyOD <https://github.com/yzhao062/pyod>`_.
See examples below for detecting anomalies with GNN in 5 lines!

**PyGOD is featured for**:

* **Unified APIs, detailed documentation, and interactive examples** across various graph-based algorithms.
* **Comprehensive coverage** of more than 10 latest graph neural networks (GNNs).
* **Full support of detections at multiple levels**, such as node-, edge-, and graph-level tasks (WIP).
* **Streamline data processing with PyG**--fully compatible with PyG data objects.

**Outlier Detection Using GNN with 5 Lines of Code**\ :


.. code-block:: python


    # train a dominant detector
    from pygod.models import DOMINANT

    model = DOMINANT()  # hyperparameters can be set here
    model.fit(data)  # data is a Pytorch Geometric data object

    # get outlier scores on the input data
    outlier_scores = model.decision_scores # raw outlier scores on the input data

    # predict on the new data
    outlier_scores = model.decision_function(test_data) # raw outlier scores on the input data  # predict raw outlier scores on test

**Citing PyGOD**\ :

`PyGOD paper <http://tbd>`_ is available on arxiv and under review in xxxx.
If you use PyGOD in a scientific publication, we would appreciate
citations to the following paper::

    @article{tbd,
      author  = {tbd},
      title   = {PyGOD: A Comprehensive Python Library for Graph Outlier Detection},
      journal = {tbd},
      year    = {2022},
      url     = {tbd}
    }

or::

    tbd, tbd and tbd, 2022. PyGOD: A Comprehensive Python Library for Graph Outlier Detection. tbd.


----

Implemented Algorithms
^^^^^^^^^^^^^^^^^^^^^^

PyOD toolkit consists of three major functional groups:

**(i) Node-level detection** :

===================  ==================  ======================================================================================================  =====  ==============================================
Type                 Abbr                Algorithm                                                                                               Year   Class
===================  ==================  ======================================================================================================  =====  ==============================================
GNN                  Dominant            Deep anomaly detection on attributed networks                                                           2019   :class:`pygod.models.dominant.DOMINANT`
GNN                  AnomalyDAE          AnomalyDAE: Dual autoencoder for anomaly detection on attributed networks                               2020   :class:`pygod.models.anomalydae.AnomalyDAE`
GNN                  DONE                Outlier Resistant Unsupervised Deep Architectures for Attributed Network Embedding                      2020   :class:`pygod.models.done.DONE`
GNN                  AdONE               Outlier Resistant Unsupervised Deep Architectures for Attributed Network Embedding                      2020   :class:`pygod.models.adone.AdONE`
GNN                  coLA                Anomaly Detection on Attributed Networks via Contrastive Self-Supervised Learning                       2021
GNN                  FRAUDRE (TO MOVE)   Fraud Detection Dual-Resistant to Graph Inconsistency and Imbalance                                     2021
GNN                  GCNAE (change ref)  Higher-order Structure Based Anomaly Detection on Attributed Networks                                   2021   :class:`pygod.models.gcnae.GCNAE`
GNN                  MLPAE (change ref)  Higher-order Structure Based Anomaly Detection on Attributed Networks                                   2021   :class:`pygod.models.mlpae.MLPAE`
GNN                  GUIDE               Higher-order Structure Based Anomaly Detection on Attributed Networks                                   2021   :class:`pygod.models.guide.GUIDE`
GNN                  OCGNN               One-Class Graph Neural Networks for Anomaly Detection in Attributed Networks                            2021   :class:`pygod.models.ocgnn.OCGNN`
GNN                  ONE                 Outlier aware network embedding for attributed networks                                                 2019   :class:`pygod.models.one.ONE`
===================  ==================  ======================================================================================================  =====  ==============================================


----


API CheatSheet
==============

The following APIs are applicable for all detector models for easy use.

* :func:`pygod.models.base.BaseDetector.fit`: Fit detector. y is ignored in unsupervised methods.
* :func:`pygod.models.base.BaseDetector.decision_function`: Predict raw anomaly scores of PyG Graph G using the fitted detector
* :func:`pygod.models.base.BaseDetector.predict`: Predict if a particular sample is an outlier or not using the fitted detector.
* :func:`pygod.models.base.BaseDetector.predict_proba`: Predict the probability of a sample being outlier using the fitted detector.
* :func:`pygod.models.base.BaseDetector.predict_confidence`: Predict the model's sample-wise confidence (available in predict and predict_proba).
* :func:`pygod.models.base.BaseDetector.process_graph` (you do not need to call this explicitly): Process the raw PyG data object into a tuple of sub data objects needed for the underlying model.


Key Attributes of a fitted model:

* :attr:`pygod.models.base.BaseDetector.decision_scores_`: The outlier scores of the training data. The higher, the more abnormal.
  Outliers tend to have higher scores.
* :attr:`pygod.models.base.BaseDetector.labels_`: The binary labels of the training data. 0 stands for inliers and 1 for outliers/anomalies.


**Input of PyGOD**: Please pass in a `PyTorch Geometric (PyG) <https://www.pyg.org/>`_ data object.
See `PyG data processing examples <https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html#data-handling-of-graphs>`_.



----


.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Getting Started

   install

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: API References

   api_cc
   pygod

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Additional Information

   team


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

