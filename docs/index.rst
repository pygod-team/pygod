.. TDC documentation master file, created by
   sphinx-quickstart on Wed Jul  7 12:08:39 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


.. figure:: pygod_logo.png
    :scale: 30%
    :alt: logo

----


.. image:: https://img.shields.io/pypi/v/pygod.svg?color=brightgreen
   :target: https://pypi.org/project/pygod/
   :alt: PyPI version

.. image:: https://readthedocs.org/projects/pygod/badge/?version=latest
   :target: https://py-god.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation status

.. image:: https://img.shields.io/github/stars/pygod-team/pygod.svg
   :target: https://github.com/pygod-team/pygod/stargazers
   :alt: GitHub stars

.. image:: https://img.shields.io/github/forks/pygod-team/pygod.svg?color=blue
   :target: https://github.com/pygod-team/pygod/network
   :alt: GitHub forks

.. image:: https://github.com/pygod-team/pygod/actions/workflows/testing.yml/badge.svg
   :target: https://github.com/pygod-team/pygod/actions/workflows/testing.yml
   :alt: testing

.. image:: https://coveralls.io/repos/github/pygod-team/pygod/badge.svg?branch=main
   :target: https://coveralls.io/github/pygod-team/pygod?branch=main
   :alt: Coverage Status

.. image:: https://img.shields.io/github/license/pygod-team/pygod.svg
   :target: https://github.com/pygod-team/pygod/blob/master/LICENSE
   :alt: License

----

PyGOD is a **Python library** for **graph outlier detection** (anomaly detection).
This exciting yet challenging field has many key applications, e.g., detecting
suspicious activities in social networks :cite:`dou2020enhancing`  and security systems :cite:`cai2021structural`.

PyGOD includes more than **10** latest graph-based detection algorithms,
such as Dominant (SDM'19) and GUIDE (BigData'21).
For consistently and accessibility, PyGOD is developed on top of `PyTorch Geometric (PyG) <https://www.pyg.org/>`_
and `PyTorch <https://pytorch.org/>`_, and follows the API design of `PyOD <https://github.com/yzhao062/pyod>`_.
See examples below for detecting anomalies with GNN in 5 lines!

**PyGOD** is under actively developed and will be updated frequently!
Please **star**, **watch**, and **fork**.

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



**Citing PyGOD (to be announced soon)**\ :

`PyGOD paper <https://pygod.org>`_ will be available on arxiv soon.
If you use PyGOD in a scientific publication, we would appreciate
citations to the following paper (to be announced)::

    @article{tba,
      author  = {tba},
      title   = {PyGOD: A Comprehensive Python Library for Graph Outlier Detection},
      journal = {tba},
      year    = {2022},
    }

or::

    tba, 2022. PyGOD: A Comprehensive Python Library for Graph Outlier Detection. tba.


----

Implemented Algorithms
^^^^^^^^^^^^^^^^^^^^^^

PyOD toolkit consists of three major functional groups:

**(i) Node-level detection** :

===================  ===================  ==================  ======================================================================================================  =====  ==============================================
Type                 Backbone             Abbr                Algorithm                                                                                               Year   Class
===================  ===================  ==================  ======================================================================================================  =====  ==============================================
Unsupervised         GNN                  Dominant            Deep anomaly detection on attributed networks                                                           2019   :class:`pygod.models.dominant.DOMINANT`
Unsupervised         GNN                  AnomalyDAE          AnomalyDAE: Dual autoencoder for anomaly detection on attributed networks                               2020   :class:`pygod.models.anomalydae.AnomalyDAE`
Unsupervised         GNN                  DONE                Outlier Resistant Unsupervised Deep Architectures for Attributed Network Embedding                      2020   :class:`pygod.models.done.DONE`
Unsupervised         GNN                  AdONE               Outlier Resistant Unsupervised Deep Architectures for Attributed Network Embedding                      2020   :class:`pygod.models.adone.AdONE`
Unsupervised         GNN                  GCNAE               Variational Graph Auto-Encoders                                                                         2021   :class:`pygod.models.gcnae.GCNAE`
Unsupervised         NN                   MLPAE               Neural Networks and Deep Learning                                                                       2021   :class:`pygod.models.mlpae.MLPAE`
Unsupervised         GNN                  GUIDE               Higher-order Structure Based Anomaly Detection on Attributed Networks                                   2021   :class:`pygod.models.guide.GUIDE`
Unsupervised         GNN                  OCGNN               One-Class Graph Neural Networks for Anomaly Detection in Attributed Networks                            2021   :class:`pygod.models.ocgnn.OCGNN`
Unsupervised         MF                   ONE                 Outlier aware network embedding for attributed networks                                                 2019   :class:`pygod.models.one.ONE`
Unsupervised         GAN                  GAAN                Generative Adversarial Attributed Network Anomaly Detection                                             2020   :class:`pygod.models.gaan.GAAN`
===================  ===================  ==================  ======================================================================================================  =====  ==============================================


**(ii) Utility functions** :

===================  ======================  ==================================  ======================================================================================================================================
Type                 Name                    Function                            Documentation
===================  ======================  ==================================  ======================================================================================================================================
Metric               eval_precision_at_k     Calculating Precision@k             `eval_precision_at_k <https://docs.pygod.org/en/latest/pygod.utils.html#pygod.utils.metric.eval_precision_at_k>`_
Metric               eval_recall_at_k        Calculating Recall@k                `eval_recall_at_k <https://docs.pygod.org/en/latest/pygod.utils.html#pygod.utils.metric.eval_recall_at_k>`_
Metric               eval_roc_auc            Calculating ROC-AUC Score           `eval_roc_auc <https://docs.pygod.org/en/latest/pygod.utils.html#pygod.utils.metric.eval_roc_auc>`_
Data                 gen_structure_outliers  Generating structural outliers      `gen_structure_outliers <https://docs.pygod.org/en/latest/pygod.utils.html#pygod.utils.outlier_generator.gen_structure_outliers>`_
Data                 gen_attribute_outliers  Generating attribute outliers       `gen_attribute_outliers <https://docs.pygod.org/en/latest/pygod.utils.html#pygod.utils.outlier_generator.gen_attribute_outliers>`_
===================  ======================  ==================================  ======================================================================================================================================


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
   tutorials/index

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

