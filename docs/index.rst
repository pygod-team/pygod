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
   :target: https://docs.pygod.org/en/latest/?badge=latest
   :alt: Documentation status

.. image:: https://img.shields.io/github/stars/pygod-team/pygod.svg
   :target: https://github.com/pygod-team/pygod/stargazers
   :alt: GitHub stars

.. image:: https://img.shields.io/github/forks/pygod-team/pygod.svg?color=blue
   :target: https://github.com/pygod-team/pygod/network
   :alt: GitHub forks

.. image:: https://static.pepy.tech/personalized-badge/pygod?period=total&units=international_system&left_color=black&right_color=blue&left_text=Downloads
   :target: https://pepy.tech/project/pygod
   :alt: PyPI downloads

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
For consistency and accessibility, PyGOD is developed on top of `PyTorch Geometric (PyG) <https://www.pyg.org/>`_
and `PyTorch <https://pytorch.org/>`_, and follows the API design of `PyOD <https://github.com/yzhao062/pyod>`_.
See examples below for detecting anomalies with PyGOD in 5 lines!


**PyGOD is featured for**:

* **Unified APIs, detailed documentation, and interactive examples** across various graph-based algorithms.
* **Comprehensive coverage** of more than 10 latest graph outlier detectors.
* **Full support of detections at multiple levels**, such as node-, edge- (WIP), and graph-level tasks (WIP).
* **Scalable design for processing large graphs** via mini-batch and sampling.
* **Streamline data processing with PyG**--fully compatible with PyG data objects.

**Outlier Detection Using PyGOD with 5 Lines of Code**\ :


.. code-block:: python


    # train a dominant detector
    from pygod.models import DOMINANT

    model = DOMINANT(num_layers=4, epoch=20)  # hyperparameters can be set here
    model.fit(data)  # data is a Pytorch Geometric data object

    # get outlier scores on the input data
    outlier_scores = model.decision_scores # raw outlier scores on the input data

    # predict on the new data in the inductive setting
    outlier_scores = model.decision_function(test_data) # raw outlier scores on the input data  # predict raw outlier scores on test



**Citing PyGOD**\ :

`PyGOD paper <https://arxiv.org/abs/2204.12095>`_ is available on arxiv :cite:`pygod2022`.
If you use PyGOD in a scientific publication, we would appreciate
citations to the following paper::

    @article{pygod2022,
      author  = {Liu, Kay and Dou, Yingtong and Zhao, Yue and Ding, Xueying and Hu, Xiyang and Zhang, Ruitong and Ding, Kaize and Chen, Canyu and Peng, Hao and Shu, Kai and Chen, George H. and Jia, Zhihao and Yu, Philip S.},
      title   = {PyGOD: A Python Library for Graph Outlier Detection},
      journal = {arXiv preprint arXiv:2204.12095},
      year    = {2022},
    }

or::

    Liu, K., Dou, Y., Zhao, Y., Ding, X., Hu, X., Zhang, R., Ding, K., Chen, C., Peng, H., Shu, K., Chen, G.H., Jia, Z., and Yu, P.S. 2022. PyGOD: A Python Library for Graph Outlier Detection. arXiv preprint arXiv:2204.12095.



----



Implemented Algorithms
======================

PyGOD toolkit consists of two major functional groups:

**(i) Node-level detection** :

===================  ===================  ==================  =====  ===========  ==============================================
Type                 Backbone             Abbr                Year   Sampling     Class
===================  ===================  ==================  =====  ===========  ==============================================
Unsupervised         NN                   MLPAE               2014   Yes          :class:`pygod.models.MLPAE`
Unsupervised         GNN                  GCNAE               2016   Yes          :class:`pygod.models.GCNAE`
Unsupervised         MF                   ONE                 2019   No           :class:`pygod.models.ONE`
Unsupervised         GNN                  DOMINANT            2019   Yes          :class:`pygod.models.DOMINANT`
Unsupervised         GNN                  DONE                2020   Yes          :class:`pygod.models.DONE`
Unsupervised         GNN                  AdONE               2020   Yes          :class:`pygod.models.AdONE`
Unsupervised         GNN                  AnomalyDAE          2020   Yes          :class:`pygod.models.AnomalyDAE`
Unsupervised         GAN                  GAAN                2020   Yes          :class:`pygod.models.GAAN`
Unsupervised         GNN                  OCGNN               2021   Yes          :class:`pygod.models.OCGNN`
Unsupervised/SSL     GNN                  CoLA (beta)         2021   In progress  :class:`pygod.models.CoLA`
Unsupervised/SSL     GNN                  ANEMONE (beta)      2021   In progress  :class:`pygod.models.ANEMONE`
Unsupervised         GNN                  GUIDE               2021   Yes          :class:`pygod.models.GUIDE`
Unsupervised/SSL     GNN                  CONAD               2022   Yes          :class:`pygod.models.CONAD`
===================  ===================  ==================  =====  ===========  ==============================================


**(ii) Utility functions** :

===================  ======================  ==================================  ======================================================================================================================================
Type                 Name                    Function                            Documentation
===================  ======================  ==================================  ======================================================================================================================================
Metric               eval_precision_at_k     Calculating Precision@k             `eval_precision_at_k <https://docs.pygod.org/en/latest/pygod.utils.html#pygod.utils.metric.eval_precision_at_k>`_
Metric               eval_recall_at_k        Calculating Recall@k                `eval_recall_at_k <https://docs.pygod.org/en/latest/pygod.utils.html#pygod.utils.metric.eval_recall_at_k>`_
Metric               eval_roc_auc            Calculating ROC-AUC Score           `eval_roc_auc <https://docs.pygod.org/en/latest/pygod.utils.html#pygod.utils.metric.eval_roc_auc>`_
Metric               eval_average_precision  Calculating average precision       `eval_average_precision <https://docs.pygod.org/en/latest/pygod.utils.html#pygod.utils.metric.eval_average_precision>`_
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
