Python Graph Outlier Detection (PyGOD)
======================================


-----


.. image:: pygod_logo.png
   :width: 600
   :alt: PyGOD Logo
   :align: center

**<tba>Add badges here<tba>**


-----

PyGOD is a comprehensive **Python library** for **detecting outlying objects**
in **graphs**. This exciting yet challenging field has many key applications
in financial fraud detection and fake news detection.

PyGOD includes more than **10** detection algorithms, from classical ??? (SIGMOD 2000) to
the latest ??? (AAAI 2022). To support easy usage, PyGOD is developed on top
of `PyTorch Geometric (PyG) <https://www.pyg.org/>`_ and `PyTorch <https://pytorch.org/>`_.
See examples below for detecting anomalies with GNN in 5 lines!


PyGOD is featured for:

* **Unified APIs, detailed documentation, and interactive examples** across various graph-based algorithms.
* **Comprehensive coverage of more than 10 algorithms**\ , including both **classical graph algorithms** and **latest graph neural networks (GNNs)**.
* **Full support of various levels of detection**, such as node-, edge-, and graph-level tasks (wip).
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

`PyGOD paper <http://tbd>`_ is accessible on arxiv and under review in xxxx.
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

Installation
^^^^^^^^^^^^

It is recommended to use **pip** or **conda** for installation. Please make sure
**the latest version** is installed, as PyGOD is updated frequently:

.. code-block:: bash

   pip install pygod            # normal install
   pip install --upgrade pygod  # or update if needed

.. code-block:: bash

   conda install -c conda-forge pygod

Alternatively, you could clone and run setup.py file:

.. code-block:: bash

   git clone https://github.com/pygod-team/pygod.git
   cd pygod
   pip install .

**<tba>Need some basic instructions here<tba>**


----


API Cheatsheet & Reference
^^^^^^^^^^^^^^^^^^^^^^^^^^

Full API Reference: (https://pygod.readthedocs.io/en/latest/pygod.html). API cheatsheet for all detectors:


* **fit(X)**\ : Fit detector.
* **decision_function(G)**\ : Predict raw anomaly score of PyG data G using the fitted detector.
* **predict(G)**\ : Predict if nodes in PyG data G is an outlier or not using the fitted detector.
* **predict_proba(G)**\ : Predict the probability of nodes in PyG data G being outlier using the fitted detector.
* **predict_confidence(G)**\ : Predict the model's node-wise confidence (available in predict and predict_proba) [#Perini2020Quantifying]_.


Key Attributes of a fitted model:


* **decision_scores_**\ : The outlier scores of the training data. The higher, the more abnormal.
  Outliers tend to have higher scores.
* **labels_**\ : The binary labels of the training data. 0 stands for inliers and 1 for outliers/anomalies.


Implemented Algorithms
^^^^^^^^^^^^^^^^^^^^^^

PyOD toolkit consists of three major functional groups:

**(i) Node-level detection** :

===================  ==================  ======================================================================================================  =====  ========================================
Type                 Abbr                Algorithm                                                                                               Year   Ref
===================  ==================  ======================================================================================================  =====  ========================================
GNN                  Dominant            Deep anomaly detection on attributed networks                                                           2019   [#Ding2019Deep]_
GNN                  AnomalyDAE          AnomalyDAE: Dual autoencoder for anomaly detection on attributed networks                               2020   [#Fan2020AnomalyDAE]_
GNN                  DONE                Outlier Resistant Unsupervised Deep Architectures for Attributed Network Embedding                      2020   [#Bandyopadhyay2020Outlier]_
GNN                  AdONE               Outlier Resistant Unsupervised Deep Architectures for Attributed Network Embedding                      2020   [#Bandyopadhyay2020Outlier]_
GNN                  coLA                Anomaly Detection on Attributed Networks via Contrastive Self-Supervised Learning                       2021   [#Liu2021Anomaly]_
GNN                  FRAUDRE (TO MOVE)   Fraud Detection Dual-Resistant to Graph Inconsistency and Imbalance                                     2021   [#Zhang2021FRAUDRE]_
GNN                  GCNAE (change ref)  Higher-order Structure Based Anomaly Detection on Attributed Networks                                   2021   [#Yuan2021Higher]_
GNN                  MLPAE (change ref)  Higher-order Structure Based Anomaly Detection on Attributed Networks                                   2021   [#Yuan2021Higher]_
GNN                  GUIDE               Higher-order Structure Based Anomaly Detection on Attributed Networks                                   2021   [#Yuan2021Higher]_
GNN                  OCGNN               One-Class Graph Neural Networks for Anomaly Detection in Attributed Networks                            2021   [#Wang2021One]_
GNN                  ONE                 Outlier aware network embedding for attributed networks                                                 2019   [#Bandyopadhyay2019Outlier]_
===================  ==================  ======================================================================================================  =====  ========================================

**(ii) Graph-level detection** :

**<tba>Add then<tba>**


**(iii) Utility functions** :

**<tba>Add then<tba>**

===================  ======================  =====================================================================================================================================================  ======================================================================================================================================
Type                 Name                    Function                                                                                                                                               Documentation
===================  ======================  =====================================================================================================================================================  ======================================================================================================================================
Data                 generate_data           Synthesized data generation; normal data is generated by a multivariate Gaussian and outliers are generated by a uniform distribution                  `generate_data <https://pyod.readthedocs.io/en/latest/pyod.utils.html#module-pyod.utils.data.generate_data>`_
Data                 generate_data_clusters  Synthesized data generation in clusters; more complex data patterns can be created with multiple clusters                                              `generate_data_clusters <https://pyod.readthedocs.io/en/latest/pyod.utils.html#pyod.utils.data.generate_data_clusters>`_
Stat                 wpearsonr               Calculate the weighted Pearson correlation of two samples                                                                                              `wpearsonr <https://pyod.readthedocs.io/en/latest/pyod.utils.html#module-pyod.utils.stat_models.wpearsonr>`_
Utility              get_label_n             Turn raw outlier scores into binary labels by assign 1 to top n outlier scores                                                                         `get_label_n <https://pyod.readthedocs.io/en/latest/pyod.utils.html#module-pyod.utils.utility.get_label_n>`_
Utility              precision_n_scores      calculate precision @ rank n                                                                                                                           `precision_n_scores <https://pyod.readthedocs.io/en/latest/pyod.utils.html#module-pyod.utils.utility.precision_n_scores>`_
===================  ======================  =====================================================================================================================================================  ======================================================================================================================================


----


Quick Start for Outlier Detection with PyGOD
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`"examples/dominant_example.py" <https://github.com/pygod-team/pygod/blob/master/examples/dominant_example.py>`_
demonstrates the basic API of using the dominant detector. **It is noted that the API across all other algorithms are consistent/similar**.

More detailed instructions for running examples can be found in `examples directory <https://github.com/pygod-team/pygod/blob/master/examples/>`_.

#. Initialize a dominant detector, fit the model, and make the prediction.

   .. code-block:: python

#. Evaluate the prediction by ROC and Precision @ Rank n (p@n).

   .. code-block:: python


----

How to Contribute
^^^^^^^^^^^^^^^^^

You are welcome to contribute to this exciting project:

See `contribution guide <https://github.com/pygod-team/pygod/blob/master/contributing.md>`_ for more information.


----

PyGOD Team
^^^^^^^^^^

PyGOD is collaboratively developed by researchers from UIC, IIT, BUAA, ASU, and CMU.

Our core team members include (alphabetical order):

`Tom Davidson (Yale) <http://tbd>`_, `Tom Davidson (Yale) <http://tbd>`_, `Tom Davidson (Yale) <http://tbd>`_,

Reach out us by submitting an issue report or email us at **<tba>add an email<tba>**

----

Reference
^^^^^^^^^

.. [#Bandyopadhyay2019Outlier] Bandyopadhyay, S., Lokesh, N. and Murty, M.N., 2019, July. Outlier aware network embedding for attributed networks. In Proceedings of the AAAI conference on artificial intelligence (Vol. 33, No. 01, pp. 12-19).

.. [#Bandyopadhyay2020Outlier] Bandyopadhyay, S., Vivek, S.V. and Murty, M.N., 2020, January. Outlier resistant unsupervised deep architectures for attributed network embedding. In Proceedings of the 13th International Conference on Web Search and Data Mining (pp. 25-33).

.. [#Ding2019Deep] Ding, K., Li, J., Bhanushali, R. and Liu, H., 2019, May. Deep anomaly detection on attributed networks. In Proceedings of the 2019 SIAM International Conference on Data Mining (pp. 594-602). Society for Industrial and Applied Mathematics.

.. [#Fan2020AnomalyDAE] Fan, H., Zhang, F. and Li, Z., 2020, May. AnomalyDAE: Dual autoencoder for anomaly detection on attributed networks. In ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (pp. 5685-5689). IEEE.

.. [#Liu2021Anomaly] Liu, Y., Li, Z., Pan, S., Gong, C., Zhou, C. and Karypis, G., 2021. Anomaly detection on attributed networks via contrastive self-supervised learning. IEEE transactions on neural networks and learning systems.

.. [#Wang2021One] Wang, X., Jin, B., Du, Y., Cui, P., Tan, Y. and Yang, Y., 2021. One-class graph neural networks for anomaly detection in attributed networks. Neural computing and applications, 33(18), pp.12073-12085.

.. [#Yuan2021Higher] Yuan, X., Zhou, N., Yu, S., Huang, H., Chen, Z. and Xia, F., 2021, December. Higher-order Structure Based Anomaly Detection on Attributed Networks. In 2021 IEEE International Conference on Big Data (Big Data) (pp. 2691-2700). IEEE.

.. [#Zhang2021FRAUDRE] Zhang, G., Wu, J., Yang, J., Beheshti, A., Xue, S., Zhou, C. and Sheng, Q.Z., 2021, December. FRAUDRE: Fraud Detection Dual-Resistant to Graph Inconsistency and Imbalance. In 2021 IEEE International Conference on Data Mining (ICDM) (pp. 867-876). IEEE.