Python Graph Outlier Detection (PyGOD)
======================================


.. image:: https://github.com/pygod-team/pygod/blob/main/docs/pygod_logo.png
   :width: 1050
   :alt: PyGOD Logo
   :align: center

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

.. image:: https://img.shields.io/github/license/pygod-team/pygod.svg
   :target: https://github.com/pygod-team/pygod/blob/master/LICENSE
   :alt: License


-----

PyGOD is a comprehensive **Python library** for **detecting outlying objects**
in **graphs**. This exciting yet challenging field has many key applications
in fraud detection [#Liu2020Alleviating]_  and fake news detection [#Dou2021User]_.

PyGOD includes more than **10** latest graph-based detection algorithms,
such as Dominant (SDM'19) and coLA (TNNLS'21).
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

**Citing PyGOD (to be updated soon)**\ :

`PyGOD paper <http://tbd>`_ is available on arxiv and under review.
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

It is recommended to use **pip** or **conda** (wip) for installation.
Please make sure **the latest version** is installed, as PyGOD is updated frequently:

.. code-block:: bash

   pip install pygod            # normal install
   pip install --upgrade pygod  # or update if needed


Alternatively, you could clone and run setup.py file:

.. code-block:: bash

   git clone https://github.com/pygod-team/pygod.git
   cd pygod
   pip install .

**Required Dependencies**\ :


* Python 3.6 +
* argparse>=1.4.0
* numpy>=1.19.4
* scikit-learn>=0.22.1
* networkx>=2.6.3
* scipy>=1.5.2
* pandas>=1.1.3
* setuptools>=50.3.1.post20201107


**Note and PyG and PyTorch Installation**\ :
PyGOD depends on `PyTorch Geometric (PyG) <https://www.pyg.org/>`_, `PyTorch <https://pytorch.org/>`_, and `networkx <https://networkx.org/>`_. 
To streamline the installation, PyGOD does **NOT** install these libraries for you. Please install them
from the above links for running PyGOD:

* torch>=1.10
* pytorch_geometric>=2.0.3
* networkx>=2.6.3


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
GNN                  GCNAE               Variational Graph Auto-Encoders                                                                         2021   [#Yuan2021Higher]_
GNN                  MLPAE (change ref)  Higher-order Structure Based Anomaly Detection on Attributed Networks                                   2021   [#Yuan2021Higher]_
GNN                  GUIDE               Higher-order Structure Based Anomaly Detection on Attributed Networks                                   2021   [#Yuan2021Higher]_
GNN                  OCGNN               One-Class Graph Neural Networks for Anomaly Detection in Attributed Networks                            2021   [#Wang2021One]_
GNN                  ONE                 Outlier aware network embedding for attributed networks                                                 2019   [#Bandyopadhyay2019Outlier]_
===================  ==================  ======================================================================================================  =====  ========================================

**(ii) Utility functions** :

===================  ======================  =====================================================================================================================================================  ======================================================================================================================================
Type                 Name                    Function                                                                                                                                               Documentation
===================  ======================  =====================================================================================================================================================  ======================================================================================================================================
Metric               eval_roc_auc            ROC-AUC score for binary classification.                                                                                                               `eval_roc_auc <https://github.com/pygod-team/pygod>`_
Data                 gen_structure_outliers  Generating structural outliers                                                                                                                         `gen_structure_outliers <https://github.com/pygod-team/pygod>`_
Data                 gen_attribute_outliers  Generating attribute outliers                                                                                                                          `gen_attribute_outliers <https://github.com/pygod-team/pygod>`_
Data                 gen_combined_outliers   Generating combined outliers                                                                                                                           `gen_combined_outliers <https://github.com/pygod-team/pygod>`_
===================  ======================  =====================================================================================================================================================  ======================================================================================================================================


----


Quick Start for Outlier Detection with PyGOD
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`"examples/dominant_example.py" <https://github.com/pygod-team/pygod/blob/master/examples/dominant_example.py>`_
demonstrates the basic API of using the dominant detector. **It is noted that the API across all other algorithms are consistent/similar**.

More detailed instructions for running examples can be found in `examples directory <https://github.com/pygod-team/pygod/blob/master/examples/>`_.

#. Initialize a dominant detector, fit the model, and make the prediction.



#. Evaluate the prediction by ROC and Precision @ Rank n (p@n).


----

How to Contribute
^^^^^^^^^^^^^^^^^

You are welcome to contribute to this exciting project:

See `contribution guide <https://github.com/pygod-team/pygod/blob/master/contributing.md>`_ for more information.


----

PyGOD Team
^^^^^^^^^^

PyGOD is a great team effort by researchers from UIC, IIT, BUAA, ASU, and CMU.
Our core team members include:

`Kay Liu (UIC) <https://kayzliu.com/>`_,
`Yingtong Dou (UIC) <http://ytongdou.com/>`_,
`Yue Zhao (CMU) <https://www.andrew.cmu.edu/user/yuezhao2/>`_,
`Xueying Ding (CMU) <https://scholar.google.com/citations?user=U9CMsh0AAAAJ&hl=en>`_,
`Xiyang Hu (CMU) <https://www.andrew.cmu.edu/user/xiyanghu/>`_,
`Ruitong Zhang (BUAA) <https://github.com/pygod-team/pygod>`_,
`Kaize Ding (ASU) <https://www.public.asu.edu/~kding9/>`_,
`Canyu Chen (IIT) <https://github.com/pygod-team/pygod>`_,

Reach out us by submitting an issue report or email us at **<tba>add an email<tba>**

----

Reference
^^^^^^^^^

.. [#Bandyopadhyay2019Outlier] Bandyopadhyay, S., Lokesh, N. and Murty, M.N., 2019, July. Outlier aware network embedding for attributed networks. In Proceedings of the AAAI conference on artificial intelligence (AAAI).

.. [#Bandyopadhyay2020Outlier] Bandyopadhyay, S., Vivek, S.V. and Murty, M.N., 2020, January. Outlier resistant unsupervised deep architectures for attributed network embedding. In Proceedings of the International Conference on Web Search and Data Mining (WSDM).

.. [#Ding2019Deep] Ding, K., Li, J., Bhanushali, R. and Liu, H., 2019, May. Deep anomaly detection on attributed networks. In Proceedings of the SIAM International Conference on Data Mining (SDM).

.. [#Dou2021User] Dou, Y., Shu, K., Xia, C., Yu, P.S. and Sun, L., 2021, July. User preference-aware fake news detection. In Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval (pp. 2051-2055).

.. [#Fan2020AnomalyDAE] Fan, H., Zhang, F. and Li, Z., 2020, May. AnomalyDAE: Dual autoencoder for anomaly detection on attributed networks. In Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP).

.. [#Liu2020Alleviating] Liu, Z., Dou, Y., Yu, P.S., Deng, Y. and Peng, H., 2020, July. Alleviating the inconsistency problem of applying graph neural network to fraud detection. In Proceedings of the 43rd international ACM SIGIR conference on research and development in information retrieval (pp. 1569-1572).

.. [#Liu2021Anomaly] Liu, Y., Li, Z., Pan, S., Gong, C., Zhou, C. and Karypis, G., 2021. Anomaly detection on attributed networks via contrastive self-supervised learning. IEEE transactions on neural networks and learning systems (TNNLS).

.. [#Perini2020Quantifying] Perini, L., Vercruyssen, V., Davis, J. Quantifying the confidence of anomaly detectors in their example-wise predictions. In *Joint European Conference on Machine Learning and Knowledge Discovery in Databases (ECML-PKDD)*, 2020.

.. [#Wang2021One] Wang, X., Jin, B., Du, Y., Cui, P., Tan, Y. and Yang, Y., 2021. One-class graph neural networks for anomaly detection in attributed networks. Neural computing and applications.

.. [#Yuan2021Higher] Yuan, X., Zhou, N., Yu, S., Huang, H., Chen, Z. and Xia, F., 2021, December. Higher-order Structure Based Anomaly Detection on Attributed Networks. In 2021 IEEE International Conference on Big Data (Big Data).

.. [#Zhang2021FRAUDRE] Zhang, G., Wu, J., Yang, J., Beheshti, A., Xue, S., Zhou, C. and Sheng, Q.Z., 2021, December. FRAUDRE: Fraud Detection Dual-Resistant to Graph Inconsistency and Imbalance. In 2021 IEEE International Conference on Data Mining (ICDM).
