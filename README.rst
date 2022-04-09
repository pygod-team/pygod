.. image:: https://raw.githubusercontent.com/pygod-team/pygod/main/docs/pygod_logo.png
   :width: 1050
   :alt: PyGOD Logo
   :align: center

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

.. image:: https://github.com/pygod-team/pygod/actions/workflows/testing.yml/badge.svg
   :target: https://github.com/pygod-team/pygod/actions/workflows/testing.yml
   :alt: testing

.. image:: https://coveralls.io/repos/github/pygod-team/pygod/badge.svg?branch=main
   :target: https://coveralls.io/github/pygod-team/pygod?branch=main
   :alt: Coverage Status

.. image:: https://img.shields.io/github/license/pygod-team/pygod.svg
   :target: https://github.com/pygod-team/pygod/blob/master/LICENSE
   :alt: License


-----

PyGOD is a **Python library** for **graph outlier detection** (anomaly detection).
This exciting yet challenging field has many key applications, e.g., detecting
suspicious activities in social networks [#Dou2020Enhancing]_  and security systems [#Cai2021Structural]_.

PyGOD includes more than **10** latest graph-based detection algorithms,
such as DOMINANT (SDM'19) and GUIDE (BigData'21).
For consistently and accessibility, PyGOD is developed on top of `PyTorch Geometric (PyG) <https://www.pyg.org/>`_
and `PyTorch <https://pytorch.org/>`_, and follows the API design of `PyOD <https://github.com/yzhao062/pyod>`_.
See examples below for detecting outliers with PyGOD in 5 lines!

**PyGOD** is under actively developed and will be updated frequently!
Please **star**, **watch**, and **fork**.


**PyGOD is featured for**:

* **Unified APIs, detailed documentation, and interactive examples** across various graph-based algorithms.
* **Comprehensive coverage** of more than 10 latest graph outlier detectors.
* **Full support of detections at multiple levels**, such as node-, edge-, and graph-level tasks (WIP).
* **Streamline data processing with PyG**--fully compatible with PyG data objects.

**Outlier Detection Using PyGOD with 5 Lines of Code**\ :

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
* scipy>=1.5.2
* pandas>=1.1.3
* setuptools>=50.3.1.post20201107


**Note on PyG and PyTorch Installation**\ :
PyGOD depends on `PyTorch Geometric (PyG) <https://www.pyg.org/>`_, `PyTorch <https://pytorch.org/>`_, and `networkx <https://networkx.org/>`_.
To streamline the installation, PyGOD does **NOT** install these libraries for you.
Please install them from the above links for running PyGOD:

* torch>=1.10
* pytorch_geometric>=2.0.3
* networkx>=2.6.3

----


API Cheatsheet & Reference
^^^^^^^^^^^^^^^^^^^^^^^^^^

Full API Reference: (https://docs.pygod.org). API cheatsheet for all detectors:


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

PyGOD toolkit consists of two major functional groups:

**(i) Node-level detection** :

===================  ===================  ==================  ======================================================================================================  =====  ========================================
Type                 Backbone             Abbr                Algorithm                                                                                               Year   Ref
===================  ===================  ==================  ======================================================================================================  =====  ========================================
Unsupervised         MLP                  MLPAE               Anomaly Detection Using Autoencoders with Nonlinear Dimensionality Reduction                            2014   [#Sakurada2014Anomaly]_
Unsupervised         GNN                  GCNAE               Variational Graph Auto-Encoders                                                                         2016   [#Kipf2016Variational]_
Unsupervised         MF                   ONE                 Outlier aware network embedding for attributed networks                                                 2019   [#Bandyopadhyay2019Outlier]_
Unsupervised         GNN                  DOMINANT            Deep anomaly detection on attributed networks                                                           2019   [#Ding2019Deep]_
Unsupervised         GNN                  DONE                Outlier Resistant Unsupervised Deep Architectures for Attributed Network Embedding                      2020   [#Bandyopadhyay2020Outlier]_
Unsupervised         GNN                  AdONE               Outlier Resistant Unsupervised Deep Architectures for Attributed Network Embedding                      2020   [#Bandyopadhyay2020Outlier]_
Unsupervised         GNN                  AnomalyDAE          AnomalyDAE: Dual autoencoder for anomaly detection on attributed networks                               2020   [#Fan2020AnomalyDAE]_
Unsupervised         GAN                  GAAN                Generative Adversarial Attributed Network Anomaly Detection                                             2020   [#Chen2020Generative]_
Unsupervised         GNN                  OCGNN               One-Class Graph Neural Networks for Anomaly Detection in Attributed Networks                            2021   [#Wang2021One]_
Unsupervised         GNN                  GUIDE               Higher-order Structure Based Anomaly Detection on Attributed Networks                                   2021   [#Yuan2021Higher]_
===================  ===================  ==================  ======================================================================================================  =====  ========================================

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


Quick Start for Outlier Detection with PyGOD
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`"A Blitz Introduction" <https://docs.pygod.org/en/latest/tutorials/intro.html>`_
demonstrates the basic API of PyGOD using the dominant detector. **It is noted that the API across all other algorithms are consistent/similar**.

----

How to Contribute
^^^^^^^^^^^^^^^^^

You are welcome to contribute to this exciting project:

See `contribution guide <https://github.com/pygod-team/pygod/blob/master/contributing.RST>`_ for more information.


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

Reach out us by submitting an issue report or send an email to dev@pygod.org.

----

Reference
^^^^^^^^^

.. [#Dou2020Enhancing] Dou, Y., Liu, Z., Sun, L., Deng, Y., Peng, H. and Yu, P.S., 2020, October. Enhancing graph neural network-based fraud detectors against camouflaged fraudsters. In Proceedings of the 29th ACM International Conference on Information & Knowledge Management (pp. 315-324).

.. [#Cai2021Structural] Cai, L., Chen, Z., Luo, C., Gui, J., Ni, J., Li, D. and Chen, H., 2021, October. Structural temporal graph neural networks for anomaly detection in dynamic graphs. In Proceedings of the 30th ACM International Conference on Information & Knowledge Management (pp. 3747-3756).

.. [#Perini2020Quantifying] Perini, L., Vercruyssen, V., Davis, J. Quantifying the confidence of anomaly detectors in their example-wise predictions. In *Joint European Conference on Machine Learning and Knowledge Discovery in Databases (ECML-PKDD)*, 2020.

.. [#Sakurada2014Anomaly] Sakurada, M. and Yairi, T., 2014, December. Anomaly detection using autoencoders with nonlinear dimensionality reduction. In Proceedings of the MLSDA 2014 2nd workshop on machine learning for sensory data analysis (pp. 4-11).

.. [#Kipf2016Variational] Kipf, T.N. and Welling, M., 2016. Variational graph auto-encoders. arXiv preprint arXiv:1611.07308.

.. [#Bandyopadhyay2019Outlier] Bandyopadhyay, S., Lokesh, N. and Murty, M.N., 2019, July. Outlier aware network embedding for attributed networks. In Proceedings of the AAAI conference on artificial intelligence (AAAI).

.. [#Ding2019Deep] Ding, K., Li, J., Bhanushali, R. and Liu, H., 2019, May. Deep anomaly detection on attributed networks. In Proceedings of the SIAM International Conference on Data Mining (SDM).

.. [#Bandyopadhyay2020Outlier] Bandyopadhyay, S., Vivek, S.V. and Murty, M.N., 2020, January. Outlier resistant unsupervised deep architectures for attributed network embedding. In Proceedings of the International Conference on Web Search and Data Mining (WSDM).

.. [#Fan2020AnomalyDAE] Fan, H., Zhang, F. and Li, Z., 2020, May. AnomalyDAE: Dual autoencoder for anomaly detection on attributed networks. In Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP).

.. [#Chen2020Generative] Chen, Z., Liu, B., Wang, M., Dai, P., Lv, J. and Bo, L., 2020, October. Generative adversarial attributed network anomaly detection. In Proceedings of the 29th ACM International Conference on Information & Knowledge Management (pp. 1989-1992).

.. [#Wang2021One] Wang, X., Jin, B., Du, Y., Cui, P., Tan, Y. and Yang, Y., 2021. One-class graph neural networks for anomaly detection in attributed networks. Neural computing and applications.

.. [#Yuan2021Higher] Yuan, X., Zhou, N., Yu, S., Huang, H., Chen, Z. and Xia, F., 2021, December. Higher-order Structure Based Anomaly Detection on Attributed Networks. In 2021 IEEE International Conference on Big Data (Big Data).
