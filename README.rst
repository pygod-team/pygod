.. image:: https://raw.githubusercontent.com/pygod-team/pygod/main/docs/pygod_logo.png
   :width: 1050
   :alt: PyGOD Logo
   :align: center

|badge_pypi| |badge_docs| |badge_stars| |badge_forks| |badge_downloads| |badge_testing| |badge_coverage| |badge_license|

.. |badge_pypi| image:: https://img.shields.io/pypi/v/pygod.svg?color=brightgreen
   :target: https://pypi.org/project/pygod/
   :alt: PyPI version

.. |badge_docs| image:: https://readthedocs.org/projects/py-god/badge/?version=latest
   :target: https://docs.pygod.org/en/latest/?badge=latest
   :alt: Documentation status

.. |badge_stars| image:: https://img.shields.io/github/stars/pygod-team/pygod?style=flat
   :target: https://github.com/pygod-team/pygod/stargazers
   :alt: GitHub stars

.. |badge_forks| image:: https://img.shields.io/github/forks/pygod-team/pygod?style=flat
   :target: https://github.com/pygod-team/pygod/network
   :alt: GitHub forks

.. |badge_downloads| image:: https://static.pepy.tech/personalized-badge/pygod?period=total&units=international_system&left_color=grey&right_color=blue&left_text=Downloads
   :target: https://pepy.tech/project/pygod
   :alt: PyPI downloads

.. |badge_testing| image:: https://github.com/pygod-team/pygod/actions/workflows/testing.yml/badge.svg
   :target: https://github.com/pygod-team/pygod/actions/workflows/testing.yml
   :alt: testing

.. |badge_coverage| image:: https://coveralls.io/repos/github/pygod-team/pygod/badge.svg?branch=main
   :target: https://coveralls.io/github/pygod-team/pygod?branch=main
   :alt: Coverage Status

.. |badge_license| image:: https://img.shields.io/github/license/pygod-team/pygod.svg
   :target: https://github.com/pygod-team/pygod/blob/master/LICENSE
   :alt: License


-----

PyGOD is a **Python library** for **graph outlier detection** (anomaly detection).
This exciting yet challenging field has many key applications, e.g., detecting
suspicious activities in social networks [#Dou2020Enhancing]_  and security systems [#Cai2021Structural]_.

PyGOD includes **10+** graph outlier detection algorithms.
For consistency and accessibility, PyGOD is developed on top of `PyTorch Geometric (PyG) <https://www.pyg.org/>`_
and `PyTorch <https://pytorch.org/>`_, and follows the API design of `PyOD <https://github.com/yzhao062/pyod>`_.
See examples below for detecting outliers with PyGOD in 5 lines!


**PyGOD is featured for**:

* **Unified APIs, detailed documentation, and interactive examples** across various graph-based algorithms.
* **Comprehensive coverage** of 10+ graph outlier detectors.
* **Full support of detections at multiple levels**, such as node-, edge-, and graph-level tasks.
* **Scalable design for processing large graphs** via mini-batch and sampling.
* **Streamline data processing with PyG**--fully compatible with PyG data objects.

**Outlier Detection Using PyGOD with 5 Lines of Code**\ :

.. code-block:: python


    # train a dominant detector
    from pygod.detector import DOMINANT

    model = DOMINANT(num_layers=4, epoch=20)  # hyperparameters can be set here
    model.fit(train_data)  # input data is a PyG data object

    # get outlier scores on the training data (transductive setting)
    score = model.decision_score_

    # predict labels and scores on the testing data (inductive setting)
    pred, score = model.predict(test_data, return_score=True)


**Citing PyGOD**\ :

Our `software paper <https://jmlr.org/papers/v25/23-0963.html>`_ and `benchmark paper <https://proceedings.neurips.cc/paper_files/paper/2022/hash/acc1ec4a9c780006c9aafd595104816b-Abstract-Datasets_and_Benchmarks.html>`_ are publicly available.
If you use PyGOD or BOND in a scientific publication, we would appreciate citations to the following papers::

    @article{JMLR:v25:23-0963,
      author  = {Kay Liu and Yingtong Dou and Xueying Ding and Xiyang Hu and Ruitong Zhang and Hao Peng and Lichao Sun and Philip S. Yu},
      title   = {{PyGOD}: A {Python} Library for Graph Outlier Detection},
      journal = {Journal of Machine Learning Research},
      year    = {2024},
      volume  = {25},
      number  = {141},
      pages   = {1--9},
      url     = {http://jmlr.org/papers/v25/23-0963.html}
    }
    @inproceedings{NEURIPS2022_acc1ec4a,
     author = {Liu, Kay and Dou, Yingtong and Zhao, Yue and Ding, Xueying and Hu, Xiyang and Zhang, Ruitong and Ding, Kaize and Chen, Canyu and Peng, Hao and Shu, Kai and Sun, Lichao and Li, Jundong and Chen, George H and Jia, Zhihao and Yu, Philip S},
     booktitle = {Advances in Neural Information Processing Systems},
     editor = {S. Koyejo and S. Mohamed and A. Agarwal and D. Belgrave and K. Cho and A. Oh},
     pages = {27021--27035},
     publisher = {Curran Associates, Inc.},
     title = {{BOND}: Benchmarking Unsupervised Outlier Node Detection on Static Attributed Graphs},
     url = {https://proceedings.neurips.cc/paper_files/paper/2022/file/acc1ec4a9c780006c9aafd595104816b-Paper-Datasets_and_Benchmarks.pdf},
     volume = {35},
     year = {2022}
    }

or::

    Liu, K., Dou, Y., Ding, X., Hu, X., Zhang, R., Peng, H., Sun, L. and Yu, P.S., 2024. PyGOD: A Python library for graph outlier detection. Journal of Machine Learning Research, 25(141), pp.1-9.
    Liu, K., Dou, Y., Zhao, Y., Ding, X., Hu, X., Zhang, R., Ding, K., Chen, C., Peng, H., Shu, K., Sun, L., Li, J., Chen, G.H., Jia, Z., and Yu, P.S., 2022. BOND: Benchmarking unsupervised outlier node detection on static attributed graphs. Advances in Neural Information Processing Systems, 35, pp.27021-27035.

----

Installation
^^^^^^^^^^^^

**Note on PyG and PyTorch Installation**\ :
PyGOD depends on `torch <https://https://pytorch.org/get-started/locally/>`_ and `torch_geometric (including its optional dependencies) <https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html#>`_.
To streamline the installation, PyGOD does **NOT** install these libraries for you.
Please install them from the above links for running PyGOD:

* torch>=2.0.0
* torch_geometric>=2.3.0

It is recommended to use **pip** for installation.
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

* python>=3.8
* numpy>=1.24.3
* scikit-learn>=1.2.2
* scipy>=1.10.1
* networkx>=3.1


----


Quick Start for Outlier Detection with PyGOD
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`"A Blitz Introduction" <https://docs.pygod.org/en/latest/tutorials/1_intro.html#sphx-glr-tutorials-1-intro-py>`_
demonstrates the basic API of PyGOD using the DOMINANT detector. **It is noted that the API across all other algorithms are consistent/similar**.


----


API Cheatsheet & Reference
^^^^^^^^^^^^^^^^^^^^^^^^^^

Full API Reference: (https://docs.pygod.org). API cheatsheet for all detectors:

* **fit(data)**\ : Fit the detector with train data.
* **predict(data)**\ : Predict on test data (train data if not provided) using the fitted detector.

Key Attributes of a fitted detector:

* **decision_score_**\ : The outlier scores of the input data. Outliers tend to have higher scores.
* **label_**\ : The binary labels of the input data. 0 stands for inliers and 1 for outliers.
* **threshold_**\ : The determined threshold for binary classification. Scores above the threshold are outliers.

**Input of PyGOD**: Please pass in a `PyG Data object <https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.Data.html#torch_geometric.data.Data>`_.
See `PyG data processing examples <https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html#data-handling-of-graphs>`_.


Implemented Algorithms
^^^^^^^^^^^^^^^^^^^^^^

==================  =====  ===========  ===========  ========================================
Abbr                Year   Backbone     Sampling      Ref
==================  =====  ===========  ===========  ========================================
SCAN                2007   Clustering   No           [#Xu2007Scan]_
GAE                 2016   GNN+AE       Yes          [#Kipf2016Variational]_
Radar               2017   MF           No           [#Li2017Radar]_
ANOMALOUS           2018   MF           No           [#Peng2018Anomalous]_
ONE                 2019   MF           No           [#Bandyopadhyay2019Outlier]_
DOMINANT            2019   GNN+AE       Yes          [#Ding2019Deep]_
DONE                2020   MLP+AE       Yes          [#Bandyopadhyay2020Outlier]_
AdONE               2020   MLP+AE       Yes          [#Bandyopadhyay2020Outlier]_
AnomalyDAE          2020   GNN+AE       Yes          [#Fan2020AnomalyDAE]_
GAAN                2020   GAN          Yes          [#Chen2020Generative]_
DMGD                2020   GNN+AE       Yes          [#Bandyopadhyay2020Integrating]_
OCGNN               2021   GNN          Yes          [#Wang2021One]_
CoLA                2021   GNN+AE+SSL   Yes          [#Liu2021Anomaly]_
GUIDE               2021   GNN+AE       Yes          [#Yuan2021Higher]_
CONAD               2022   GNN+AE+SSL   Yes          [#Xu2022Contrastive]_
GADNR               2024   GNN+AE       Yes          [#Roy2024Gadnr]_
CARD                2024   GNN+SSL+AE   Yes          [#Wang2024Card]_
==================  =====  ===========  ===========  ========================================


----

How to Contribute
^^^^^^^^^^^^^^^^^

You are welcome to contribute to this exciting project:

See `contribution guide <https://github.com/pygod-team/pygod/blob/main/CONTRIBUTING.rst>`_ for more information.


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

.. [#Dou2020Enhancing] Dou, Y., Liu, Z., Sun, L., Deng, Y., Peng, H. and Yu, P.S., 2020, October. Enhancing graph neural network-based fraud detectors against camouflaged fraudsters. In Proceedings of the 29th ACM International Conference on Information & Knowledge Management (CIKM).

.. [#Cai2021Structural] Cai, L., Chen, Z., Luo, C., Gui, J., Ni, J., Li, D. and Chen, H., 2021, October. Structural temporal graph neural networks for anomaly detection in dynamic graphs. In Proceedings of the 30th ACM International Conference on Information & Knowledge Management (CIKM).

.. [#Xu2007Scan] Xu, X., Yuruk, N., Feng, Z. and Schweiger, T.A., 2007, August. Scan: a structural clustering algorithm for networks. In Proceedings of the 13th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD).

.. [#Kipf2016Variational] Kipf, T.N. and Welling, M., 2016. Variational graph auto-encoders. arXiv preprint arXiv:1611.07308.

.. [#Li2017Radar] Li, J., Dani, H., Hu, X. and Liu, H., 2017, August. Radar: Residual Analysis for Anomaly Detection in Attributed Networks. In Proceedings of the Twenty-Sixth International Joint Conference on Artificial Intelligence (IJCAI).

.. [#Peng2018Anomalous] Peng, Z., Luo, M., Li, J., Liu, H. and Zheng, Q., 2018, July. ANOMALOUS: A Joint Modeling Approach for Anomaly Detection on Attributed Networks. In Proceedings of the Twenty-Seventh International Joint Conference on Artificial Intelligence (IJCAI).

.. [#Bandyopadhyay2019Outlier] Bandyopadhyay, S., Lokesh, N. and Murty, M.N., 2019, July. Outlier aware network embedding for attributed networks. In Proceedings of the AAAI conference on artificial intelligence (AAAI).

.. [#Ding2019Deep] Ding, K., Li, J., Bhanushali, R. and Liu, H., 2019, May. Deep anomaly detection on attributed networks. In Proceedings of the SIAM International Conference on Data Mining (SDM).

.. [#Bandyopadhyay2020Outlier] Bandyopadhyay, S., Vivek, S.V. and Murty, M.N., 2020, January. Outlier resistant unsupervised deep architectures for attributed network embedding. In Proceedings of the International Conference on Web Search and Data Mining (WSDM).

.. [#Fan2020AnomalyDAE] Fan, H., Zhang, F. and Li, Z., 2020, May. AnomalyDAE: Dual autoencoder for anomaly detection on attributed networks. In Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP).

.. [#Chen2020Generative] Chen, Z., Liu, B., Wang, M., Dai, P., Lv, J. and Bo, L., 2020, October. Generative adversarial attributed network anomaly detection. In Proceedings of the 29th ACM International Conference on Information & Knowledge Management (CIKM).

.. [#Bandyopadhyay2020Integrating] Bandyopadhyay, S., Vishal Vivek, S. and Murty, M.N., 2020. Integrating network embedding and community outlier detection via multiclass graph description. Frontiers in Artificial Intelligence and Applications, (FAIA).

.. [#Wang2021One] Wang, X., Jin, B., Du, Y., Cui, P., Tan, Y. and Yang, Y., 2021. One-class graph neural networks for anomaly detection in attributed networks. Neural computing and applications.

.. [#Liu2021Anomaly] Liu, Y., Li, Z., Pan, S., Gong, C., Zhou, C. and Karypis, G., 2021. Anomaly detection on attributed networks via contrastive self-supervised learning. IEEE transactions on neural networks and learning systems (TNNLS).

.. [#Yuan2021Higher] Yuan, X., Zhou, N., Yu, S., Huang, H., Chen, Z. and Xia, F., 2021, December. Higher-order Structure Based Anomaly Detection on Attributed Networks. In 2021 IEEE International Conference on Big Data (Big Data).

.. [#Xu2022Contrastive] Xu, Z., Huang, X., Zhao, Y., Dong, Y., and Li, J., 2022. Contrastive Attributed Network Anomaly Detection with Data Augmentation. In Proceedings of the 26th Pacific-Asia Conference on Knowledge Discovery and Data Mining (PAKDD).

.. [#Roy2024Gadnr] Roy, A., Shu, J., Li, J., Yang, C., Elshocht, O., Smeets, J. and Li, P., 2024. GAD-NR: Graph Anomaly Detection via Neighborhood Reconstruction. In Proceedings of the 17th ACM International Conference on Web Search and Data Mining (WSDM).

.. [#Wang2024Card] Wang Y., Wang X., He C., Chen X., Luo Z., Duan L., Zuo J., 2024. Community-Guided Contrastive Learning with Anomaly-Aware Reconstruction for Anomaly Detection on Attributed Networks. Database Systems for Advanced Applications (DASFAA).