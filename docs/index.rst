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
in financial fraud detection and fake news detection.

PyGOD includes more than **10** latest graph-based detection algorithms,
such as Dominant (SDM'19) and coLA (TNNLS'21).
For consistently and accessibility, PyGOD is developed on top of `PyTorch Geometric (PyG) <https://www.pyg.org/>`_
and `PyTorch <https://pytorch.org/>`_. See examples below for detecting anomalies with GNN in 5 lines!


PyGOD is featured for:

* **Unified APIs, detailed documentation, and interactive examples** across various graph-based algorithms.
* **Comprehensive coverage of more than 10 algorithms**\ , including the latest graph neural networks (GNNs).
* **Full support of various levels of detection**, such as node-, edge-, and graph-level tasks (WIP).
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



.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Getting Started

   install

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: API References

   pygod.models

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

