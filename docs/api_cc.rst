API CheatSheet
==============

The following APIs are applicable for all detectors for easy use.

* :func:`pygod.detector.Detector.fit`: Fit detector.
* :func:`pygod.detector.Detector.decision_function`: Predict raw anomaly scores of PyG data using the fitted detector

Key Attributes of a fitted detector:

* :attr:`pygod.detector.Detector.decision_score_`: The outlier scores of the input data. Outliers tend to have higher scores.
* :attr:`pygod.detector.Detector.label_`: The binary labels of the input data. 0 stands for inliers and 1 for outliers.

For the inductive setting:

* :func:`pygod.detector.Detector.predict`: Predict if a particular sample is an outlier or not using the fitted detector.

**Input of PyGOD**: Please pass in a `PyTorch Geometric (PyG) <https://www.pyg.org/>`_ data object.
See `PyG data processing examples <https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html#data-handling-of-graphs>`_.


Base Detector
-------------

``Detector`` is the abstract class for all detectors:

.. autoclass:: pygod.detector.Detector
    :members:
    :undoc-members:
    :show-inheritance:
    :inherited-members:

Deep Detector
-------------

By inherit ``Detector`` class, we also provide base deep detector class for deep learning based detectors to ease the implementation.

.. autoclass:: pygod.detector.DeepDetector
    :members: emb, init_model, forward_model, process_graph
    :undoc-members: fit, decision_function, predict
