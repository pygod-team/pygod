API CheatSheet
==============

The following APIs are applicable for all detectors for easy use.

* :func:`pygod.detectors.Detector.fit`: Fit detector. y is ignored in unsupervised methods.
* :func:`pygod.detectors.Detector.decision_function`: Predict raw anomaly scores of PyG Graph G using the fitted detector

Key Attributes of a fitted model:

* :attr:`pygod.detectors.Detector.decision_score_`: The outlier scores of the training data. The higher, the more abnormal.
  Outliers tend to have higher scores.
* :attr:`pygod.detectors.Detector.label_`: The binary labels of the training data. 0 stands for inliers and 1 for outliers/anomalies.

For the inductive setting:

* :func:`pygod.detectors.BaseDetector.predict`: Predict if a particular sample is an outlier or not using the fitted detector.

**Input of PyGOD**: Please pass in a `PyTorch Geometric (PyG) <https://www.pyg.org/>`_ data object.
See `PyG data processing examples <https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html#data-handling-of-graphs>`_.

* :func:`pygod.models.base.BaseDetector.process_graph` (you do not need to call this explicitly): Process the raw PyG data object into a tuple of sub data objects needed for the underlying model.


See base class definition below:

pygod.detectors.base module
---------------------------

.. automodule:: pygod.detectors.base
    :members:
    :undoc-members:
    :show-inheritance:
    :inherited-members: