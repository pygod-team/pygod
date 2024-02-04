API CheatSheet
==============

The following APIs are applicable for all detectors for easy use.

* :func:`pygod.detector.Detector.fit`: Fit the detector with train data.
* :func:`pygod.detector.Detector.predict`: Predict on test data (train data if not provided) using the fitted detector.

Key Attributes of a fitted detector:

* :attr:`pygod.detector.Detector.decision_score_`: The outlier scores of the input data. Outliers tend to have higher scores.
* :attr:`pygod.detector.Detector.label_`: The binary labels of the input data. 0 stands for inliers and 1 for outliers.
* :attr:`pygod.detector.Detector.threshold_` : The determined threshold for binary classification. Scores above the threshold are outliers.

**Input of PyGOD**: Please pass in a `PyG Data object <https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.Data.html#torch_geometric.data.Data>`_.
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
