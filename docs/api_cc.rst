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


See base class definition below:

pygod.models.base module
------------------------

.. automodule:: pygod.models.base
    :members:
    :undoc-members:
    :show-inheritance:
    :inherited-members: