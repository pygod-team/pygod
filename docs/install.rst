Installation
============

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

**Required Dependencies**\ :


* Python 3.6 +
* numpy>=1.13
* scipy>=0.19.1
* scikit_learn>=0.20.0
* to be finished


**Note and PyG and PyTorch Installation**\ :
PyGOD depends on `PyTorch Geometric (PyG) <https://www.pyg.org/>`_
and `PyTorch <https://pytorch.org/>`_. To streamline the installation,
PyGOD does **NOT** install these libraries for you. Please install them
from the above links for running PyGOD:

* torch>=?
* pytorch_geometric>=?