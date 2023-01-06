Installation
============


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

* Python 3.7+
* numpy>=1.19.4
* scikit-learn>=0.22.1
* scipy>=1.5.2
* setuptools>=50.3.1.post20201107


**Note on PyG and PyTorch Installation**\ :
PyGOD depends on `PyTorch Geometric (PyG) <https://www.pyg.org/>`_, `PyTorch <https://pytorch.org/>`_, and `networkx <https://networkx.org/>`_.
To streamline the installation, PyGOD does **NOT** install these libraries for you.
Please install them from the above links for running PyGOD:

* torch>=1.10
* pytorch_geometric>=2.0.3
* networkx>=2.6.3