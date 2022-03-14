Contribute to PyGOD
===================

This guide will tell how to contribute to pyGOD at the beginning stage. This guide may change subject to the development process. Part of this guide is  from [DGL docs](https://docs.dgl.ai/contribute.html).

Coding styles
-------------

For python codes, we generally follow the [PEP8 style guide](https://www.python.org/dev/peps/pep-0008). The python comments follow [NumPy style python docstrings](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html).

We tweak it a little bit from the standard. For example, following variable names are accepted:

* ``i,j,k``: for loop variables
* ``u,v``: for representing nodes
* ``e``: for representing edges
* ``g``: for representing graph
* ``fn``: for representing functions
* ``n,m``: for representing sizes
* ``w,x,y``: for representing weight, input, output tensors
* ``_``: for unused variables

Development Environment
-------------

To prevent the problems induced by inconsistent versions of dependencies, following requirements are suggested.

python>=3.7
torch>=1.10.1
torch_geometry>=2.0.4

Contributing New Models
-----------------------------------

To contribute a new model , simply

1. Make a directory with the name of your model (say ``awesome-gnn``) within the directory ``pygod/models``.
2. Populate it with your work, along with a README.  Make a pull request once you are done.  Your README should contain the instructions for running your program.

3. Commit and push to the master branch (only at the very beginning stage).
