Contribute to PyGOD
===================

This guide will tell how to contribute to PyGOD at the beginning stage.
This guide may change subject to the development process.


Where to start?
---------------

You are welcome to contribute to this exciting project:

- Please first check Issue lists for "help wanted" tag and comment the one you are interested. We will assign the issue to you.

- Fork the **main branch** and add your improvement/modification/fix.

- Create a pull request to **dev branch** and follow the pull request template PR template

- Automatic tests will be triggered. Make sure all tests are passed. Please make sure all added modules are accompanied by proper test functions.

- To make sure the code has the same style and standard, please refer to gcnae.py and dominant.py, for example.


Coding styles
-------------


For python codes, we generally follow the `PEP8 style guide <https://www.python.org/dev/peps/pep-0008>`_.
The python comments follow `NumPy style python docstrings <https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html>`_.

We tweak it a little from the standard. For example, following variable names are accepted:

* ``i,j,k``: for loop variables
* ``u,v``: for representing nodes
* ``e``: for representing edges
* ``g``: for representing graph
* ``fn``: for representing functions
* ``n,m``: for representing sizes
* ``w,x,y``: for representing weight, input, output tensors
* ``_``: for unused variables


Development Environment
-----------------------

To prevent the problems induced by inconsistent versions of dependencies, following requirements are suggested.

- python>=3.8
- torch>=2.0.0
- torch_geometry>=2.3.0

Please follow the `installation guide <https://docs.pygod.org/en/latest/install.html>`_ for more details.
