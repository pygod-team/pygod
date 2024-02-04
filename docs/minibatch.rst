Efficient GPU Training
======================

To train deep detectors efficiently, we usually use
`CUDA <https://developer.nvidia.com/cuda-toolkit>`_ to accelerate
the detector training on GPU. PyGOD provides ``gpu`` parameter for
``DeepDetector``. During initialization, we can set ``gpu`` to the index
of the GPU that is available. By default, ``gpu=-1``, which means train
the detector on CPU. Here is an example of initialize ``DOMINANT`` with
the first GPU (index of ``0``):

.. code:: python

    DOMINANT(gpu=0)

However, training deep detectors on large-scale graphs can be
memory-intensive, especially on the detectors relying on adjacency
matrix reconstruction. At this time, full batch training may result in
out-of-memory (OOM) error. As such, we divide the large graph into
minibatches, and train the detector on each batch. PyGOD provides
``batch_size`` parameter for ``DeepDetector``, where users are able to
adjust the size of each batch for various GPU memory. We recommend users
setting ``batch_size`` to largest value that will not cause OOM. For
instance, we would like to train ``DOMINANT`` with the batches of 64
nodes:

.. code:: python

    DOMINANT(gpu=0, batch_size=64)

Unlike other data modalities, the output of each node in graphs rely on
its neighbors. In PyGOD implementation, we adopt the data loader
``torch_geometric.loader.NeighborLoader`` in PyG to load both the center
nodes and the neighbor nodes for minibatches. But the computation on
neighbor nodes will lead to significant overhead and reduce the
efficiency in the detector training. Thus, we neighbor sampling is
crucial to reduce the overhead. PyGOD provides ``num_neigh`` parameter
for ``DeepDetector``. We can specify how many neighbors are sampled at
each layer of the detector. The default value of ``num_neigh`` is
``-1``, indicating sample all neighbors of the center node. If we want
to sample 5 neighbors at each layer, we can initialize ``DOMINANT``
like:

.. code:: python

    DOMINANT(gpu=0, batch_size=64, num_neigh=5)

We can also sample different number of neighbors at each layer by
setting ``num_neigh`` as a list, but the length of the list has to match
with the number of layers ``num_layers``:

.. code:: python

    DOMINANT(gpu=0, batch_size=64, num_layers=2, num_neigh=[5, 3])

To learn more, read PyG's tutorial on
`Scaling GNNs via Neighbor Sampling <https://pytorch-geometric.readthedocs.io/en/latest/tutorial/neighbor_loader.html>`_.
