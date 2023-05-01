# -*- coding: utf-8 -*-
"""Higher-order Structure based Anomaly Detection on Attributed
    Networks (GUIDE)"""
# Author: Kay Liu <zliu234@uic.edu>
# License: BSD 2 clause

import os
import warnings

import torch
import torch.nn.functional as F

from . import DeepDetector
from ..nn import GUIDEBase


class GUIDE(DeepDetector):
    """
    GUIDE (Higher-order Structure based Anomaly Detection on Attributed
    Networks) is an anomaly detector consisting of an attribute graph
    convolutional autoencoder, and a structure graph attentive
    autoencoder (not the same as the graph attention networks). Instead
    of the adjacency matrix, node motif degree is used as input of
    structure autoencoder. The reconstruction mean square error of the
    autoencoders are defined as structure anomaly score and attribute
    anomaly score, respectively.

    Note: The calculation of node motif degree in preprocessing has
    high time complexity. It may take longer than you expect.

    See :cite:`yuan2021higher` for details.

    Parameters
    ----------
    hid_x :  int, optional
        Hidden dimension for attribute autoencoder. Default: ``32``.
    hid_s :  int, optional
        Hidden dimension for structure autoencoder. Default: ``4``.
    num_layers : int, optional
        Total number of layers in autoencoders. Default: ``4``.
    dropout : float, optional
        Dropout rate. Default: ``0.``.
    weight_decay : float, optional
        Weight decay (L2 penalty). Default: ``0.``.
    act : callable activation function or None, optional
        Activation function if not None.
        Default: ``torch.nn.functional.relu``.
    alpha : float, optional
        Loss balance weight for attribute and structure. ``None`` for
        balancing by standard deviation. Default: ``None``.
    contamination : float, optional
        Valid in (0., 0.5). The proportion of outliers in the data set.
        Used when fitting to define the threshold on the decision
        function. Default: ``0.1``.
    lr : float, optional
        Learning rate. Default: ``0.004``.
    epoch : int, optional
        Maximum number of training epoch. Default: ``10``.
    gpu : int, optional
        GPU Index, -1 for using CPU. Default: ``0``.
    batch_size : int, optional
        Minibatch size, 0 for full batch training. Default: ``0``.
    num_neigh : int, optional
        Number of neighbors in sampling, -1 for all neighbors.
        Default: ``-1``.
    graphlet_size : int, optional
        The maximum graphlet size used to compute structure input.
        Default: ``4``.
    selected_motif : bool, optional
        Use selected motifs which are defined in the original paper.
        Default: ``True``.
    cache_dir : str, option
        The directory for the node motif degree caching.
        Default: ``None``.
    verbose : bool, optional
        Verbosity mode. Turn on to print out log information.
        Default: ``False``.

    Examples
    --------
    >>> from pygod.detectors import GUIDE
    >>> model = GUIDE()
    >>> model.fit(data) # PyG graph data object
    >>> prediction = model.predict(data)
    """

    def __init__(self,
                 hid_x=64,
                 hid_s=4,
                 num_layers=4,
                 dropout=0.,
                 weight_decay=0.,
                 act=F.relu,
                 backbone=None,
                 alpha=0.5,
                 contamination=0.1,
                 lr=0.004,
                 epoch=100,
                 gpu=-1,
                 batch_size=0,
                 num_neigh=-1,
                 graphlet_size=4,
                 selected_motif=True,
                 cache_dir=None,
                 verbose=0,
                 **kwargs):

        if backbone is not None:
            warnings.warn("Backbone and num_layers are not used in AnomalyDAE")

        super(GUIDE, self).__init__(hid_dim=(hid_x, hid_s),
                                    num_layers=num_layers,
                                    dropout=dropout,
                                    weight_decay=weight_decay,
                                    act=act,
                                    backbone=backbone,
                                    contamination=contamination,
                                    lr=lr,
                                    epoch=epoch,
                                    gpu=gpu,
                                    batch_size=batch_size,
                                    num_neigh=num_neigh,
                                    verbose=verbose,
                                    **kwargs)

        self.dim_s = None
        self.alpha = alpha
        self.graphlet_size = graphlet_size
        if selected_motif:
            assert self.graphlet_size == 4, \
                "Graphlet size is fixed when using selected motif"
        self.selected_motif = selected_motif
        self.verbose = verbose

        if cache_dir is None:
            cache_dir = os.path.join(os.path.expanduser('~'), '.pygod')
        self.cache_dir = cache_dir

    def process_graph(self, data):

        data.s = GUIDEBase.calc_gdd(data,
                                    self.cache_dir,
                                    graphlet_size=self.graphlet_size,
                                    selected_motif=self.selected_motif)
        self.dim_s = data.s.shape[1]

    def init_model(self, **kwargs):

        return GUIDEBase(dim_x=self.in_dim,
                         dim_s=self.dim_s,
                         hid_x=self.hid_dim[0],
                         hid_s=self.hid_dim[1],
                         num_layers=self.num_layers,
                         dropout=self.dropout,
                         act=self.act,
                         **kwargs).to(self.device)

    def forward_model(self, data):

        batch_size = data.batch_size

        x = data.x.to(self.device)
        s = data.s.to(self.device)
        edge_index = data.edge_index.to(self.device)

        x_, s_ = self.model(x, s, edge_index)

        score = self.model.loss_func(x[:batch_size],
                                     x_[:batch_size],
                                     s[:batch_size],
                                     s_[:batch_size],
                                     self.alpha)

        loss = torch.mean(score)

        return loss, score.detach().cpu()
