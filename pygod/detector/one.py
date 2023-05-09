# -*- coding: utf-8 -*-
"""Outlier Aware Network Embedding for Attributed Networks (ONE)
"""
# Author: Xiyang Hu <xiyanghu@cmu.edu>, Kay Liu <zliu234@uic.edu>
# License: BSD 2 clause

import time
import torch
from torch_geometric.utils import to_dense_adj

from . import Detector
from ..utils import logger, validate_device


class ONE(Detector):
    """
    Outlier Aware Network Embedding for Attributed Networks

    See :cite:`bandyopadhyay2019outlier` for details.

    Parameters
    ----------
    hid_a : int, optional
        Hidden dimension for the attribute. Default: ``36``.
    hid_s : int, optional
        Hidden dimension for the structure. Default: ``36``.
    alpha : float, optional
        Weight for the attribute loss. Default: ``1.``.
    beta : float, optional
        Weight for the structural loss. Default: ``1.``.
    gamma : float, optional
        Weight for the combined loss. Default: ``1.``.
    weight_decay : float, optional
        Weight decay (L2 penalty). Default: ``0.``.
    contamination : float, optional
        Valid in (0., 0.5). The proportion of outliers in the data set.
        Used when fitting to define the threshold on the decision
        function. Default: ``0.1``.
    lr : float, optional
        Learning rate. Default: ``0.004``.
    epoch : int, optional
        Maximum number of training epoch. Default: ``5``.
    gpu : int
        GPU Index, -1 for using CPU. Default: ``-1``.
    verbose : int, optional
        Verbosity mode. Range in [0, 3]. Larger value for printing out
        more log information. Default: ``0``.
    """

    def __init__(self,
                 hid_a=36,
                 hid_s=36,
                 alpha=1.,
                 beta=1.,
                 gamma=1.,
                 weight_decay=0.,
                 contamination=0.1,
                 lr=0.004,
                 epoch=5,
                 gpu=-1,
                 verbose=0):
        super(ONE, self).__init__(contamination=contamination)

        self.hid_a = hid_a
        self.hid_s = hid_s
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.weight_decay = weight_decay
        self.lr = lr
        self.epoch = epoch
        self.device = validate_device(gpu)
        self.verbose = verbose

        self.attribute_score_ = None
        self.structural_score_ = None
        self.combined_score_ = None

        self.model = None

    def fit(self, data, label=None):

        self.process_graph(data)

        num_nodes, in_dim = data.x.shape
        x = data.x
        s = data.s

        w = torch.randn(self.hid_a, self.hid_s)

        u = torch.randn(num_nodes, self.hid_a)
        v = torch.randn(self.hid_a, in_dim)

        g = torch.randn(num_nodes, self.hid_s)
        h = torch.randn(self.hid_a, num_nodes)

        self.model = ONEBase(g, h, u, v, w, self.alpha, self.beta, self.gamma)
        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=self.lr,
                                     weight_decay=self.weight_decay)

        for epoch in range(self.epoch):
            start_time = time.time()
            x_, s_, diff = self.model()
            loss, o1, o2, o3 = self.model.loss_func(x,
                                                    x_,
                                                    s,
                                                    s_,
                                                    diff)

            self.attribute_score_ = o1.detach().cpu()
            self.structural_score_ = o2.detach().cpu()
            self.combined_score_ = o3.detach().cpu()
            self.decision_score_ = ((o1 + o2 + o3) / 3).detach().cpu()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logger(epoch=epoch,
                   loss=loss.item(),
                   score=self.decision_score_,
                   target=label,
                   time=time.time() - start_time,
                   verbose=self.verbose,
                   train=True)

        self._process_decision_score()
        return self

    def decision_function(self, data, label=None):
        if data is not None:
            self.fit(data, label)
        return self.decision_score_

    def process_graph(self, data):

        data.s = to_dense_adj(data.edge_index)[0]


class ONEBase(torch.nn.Module):
    def __init__(self, g, h, u, v, w, alpha=1., beta=1., gamma=1.):

        super(ONEBase, self).__init__()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.g = torch.nn.Parameter(g)
        self.h = torch.nn.Parameter(h)
        self.u = torch.nn.Parameter(u)
        self.v = torch.nn.Parameter(v)
        self.w = torch.nn.Parameter(w)

    def forward(self):
        x_ = self.u @ self.v
        s_ = self.g @ self.h
        diff = self.g - self.u @ self.w
        return x_, s_, diff

    def loss_func(self, x, x_, s, s_, diff):
        dx = torch.sum(torch.pow(x - x_, 2), 1)
        o1 = dx / torch.sum(dx)
        loss_a = torch.mean(torch.log(torch.pow(o1, -1)) * dx)

        ds = torch.sum(torch.pow(s - s_, 2), 1)
        o2 = ds / torch.sum(ds)
        loss_s = torch.mean(torch.log(torch.pow(o2, -1)) * ds)

        dc = torch.sum(torch.pow(diff, 2), 1)
        o3 = dc / torch.sum(dc)
        loss_c = torch.mean(torch.log(torch.pow(o3, -1)) * dc)

        loss = self.alpha * loss_a + self.beta * loss_s + self.gamma * loss_c

        return loss, o1, o2, o3
