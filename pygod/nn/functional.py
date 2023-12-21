# -*- coding: utf-8 -*-
"""Functional Interface for PyGOD"""
# Author: Kay Liu <zliu234@uic.edu>, Yingtong Dou <ytongdou@gmail.com>
# License: BSD 2 clause

import torch
import torch.nn.functional as F
from scipy.linalg import sqrtm
import math


def double_recon_loss(x,
                      x_,
                      s,
                      s_,
                      weight=0.5,
                      pos_weight_a=0.5,
                      pos_weight_s=0.5,
                      bce_s=False):
    r"""
    Double reconstruction loss function for feature and structure.
    The loss function is defined as :math:`\alpha \symbf{E_a} +
    (1-\alpha) \symbf{E_s}`, where :math:`\alpha` is the weight between
    0 and 1 inclusive, and :math:`\symbf{E_a}` and :math:`\symbf{E_s}`
    are the reconstruction loss for feature and structure, respectively.
    The first dimension is kept for outlier scores of each node.

    For feature reconstruction, we use mean squared error loss:
    :math:`\symbf{E_a} = \|\symbf{X}-\symbf{X}'\|\odot H`,
    where :math:`H=\begin{cases}1 - \eta &
    \text{if }x_{ij}=0\\ \eta & \text{if }x_{ij}>0\end{cases}`, and
    :math:`\eta` is the positive weight for feature.

    For structure reconstruction, we use mean squared error loss by
    default: :math:`\symbf{E_s} = \|\symbf{S}-\symbf{S}'\|\odot
    \Theta`, where :math:`\Theta=\begin{cases}1 -
    \theta & \text{if }s_{ij}=0\\ \theta & \text{if }s_{ij}>0
    \end{cases}`, and :math:`\theta` is the positive weight for
    structure. Alternatively, we can use binary cross entropy loss
    for structure reconstruction: :math:`\symbf{E_s} =
    \text{BCE}(\symbf{S}, \symbf{S}' \odot \Theta)`.

    Parameters
    ----------
    x : torch.Tensor
        Ground truth node feature
    x_ : torch.Tensor
        Reconstructed node feature
    s : torch.Tensor
        Ground truth node structure
    s_ : torch.Tensor
        Reconstructed node structure
    weight : float, optional
        Balancing weight :math:`\alpha` between 0 and 1 inclusive between node feature
        and graph structure. Default: ``0.5``.
    pos_weight_a : float, optional
        Positive weight for feature :math:`\eta`. Default: ``0.5``.
    pos_weight_s : float, optional
        Positive weight for structure :math:`\theta`. Default: ``0.5``.
    bce_s : bool, optional
        Use binary cross entropy for structure reconstruction loss.

    Returns
    -------
    score : torch.tensor
        Outlier scores of shape :math:`N` with gradients.
    """

    assert 0 <= weight <= 1, "weight must be a float between 0 and 1."
    assert 0 <= pos_weight_a <= 1 and 0 <= pos_weight_s <= 1, \
        "positive weight must be a float between 0 and 1."

    # attribute reconstruction loss
    diff_attr = torch.pow(x - x_, 2)

    if pos_weight_a != 0.5:
        diff_attr = torch.where(x > 0, 
                                diff_attr * pos_weight_a, 
                                diff_attr * (1 - pos_weight_a))

    attr_error = torch.sqrt(torch.sum(diff_attr, 1))

    # structure reconstruction loss
    if bce_s:
        diff_stru = F.binary_cross_entropy(s_, s, reduction='none')
    else:
        diff_stru = torch.pow(s - s_, 2)

    if pos_weight_s != 0.5:
        diff_stru = torch.where(s > 0, 
                                diff_stru * pos_weight_s, 
                                diff_stru * (1 - pos_weight_s))

    stru_error = torch.sqrt(torch.sum(diff_stru, 1))

    score = weight * attr_error + (1 - weight) * stru_error

    return score

def KL_neighbor_loss(predictions, targets, mask_len, device):
    """
    The local neighor distribution KL divergence loss used in GAD-NR.
    Source:
    https://github.com/Graph-COM/GAD-NR/blob/master/GAD-NR_inj_cora.ipynb
    """

    x1 = predictions.squeeze().cpu().detach()[:mask_len, :]
    x2 = targets.squeeze().cpu().detach()[:mask_len, :]

    mean_x1 = x1.mean(0)
    mean_x2 = x2.mean(0)

    nn = x1.shape[0]
    h_dim = x1.shape[1]

    cov_x1 = (x1-mean_x1).transpose(1,0).matmul(x1-mean_x1) / max((nn-1),1)
    cov_x2 = (x2-mean_x2).transpose(1,0).matmul(x2-mean_x2) / max((nn-1),1)

    eye = torch.eye(h_dim)
    cov_x1 = cov_x1 + eye
    cov_x2 = cov_x2 + eye

    KL_loss = 0.5 * (math.log(torch.det(cov_x1) / torch.det(cov_x2)) - h_dim
    + torch.trace(torch.inverse(cov_x2).matmul(cov_x1)) + (mean_x2 - 
    mean_x1).reshape(1,-1).matmul(torch.inverse(cov_x2)).matmul(mean_x2 - 
                                                                mean_x1))
    KL_loss = KL_loss.to(device)
    return KL_loss

def W2_neighbor_loss(predictions, targets, mask_len, device):
    """
    The local neighor distribution W2 loss used in GAD-NR.
    Source:
    https://github.com/Graph-COM/GAD-NR/blob/master/GAD-NR_inj_cora.ipynb
    """

    x1 = predictions.squeeze().cpu().detach()[:mask_len, :]
    x2 = targets.squeeze().cpu().detach()[:mask_len, :]

    mean_x1 = x1.mean(0)
    mean_x2 = x2.mean(0)

    nn = x1.shape[0]

    cov_x1 = (x1-mean_x1).transpose(1,0).matmul(x1-mean_x1) / max((nn-1),1)
    cov_x2 = (x2-mean_x2).transpose(1,0).matmul(x2-mean_x2) / max((nn-1),1)

    W2_loss = torch.square(mean_x1-mean_x2).sum() 
    + torch.trace(cov_x1 + cov_x2 
    + 2 * sqrtm(sqrtm(cov_x1) @ (cov_x2.numpy()) @ (sqrtm(cov_x1))))

    W2_loss = W2_loss.to(device)

    return W2_loss
