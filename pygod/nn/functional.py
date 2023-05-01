import torch
import torch.nn.functional as F


def double_recon_loss(x,
                      x_,
                      s,
                      s_,
                      weight=0.5,
                      pos_weight_a=0.5,
                      pos_weight_s=0.5,
                      bce_s=False):
    r"""
    Mean squared error reconstruction loss for feature and structure.
    :math:`\alpha \|X−X'\odot\eta\|+(1-\alpha)\|S-S'\odot\theta\|`,
    where :math:`\odot` is element-wise multiplication and :math:`\eta`
    and :math:`\theta` are defined as follows:
    :math:`\eta=\begin{cases}1&\text{if }x_i=0\\
    \eta&\text{if }x_i>0\end{cases}`
    and
    :math:`\theta=\begin{cases}1&\text{if }s_{ij}=0\\
    \theta&\text{if }s_{ij}>0\end{cases}`
    where :math:`x_i` is the :math:`i`-th node feature and
    :math:`s_{ij}` is the :math:`ij`-th element of the adjacency matrix.

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
        Balancing weight between 0 and 1 inclusive between node feature
        and graph structure. Default: ``0.5``.
    pos_weight_a : float, optional
        Non-zero penalty for feature. Default: ``1``.
    pos_weight_s : float, optional
        Non-zero penalty for structure. Default: ``1``.
    bce_s : bool, optional
        Use binary cross entropy for structure reconstruction loss.

    Returns
    -------
    score : torch.tensor
        Outlier scores of shape :math:`N` with gradients.
    """

    assert 0 <= weight <= 1, "weight must be a float between 0 and 1."
    assert 0 <= pos_weight_a <= 1, "eta must be greater than or equal to 1."
    assert 0 <= pos_weight_s <= 1, "theta must be greater than or equal to 1."

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
