import torch


def double_mse_loss(x, x_, s, s_, weight=0.5):
    """
    Mean squared error reconstruction loss for feature and structure.

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
        Balancing weight between 0 and 1 inclusive. Default: ``0.5``.

    Returns
    -------
    score : torch.tensor
        Outlier scores of shape :math:`N` with gradients.
    """

    assert 0 <= weight <= 1, "weight must be a float between 0 and 1."

    # attribute reconstruction loss
    diff_attribute = torch.pow(x - x_, 2)
    attribute_error = torch.sqrt(torch.sum(diff_attribute, 1))

    # structure reconstruction loss
    diff_structure = torch.pow(s - s_, 2)
    structure_error = torch.sqrt(torch.sum(diff_structure, 1))

    score = weight * attribute_error + (1 - weight) * structure_error

    return score
