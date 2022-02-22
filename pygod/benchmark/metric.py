"""
Metrics used to evaluate the anomaly detection performance
"""

def eval_rocauc(pred, labels):
    """
    Description
    -----------
    ROC-AUC score for multi-label node classification.
    Parameters
    ----------
    pred : torch.Tensor
        Output logits of model in form of ``N * 1``.
    labels : torch.LongTensor
        Labels in form of ``N * 1``.

    Returns
    -------
    rocauc : float
        Average ROC-AUC score across different labels.
    """

    rocauc_list = []
    if mask is not None:
        pred, labels = pred[mask], labels[mask]
        if pred is None or labels is None:
            return 0.0
    pred = pred.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    for i in range(labels.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(labels[:, i] == 1) > 0 and np.sum(labels[:, i] == 0) > 0:
            rocauc_list.append(roc_auc_score(y_true=labels[:, i],
                                             y_score=pred[:, i]))

    if len(rocauc_list) == 0:
        raise RuntimeError('No positively labeled data available. Cannot compute ROC-AUC.')

    rocauc = sum(rocauc_list) / len(rocauc_list)

    return rocauc


def eval_recall_at_k(pred, labels, k):
    pass


def eval_precision_at_k(pred, labels, k):
    pass
