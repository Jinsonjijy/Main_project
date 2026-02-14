import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def compute_auc(labels, scores):
    """
    ROC-AUC
    """
    return roc_auc_score(labels, scores)


def compute_pr_auc(labels, scores):
    """
    Precision-Recall AUC
    """
    return average_precision_score(labels, scores)


def precision_at_k(labels, scores, k=10):
    """
    Precision@K
    """
    idx = np.argsort(scores)[::-1][:k]
    return np.sum(labels[idx]) / k


def recall_at_k(labels, scores, k=10):
    """
    Recall@K
    """
    idx = np.argsort(scores)[::-1][:k]
    total_pos = np.sum(labels)

    if total_pos == 0:
        return 0

    return np.sum(labels[idx]) / total_pos
