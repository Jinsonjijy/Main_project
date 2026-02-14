from .roc_evaluator import get_labels_and_scores
from .metrics import (
    compute_auc,
    compute_pr_auc,
    precision_at_k,
    recall_at_k
)

from gnn_run import model, data, disease_rev, normalize


import numpy as np


all_auc = []
all_pr_auc = []
all_p10 = []
all_r10 = []
all_r20 = []

print("Running ROC evaluation...\n")

for disease_norm in disease_rev.keys():

    result = get_labels_and_scores(
        disease=disease_norm,
        model=model,
        data=data,
        disease_rev=disease_rev,
        normalize=lambda x: x   # already normalized keys
    )

    if result is None:
        continue

    labels, scores = result

    try:
        auc = compute_auc(labels, scores)
        pr_auc = compute_pr_auc(labels, scores)
        p10 = precision_at_k(labels, scores, k=10)
        r10 = recall_at_k(labels, scores, k=10)
        r20 = recall_at_k(labels, scores, k=20)

        all_auc.append(auc)
        all_pr_auc.append(pr_auc)
        all_p10.append(p10)
        all_r10.append(r10)
        all_r20.append(r20)

    except Exception:
        continue


print("Diseases evaluated:", len(all_auc))
print("====================================")
print("Average ROC-AUC:", np.mean(all_auc))
print("Std ROC-AUC:", np.std(all_auc))
print("------------------------------------")
print("Average PR-AUC:", np.mean(all_pr_auc))
print("------------------------------------")
print("Average Precision@10:", np.mean(all_p10))
print("Average Recall@10:", np.mean(all_r10))
print("Average Recall@20:", np.mean(all_r20))
