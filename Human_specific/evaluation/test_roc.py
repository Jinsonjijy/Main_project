import sys
import os
import torch
import numpy as np
import random

# ---------------------------------
# Add parent directory to path
# ---------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from roc_evaluator import get_labels_and_scores
from metrics import (
    compute_auc,
    compute_pr_auc,
    precision_at_k,
    recall_at_k
)

from gnn_run import model, data, disease_rev, normalize


# ---------------------------------
# Reproducibility
# ---------------------------------
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

model.eval()

# ---------------------------------
# Compute embeddings ONCE
# ---------------------------------
with torch.no_grad():
    emb = model(data.x_dict, data.edge_index_dict)


all_auc = []
all_pr_auc = []
all_p10 = []
all_r10 = []
all_r20 = []

print("Running ROC evaluation (Zero Disease Init)...\n")

for disease_norm in disease_rev.keys():

    result = get_labels_and_scores(
        disease=disease_norm,
        emb=emb,
        data=data,
        disease_rev=disease_rev,
        normalize=lambda x: x
    )

    if result is None:
        continue

    labels, scores = result

    try:
        all_auc.append(compute_auc(labels, scores))
        all_pr_auc.append(compute_pr_auc(labels, scores))
        all_p10.append(precision_at_k(labels, scores, k=10))
        all_r10.append(recall_at_k(labels, scores, k=10))
        all_r20.append(recall_at_k(labels, scores, k=20))

    except Exception:
        continue


print("Diseases evaluated:", len(all_auc))
print("====================================")
print("Average ROC-AUC:", round(np.mean(all_auc), 4))
print("Std ROC-AUC:", round(np.std(all_auc), 4))
print("------------------------------------")
print("Average PR-AUC:", round(np.mean(all_pr_auc), 4))
print("------------------------------------")
print("Average Precision@10:", round(np.mean(all_p10), 4))
print("Average Recall@10:", round(np.mean(all_r10), 4))
print("Average Recall@20:", round(np.mean(all_r20), 4))