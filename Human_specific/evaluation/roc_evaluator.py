import torch
import numpy as np


def get_labels_and_scores(
    disease,
    emb,
    data,
    disease_rev,
    normalize
):
    """
    Returns:
        labels (numpy array)
        scores (numpy array)
    """

    d_norm = normalize(disease)
    if d_norm not in disease_rev:
        return None

    d_id = disease_rev[d_norm]

    # Get embeddings directly (already computed)
    disease_emb = emb["disease"][d_id]     # (hidden_dim,)
    drug_embs = emb["drug"]                # (num_drugs, hidden_dim)

    # Dot product score
    scores = torch.matmul(drug_embs, disease_emb)
    scores = scores.cpu().numpy()

    # Ground truth labels
    treat_edges = data["drug", "treats", "disease"].edge_index

    true_drugs = treat_edges[0][
        treat_edges[1] == d_id
    ].cpu().numpy()

    labels = np.zeros(len(scores))
    labels[true_drugs] = 1

    if np.sum(labels) == 0:
        return None

    return labels, scores