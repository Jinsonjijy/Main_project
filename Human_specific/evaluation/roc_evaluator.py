import torch
import numpy as np


def get_labels_and_scores(
    disease,
    model,
    data,
    disease_rev,
    normalize
):
    """
    Returns:
        labels (numpy array)
        scores (numpy array)
    """

    model.eval()

    d_norm = normalize(disease)
    if d_norm not in disease_rev:
        return None

    d_id = disease_rev[d_norm]

    with torch.no_grad():
        emb = model(data.x_dict, data.edge_index_dict)

        disease_emb = emb["disease"][d_id]            # (hidden_dim,)
        drug_embs = emb["drug"]                      # (num_drugs, hidden_dim)

        # Dot product score
        scores = torch.matmul(drug_embs, disease_emb)
        scores = scores.cpu().numpy()

    # Build ground truth labels
    treat_edges = data["drug", "treats", "disease"].edge_index

    true_drugs = treat_edges[0][
        treat_edges[1] == d_id
    ].cpu().numpy()

    labels = np.zeros(len(scores))
    labels[true_drugs] = 1

    # If no positive drugs exist, skip
    if np.sum(labels) == 0:
        return None

    return labels, scores
