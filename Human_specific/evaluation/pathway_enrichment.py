from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests


def pathway_enrichment(
    disease,
    predict_function,
    data,
    gene_map,
    drug_rev,
    pathway_to_genes,
    disease_rev,
    normalize,
    top_k=10
):

    results, selected_pathways = predict_function(disease, top_k)

    d_norm = normalize(disease)
    if d_norm not in disease_rev:
        return []

    d_id = disease_rev[d_norm]

    # Get disease genes
    gene_ids = data["disease", "associates", "gene"].edge_index[1][
        data["disease", "associates", "gene"].edge_index[0] == d_id
    ]

    disease_genes = {gene_map[g.item()] for g in gene_ids}

    enrichment_results = []

    total_genes = len(gene_map)

    for name, dbid, score in results:

        drug_id = drug_rev.get(dbid)
        if drug_id is None:
            continue

        # Get drug targets
        drug_gene_ids = data["gene", "targets", "drug"].edge_index[0][
            data["gene", "targets", "drug"].edge_index[1] == drug_id
        ]

        drug_targets = {gene_map[g.item()] for g in drug_gene_ids}

        # Count overlap
        overlap = len(disease_genes & drug_targets)

        a = overlap
        b = len(drug_targets) - overlap
        c = len(disease_genes) - overlap
        d = max(total_genes - (a + b + c), 1)

        table = [[a, b], [c, d]]

        _, p_value = fisher_exact(table)

        enrichment_results.append({
            "drug": name,
            "drugbank_id": dbid,
            "score": score,
            "overlap_genes": overlap,
            "p_value": p_value
        })

    # Apply FDR correction
    pvals = [r["p_value"] for r in enrichment_results]

    if len(pvals) > 0:
        _, corrected, _, _ = multipletests(pvals, method="fdr_bh")

        for i in range(len(enrichment_results)):
            enrichment_results[i]["adjusted_p"] = corrected[i]

    return enrichment_results
