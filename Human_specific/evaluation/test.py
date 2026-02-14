from evaluation.pathway_enrichment import pathway_enrichment

from gnn_run import (
    predict_drugs,
    data,
    gene_map,
    drug_rev,
    pathway_to_genes,
    disease_rev,
    normalize
)

results = pathway_enrichment(
    disease="Glioma",
    predict_function=predict_drugs,
    data=data,
    gene_map=gene_map,
    drug_rev=drug_rev,
    pathway_to_genes=pathway_to_genes,
    disease_rev=disease_rev,
    normalize=normalize,
    top_k=10
)

for r in results:
    print(
        r["drug"],
        "| overlap:", r["overlap_genes"],
        "| adj_p:", round(r["adjusted_p"], 6)
    )

