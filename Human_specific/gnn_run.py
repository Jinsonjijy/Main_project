import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from gnn_model import DrugRepurposingHeteroGNN

# ==================================================
# CONFIG
# ==================================================
INPUT_DIM = 640
HIDDEN_DIM = 256
TOP_PATHWAYS = 5
ALPHA = 0.7

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ==================================================
# Helpers
# ==================================================
def encode(series):
    codes, uniques = pd.factorize(series)
    mapping = dict(enumerate(uniques))
    reverse = {v: k for k, v in mapping.items()}
    return codes, mapping, reverse

def normalize(x):
    return str(x).lower().replace(",", "").replace(" ", "").strip()

# ==================================================
# Load Drug Names
# ==================================================
drug_names_df = pd.read_csv("data/uniprot_links.csv")
drug_names_df.columns = drug_names_df.columns.str.strip()
drug_id_to_name = dict(zip(drug_names_df["DrugBank ID"], drug_names_df["Name"]))

# ==================================================
# Load Datasets
# ==================================================
drug_gene = pd.read_csv("data/pharmacologically_active.csv")
gene_disease = pd.read_csv("data/core_plus_disease_gene.csv")
drug_disease = pd.read_csv("data/drug_parser.csv")

drug_gene = drug_gene[drug_gene["Species"] == "Humans"]
drug_gene = drug_gene.rename(columns={
    "GeneName": "GeneSymbol",
    "DrugIDs": "DrugID"
})

gene_disease["DiseaseName_norm"] = gene_disease["DiseaseName"].apply(normalize)
drug_disease["DiseaseName_norm"] = drug_disease["DiseaseName"].apply(normalize)

gene_disease["disease_id"], disease_map, disease_rev = encode(gene_disease["DiseaseName_norm"])
gene_disease["gene_id"], gene_map, gene_rev = encode(gene_disease["GeneSymbol"])

drug_gene["gene_id"] = drug_gene["GeneSymbol"].map(gene_rev)
drug_gene = drug_gene.dropna(subset=["gene_id"])
drug_gene["drug_id"], drug_map, drug_rev = encode(drug_gene["DrugID"])

drug_disease["disease_id"] = drug_disease["DiseaseName_norm"].map(disease_rev)
drug_disease["drug_id"] = drug_disease["DrugBankID"].map(drug_rev)
drug_disease = drug_disease.dropna(subset=["disease_id", "drug_id"])

print(f"Diseases: {len(disease_map)}, Genes: {len(gene_map)}, Drugs: {len(drug_map)}")

# ==================================================
# Load Protein Embeddings
# ==================================================
protein_emb = torch.load("protein_embeddings.pt", map_location="cpu")

# ==================================================
# Load KEGG Pathways
# ==================================================
kegg_df = pd.read_csv("data/kegg_pathway_genes.csv")
kegg_df["GeneSymbols"] = kegg_df["GeneSymbols"].astype(str)

pathway_to_genes = {}

for _, row in kegg_df.iterrows():
    pname = row["PathwayName"]
    genes = [g.strip() for g in row["GeneSymbols"].split(";") if g.strip()]
    pathway_to_genes[pname] = set(genes)

# ==================================================
# Build Graph
# ==================================================
data = HeteroData()

data["drug"].x = torch.randn(len(drug_map), INPUT_DIM) * 0.01

gene_features = []
missing = 0

for gene in gene_map.values():
    if gene in protein_emb and protein_emb[gene].shape[0] == INPUT_DIM:
        gene_features.append(protein_emb[gene])
    else:
        gene_features.append(torch.zeros(INPUT_DIM))
        missing += 1

print("Missing gene embeddings:", missing)

data["gene"].x = torch.stack(gene_features)
data["disease"].x = torch.randn(len(disease_map), INPUT_DIM) * 0.01

dg = torch.tensor(gene_disease[["disease_id", "gene_id"]].values.T, dtype=torch.long)
gd = torch.tensor(drug_gene[["gene_id", "drug_id"]].values.T, dtype=torch.long)
dd = torch.tensor(drug_disease[["drug_id", "disease_id"]].values.T, dtype=torch.long)

data["disease", "associates", "gene"].edge_index = dg
data["gene", "rev_associates", "disease"].edge_index = dg.flip(0)
data["gene", "targets", "drug"].edge_index = gd
data["drug", "rev_targets", "gene"].edge_index = gd.flip(0)
data["drug", "treats", "disease"].edge_index = dd
data["disease", "rev_treats", "drug"].edge_index = dd.flip(0)

data = data.to(device)

# ==================================================
# Load Model
# ==================================================
model = DrugRepurposingHeteroGNN(
    input_dim=INPUT_DIM,
    hidden_dim=HIDDEN_DIM
).to(device)

model.load_state_dict(torch.load("drug_repurposing_gnn.pt", map_location=device))
model.eval()

print("Model loaded successfully\n")

# ==================================================
# Prediction
# ==================================================
@torch.no_grad()
def predict_drugs(disease, top_k=10):

    d_norm = normalize(disease)
    if d_norm not in disease_rev:
        raise ValueError("Disease not found")

    d_id = disease_rev[d_norm]

    emb = model(data.x_dict, data.edge_index_dict)

    disease_emb = emb["disease"]
    gene_emb = emb["gene"]
    drug_emb = emb["drug"]

    gene_ids = data["disease", "associates", "gene"].edge_index[1][
        data["disease", "associates", "gene"].edge_index[0] == d_id
    ]

    if gene_ids.numel() == 0:
        return [], []

    disease_genes = {gene_map[g.item()] for g in gene_ids}

    # -------- Pathway overlap --------
    pathway_scores = []
    for pname, pgenes in pathway_to_genes.items():
        overlap = disease_genes & pgenes
        if overlap:
            pathway_scores.append((pname, len(overlap)))

    pathway_scores.sort(key=lambda x: x[1], reverse=True)
    selected_pathways = pathway_scores[:TOP_PATHWAYS]

    # -------- Context embedding --------
    gene_context = gene_emb[gene_ids].mean(dim=0)
    final_context = F.normalize(
        ALPHA * disease_emb[d_id] + (1 - ALPHA) * gene_context,
        dim=0
    )

    drug_ids = data["gene", "targets", "drug"].edge_index[1][
        torch.isin(data["gene", "targets", "drug"].edge_index[0], gene_ids)
    ].unique()

    scores = torch.matmul(drug_emb[drug_ids], final_context)
    top_vals, top_idx = torch.topk(scores, min(len(scores), 50))

    # -------- Cleaner --------
    bad_keywords = [
        "cell", "keratinocyte", "fibroblast",
        "alcohol", "serum", "plasma",
        "culture", "medium", "reagent",
        "buffer", "vehicle"
    ]

    results = []

    for i, score in zip(top_idx.tolist(), top_vals.tolist()):
        dbid = drug_map[drug_ids[i].item()]
        name = drug_id_to_name.get(dbid, "Unknown")

        if any(k in name.lower() for k in bad_keywords):
            continue

        results.append((name, dbid, float(score)))

        if len(results) == top_k:
            break

    return results, selected_pathways


# ==================================================
# Interactive Loop
# ==================================================
if __name__ == "__main__":

    print("=" * 60)
    print("Drug Repurposing System")
    print("=" * 60)

    while True:
        disease = input("\nEnter disease name (or 'exit'): ").strip()

        if disease.lower() in ["exit", "quit", "q"]:
            break

        try:
            results, pathways = predict_drugs(disease)

            print("\nTop KEGG Pathways:")
            for p, count in pathways:
                print(f"- {p} ({count} genes)")

            print("\nTop Candidate Drugs:\n")
            for i, (name, dbid, score) in enumerate(results, 1):
                print(f"{i}. {name} ({dbid}) | Score: {score:.4f}")

        except Exception as e:
            print("Error:", e)

# ==================================================
# EXPORT GRAPH + RESULTS (FOR FLASK)
# ==================================================

import os
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.abspath(
    os.path.join(BASE_DIR, "..", "Frontend")
)

GRAPH_JSON_PATH = os.path.join(FRONTEND_DIR, "graph.json")
RESULTS_JSON_PATH = os.path.join(FRONTEND_DIR, "results.json")

os.makedirs(FRONTEND_DIR, exist_ok=True)


def export_disease_subgraph(disease, top_k=10, max_genes=15):

    results, pathways = predict_drugs(disease, top_k=top_k)

    if not results:
        raise ValueError("No results found")

    d_norm = normalize(disease)
    d_id = disease_rev[d_norm]

    gene_ids = data["disease", "associates", "gene"].edge_index[1][
        data["disease", "associates", "gene"].edge_index[0] == d_id
    ].unique()[:max_genes]

    nodes = []
    links = []

    # -----------------------------
    # Disease node
    # -----------------------------
    nodes.append({
        "id": disease,
        "label": disease,
        "type": "disease"
    })

    # -----------------------------
    # Gene nodes
    # -----------------------------
    for g in gene_ids.tolist():
        gene_name = gene_map[g]

        nodes.append({
            "id": gene_name,
            "label": gene_name,
            "type": "gene"
        })

        links.append({
            "source": disease,
            "target": gene_name,
            "type": "disease-gene"
        })

    # -----------------------------
    # Drug nodes
    # -----------------------------
    for name, dbid, score in results:

        nodes.append({
            "id": dbid,
            "label": name,
            "type": "drug",
            "score": round(score, 4)
        })

        for g in gene_ids.tolist():
            links.append({
                "source": gene_map[g],
                "target": dbid,
                "type": "gene-drug"
            })

    # -----------------------------
    # Save graph.json
    # -----------------------------
    with open(GRAPH_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump({"nodes": nodes, "links": links}, f, indent=2)

    # -----------------------------
    # Save results.json (exact format you want)
    # -----------------------------
    results_output = {
        "disease": disease,
        "pathways_used": [p[0] if isinstance(p, tuple) else p for p in pathways],
        "top_drugs": [
            {
                "drug_name": name,
                "drugbank_id": dbid,
                "score": round(score, 4)
            }
            for name, dbid, score in results
        ]
    }

    with open(RESULTS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(results_output, f, indent=2)

    return results, pathways
