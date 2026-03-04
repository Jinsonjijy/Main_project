import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from gnn_model import DrugRepurposingHeteroGNN
import os
import json
import random
import numpy as np
 

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


# ==================================================
# CONFIG
# ==================================================
INPUT_DIM = 640
HIDDEN_DIM = 256
TOP_PATHWAYS = 5

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
# Load Cleaned Metadata
# ==================================================
metadata = pd.read_csv("data/drug_target_metadata_cleaned.csv")


# ==================================================
# Load Core Datasets
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

gene_disease["disease_id"], disease_map, disease_rev = encode(
    gene_disease["DiseaseName_norm"]
)
gene_disease["gene_id"], gene_map, gene_rev = encode(
    gene_disease["GeneSymbol"]
)

drug_gene["gene_id"] = drug_gene["GeneSymbol"].map(gene_rev)
drug_gene = drug_gene.dropna(subset=["gene_id"])

unique_drugs = sorted(drug_gene["DrugID"].unique())
drug_map = {dbid: idx for idx, dbid in enumerate(unique_drugs)}
drug_rev = {idx: dbid for dbid, idx in drug_map.items()}
drug_gene["drug_id"] = drug_gene["DrugID"].map(drug_map)

drug_disease["disease_id"] = drug_disease["DiseaseName_norm"].map(disease_rev)
drug_disease["drug_id"] = drug_disease["DrugBankID"].map(drug_map)
drug_disease = drug_disease.dropna(subset=["disease_id", "drug_id"])

print(f"Diseases: {len(disease_map)}, Genes: {len(gene_map)}, Drugs: {len(drug_map)}")


# ==================================================
# Load Protein Embeddings
# ==================================================
protein_emb = torch.load("protein_embeddings.pt", map_location="cpu")


# ==================================================
# Load Drug Embeddings
# ==================================================
full_drug_features = torch.load("drug_embeddings.pt")

with open("data/drug_map.json", "r") as f:
    full_drug_map = json.load(f)

drug_features = torch.zeros((len(drug_map), INPUT_DIM))

for dbid, idx in drug_map.items():
    if dbid in full_drug_map:
        drug_features[idx] = full_drug_features[full_drug_map[dbid]]


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

data["drug"].x = drug_features

gene_features = []
for gene in gene_map.values():
    if gene in protein_emb:
        gene_features.append(protein_emb[gene])
    else:
        gene_features.append(torch.zeros(INPUT_DIM))

data["gene"].x = torch.stack(gene_features)
data["disease"].x = torch.randn(len(disease_map), INPUT_DIM)

# Core edges (INT64 FIX)
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

model.load_state_dict(
    torch.load("drug_repurposing_gnn.pt", map_location=device,weights_only=False)
)
model.eval()

print("Model loaded successfully\n")

with torch.no_grad():
    full_emb = model(data.x_dict, data.edge_index_dict)

disease_emb = full_emb["disease"]
drug_emb = full_emb["drug"]
gene_emb = full_emb["gene"]


# ==================================================
# Prediction Function
# ==================================================
@torch.no_grad()
def predict_drugs(disease, top_k=10):

    d_norm = normalize(disease)
    if d_norm not in disease_rev:
        raise ValueError("Disease not found")

    d_id = disease_rev[d_norm]

    # emb = model(data.x_dict, data.edge_index_dict)
    # disease_emb = emb["disease"]
    # drug_emb = emb["drug"]
    # gene_emb = emb["gene"]

    # Get disease genes
    gene_ids = data["disease", "associates", "gene"].edge_index[1][
        data["disease", "associates", "gene"].edge_index[0] == d_id
    ]

    if gene_ids.numel() == 0:
        return [], []

    disease_genes = {gene_map[g.item()] for g in gene_ids}

    # Pathway scoring
    pathway_scores = []
    for pname, pgenes in pathway_to_genes.items():
        overlap = disease_genes & pgenes
        if overlap:
            pathway_scores.append((pname, len(overlap)))

    pathway_scores.sort(key=lambda x: x[1], reverse=True)
    selected_pathways = pathway_scores[:TOP_PATHWAYS]

    # Cosine similarity
    disease_norm = disease_emb
    drug_norm = drug_emb

    scores = F.cosine_similarity(
        disease_norm[d_id].unsqueeze(0),
        drug_norm,
        dim=1
    )

    top_vals, top_idx = torch.topk(scores, min(len(scores), top_k))

    results = []
    for idx, score in zip(top_idx.tolist(), top_vals.tolist()):
        dbid = drug_rev[idx]
        name = drug_id_to_name.get(dbid, "Unknown")
        results.append((name, dbid, float(score)))

    return results, selected_pathways


# ==================================================
# CLI
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
# EXPORT FOR FLASK
# ==================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "Frontend"))

GRAPH_JSON_PATH = os.path.join(FRONTEND_DIR, "graph.json")
RESULTS_JSON_PATH = os.path.join(FRONTEND_DIR, "results.json")

os.makedirs(FRONTEND_DIR, exist_ok=True)


def export_disease_subgraph(disease, top_k=6):

    # -------------------------------------------------
    # Get Predictions from Real GNN
    # -------------------------------------------------
    results, pathways = predict_drugs(disease, top_k=top_k)

    # -------------------------------------------------
    # WRITE RESULTS.JSON
    # -------------------------------------------------
    with open(RESULTS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "disease": disease,
            "pathways": pathways,
            "drugs": [
                {
                    "name": name,
                    "drugbank_id": dbid,
                    "score": float(score)
                }
                for name, dbid, score in results
            ]
        }, f, indent=2)

    # -------------------------------------------------
    # BUILD GRAPH.JSON (Top 5 Genes Only)
    # -------------------------------------------------

    d_norm = normalize(disease)
    if d_norm not in disease_rev:
        raise ValueError("Disease not found")

    d_id = disease_rev[d_norm]

    # Get disease-associated genes
    gene_ids = data["disease", "associates", "gene"].edge_index[1][
        data["disease", "associates", "gene"].edge_index[0] == d_id
    ]

    # Convert to gene symbols
    all_genes = [gene_map[g.item()] for g in gene_ids]

    # Select ONLY top 5 genes
    disease_genes = all_genes[:5]

    nodes = []
    links = []

    # --------------------------
    # Disease Node
    # --------------------------
    nodes.append({
        "id": disease,
        "label": disease,
        "type": "disease"
    })

    # --------------------------
    # Gene Nodes + Disease-Gene Links
    # --------------------------
    for gene in disease_genes:
        nodes.append({
            "id": gene,
            "label": gene,
            "type": "gene"
        })

        links.append({
            "source": disease,
            "target": gene,
            "type": "disease-gene"
        })

    # --------------------------
    # Drug Nodes + Gene-Drug Links
    # --------------------------
    for name, dbid, score in results:

        nodes.append({
            "id": dbid,
            "label": name,
            "type": "drug",
            "score": float(score)
        })

        # Connect each selected gene to this drug
        for gene in disease_genes:
            links.append({
                "source": gene,
                "target": dbid,
                "type": "gene-drug"
            })

    graph_data = {
        "nodes": nodes,
        "links": links
    }

    with open(GRAPH_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(graph_data, f, indent=2)

    print("graph.json updated")
    print("results.json updated")

    return results, pathways