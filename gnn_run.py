import pandas as pd
import torch
import random
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from gnn_model import DrugRepurposingHeteroGNN

# --------------------------------------------------
# Helper: categorical encoding (NO sklearn)
# --------------------------------------------------
def encode(series):
    codes, uniques = pd.factorize(series)
    mapping = dict(enumerate(uniques))
    reverse = {v: k for k, v in mapping.items()}
    return codes, mapping, reverse

# --------------------------------------------------
# Load datasets
# --------------------------------------------------
drug_gene = pd.read_csv("data/pharmacologically_active.csv")
gene_disease = pd.read_csv("data/CTD_curated_genes_diseases.csv")

# Clean column names
drug_gene.columns = drug_gene.columns.str.strip()
gene_disease.columns = gene_disease.columns.str.strip()

# Keep only HUMAN drug–gene interactions
drug_gene = drug_gene[drug_gene["Species"] == "Humans"]

# Rename to standard internal names
drug_gene = drug_gene.rename(columns={
    "GeneName": "GeneSymbol",
    "DrugIDs": "DrugID"
})

# Keep only required columns
drug_gene = drug_gene[["DrugID", "GeneSymbol"]].drop_duplicates()
gene_disease = gene_disease[["GeneSymbol", "DiseaseName"]].drop_duplicates()

# --------------------------------------------------
# Normalize disease names
# --------------------------------------------------
def normalize(x):
    return str(x).lower().replace(",", "").replace(" ", "").strip()

gene_disease["DiseaseName_norm"] = gene_disease["DiseaseName"].apply(normalize)

# --------------------------------------------------
# Encode nodes
# --------------------------------------------------
gene_disease["disease_id"], disease_map, disease_rev = encode(
    gene_disease["DiseaseName_norm"]
)

gene_disease["gene_id"], gene_map, gene_rev = encode(
    gene_disease["GeneSymbol"]
)

# Map DrugBank genes to CTD gene IDs
drug_gene["gene_id"] = drug_gene["GeneSymbol"].map(gene_rev)

# 🔥 CRITICAL FIX: drop drug–gene rows whose genes are NOT in CTD
drug_gene = drug_gene.dropna(subset=["gene_id"])

# Encode drugs
drug_gene["drug_id"], drug_map, drug_rev = encode(
    drug_gene["DrugID"]
)

# Cast IDs to int (MANDATORY)
gene_disease["disease_id"] = gene_disease["disease_id"].astype(int)
gene_disease["gene_id"] = gene_disease["gene_id"].astype(int)
drug_gene["gene_id"] = drug_gene["gene_id"].astype(int)
drug_gene["drug_id"] = drug_gene["drug_id"].astype(int)

num_diseases = len(disease_map)
num_genes = len(gene_map)
num_drugs = len(drug_map)

# --------------------------------------------------
# Build HeteroData graph
# --------------------------------------------------
data = HeteroData()

EMB_DIM = 128
data["disease"].x = torch.randn(num_diseases, EMB_DIM)
data["gene"].x = torch.randn(num_genes, EMB_DIM)
data["drug"].x = torch.randn(num_drugs, EMB_DIM)

# Disease ↔ Gene edges
dg_edges = torch.tensor(
    gene_disease[["disease_id", "gene_id"]].values.T,
    dtype=torch.long
)

data["disease", "associates", "gene"].edge_index = dg_edges
data["gene", "rev_associates", "disease"].edge_index = dg_edges.flip(0)

# Gene ↔ Drug edges
gd_edges = torch.tensor(
    drug_gene[["gene_id", "drug_id"]].values.T,
    dtype=torch.long
)

data["gene", "targets", "drug"].edge_index = gd_edges
data["drug", "rev_targets", "gene"].edge_index = gd_edges.flip(0)

# --------------------------------------------------
# Supervised disease–drug positive pairs
# --------------------------------------------------
merged = gene_disease.merge(drug_gene, on="gene_id")

positive_pairs = list(
    set(zip(merged["disease_id"], merged["drug_id"]))
)

# --------------------------------------------------
# Model
# --------------------------------------------------
model = DrugRepurposingHeteroGNN(hidden_dim=EMB_DIM)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# --------------------------------------------------
# Training (link prediction)
# --------------------------------------------------
def train():
    model.train()
    optimizer.zero_grad()

    emb = model(data.x_dict, data.edge_index_dict)
    losses = []

    batch = random.sample(
        positive_pairs,
        min(256, len(positive_pairs))
    )

    for d_id, dr_id in batch:
        pos = torch.dot(
            emb["disease"][d_id],
            emb["drug"][dr_id]
        )

        neg_id = random.randint(0, num_drugs - 1)
        neg = torch.dot(
            emb["disease"][d_id],
            emb["drug"][neg_id]
        )

        losses.append(F.relu(1.0 - pos + neg))

    loss = torch.stack(losses).mean()
    loss.backward()
    optimizer.step()
    return loss.item()

# Train model
for _ in range(200):
    train()

# --------------------------------------------------
# Inference
# --------------------------------------------------
def predict_drugs(disease_name, top_k=5):
    d_norm = normalize(disease_name)
    if d_norm not in disease_rev:
        raise ValueError("Disease not found in dataset")

    d_id = disease_rev[d_norm]

    model.eval()
    with torch.no_grad():
        emb = model(data.x_dict, data.edge_index_dict)

    scores = torch.matmul(
        emb["drug"], emb["disease"][d_id]
    )

    top = torch.topk(scores, top_k).indices.tolist()
    return [drug_map[i] for i in top]

# --------------------------------------------------
# Test
# --------------------------------------------------
disease = "COVID-19"

print("\nDisease:", disease)
print("Predicted Drugs:\n")

for d in predict_drugs(disease):
    print(d)
