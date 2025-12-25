import pandas as pd
import torch
import random
import torch.nn.functional as F
from collections import defaultdict
from torch_geometric.data import HeteroData
from gnn_model import DrugRepurposingHeteroGNN

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
# Load datasets
# ==================================================
gene_disease = pd.read_csv("data/core_plus_disease_gene.csv")
drug_gene = pd.read_csv("data/pharmacologically_active.csv")
drug_disease = pd.read_csv("data/drug_parser.csv")

drug_gene = drug_gene[drug_gene["Species"] == "Humans"]
drug_gene = drug_gene.rename(columns={
    "GeneName": "GeneSymbol",
    "DrugIDs": "DrugID",
    "UniProtID": "UniProtID"
})

# Normalize diseases
gene_disease["DiseaseName_norm"] = gene_disease["DiseaseName"].apply(normalize)
drug_disease["DiseaseName_norm"] = drug_disease["DiseaseName"].apply(normalize)

# Encode
gene_disease["disease_id"], d_map, d_rev = encode(gene_disease["DiseaseName_norm"])
gene_disease["gene_id"], g_map, g_rev = encode(gene_disease["GeneSymbol"])

drug_gene["gene_id"] = drug_gene["GeneSymbol"].map(g_rev)
drug_gene = drug_gene.dropna(subset=["gene_id"])

drug_gene["drug_id"], dr_map, dr_rev = encode(drug_gene["DrugID"])

drug_disease["disease_id"] = drug_disease["DiseaseName_norm"].map(d_rev)
drug_disease["drug_id"] = drug_disease["DrugBankID"].map(dr_rev)
drug_disease = drug_disease.dropna(subset=["disease_id", "drug_id"])

num_d = len(d_map)
num_g = len(g_map)
num_dr = len(dr_map)

print("Diseases:", num_d, "Genes:", num_g, "Drugs:", num_dr)

# ==================================================
# Load protein embeddings (UniProt → vector)
# ==================================================
protein_emb = torch.load("protein_embeddings.pt")

gene_to_uniprot = dict(
    zip(drug_gene["GeneSymbol"], drug_gene["UniProtID"])
)

EMB_DIM = 1024

# ==================================================
# Build Graph
# ==================================================
data = HeteroData()

# Drug nodes
data["drug"].x = torch.randn(num_dr, EMB_DIM) * 0.01

# Gene nodes (FIXED)
gene_x = []
missing = 0
for gene_symbol in g_map.values():
    uid = gene_to_uniprot.get(gene_symbol)
    if uid and uid in protein_emb and protein_emb[uid].shape[0] == EMB_DIM:
        gene_x.append(protein_emb[uid])
    else:
        gene_x.append(torch.zeros(EMB_DIM))
        missing += 1

data["gene"].x = torch.stack(gene_x)
print("Missing gene embeddings:", missing)

# Disease nodes (initialized from genes)
data["disease"].x = torch.zeros(num_d, EMB_DIM)

dg_edges = torch.tensor(
    gene_disease[["disease_id", "gene_id"]].values.T,
    dtype=torch.long
)

for d in range(num_d):
    gids = dg_edges[1][dg_edges[0] == d]
    if gids.numel() > 0:
        data["disease"].x[d] = data["gene"].x[gids].mean(dim=0)

# ==================================================
# Edges (WITH reverse edges ✅)
# ==================================================
gd_edges = torch.tensor(
    drug_gene[["gene_id", "drug_id"]].values.T,
    dtype=torch.long
)

dd_edges = torch.tensor(
    drug_disease[["drug_id", "disease_id"]].values.T,
    dtype=torch.long
)

data["disease", "associates", "gene"].edge_index = dg_edges
data["gene", "rev_associates", "disease"].edge_index = dg_edges.flip(0)

data["gene", "targets", "drug"].edge_index = gd_edges
data["drug", "rev_targets", "gene"].edge_index = gd_edges.flip(0)

data["drug", "treats", "disease"].edge_index = dd_edges
data["disease", "rev_treats", "drug"].edge_index = dd_edges.flip(0)

# ==================================================
# Training pairs (NO leakage)
# ==================================================
merged = gene_disease.merge(drug_gene, on="gene_id")

by_disease = defaultdict(list)
for d, dr in zip(merged["disease_id"], merged["drug_id"]):
    by_disease[d].append(dr)

positive_pairs = [
    (d, dr)
    for d, drs in by_disease.items()
    for dr in random.sample(drs, min(10, len(drs)))
]

print("Positive pairs:", len(positive_pairs))

# ==================================================
# Train
# ==================================================
model = DrugRepurposingHeteroGNN(hidden_dim=256)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def train():
    model.train()
    optimizer.zero_grad()

    emb = model(data.x_dict, data.edge_index_dict)

    batch = random.sample(positive_pairs, min(256, len(positive_pairs)))
    batch = torch.tensor(batch)

    d, dr = batch[:, 0], batch[:, 1]

    pos = (emb["disease"][d] * emb["drug"][dr]).sum(dim=1)

    neg_dr = torch.randint(0, num_dr, (len(d),))
    neg = (emb["disease"][d] * emb["drug"][neg_dr]).sum(dim=1)

    loss = F.relu(1.0 - pos + neg).mean()
    loss.backward()
    optimizer.step()

    return loss.item()

print("Training...")
for epoch in range(150):
    loss = train()
    if epoch % 25 == 0:
        print(f"Epoch {epoch} | Loss: {loss:.4f}")

torch.save(model.state_dict(), "drug_repurposing_gnn.pt")
print("✅ Model saved")
