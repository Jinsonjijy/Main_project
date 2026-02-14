import pandas as pd
import torch
import random
import torch.nn.functional as F
from collections import defaultdict
from torch_geometric.data import HeteroData
from gnn_model import DrugRepurposingHeteroGNN

# ==============================
# CONFIG
# ==============================
SEED = 42
EMB_DIM = 640
HIDDEN_DIM = 256
EPOCHS = 100
BATCH_SIZE = 512

random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ==============================
# Helpers
# ==============================
def encode(series):
    codes, uniques = pd.factorize(series)
    mapping = dict(enumerate(uniques))
    reverse = {v: k for k, v in mapping.items()}
    return codes, mapping, reverse

def normalize(x):
    return str(x).lower().replace(",", "").replace(" ", "").strip()

# ==============================
# Load datasets
# ==============================
gene_disease = pd.read_csv("data/core_plus_disease_gene.csv")
drug_gene = pd.read_csv("data/pharmacologically_active.csv")
drug_disease = pd.read_csv("data/drug_parser.csv")

drug_gene = drug_gene[drug_gene["Species"] == "Humans"]
drug_gene = drug_gene.rename(columns={
    "GeneName": "GeneSymbol",
    "DrugIDs": "DrugID"
})

gene_disease["DiseaseName_norm"] = gene_disease["DiseaseName"].apply(normalize)
drug_disease["DiseaseName_norm"] = drug_disease["DiseaseName"].apply(normalize)

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

print(f"Diseases: {num_d}, Genes: {num_g}, Drugs: {num_dr}")

# ==============================
# Load protein embeddings
# ==============================
protein_emb = torch.load("protein_embeddings.pt", map_location="cpu")

# ==============================
# Build Graph
# ==============================
data = HeteroData()

data["drug"].x = torch.randn(num_dr, EMB_DIM) * 0.01

gene_x = []
missing = 0

for gene_symbol in g_map.values():
    if gene_symbol in protein_emb and protein_emb[gene_symbol].shape[0] == EMB_DIM:
        gene_x.append(protein_emb[gene_symbol])
    else:
        gene_x.append(torch.zeros(EMB_DIM))
        missing += 1

data["gene"].x = torch.stack(gene_x)
print("Missing gene embeddings:", missing)

data["disease"].x = torch.randn(num_d, EMB_DIM) * 0.01

dg_edges = torch.tensor(
    gene_disease[["disease_id", "gene_id"]].values.T,
    dtype=torch.long
)

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

data = data.to(device)

# ==============================
# Positive pairs
# ==============================
merged = gene_disease.merge(drug_gene, on="gene_id")

positive_pairs = list(set(zip(merged["disease_id"], merged["drug_id"])))
print("Total positive pairs:", len(positive_pairs))

# ==============================
# Model
# ==============================
model = DrugRepurposingHeteroGNN(
    input_dim=EMB_DIM,
    hidden_dim=HIDDEN_DIM
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ==============================
# Training loop
# ==============================
def train_epoch():
    model.train()
    optimizer.zero_grad()

    emb = model(data.x_dict, data.edge_index_dict)
    disease_emb = emb["disease"]
    drug_emb = emb["drug"]

    batch = random.sample(positive_pairs, min(BATCH_SIZE, len(positive_pairs)))
    batch = torch.tensor(batch, device=device)

    d = batch[:, 0]
    dr = batch[:, 1]

    pos_scores = (disease_emb[d] * drug_emb[dr]).sum(dim=1)

    neg_dr = torch.randint(0, num_dr, (len(d),), device=device)
    neg_scores = (disease_emb[d] * drug_emb[neg_dr]).sum(dim=1)

    loss = F.margin_ranking_loss(
        pos_scores,
        neg_scores,
        torch.ones_like(pos_scores),
        margin=0.5
    )

    loss.backward()
    optimizer.step()

    return loss.item()

print("\nTraining...")
for epoch in range(EPOCHS):
    loss = train_epoch()
    if epoch % 10 == 0:
        print(f"Epoch {epoch} | Loss: {loss:.4f}")

torch.save(model.state_dict(), "drug_repurposing_gnn.pt")
print("Model saved.")
