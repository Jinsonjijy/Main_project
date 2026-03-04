import pandas as pd
import torch
import random
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from gnn_model import DrugRepurposingHeteroGNN
import json

SEED = 42
EMB_DIM = 640
HIDDEN_DIM = 256
EPOCHS = 100
BATCH_SIZE = 512

random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def encode(series):
    codes, uniques = pd.factorize(series)
    mapping = dict(enumerate(uniques))
    reverse = {v: k for k, v in mapping.items()}
    return codes, mapping, reverse


def normalize(x):
    return str(x).lower().replace(",", "").replace(" ", "").strip()


# ================= DATA =================
gene_disease = pd.read_csv("data/core_plus_disease_gene.csv")
drug_gene = pd.read_csv("data/pharmacologically_active.csv")
drug_disease = pd.read_csv("data/drug_parser.csv")
metadata = pd.read_csv("data/drug_target_metadata_cleaned.csv")

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

unique_drugs = sorted(drug_gene["DrugID"].unique())
dr_map = {dbid: idx for idx, dbid in enumerate(unique_drugs)}
dr_rev = {idx: dbid for dbid, idx in dr_map.items()}
drug_gene["drug_id"] = drug_gene["DrugID"].map(dr_map)

drug_disease["disease_id"] = drug_disease["DiseaseName_norm"].map(d_rev)
drug_disease["drug_id"] = drug_disease["DrugBankID"].map(dr_map)
drug_disease = drug_disease.dropna(subset=["disease_id", "drug_id"])

num_d = len(d_map)
num_g = len(g_map)
num_dr = len(dr_map)

print(f"Diseases: {num_d}, Genes: {num_g}, Drugs: {num_dr}")

protein_emb = torch.load("protein_embeddings.pt", map_location="cpu")
full_drug_features = torch.load("drug_embeddings.pt")

with open("data/drug_map.json", "r") as f:
    full_drug_map = json.load(f)

drug_features = torch.zeros((num_dr, EMB_DIM))
for dbid, new_idx in dr_map.items():
    drug_features[new_idx] = full_drug_features[full_drug_map[dbid]]

# ============ GRAPH ============
data = HeteroData()
data["drug"].x = drug_features

gene_x = []
for gene_symbol in g_map.values():
    if gene_symbol in protein_emb:
        gene_x.append(protein_emb[gene_symbol])
    else:
        gene_x.append(torch.zeros(EMB_DIM))

data["gene"].x = torch.stack(gene_x)
data["disease"].x = torch.randn(num_d, EMB_DIM)

dg = torch.tensor(
    gene_disease[["disease_id", "gene_id"]].values.T,
    dtype=torch.long
)

gd = torch.tensor(
    drug_gene[["gene_id", "drug_id"]].values.T,
    dtype=torch.long
)

dd = torch.tensor(
    drug_disease[["drug_id", "disease_id"]].values.T,
    dtype=torch.long
)

data["disease", "associates", "gene"].edge_index = dg
data["gene", "rev_associates", "disease"].edge_index = dg.flip(0)
data["gene", "targets", "drug"].edge_index = gd
data["drug", "rev_targets", "gene"].edge_index = gd.flip(0)

# ==== Metadata Relations ====
metadata = metadata[
    metadata["DrugBankID"].isin(dr_map.keys())
]
metadata = metadata[
    metadata["TargetGene"].isin(g_map.values())
]

metadata["drug_id"] = metadata["DrugBankID"].map(dr_map)
metadata["gene_id"] = metadata["TargetGene"].map(g_rev)
metadata = metadata.dropna(subset=["drug_id", "gene_id"])


def build_edges(df):
    return torch.tensor(
        df[["gene_id", "drug_id"]].values.T,
        dtype=torch.long
    )


inhibits_df = metadata[metadata["Relation"] == "inhibits"]
activates_df = metadata[metadata["Relation"] == "activates"]

if len(inhibits_df) > 0:
    inh = build_edges(inhibits_df)
    data["gene", "inhibits", "drug"].edge_index = inh
    data["drug", "rev_inhibits", "gene"].edge_index = inh.flip(0)

if len(activates_df) > 0:
    act = build_edges(activates_df)
    data["gene", "activates", "drug"].edge_index = act
    data["drug", "rev_activates", "gene"].edge_index = act.flip(0)

data = data.to(device)

# ================= GRAPH STATISTICS =================
num_dg = dg.shape[1]
num_gd = gd.shape[1]
num_inh = len(inhibits_df)
num_act = len(activates_df)

print("\n===== Graph Statistics =====")
print(f"Diseases: {num_d}")
print(f"Genes: {num_g}")
print(f"Drugs: {num_dr}")
print(f"Disease–Gene edges: {num_dg}")
print(f"Gene–Drug (targets): {num_gd}")
print(f"Gene–Drug (inhibits): {num_inh}")
print(f"Gene–Drug (activates): {num_act}")
print(f"Total edges: {num_dg + num_gd + num_inh + num_act}")
# ============ TRAIN ============
merged = gene_disease.merge(drug_gene, on="gene_id")
positive_pairs = list(set(zip(merged["disease_id"], merged["drug_id"])))

model = DrugRepurposingHeteroGNN(
    input_dim=EMB_DIM,
    hidden_dim=HIDDEN_DIM
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


def train_epoch():
    model.train()
    optimizer.zero_grad()

    emb = model(data.x_dict, data.edge_index_dict)
    disease_emb = F.normalize(emb["disease"], dim=1)
    drug_emb = F.normalize(emb["drug"], dim=1)

    batch = random.sample(positive_pairs, min(BATCH_SIZE, len(positive_pairs)))
    batch = torch.tensor(batch, device=device)

    d = batch[:, 0]
    dr = batch[:, 1]

    pos = F.cosine_similarity(disease_emb[d], drug_emb[dr])

    neg_dr = torch.randint(0, num_dr, (len(d),), device=device)
    neg = F.cosine_similarity(disease_emb[d], drug_emb[neg_dr])

    loss = F.margin_ranking_loss(pos, neg, torch.ones_like(pos), margin=0.5)

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
