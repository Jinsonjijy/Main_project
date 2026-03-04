import torch
import pandas as pd
import random
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from model import BacteriaSpecificHeteroGNN
import json

# ================= CONFIG =================
SEED = 42
EMB_DIM = 320
HIDDEN_DIM = 256
EPOCHS = 100
BATCH_SIZE = 512

random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ================= UTIL =================
def encode(series):
    codes, uniques = pd.factorize(series)
    mapping = dict(enumerate(uniques))
    reverse = {v: k for k, v in mapping.items()}
    return codes, mapping, reverse


# ================= LOAD DATA =================
print("Loading datasets...")

disease_protein = pd.read_csv("data/disease_protein.csv")
disease_drug = pd.read_csv("data/disease_drug_positive.csv")
kegg_lookup = pd.read_csv("data/kegg_drug_lookup.csv")

# Clean column names
disease_protein.columns = disease_protein.columns.str.strip()
disease_drug.columns = disease_drug.columns.str.strip()
kegg_lookup.columns = kegg_lookup.columns.str.strip()

# Encode disease and protein
disease_protein["disease_id"], d_map, d_rev = encode(
    disease_protein["disease_name"]
)

disease_protein["protein_id"], p_map, p_rev = encode(
    disease_protein["uniprot_id"]
)

# Map disease-drug positives
disease_drug["disease_id"] = disease_drug["disease_name"].map(d_rev)

# Create KEGG drug index
unique_drugs = sorted(kegg_lookup["KEGG_ID"].unique())
dr_map = {d: i for i, d in enumerate(unique_drugs)}
dr_rev = {i: d for d, i in dr_map.items()}

disease_drug["drug_id"] = disease_drug["KEGG_ID"].map(dr_map)
disease_drug = disease_drug.dropna(subset=["disease_id", "drug_id"])

num_d = len(d_map)
num_p = len(p_map)
num_dr = len(dr_map)

print(f"Diseases: {num_d}")
print(f"Proteins: {num_p}")
print(f"Drugs: {num_dr}")


# ================= LOAD PROTEIN EMBEDDINGS =================
print("Loading protein embeddings...")

protein_embeddings = torch.load("protein_embeddings.pt", map_location="cpu")

protein_x = []
for protein in p_map.values():
    if protein in protein_embeddings:
        protein_x.append(protein_embeddings[protein])
    else:
        protein_x.append(torch.zeros(EMB_DIM))

protein_x = torch.stack(protein_x)


# ================= LOAD DRUG EMBEDDINGS =================
print("Loading drug embeddings...")

drug_features = torch.load("drug_embeddings_320.pt", map_location="cpu")

with open("drug_map.json", "r") as f:
    full_drug_map = json.load(f)

drug_x = torch.zeros((num_dr, EMB_DIM))

for kegg_id, idx in dr_map.items():
    if kegg_id in full_drug_map:
        drug_x[idx] = drug_features[full_drug_map[kegg_id]]
    else:
        drug_x[idx] = torch.zeros(EMB_DIM)

# Disease embeddings (learnable initial state)
disease_x = torch.randn(num_d, EMB_DIM)


# ================= BUILD GRAPH =================
print("Building heterogeneous graph...")

data = HeteroData()

data["disease"].x = disease_x
data["protein"].x = protein_x
data["drug"].x = drug_x

# Disease → Protein edges
dp_edge = torch.tensor(
    disease_protein[["disease_id", "protein_id"]].values.T,
    dtype=torch.long
)

data["disease", "associated_with", "protein"].edge_index = dp_edge
data["protein", "rev_associated_with", "disease"].edge_index = dp_edge.flip(0)

# Protein → Drug edges
protein_drug_df = kegg_lookup.copy()
protein_drug_df["protein_id"] = protein_drug_df["UniProt"].map(p_rev)
protein_drug_df["drug_id"] = protein_drug_df["KEGG_ID"].map(dr_map)
protein_drug_df = protein_drug_df.dropna(subset=["protein_id", "drug_id"])

pd_edge = torch.tensor(
    protein_drug_df[["protein_id", "drug_id"]].values.T,
    dtype=torch.long
)

data["protein", "targeted_by", "drug"].edge_index = pd_edge
data["drug", "rev_targeted_by", "protein"].edge_index = pd_edge.flip(0)

data = data.to(device)


# ================= TRAIN =================
print("Preparing training pairs...")

positive_pairs = list(set(
    zip(disease_drug["disease_id"], disease_drug["drug_id"])
))

model = BacteriaSpecificHeteroGNN(
    input_dim=EMB_DIM,
    hidden_dim=HIDDEN_DIM
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


def train_epoch():
    model.train()
    optimizer.zero_grad()

    emb = model(data.x_dict, data.edge_index_dict)

    disease_emb = emb["disease"]
    drug_emb = emb["drug"]

    batch = random.sample(
        positive_pairs,
        min(BATCH_SIZE, len(positive_pairs))
    )

    batch = torch.tensor(batch, device=device)

    d = batch[:, 0]
    dr = batch[:, 1]

    pos = F.cosine_similarity(disease_emb[d], drug_emb[dr])

    neg_dr = torch.randint(0, num_dr, (len(d),), device=device)
    neg = F.cosine_similarity(disease_emb[d], drug_emb[neg_dr])

    loss = F.margin_ranking_loss(
        pos,
        neg,
        torch.ones_like(pos),
        margin=0.5
    )

    loss.backward()
    optimizer.step()

    return loss.item()


print("\nTraining started...\n")

for epoch in range(EPOCHS):
    loss = train_epoch()

    if epoch % 10 == 0:
        print(f"Epoch {epoch} | Loss: {loss:.4f}")

torch.save(model.state_dict(), "bacteria_gnn.pt")

print("\nModel saved as bacteria_gnn.pt")