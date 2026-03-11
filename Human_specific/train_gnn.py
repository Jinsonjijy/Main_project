import pandas as pd
import torch
import random
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from gnn_model import DrugRepurposingHeteroGNN
import json

SEED = 42
EMB_DIM = 644
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


# ================= LOAD DATA =================

gene_disease = pd.read_csv("data/core_plus_disease_gene.csv")
drug_gene = pd.read_csv("data/pharmacologically_active.csv")
metadata = pd.read_csv("data/drug_target_metadata_cleaned.csv")
drug_disease = pd.read_csv("data/drug_disease_edges.csv")

drug_disease["disease_norm"] = drug_disease["disease"].apply(normalize)

drug_gene = drug_gene[drug_gene["Species"] == "Humans"]

drug_gene = drug_gene.rename(columns={
    "GeneName": "GeneSymbol",
    "DrugIDs": "DrugID"
})

gene_disease["DiseaseName_norm"] = gene_disease["DiseaseName"].apply(normalize)

gene_disease["disease_id"], d_map, d_rev = encode(gene_disease["DiseaseName_norm"])
gene_disease["gene_id"], g_map, g_rev = encode(gene_disease["GeneSymbol"])

drug_gene["gene_id"] = drug_gene["GeneSymbol"].map(g_rev)
drug_gene = drug_gene.dropna(subset=["gene_id"])

unique_drugs = sorted(drug_gene["DrugID"].unique())
dr_map = {dbid: idx for idx, dbid in enumerate(unique_drugs)}
dr_rev = {idx: dbid for dbid, idx in dr_map.items()}

drug_disease["disease_id"] = drug_disease["disease_norm"].map(d_rev)
drug_disease["drug_id"] = drug_disease["drug_id"].map(dr_map)

drug_disease = drug_disease.dropna(subset=["disease_id", "drug_id"])

drug_gene["drug_id"] = drug_gene["DrugID"].map(dr_map)

num_d = len(d_map)
num_g = len(g_map)
num_dr = len(dr_map)

print(f"Diseases: {num_d}, Genes: {num_g}, Drugs: {num_dr}")


# ================= LOAD FEATURES =================

protein_emb = torch.load("gene_features_full.pt", map_location="cpu")
drug_emb_all = torch.load("drug_embeddings.pt")

with open("data/drug_map.json", "r") as f:
    full_drug_map = json.load(f)

drug_features = torch.zeros((num_dr, EMB_DIM))

for dbid, idx in dr_map.items():
    if dbid in drug_emb_all:
        drug_features[idx] = drug_emb_all[dbid]


# ================= GRAPH =================

data = HeteroData()
data["drug"].x = drug_features

# -------- Gene features --------

gene_x = []
for gene_symbol in g_map.values():

    if gene_symbol in protein_emb:
        gene_x.append(protein_emb[gene_symbol])
    else:
        gene_x.append(torch.zeros(EMB_DIM))

data["gene"].x = torch.stack(gene_x)

# -------- Disease features --------

disease_emb = torch.load("disease_features_644.pt")
disease_x = []

for disease in d_map.values():
    if disease in disease_emb:
        disease_x.append(disease_emb[disease])
    else:
        disease_x.append(torch.zeros(EMB_DIM))

data["disease"].x = torch.stack(disease_x)


# ================= CORE EDGES =================

dg = torch.tensor(
    gene_disease[["disease_id", "gene_id"]].values.T,
    dtype=torch.long
)

gd = torch.tensor(
    drug_gene[["gene_id", "drug_id"]].values.T,
    dtype=torch.long
)

data["disease", "associates", "gene"].edge_index = dg
data["gene", "rev_associates", "disease"].edge_index = dg.flip(0)

data["gene", "targets", "drug"].edge_index = gd
data["drug", "rev_targets", "gene"].edge_index = gd.flip(0)


# ================= DRUG-DISEASE EDGES =================

# dd = torch.tensor(
#     drug_disease[["drug_id", "disease_id"]].values.T,
#     dtype=torch.long
# )

# data["drug", "treats", "disease"].edge_index = dd
# data["disease", "rev_treats", "drug"].edge_index = dd.flip(0)


# ================= METADATA RELATIONS =================

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


# ================= PPI EDGES =================

print("\nLoading PPI edges...")

ppi = pd.read_csv("data/ppi_edges.csv")

ppi["g1"] = ppi["Gene1"].map(g_rev)
ppi["g2"] = ppi["Gene2"].map(g_rev)

ppi = ppi.dropna()

ppi_edges = torch.tensor(
    ppi[["g1", "g2"]].values.T,
    dtype=torch.long
)

ppi_edges = torch.cat([ppi_edges, ppi_edges.flip(0)], dim=1)

data["gene", "interacts", "gene"].edge_index = ppi_edges
data["gene", "rev_interacts", "gene"].edge_index = ppi_edges.flip(0)

print("Gene-Gene PPI edges:", ppi_edges.shape[1])


data = data.to(device)


# ================= TRAIN / TEST SPLIT =================

from sklearn.model_selection import train_test_split

positive_pairs = list(zip(
    drug_disease["disease_id"].astype(int),
    drug_disease["drug_id"].astype(int)
))

train_pairs, test_pairs = train_test_split(
    positive_pairs,
    test_size=0.2,
    random_state=42
)

positive_set = set(train_pairs)

print("Train pairs:", len(train_pairs))
print("Test pairs:", len(test_pairs))


# ================= BUILD GENE LOOKUPS =================

disease_genes = {}

for d, g in zip(gene_disease["disease_id"], gene_disease["gene_id"]):
    disease_genes.setdefault(d, set()).add(g)


gene_to_drugs = {}

for g, dr in zip(drug_gene["gene_id"], drug_gene["drug_id"]):
    gene_to_drugs.setdefault(g, set()).add(dr)


# ================= MODEL =================

model = DrugRepurposingHeteroGNN(
    input_dim=EMB_DIM,
    hidden_dim=HIDDEN_DIM
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# ================= HARD NEGATIVE SAMPLING =================

def sample_hard_negative(disease_id):

    genes = disease_genes.get(disease_id, set())

    candidate_drugs = set()

    for g in gene_to_drugs:
        if g not in genes:
            candidate_drugs.update(gene_to_drugs[g])

    candidates = list(candidate_drugs)
    random.shuffle(candidates)

    for dr in candidates:
        if (disease_id, dr) not in positive_set:
            return dr

    while True:
        dr = random.randint(0, num_dr - 1)
        if (disease_id, dr) not in positive_set:
            return dr


# ================= TRAIN LOOP =================

def train_epoch():

    model.train()
    optimizer.zero_grad()

    emb = model(data.x_dict, data.edge_index_dict)

    disease_emb = F.normalize(emb["disease"], dim=1)
    drug_emb = F.normalize(emb["drug"], dim=1)

    batch = random.sample(train_pairs, min(BATCH_SIZE, len(train_pairs)))
    batch = torch.tensor(batch, device=device)

    d = batch[:, 0]
    dr = batch[:, 1]

    pos = F.cosine_similarity(
        disease_emb[d],
        drug_emb[dr]
    )

    neg_drugs = []

    for disease_id in d.tolist():
        neg_drugs.append(sample_hard_negative(disease_id))

    neg_drugs = torch.tensor(neg_drugs, device=device)

    neg = F.cosine_similarity(
        disease_emb[d],
        drug_emb[neg_drugs]
    )

    loss = F.margin_ranking_loss(
        pos,
        neg,
        torch.ones_like(pos),
        margin=0.5
    )

    loss.backward()
    optimizer.step()

    return loss.item()


# ================= TRAIN =================

print("\nTraining...")

for epoch in range(EPOCHS):

    loss = train_epoch()

    if epoch % 10 == 0:
        print(f"Epoch {epoch} | Loss: {loss:.4f}")


torch.save(model.state_dict(), "drug_repurposing_gnn.pt")

print("Model saved.")