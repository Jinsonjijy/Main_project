import pandas as pd
import torch
import random
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from gnn_model import DrugRepurposingHeteroGNN

# --------------------------------------------------
# Helper functions
# --------------------------------------------------
def encode(series):
    codes, uniques = pd.factorize(series)
    mapping = dict(enumerate(uniques))
    reverse = {v: k for k, v in mapping.items()}
    return codes, mapping, reverse

def normalize(x):
    return str(x).lower().replace(",", "").replace(" ", "").strip()

# --------------------------------------------------
# Load datasets
# --------------------------------------------------
drug_gene = pd.read_csv("data/pharmacologically_active.csv")
gene_disease = pd.read_csv("data/CTD_curated_genes_diseases.csv")

# ⭐ NEW: DrugBank disease–drug
drug_disease = pd.read_csv("data/drugbank_parse.csv")

drug_gene.columns = drug_gene.columns.str.strip()
gene_disease.columns = gene_disease.columns.str.strip()
drug_disease.columns = drug_disease.columns.str.strip()

# Humans only
drug_gene = drug_gene[drug_gene["Species"] == "Humans"]

drug_gene = drug_gene.rename(columns={
    "GeneName": "GeneSymbol",
    "DrugIDs": "DrugID",
    "UniProtID": "UniProtID"
})

drug_gene = drug_gene[["DrugID", "GeneSymbol", "UniProtID"]].drop_duplicates()
gene_disease = gene_disease[["GeneSymbol", "DiseaseName"]].drop_duplicates()

# --------------------------------------------------
# Normalize disease names
# --------------------------------------------------
gene_disease["DiseaseName_norm"] = gene_disease["DiseaseName"].apply(normalize)
drug_disease["DiseaseName_norm"] = drug_disease["DiseaseName"].apply(normalize)

# --------------------------------------------------
# Encode disease & gene
# --------------------------------------------------
gene_disease["disease_id"], disease_map, disease_rev = encode(
    gene_disease["DiseaseName_norm"]
)
gene_disease["gene_id"], gene_map, gene_rev = encode(
    gene_disease["GeneSymbol"]
)

# --------------------------------------------------
# Drug–gene mapping
# --------------------------------------------------
drug_gene["gene_id"] = drug_gene["GeneSymbol"].map(gene_rev)
drug_gene = drug_gene.dropna(subset=["gene_id"])
drug_gene["drug_id"], drug_map, drug_rev = encode(drug_gene["DrugID"])

# --------------------------------------------------
# ⭐ Map DrugBank disease–drug
# --------------------------------------------------
drug_disease["disease_id"] = drug_disease["DiseaseName_norm"].map(disease_rev)
drug_disease["drug_id"] = drug_disease["DrugBankID"].map(drug_rev)

drug_disease = drug_disease.dropna(subset=["disease_id", "drug_id"])

drug_disease = drug_disease[["disease_id", "drug_id"]].drop_duplicates()

num_diseases = len(disease_map)
num_genes = len(gene_map)
num_drugs = len(drug_map)

# --------------------------------------------------
# Load protein embeddings
# --------------------------------------------------
protein_emb = torch.load("protein_embeddings.pt")

gene_to_uniprot = dict(
    zip(drug_gene["GeneSymbol"], drug_gene["UniProtID"])
)

# --------------------------------------------------
# Build heterogeneous graph
# --------------------------------------------------
data = HeteroData()
EMB_DIM = 1024

data["disease"].x = torch.randn(num_diseases, EMB_DIM) * 0.01
data["drug"].x = torch.randn(num_drugs, EMB_DIM) * 0.01

gene_features = []
missing = 0

for gene_symbol in gene_map.values():
    if gene_symbol in gene_to_uniprot:
        uid = gene_to_uniprot[gene_symbol]
        gene_features.append(protein_emb.get(uid, torch.zeros(EMB_DIM)))
        if uid not in protein_emb:
            missing += 1
    else:
        gene_features.append(torch.zeros(EMB_DIM))
        missing += 1

data["gene"].x = torch.stack(gene_features)
print(f"⚠ Missing protein embeddings for {missing} genes")

# --------------------------------------------------
# Edges
# --------------------------------------------------
dg = torch.tensor(
    gene_disease[["disease_id", "gene_id"]].values.T, dtype=torch.long
)
gd = torch.tensor(
    drug_gene[["gene_id", "drug_id"]].values.T, dtype=torch.long
)

dd = torch.tensor(
    drug_disease[["drug_id", "disease_id"]].values.T, dtype=torch.long
)

data["disease", "associates", "gene"].edge_index = dg
data["gene", "rev_associates", "disease"].edge_index = dg.flip(0)

data["gene", "targets", "drug"].edge_index = gd
data["drug", "rev_targets", "gene"].edge_index = gd.flip(0)

# ⭐ NEW: disease–drug indication
data["drug", "treats", "disease"].edge_index = dd
data["disease", "rev_treats", "drug"].edge_index = dd.flip(0)

# --------------------------------------------------
# Training pairs (NO leakage)
# --------------------------------------------------
merged = gene_disease.merge(drug_gene, on="gene_id")
positive_pairs = list(set(zip(
    merged["disease_id"],
    merged["drug_id"]
)))

# --------------------------------------------------
# Model
# --------------------------------------------------
model = DrugRepurposingHeteroGNN(hidden_dim=256)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# --------------------------------------------------
# Training loop
# --------------------------------------------------
def train():
    model.train()
    optimizer.zero_grad()

    emb = model(data.x_dict, data.edge_index_dict)

    batch = random.sample(positive_pairs, min(256, len(positive_pairs)))
    batch = torch.tensor(batch, dtype=torch.long)

    d_ids = batch[:, 0]
    dr_ids = batch[:, 1]

    pos = (emb["disease"][d_ids] * emb["drug"][dr_ids]).sum(dim=1)

    neg_dr_ids = torch.randint(0, num_drugs, (len(d_ids),))
    neg = (emb["disease"][d_ids] * emb["drug"][neg_dr_ids]).sum(dim=1)

    loss = F.relu(1.0 - pos + neg).mean()
    loss.backward()
    optimizer.step()

    return loss.item()

print("Training model...")
for epoch in range(150):
    loss = train()
    if epoch % 25 == 0:
        print(f"Epoch {epoch} | Loss: {loss:.4f}")

torch.save(model.state_dict(), "drug_repurposing_gnn.pt")
print("Model saved as drug_repurposing_gnn.pt")
