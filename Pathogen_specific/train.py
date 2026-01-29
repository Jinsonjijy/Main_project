import pandas as pd
import torch
import random
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from model import PathogenDrugHeteroGNN

# ==================================================
# CONFIG
# ==================================================
PROTEIN_DIM = 320          # ESM2 embedding dimension
HIDDEN_DIM = 256
LR = 1e-3
EPOCHS = 200
BATCH_SIZE = 256
MARGIN = 1.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(42)
random.seed(42)

# ==================================================
# HELPER FUNCTION
# ==================================================
def encode(series):
    """
    Convert categorical labels into integer indices
    """
    codes, uniques = pd.factorize(series)
    return codes, list(uniques)

# ==================================================
# LOAD FINAL DATA FILES
# ==================================================
print("\n📌 Loading CSV files...")

proteins = pd.read_csv("data/all_bacteria.csv")
pp = pd.read_csv("data/ALL_protein_pathway_STEP2B.csv")
anchors = pd.read_csv("data/ALL_step3_FINAL.csv")
drug_direct = pd.read_csv("data/ALL_step4_protein_drug.csv")
drug_infer = pd.read_csv("data/ALL_step5_KO_inferred_drugs.csv")

# Fix column naming
drug_direct = drug_direct.rename(columns={"drugbank_id": "drug_id"})
drug_infer = drug_infer.rename(columns={"drugbank_id": "drug_id"})

print("✅ CSV files loaded successfully!")

# ==================================================
# BUILD PROTEIN–PATHWAY EDGES
# ==================================================
print("\n📌 Building Protein–Pathway edges...")

pp_main = pp[["uniprot_id", "map_pathway_id"]]

pp_anchor = anchors[["uniprot_id", "pathway_id"]].rename(
    columns={"pathway_id": "map_pathway_id"}
)

protein_pathway = (
    pd.concat([pp_main, pp_anchor], ignore_index=True)
    .drop_duplicates()
)

assert len(protein_pathway) > 0, "❌ Protein–Pathway edges empty!"

print("✅ Protein–Pathway edges created:", len(protein_pathway))

# ==================================================
# BUILD PATHWAY–DRUG EDGES
# ==================================================
print("\n📌 Building Pathway–Drug edges...")

pd_direct = (
    drug_direct.merge(protein_pathway, on="uniprot_id", how="inner")
    [["map_pathway_id", "drug_id"]]
)

pd_infer = drug_infer[["pathway_id", "drug_id"]].rename(
    columns={"pathway_id": "map_pathway_id"}
)

pathway_drug = (
    pd.concat([pd_direct, pd_infer], ignore_index=True)
    .drop_duplicates()
)

assert len(pathway_drug) > 0, "❌ Pathway–Drug edges empty!"

print("✅ Pathway–Drug edges created:", len(pathway_drug))

# ==================================================
# ENCODE NODE IDS
# ==================================================
print("\n📌 Encoding node IDs...")

proteins["pathogen_idx"], pathogen_ids = encode(proteins["pathogen"])
proteins["protein_idx"], protein_ids = encode(proteins["uniprot_id"])

# Protein → Pathway encoding
protein_pathway["protein_idx"] = protein_pathway["uniprot_id"].map(
    {p: i for i, p in enumerate(protein_ids)}
)
protein_pathway.dropna(inplace=True)

protein_pathway["pathway_idx"], pathway_ids = encode(
    protein_pathway["map_pathway_id"]
)

# Pathway → Drug encoding
pathway_drug["pathway_idx"] = pathway_drug["map_pathway_id"].map(
    {p: i for i, p in enumerate(pathway_ids)}
)
pathway_drug.dropna(inplace=True)

pathway_drug["drug_idx"], drug_ids = encode(pathway_drug["drug_id"])

print(
    f"\n✅ Graph Nodes Summary:\n"
    f"Pathogens: {len(pathogen_ids)}\n"
    f"Proteins : {len(protein_ids)}\n"
    f"Pathways : {len(pathway_ids)}\n"
    f"Drugs    : {len(drug_ids)}"
)

# ==================================================
# LOAD PROTEIN EMBEDDINGS
# ==================================================
print("\n📌 Loading ESM2 protein embeddings...")

protein_emb = torch.load("protein_embeddings_all.pt")

print("✅ Protein embeddings loaded!")

# ==================================================
# BUILD HETEROGENEOUS GRAPH
# ==================================================
print("\n📌 Building Heterogeneous Graph...")

data = HeteroData()

# ---------------- Protein Features ----------------
data["protein"].x = torch.stack([
    protein_emb.get(pid, torch.zeros(PROTEIN_DIM))
    for pid in protein_ids
])

# ---------------- Node Counts (IMPORTANT FIX) ----------------
data["drug"].num_nodes = len(drug_ids)
data["pathway"].num_nodes = len(pathway_ids)
data["pathogen"].num_nodes = len(pathogen_ids)

# ------------------ Edge Index ------------------

# Pathogen → Protein
pp_edge = torch.tensor(
    proteins[["pathogen_idx", "protein_idx"]].values.T,
    dtype=torch.long
)

# Protein → Pathway
pw_edge = torch.tensor(
    protein_pathway[["protein_idx", "pathway_idx"]].values.T,
    dtype=torch.long
)

# Pathway → Drug
pd_edge = torch.tensor(
    pathway_drug[["pathway_idx", "drug_idx"]].values.T,
    dtype=torch.long
)

# Assign relations
data["pathogen", "rev_has", "protein"].edge_index = pp_edge
data["protein", "has", "pathogen"].edge_index = pp_edge.flip(0)

data["protein", "rev_in_pathway", "pathway"].edge_index = pw_edge
data["pathway", "in_pathway", "protein"].edge_index = pw_edge.flip(0)

data["pathway", "targeted_by", "drug"].edge_index = pd_edge
data["drug", "rev_targeted_by", "pathway"].edge_index = pd_edge.flip(0)

data = data.to(DEVICE)

print("✅ Graph built successfully!")

# ==================================================
# BUILD POSITIVE TRAINING PAIRS (PATHOGEN, DRUG)
# ==================================================
print("\n📌 Building positive pathogen–drug pairs...")

merged = (
    proteins.merge(protein_pathway, on="protein_idx")
    .merge(pathway_drug, on="pathway_idx")
)

pos_pairs = list(set(zip(
    merged["pathogen_idx"],
    merged["drug_idx"]
)))

assert len(pos_pairs) > 0, "❌ No supervision signal found!"

print("✅ Positive pathogen–drug pairs:", len(pos_pairs))

# ==================================================
# INITIALIZE MODEL
# ==================================================
print("\n📌 Initializing model...")

model = PathogenDrugHeteroGNN(
    num_drugs=len(drug_ids),
    num_pathways=len(pathway_ids),
    num_pathogens=len(pathogen_ids),
    protein_dim=PROTEIN_DIM,
    hidden_dim=HIDDEN_DIM
).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

print("✅ Model initialized successfully!")

# ==================================================
# TRAINING STEP
# ==================================================
def train_step():
    model.train()
    optimizer.zero_grad()

    emb = model(data)

    # Sample batch
    batch = random.sample(pos_pairs, min(BATCH_SIZE, len(pos_pairs)))
    batch = torch.tensor(batch, device=DEVICE)

    p, d_pos = batch[:, 0], batch[:, 1]

    # Negative sampling
    d_neg = torch.randint(
        0, emb["drug"].size(0),
        (len(p),),
        device=DEVICE
    )

    # Dot-product scores
    pos_score = (emb["pathogen"][p] * emb["drug"][d_pos]).sum(dim=1)
    neg_score = (emb["pathogen"][p] * emb["drug"][d_neg]).sum(dim=1)

    # Margin ranking loss
    loss = F.relu(MARGIN - pos_score + neg_score).mean()

    loss.backward()
    optimizer.step()

    return loss.item()

# ==================================================
# TRAIN LOOP
# ==================================================
print("\n🚀 Training Started...\n")

for epoch in range(EPOCHS):
    loss = train_step()

    if epoch % 25 == 0:
        print(f"Epoch {epoch:03d} | Loss = {loss:.4f}")

# ==================================================
# SAVE MODEL
# ==================================================
torch.save(model.state_dict(), "pathogen_gnn.pt")
print("\n✅ Training complete! Model saved as pathogen_gnn.pt")
