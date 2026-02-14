import pandas as pd
import torch
import random
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from model import PathogenDrugHeteroGNN

# ==================================================
# CONFIG
# ==================================================
PROTEIN_DIM = 320
HIDDEN_DIM = 256
LR = 1e-3
EPOCHS = 200
BATCH_SIZE = 256
MARGIN = 1.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(42)
random.seed(42)

# ==================================================
# HELPER
# ==================================================
def encode(series):
    codes, uniques = pd.factorize(series)
    return codes, list(uniques)

# ==================================================
# LOAD DATA
# ==================================================
print("\n📌 Loading CSV files...")

proteins = pd.read_csv("data/all_bacteria.csv")
pp = pd.read_csv("data/ALL_protein_pathway_STEP2B.csv")
anchors = pd.read_csv("data/ALL_step3_FINAL.csv")
drug_direct = pd.read_csv("data/ALL_step4_protein_drug.csv")
drug_infer = pd.read_csv("data/ALL_step5_KO_inferred_drugs.csv")

drug_direct = drug_direct.rename(columns={"drugbank_id": "drug_id"})
drug_infer = drug_infer.rename(columns={"drugbank_id": "drug_id"})

print("✅ CSV files loaded!")

# ==================================================
# BUILD EDGES
# ==================================================
pp_main = pp[["uniprot_id", "map_pathway_id"]]
pp_anchor = anchors[["uniprot_id", "pathway_id"]].rename(
    columns={"pathway_id": "map_pathway_id"}
)

protein_pathway = pd.concat([pp_main, pp_anchor], ignore_index=True).drop_duplicates()

pd_direct = (
    drug_direct.merge(protein_pathway, on="uniprot_id", how="inner")
    [["map_pathway_id", "drug_id"]]
)

pd_infer = drug_infer[["pathway_id", "drug_id"]].rename(
    columns={"pathway_id": "map_pathway_id"}
)

pathway_drug = pd.concat([pd_direct, pd_infer], ignore_index=True).drop_duplicates()

# ==================================================
# ENCODE IDS
# ==================================================
proteins["pathogen_idx"], pathogen_ids = encode(proteins["pathogen"])
proteins["protein_idx"], protein_ids = encode(proteins["uniprot_id"])

protein_pathway["protein_idx"] = protein_pathway["uniprot_id"].map(
    {p: i for i, p in enumerate(protein_ids)}
)
protein_pathway.dropna(inplace=True)

protein_pathway["pathway_idx"], pathway_ids = encode(
    protein_pathway["map_pathway_id"]
)

pathway_drug["pathway_idx"] = pathway_drug["map_pathway_id"].map(
    {p: i for i, p in enumerate(pathway_ids)}
)
pathway_drug.dropna(inplace=True)

pathway_drug["drug_idx"], drug_ids = encode(pathway_drug["drug_id"])

print(f"""
Graph Summary:
Pathogens: {len(pathogen_ids)}
Proteins : {len(protein_ids)}
Pathways : {len(pathway_ids)}
Drugs    : {len(drug_ids)}
""")

# ==================================================
# LOAD PROTEIN EMBEDDINGS
# ==================================================
protein_emb = torch.load("protein_embeddings_all.pt")

# ==================================================
# BUILD GRAPH
# ==================================================
data = HeteroData()

data["protein"].x = torch.stack([
    protein_emb.get(pid, torch.zeros(PROTEIN_DIM))
    for pid in protein_ids
])

data["drug"].num_nodes = len(drug_ids)
data["pathway"].num_nodes = len(pathway_ids)
data["pathogen"].num_nodes = len(pathogen_ids)

pp_edge = torch.tensor(
    proteins[["pathogen_idx", "protein_idx"]].values.T,
    dtype=torch.long
)

pw_edge = torch.tensor(
    protein_pathway[["protein_idx", "pathway_idx"]].values.T,
    dtype=torch.long
)

pd_edge = torch.tensor(
    pathway_drug[["pathway_idx", "drug_idx"]].values.T,
    dtype=torch.long
)

data["pathogen", "rev_has", "protein"].edge_index = pp_edge
data["protein", "has", "pathogen"].edge_index = pp_edge.flip(0)

data["protein", "rev_in_pathway", "pathway"].edge_index = pw_edge
data["pathway", "in_pathway", "protein"].edge_index = pw_edge.flip(0)

data["pathway", "targeted_by", "drug"].edge_index = pd_edge
data["drug", "rev_targeted_by", "pathway"].edge_index = pd_edge.flip(0)

data = data.to(DEVICE)

# ==================================================
# BUILD POSITIVE PAIRS
# ==================================================
merged = (
    proteins.merge(protein_pathway, on="protein_idx")
    .merge(pathway_drug, on="pathway_idx")
)

pos_pairs = list(set(zip(
    merged["pathogen_idx"],
    merged["drug_idx"]
)))

pos_set = set(pos_pairs)

print("Positive pairs:", len(pos_pairs))

# ==================================================
# MODEL
# ==================================================
model = PathogenDrugHeteroGNN(
    num_drugs=len(drug_ids),
    num_pathways=len(pathway_ids),
    num_pathogens=len(pathogen_ids),
    protein_dim=PROTEIN_DIM,
    hidden_dim=HIDDEN_DIM
).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ==================================================
# TRAIN STEP
# ==================================================
def train_step():
    model.train()
    optimizer.zero_grad()

    emb = model(data)

    batch = random.sample(pos_pairs, min(BATCH_SIZE, len(pos_pairs)))
    batch = torch.tensor(batch, device=DEVICE)

    p, d_pos = batch[:, 0], batch[:, 1]

    # --- Negative sampling (avoid true positives) ---
    d_neg = []
    for pi in p.tolist():
        while True:
            candidate = random.randint(0, len(drug_ids) - 1)
            if (pi, candidate) not in pos_set:
                d_neg.append(candidate)
                break

    d_neg = torch.tensor(d_neg, device=DEVICE)

    # --- Normalize embeddings ---
    p_emb = F.normalize(emb["pathogen"][p], dim=1)
    d_pos_emb = F.normalize(emb["drug"][d_pos], dim=1)
    d_neg_emb = F.normalize(emb["drug"][d_neg], dim=1)

    pos_score = (p_emb * d_pos_emb).sum(dim=1)
    neg_score = (p_emb * d_neg_emb).sum(dim=1)

    loss = F.relu(MARGIN - pos_score + neg_score).mean()

    loss.backward()
    optimizer.step()

    return loss.item(), pos_score.mean().item(), neg_score.mean().item()

# ==================================================
# TRAIN LOOP
# ==================================================
print("\n🚀 Training Started...\n")

for epoch in range(EPOCHS):

    loss, pos_m, neg_m = train_step()

    if epoch % 25 == 0:
        print(
            f"Epoch {epoch:03d} | "
            f"Loss {loss:.4f} | "
            f"Pos {pos_m:.4f} | "
            f"Neg {neg_m:.4f}"
        )

# ==================================================
# SAVE
# ==================================================
torch.save(model.state_dict(), "pathogen_gnn.pt")
print("\n✅ Training complete! Model saved as pathogen_gnn.pt")
