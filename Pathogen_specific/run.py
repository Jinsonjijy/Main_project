import torch
import torch.nn.functional as F
import pandas as pd
from torch_geometric.data import HeteroData
from model import PathogenDrugHeteroGNN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROTEIN_DIM = 320

# ==================================================
# Helper
# ==================================================
def encode(series):
    codes, uniques = pd.factorize(series)
    return codes, list(uniques)

# ==================================================
# LOAD SAME DATA AS TRAINING
# ==================================================
proteins = pd.read_csv("data/all_bacteria.csv")
pp = pd.read_csv("data/ALL_protein_pathway_STEP2B.csv")
anchors = pd.read_csv("data/ALL_step3_FINAL.csv")
drug_direct = pd.read_csv("data/ALL_step4_protein_drug.csv")
drug_infer = pd.read_csv("data/ALL_step5_KO_inferred_drugs.csv")

# Fix column names
drug_direct = drug_direct.rename(columns={"drugbank_id": "drug_id"})
drug_infer = drug_infer.rename(columns={"drugbank_id": "drug_id"})

# ==================================================
# BUILD PROTEIN–PATHWAY
# ==================================================
pp_main = pp[["uniprot_id", "map_pathway_id"]]
pp_anchor = anchors[["uniprot_id", "pathway_id"]] \
    .rename(columns={"pathway_id": "map_pathway_id"})

protein_pathway = (
    pd.concat([pp_main, pp_anchor], ignore_index=True)
    .drop_duplicates()
)

# ==================================================
# BUILD PATHWAY–DRUG
# ==================================================
pd_direct = (
    drug_direct
    .merge(protein_pathway, on="uniprot_id", how="inner")
    [["map_pathway_id", "drug_id"]]
)

pd_infer = drug_infer[["pathway_id", "drug_id"]] \
    .rename(columns={"pathway_id": "map_pathway_id"})

pathway_drug = (
    pd.concat([pd_direct, pd_infer], ignore_index=True)
    .drop_duplicates()
)

# ==================================================
# ENCODE NODES (MUST MATCH TRAINING)
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

pathogen_rev = {p: i for i, p in enumerate(pathogen_ids)}
drug_map = {i: d for i, d in enumerate(drug_ids)}

print(
    f"Loaded graph → "
    f"Pathogens {len(pathogen_ids)}, "
    f"Proteins {len(protein_ids)}, "
    f"Pathways {len(pathway_ids)}, "
    f"Drugs {len(drug_ids)}"
)

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
# LOAD MODEL
# ==================================================
model = PathogenDrugHeteroGNN(
    num_drugs=len(drug_ids),
    num_pathways=len(pathway_ids),
    num_pathogens=len(pathogen_ids),
    protein_dim=PROTEIN_DIM,
    hidden_dim=256
).to(DEVICE)

model.load_state_dict(torch.load("pathogen_gnn.pt", map_location=DEVICE))
model.eval()

print("✅ Model loaded")

# ==================================================
# PREDICT
# ==================================================
def predict(pathogen, top_k=5):
    if pathogen not in pathogen_rev:
        raise ValueError(f"Unknown pathogen: {pathogen}")

    pid = pathogen_rev[pathogen]

    with torch.no_grad():
        emb = model(data)

    scores = torch.matmul(
        emb["drug"],
        emb["pathogen"][pid]
    )

    vals, idx = torch.topk(scores, top_k)

    return [
        (drug_map[i], float(v))
        for i, v in zip(idx.cpu().tolist(), vals.cpu().tolist())
    ]

# ==================================================
# CLI
# ==================================================
# ==================================================
# CLI (IMPROVED INPUT HANDLING)
# ==================================================
# ==================================================
# CLI (FINAL FIXED)
# ==================================================
if __name__ == "__main__":

    print("\n✅ Example valid pathogen inputs:")
    print(pathogen_ids[:5])

    while True:

        # ✅ Proper clean input
        q = input("\nEnter pathogen (or exit): ").strip().lower()

        if q in {"exit", "quit"}:
            break

        # Convert scientific name → dataset format
        q_clean = q.replace(" ", "_")

        # If user entered disease name → map it
        if q_clean not in pathogen_rev:

            disease_match = proteins[
                proteins["disease_name"].str.lower() == q
            ]["pathogen"].unique()

            if len(disease_match) > 0:
                q_clean = disease_match[0]
                print(f"✅ Disease mapped to pathogen: {q_clean}")

        try:
            results = predict(q_clean, top_k=5)

            print("\nTop 5 candidate drugs:")
            for i, (drug, score) in enumerate(results, 1):
                print(f"{i}. {drug} | score={score:.4f}")

        except Exception as e:
            print("\n❌ Error:", e)
            print("Try one of these pathogen IDs:")
            print(pathogen_ids)
