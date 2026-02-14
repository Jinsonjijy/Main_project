import torch
import torch.nn.functional as F
import pandas as pd
from torch_geometric.data import HeteroData
from model import PathogenDrugHeteroGNN

# ==================================================
# CONFIG
# ==================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROTEIN_DIM = 320
TOP_K = 5

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

# ==================================================
# BUILD PROTEIN–PATHWAY
# ==================================================
pp_main = pp[["uniprot_id", "map_pathway_id"]]
pp_anchor = anchors[["uniprot_id", "pathway_id"]].rename(
    columns={"pathway_id": "map_pathway_id"}
)

protein_pathway = (
    pd.concat([pp_main, pp_anchor], ignore_index=True)
    .drop_duplicates()
)

# ==================================================
# BUILD PATHWAY–DRUG
# ==================================================
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

# ==================================================
# ENCODE
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

print(f"""
Graph Summary:
Pathogens: {len(pathogen_ids)}
Proteins : {len(protein_ids)}
Pathways : {len(pathway_ids)}
Drugs    : {len(drug_ids)}
""")

# ==================================================
# LOAD EMBEDDINGS
# ==================================================
protein_emb = torch.load("protein_embeddings_all.pt")

# ==================================================
# LOAD DRUG NAMES
# ==================================================
drug_lookup = pd.read_csv("data/drug_lookup_enriched.csv")

drug_name_map = {}
for idx, drug_id in drug_map.items():
    row = drug_lookup.loc[drug_lookup["drug_id"] == drug_id]
    if len(row) > 0:
        drug_name_map[idx] = row["drug_name"].values[0]
    else:
        drug_name_map[idx] = drug_id

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

print("✅ Model loaded successfully")

with torch.no_grad():
    cached_emb = model(data)

# ==================================================
# PREDICT
# ==================================================
def predict(pathogen, top_k=TOP_K):

    if pathogen not in pathogen_rev:
        raise ValueError(f"Unknown pathogen: {pathogen}")

    pid = pathogen_rev[pathogen]

    drug_emb = F.normalize(cached_emb["drug"], dim=1)
    pathogen_emb = F.normalize(
        cached_emb["pathogen"][pid], dim=0
    )

    scores = torch.matmul(drug_emb, pathogen_emb)
    probs = torch.sigmoid(scores)

    vals, idx = torch.topk(probs, 50)

    # Pathogen pathways
    pp_edges = data["pathogen", "rev_has", "protein"].edge_index
    pathogen_proteins = pp_edges[1][pp_edges[0] == pid]

    pw_edges = data["protein", "rev_in_pathway", "pathway"].edge_index
    pathogen_pathways = pw_edges[1][
        torch.isin(pw_edges[0], pathogen_proteins)
    ]

    pathogen_pw_set = set(pathogen_pathways.cpu().tolist())
    pd_edges = data["pathway", "targeted_by", "drug"].edge_index

    # Strong biochemical filtering
    bad_keywords = [
        "acid", "phosphate", "aldehyde", "triphosphate",
        "adenine", "ribose", "cofactor", "nucleotide",
        "metabolite", "glycer", "pyruvate"
    ]

    results = []

    for drug_idx, score in zip(idx.cpu().tolist(), vals.cpu().tolist()):

        name = drug_name_map[drug_idx]
        name_lower = name.lower()

        # Remove biochemical junk
        if any(k in name_lower for k in bad_keywords):
            continue

        drug_pathways = pd_edges[0][pd_edges[1] == drug_idx]
        shared = pathogen_pw_set.intersection(
            set(drug_pathways.cpu().tolist())
        )

        # Remove generic pathway
        pathway_names = [
            pathway_ids[p]
            for p in shared
            if pathway_ids[p] != "map01100"
        ][:3]

        results.append({
            "name": name,
            "score": float(score),
            "pathways": pathway_names
        })

        if len(results) == top_k:
            break

    return results

# ==================================================
# CLI
# ==================================================
if __name__ == "__main__":

    print("\nEnter pathogen (type 'exit' to quit)")

    while True:

        query = input("\nPathogen: ").strip().lower()

        if query in {"exit", "quit"}:
            break

        query_clean = query.replace(" ", "_")

        try:
            results = predict(query_clean)

            print("\n" + "=" * 60)
            print(" DRUG REPURPOSING – PATHOGEN GNN")
            print(f" Input Pathogen : {query_clean}")
            print("=" * 60)

            print("\nTop Candidate Drugs:\n")

            if not results:
                print("No candidates found.")
                continue

            for i, r in enumerate(results, 1):
                print(
                    f"{i:>2}. {r['name']:<30} "
                    f"| Score: {r['score']:.4f}"
                )

                if r["pathways"]:
                    print("     Top Supporting Pathways:")
                    for p in r["pathways"]:
                        print("       -", p)

            print()

        except Exception as e:
            print("\n❌ Error:", e)
            print("Available pathogen examples:")
            print(pathogen_ids[:10])
