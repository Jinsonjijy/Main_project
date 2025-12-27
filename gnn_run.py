import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from gnn_model import DrugRepurposingHeteroGNN

# ==================================================
# GLOBAL CONFIG
# ==================================================
INPUT_DIM = 1024
HIDDEN_DIM = 256
SEED = 42

torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ==================================================
# Helper functions
# ==================================================
def encode(series):
    codes, uniques = pd.factorize(series)
    mapping = dict(enumerate(uniques))
    reverse = {v: k for k, v in mapping.items()}
    return codes, mapping, reverse

def normalize(x):
    return str(x).lower().replace(",", "").replace(" ", "").strip()

# ==================================================
# Load DrugBank ID → Drug Name mapping
# ==================================================
drug_names_df = pd.read_csv("data/uniprot_links.csv")
drug_names_df.columns = drug_names_df.columns.str.strip()

drug_id_to_name = dict(
    zip(drug_names_df["DrugBank ID"], drug_names_df["Name"])
)

# ==================================================
# BUILD VALID DRUG FILTER
# ==================================================
valid_drugs_df = drug_names_df.copy()
valid_drugs_df["Name"] = valid_drugs_df["Name"].astype(str)

invalid_keywords = [
    "keratinocyte", "cell", "tissue", "fibroblast",
    "stem", "epithelial", "neonatal", "foreskin"
]

mask = ~valid_drugs_df["Name"].str.lower().str.contains(
    "|".join(invalid_keywords)
)

valid_drugbank_ids = set(
    valid_drugs_df.loc[mask, "DrugBank ID"]
)

print("Valid drugs after filtering:", len(valid_drugbank_ids))

# ==================================================
# Load datasets
# ==================================================
drug_gene = pd.read_csv("data/pharmacologically_active.csv")
gene_disease = pd.read_csv("data/core_plus_disease_gene.csv")
drug_disease = pd.read_csv("data/drug_parser.csv")

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

# ==================================================
# Normalize disease names
# ==================================================
gene_disease["DiseaseName_norm"] = gene_disease["DiseaseName"].apply(normalize)
drug_disease["DiseaseName_norm"] = drug_disease["DiseaseName"].apply(normalize)

# ==================================================
# Encode diseases and genes
# ==================================================
gene_disease["disease_id"], disease_map, disease_rev = encode(
    gene_disease["DiseaseName_norm"]
)
gene_disease["gene_id"], gene_map, gene_rev = encode(
    gene_disease["GeneSymbol"]
)

# ==================================================
# Drug–gene mapping
# ==================================================
drug_gene["gene_id"] = drug_gene["GeneSymbol"].map(gene_rev)
drug_gene = drug_gene.dropna(subset=["gene_id"])

drug_gene["drug_id"], drug_map, drug_rev = encode(drug_gene["DrugID"])

# ==================================================
# Drug–disease mapping
# ==================================================
drug_disease["disease_id"] = drug_disease["DiseaseName_norm"].map(disease_rev)
drug_disease["drug_id"] = drug_disease["DrugBankID"].map(drug_rev)
drug_disease = drug_disease.dropna(subset=["disease_id", "drug_id"])
drug_disease = drug_disease[["drug_id", "disease_id"]].drop_duplicates()

num_diseases = len(disease_map)
num_genes = len(gene_map)
num_drugs = len(drug_map)

print("Diseases:", num_diseases)
print("Genes:", num_genes)
print("Drugs:", num_drugs)

# ==================================================
# Load protein embeddings
# ==================================================
protein_emb = torch.load("protein_embeddings.pt", map_location="cpu")

gene_to_uniprot = dict(
    zip(drug_gene["GeneSymbol"], drug_gene["UniProtID"])
)

# ==================================================
# Build Heterogeneous Graph
# ==================================================
data = HeteroData()

data["disease"].x = torch.randn(num_diseases, INPUT_DIM) * 0.01
data["drug"].x = torch.randn(num_drugs, INPUT_DIM) * 0.01

# Gene node features
gene_features = []
for gene_symbol in gene_map.values():
    if gene_symbol in gene_to_uniprot:
        uid = gene_to_uniprot[gene_symbol]
        vec = protein_emb.get(uid)
        if vec is not None and vec.shape[0] == INPUT_DIM:
            gene_features.append(vec)
        else:
            gene_features.append(torch.zeros(INPUT_DIM))
    else:
        gene_features.append(torch.zeros(INPUT_DIM))

data["gene"].x = torch.stack(gene_features)

# Edges
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

data["drug", "treats", "disease"].edge_index = dd
data["disease", "rev_treats", "drug"].edge_index = dd.flip(0)

data = data.to(device)

# ==================================================
# Load trained model
# ==================================================
model = DrugRepurposingHeteroGNN(hidden_dim=HIDDEN_DIM).to(device)
model.load_state_dict(
    torch.load("drug_repurposing_gnn.pt", map_location=device)
)
model.eval()

print("Model loaded successfully")

# ==================================================
# Prediction function 
# ==================================================
def predict_drugs(disease, top_k=5, alpha=0.7):

    d_norm = normalize(disease)
    if d_norm not in disease_rev:
        raise ValueError("Disease not found in dataset")

    d_id = disease_rev[d_norm]

    with torch.no_grad():
        emb = model(data.x_dict, data.edge_index_dict)

    drug_emb = F.normalize(emb["drug"], dim=1)
    disease_emb = F.normalize(emb["disease"], dim=1)
    gene_emb = F.normalize(emb["gene"], dim=1)

    # Disease → Gene
    dg_edges = data["disease", "associates", "gene"].edge_index
    gene_ids = dg_edges[1][dg_edges[0] == d_id]

    if gene_ids.numel() == 0:
        return []

    # Gene → Drug
    gd_edges = data["gene", "targets", "drug"].edge_index
    mask = torch.isin(gd_edges[0], gene_ids)
    drug_ids = gd_edges[1][mask].unique()

    if drug_ids.numel() == 0:
        return []

    gene_context = gene_emb[gene_ids].mean(dim=0)

    final_context = (
        alpha * disease_emb[d_id] +
        (1 - alpha) * gene_context
    )
    final_context = F.normalize(final_context, dim=0)

    scores = torch.matmul(drug_emb[drug_ids], final_context)
    top_vals, top_idx = torch.topk(scores, min(top_k, scores.size(0)))

    results = []
    for i, s in zip(top_idx.tolist(), top_vals.tolist()):
        drug_global_id = drug_ids[i].item()
        dbid = drug_map[drug_global_id]

        # FINAL DRUG FILTER
        if dbid not in valid_drugbank_ids:
            continue

        name = drug_id_to_name.get(dbid, "Unknown")
        results.append((name, dbid, float(s)))

    return results

# ==================================================
# Run
# ==================================================
if __name__ == "__main__":
    disease = input("Enter the Disease: ").strip()

    results = predict_drugs(disease)

    print("\n" + "=" * 60)
    print("💊 DRUG REPURPOSING – NORMAL INFERENCE")
    print(f"🦠 Input Disease : {disease}")
    print("🧠 Mode          : Disease + Gene Network")
    print("=" * 60 + "\n")
    if not results:
        print("❌ No candidate drugs found.")
    else:
        for i, (name, dbid, score) in enumerate(results, start=1):
            print(f"{i:>2}. {name:<25} ({dbid})  |  Score: {score:.4f}")

        
        top_name, top_dbid, top_score = results[0]
        print("\n⭐ Top Repurposing Candidate:")
        print(f"→ {top_name} ({top_dbid})  |  Score: {top_score:.4f}")

       
        print("\n🧠 Interpretation:")
        print("• Predictions are based on disease–gene–drug network inference.")
        print("• Higher scores indicate stronger mechanistic relevance.")
