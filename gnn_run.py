import pandas as pd
import torch
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
# Load DrugBank ID → Drug Name mapping (ADD)
# --------------------------------------------------
drug_names_df = pd.read_csv("data/uniprot_links.csv")
drug_names_df.columns = drug_names_df.columns.str.strip()

drug_id_to_name = dict(
    zip(drug_names_df["DrugBank ID"], drug_names_df["Name"])
)

# --------------------------------------------------
# Load datasets (SAME AS TRAINING)
# --------------------------------------------------
drug_gene = pd.read_csv("data/pharmacologically_active.csv")
gene_disease = pd.read_csv("data/CTD_curated_genes_diseases.csv")

drug_gene.columns = drug_gene.columns.str.strip()
gene_disease.columns = gene_disease.columns.str.strip()

drug_gene = drug_gene[drug_gene["Species"] == "Humans"]

drug_gene = drug_gene.rename(columns={
    "GeneName": "GeneSymbol",
    "DrugIDs": "DrugID",
    "UniProtID": "UniProtID"
})

drug_gene = drug_gene[["DrugID", "GeneSymbol", "UniProtID"]].drop_duplicates()
gene_disease = gene_disease[["GeneSymbol", "DiseaseName"]].drop_duplicates()

gene_disease["DiseaseName_norm"] = gene_disease["DiseaseName"].apply(normalize)

gene_disease["disease_id"], disease_map, disease_rev = encode(
    gene_disease["DiseaseName_norm"]
)
gene_disease["gene_id"], gene_map, gene_rev = encode(
    gene_disease["GeneSymbol"]
)

drug_gene["gene_id"] = drug_gene["GeneSymbol"].map(gene_rev)
drug_gene = drug_gene.dropna(subset=["gene_id"])
drug_gene["drug_id"], drug_map, drug_rev = encode(drug_gene["DrugID"])

num_diseases = len(disease_map)
num_genes = len(gene_map)
num_drugs = len(drug_map)

# --------------------------------------------------
# Load ProtBERT embeddings
# --------------------------------------------------
protein_emb = torch.load("protein_embeddings.pt")

gene_to_uniprot = dict(
    zip(drug_gene["GeneSymbol"], drug_gene["UniProtID"])
)

# --------------------------------------------------
# Build graph (INPUT DIM = 1024, PROJECTED INTERNALLY)
# --------------------------------------------------
data = HeteroData()
INPUT_DIM = 1024

data["disease"].x = torch.zeros(num_diseases, INPUT_DIM)
data["drug"].x = torch.zeros(num_drugs, INPUT_DIM)

gene_features = []
for gene_symbol in gene_map.values():
    if gene_symbol in gene_to_uniprot:
        uid = gene_to_uniprot[gene_symbol]
        gene_features.append(protein_emb.get(uid, torch.zeros(INPUT_DIM)))
    else:
        gene_features.append(torch.zeros(INPUT_DIM))

data["gene"].x = torch.stack(gene_features)

dg = torch.tensor(
    gene_disease[["disease_id", "gene_id"]].values.T, dtype=torch.long
)
gd = torch.tensor(
    drug_gene[["gene_id", "drug_id"]].values.T, dtype=torch.long
)

data["disease", "associates", "gene"].edge_index = dg
data["gene", "rev_associates", "disease"].edge_index = dg.flip(0)
data["gene", "targets", "drug"].edge_index = gd
data["drug", "rev_targets", "gene"].edge_index = gd.flip(0)

# --------------------------------------------------
# LOAD TRAINED MODEL (MUST MATCH TRAINING)
# --------------------------------------------------
model = DrugRepurposingHeteroGNN(hidden_dim=256)
model.load_state_dict(
    torch.load("drug_repurposing_gnn.pt", map_location="cpu")
)
model.eval()

# --------------------------------------------------
# Prediction
# --------------------------------------------------
def predict_drugs(disease, top_k=5):
    d_norm = normalize(disease)

    if d_norm not in disease_rev:
        raise ValueError("Disease not found in dataset")

    d_id = disease_rev[d_norm]

    with torch.no_grad():
        emb = model(data.x_dict, data.edge_index_dict)

    scores = torch.matmul(
        emb["drug"], emb["disease"][d_id]
    )

    top = torch.topk(scores, top_k).indices.tolist()

    results = []
    for i in top:
        dbid = drug_map[i]
        name = drug_id_to_name.get(dbid, "Unknown")
        results.append((name, dbid, scores[i].item()))

    return results

# --------------------------------------------------
# Run
# --------------------------------------------------
disease = input("Enter the Disease: ")

print("\nDisease:", disease)
print("Predicted Drugs:\n")

for name, dbid, score in predict_drugs(disease):
    print(f"{name} ({dbid})")
    # print( f"| score: {score:.4f}")
