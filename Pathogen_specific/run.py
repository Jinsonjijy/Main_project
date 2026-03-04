import torch
import pandas as pd
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from model import BacteriaSpecificHeteroGNN

EMB_DIM = 320
HIDDEN_DIM = 256
TOP_K = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load datasets
disease_protein = pd.read_csv("data/disease_protein.csv")
kegg_lookup = pd.read_csv("data/kegg_drug_lookup.csv")

def encode(series):
    codes, uniques = pd.factorize(series)
    mapping = dict(enumerate(uniques))
    reverse = {v: k for k, v in mapping.items()}
    return codes, mapping, reverse

disease_protein["disease_id"], d_map, d_rev = encode(disease_protein["Disease"])
disease_protein["protein_id"], p_map, p_rev = encode(disease_protein["UniProt"])

unique_drugs = sorted(kegg_lookup["KEGG_ID"].unique())
dr_map = {d: i for i, d in enumerate(unique_drugs)}
dr_rev = {i: d for d, i in dr_map.items()}


# Load embeddings
protein_embeddings = torch.load("protein_embeddings.pt")

protein_x = torch.stack([
    protein_embeddings[p] if p in protein_embeddings
    else torch.zeros(EMB_DIM)
    for p in p_map.values()
])

drug_x = torch.randn(len(dr_map), EMB_DIM)
disease_x = torch.randn(len(d_map), EMB_DIM)


# Build graph
data = HeteroData()
data["protein"].x = protein_x
data["drug"].x = drug_x
data["disease"].x = disease_x

dp_edge = torch.tensor(
    disease_protein[["disease_id", "protein_id"]].values.T,
    dtype=torch.long
)

data["disease", "associated_with", "protein"].edge_index = dp_edge
data["protein", "rev_associated_with", "disease"].edge_index = dp_edge.flip(0)

data = data.to(device)


# Load model
model = BacteriaSpecificHeteroGNN(
    input_dim=EMB_DIM,
    hidden_dim=HIDDEN_DIM
).to(device)

model.load_state_dict(torch.load("bacteria_gnn.pt", map_location=device))
model.eval()


@torch.no_grad()
def predict(disease_name):

    if disease_name not in d_rev:
        raise ValueError("Disease not found")

    d_id = d_rev[disease_name]

    emb = model(data.x_dict, data.edge_index_dict)

    disease_emb = emb["disease"]
    drug_emb = emb["drug"]

    scores = F.cosine_similarity(
        disease_emb[d_id].unsqueeze(0),
        drug_emb,
        dim=1
    )

    top_vals, top_idx = torch.topk(scores, TOP_K)

    print("\nTop Predicted KEGG Drugs:\n")
    for rank, (idx, score) in enumerate(
        zip(top_idx.tolist(), top_vals.tolist()), 1
    ):
        print(f"{rank}. {dr_rev[idx]} | Score: {score:.4f}")


# CLI
if __name__ == "__main__":
    while True:
        d = input("Enter bacterial infection (or exit): ")
        if d.lower() == "exit":
            break
        predict(d)