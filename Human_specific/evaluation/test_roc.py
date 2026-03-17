import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
import random
from torch_geometric.data import HeteroData
from sklearn.metrics import roc_auc_score, average_precision_score
from gnn_model import DrugRepurposingHeteroGNN

# ================= FIX RANDOMNESS =================

SEED = 47

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


EMB_DIM = 644
HIDDEN_DIM = 256
TOPK = 20
NEG_RATIO = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def normalize(x):
    return str(x).lower().replace(",", "").replace(" ", "").strip()


# ================= LOAD DATA =================

gene_disease = pd.read_csv("data/core_plus_disease_gene.csv")
drug_gene = pd.read_csv("data/pharmacologically_active.csv")
drug_disease = pd.read_csv("data/drug_disease_edges.csv")

drug_gene = drug_gene[drug_gene["Species"] == "Humans"]

drug_gene = drug_gene.rename(columns={
    "GeneName": "GeneSymbol",
    "DrugIDs": "DrugID"
})

gene_disease["DiseaseName_norm"] = gene_disease["DiseaseName"].apply(normalize)

d_uniques = gene_disease["DiseaseName_norm"].unique()
g_uniques = gene_disease["GeneSymbol"].unique()

disease_map = {d: i for i, d in enumerate(d_uniques)}
gene_map = {g: i for i, g in enumerate(g_uniques)}


# ================= DRUG MAPPING =================

drug_gene["gene_id"] = drug_gene["GeneSymbol"].map(gene_map)
drug_gene = drug_gene.dropna(subset=["gene_id"])

unique_drugs = sorted(drug_gene["DrugID"].unique())

dr_map = {d: i for i, d in enumerate(unique_drugs)}
dr_rev = {i: d for d, i in dr_map.items()}

drug_gene["drug_id"] = drug_gene["DrugID"].map(dr_map)

num_dr = len(dr_map)

print("Drugs:", num_dr)


# ================= DRUG-DISEASE SPLIT =================

drug_disease["disease_norm"] = drug_disease["disease"].apply(normalize)

drug_disease["disease_id"] = drug_disease["disease_norm"].map(disease_map)
drug_disease["drug_id"] = drug_disease["drug_id"].map(dr_map)

drug_disease = drug_disease.dropna(subset=["disease_id", "drug_id"])

pairs = list(zip(
    drug_disease["disease_id"].astype(int),
    drug_disease["drug_id"].astype(int)
))

from sklearn.model_selection import train_test_split

train_pairs, test_pairs = train_test_split(
    pairs,
    test_size=0.2,
    random_state=42
)

print("Train pairs:", len(train_pairs))
print("Test pairs:", len(test_pairs))


# ================= LOAD FEATURES =================

drug_emb_all = torch.load("drug_embeddings.pt", map_location="cpu")
protein_emb = torch.load("gene_features_full.pt", map_location="cpu")
disease_emb_file = torch.load("disease_features_644.pt", map_location="cpu")

data = HeteroData()


# ================= DRUG FEATURES =================

drug_x = torch.zeros((num_dr, EMB_DIM))

for dbid, idx in dr_map.items():
    if dbid in drug_emb_all:
        drug_x[idx] = drug_emb_all[dbid]

data["drug"].x = drug_x


# ================= GENE FEATURES =================

gene_x = []

for g in g_uniques:
    if g in protein_emb:
        gene_x.append(protein_emb[g])
    else:
        gene_x.append(torch.zeros(EMB_DIM))

data["gene"].x = torch.stack(gene_x)


# ================= DISEASE FEATURES =================

disease_x = []

for d in d_uniques:
    if d in disease_emb_file:
        disease_x.append(disease_emb_file[d])
    else:
        disease_x.append(torch.zeros(EMB_DIM))

data["disease"].x = torch.stack(disease_x)


# ================= CORE EDGES =================

dg_edges = []

for _, row in gene_disease.iterrows():
    dg_edges.append([
        disease_map[row["DiseaseName_norm"]],
        gene_map[row["GeneSymbol"]]
    ])

dg = torch.tensor(dg_edges, dtype=torch.long).T


gd_edges = []

for _, row in drug_gene.iterrows():
    gd_edges.append([
        row["gene_id"],
        row["drug_id"]
    ])

gd = torch.tensor(gd_edges, dtype=torch.long).T


data["disease", "associates", "gene"].edge_index = dg
data["gene", "rev_associates", "disease"].edge_index = dg.flip(0)

data["gene", "targets", "drug"].edge_index = gd
data["drug", "rev_targets", "gene"].edge_index = gd.flip(0)


# ================= PPI EDGES =================

ppi = pd.read_csv("data/ppi_edges.csv")

ppi["g1"] = ppi["Gene1"].map(gene_map)
ppi["g2"] = ppi["Gene2"].map(gene_map)

ppi = ppi.dropna()

ppi_edges = torch.tensor(
    ppi[["g1", "g2"]].values.T,
    dtype=torch.long
)

ppi_edges = torch.cat([ppi_edges, ppi_edges.flip(0)], dim=1)

data["gene", "interacts", "gene"].edge_index = ppi_edges
data["gene", "rev_interacts", "gene"].edge_index = ppi_edges.flip(0)

data = data.to(device)


# ================= LOAD MODEL =================

model = DrugRepurposingHeteroGNN(
    input_dim=EMB_DIM,
    hidden_dim=HIDDEN_DIM
).to(device)

model.load_state_dict(
    torch.load("drug_repurposing_gnn.pt", map_location=device)
)

model.eval()

print("Model loaded")


# ================= EMBEDDINGS =================

with torch.no_grad():
    emb = model(data.x_dict, data.edge_index_dict)

    disease_emb = F.normalize(emb["disease"], dim=1)
    drug_emb = F.normalize(emb["drug"], dim=1)


# ================= BUILD TEST DATA =================

scores = []
labels = []

train_set = set(train_pairs)


# ================= POSITIVE PAIRS =================

for d, dr in test_pairs:

    score = F.cosine_similarity(
        disease_emb[d].unsqueeze(0),
        drug_emb[dr].unsqueeze(0)
    ).item()

    scores.append(score)
    labels.append(1)


# ================= NEGATIVE PAIRS =================

random.seed(SEED)

negatives = []

while len(negatives) < NEG_RATIO * len(test_pairs):

    d = random.randint(0, len(d_uniques) - 1)
    dr = random.randint(0, num_dr - 1)

    if (d, dr) not in train_set and (d, dr) not in test_pairs:
        negatives.append((d, dr))


for d, dr in negatives:

    score = F.cosine_similarity(
        disease_emb[d].unsqueeze(0),
        drug_emb[dr].unsqueeze(0)
    ).item()

    scores.append(score)
    labels.append(0)


scores = np.array(scores)
labels = np.array(labels)


# ================= METRICS =================

roc = roc_auc_score(labels, scores)
pr = average_precision_score(labels, scores)

print("\nEvaluation Results")
print("----------------------")
print("AUROC:", round(roc, 4))
print("PR-AUC:", round(pr, 4))


# ================= TOP-K =================

ranked = np.argsort(-scores)

topk = ranked[:TOPK]

hits = np.sum(labels[topk])

precision = hits / TOPK
recall = hits / np.sum(labels)

print("Hits@20:", hits)
print("Precision@20:", round(precision, 4))
print("Recall@20:", round(recall, 4))


# ================= ROC CURVE =================

from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

fpr, tpr, thresholds = roc_curve(labels, scores)

plt.figure()
plt.plot(fpr, tpr, label="ROC Curve (AUC = %.3f)" % roc)
plt.plot([0,1],[0,1],'--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for Drug-Disease Prediction")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig("roc_curve.png", dpi=300)
plt.show()


# ================= PR CURVE =================

from sklearn.metrics import precision_recall_curve

precision_curve, recall_curve, _ = precision_recall_curve(labels, scores)

plt.figure()
plt.plot(recall_curve, precision_curve, label="PR Curve (AUC = %.3f)" % pr)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve for Drug-Disease Prediction")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig("pr_curve.png", dpi=300)
plt.show()