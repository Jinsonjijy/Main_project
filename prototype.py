import pandas as pd
from collections import Counter

# --------------------------------------------------
# Load dataset
# --------------------------------------------------
df = pd.read_csv("data/dataset.csv")

# --------------------------------------------------
# Normalize DiseaseName
# --------------------------------------------------
def normalize(text):
    return (
        str(text)
        .lower()
        .replace(",", "")
        .replace("_", "")
        .replace(" ", "")
        .strip()
    )

df["DiseaseName_norm"] = df["DiseaseName"].apply(normalize)

# --------------------------------------------------
# Drug repurposing prototype
# --------------------------------------------------
def repurpose_drugs(disease_name, top_k=5):

    disease_name_norm = normalize(disease_name)

    # Find disease rows
    disease_rows = df[df["DiseaseName_norm"] == disease_name_norm]

    if disease_rows.empty:
        return []

    # Get genes
    disease_genes = disease_rows["GeneSymbol"].unique()

    # Find all rows with those genes
    related_rows = df[df["GeneSymbol"].isin(disease_genes)]

    # Collect drug IDs
    drug_list = related_rows["DrugIDs"].astype(str).tolist()

    # Rank drugs
    drug_ranking = Counter(drug_list)

    return drug_ranking.most_common(top_k)

# --------------------------------------------------
# Run prototype
# --------------------------------------------------
if __name__ == "__main__":

    disease_input = "Hidradenitissuppurativa,familial"

    results = repurpose_drugs(disease_input, top_k=5)

    print(f"\nCandidate drugs for {disease_input}:\n")

    if not results:
        print("No drugs found.")
    else:
        for drug, score in results:
            print(f"{drug}  | score: {score}")
