import pandas as pd
import requests
import time

# ================= CONFIG =================
DISEASE_PROTEIN_FILE = "data/disease_protein.csv"

ANTIBIOTIC_FILE = "data/kegg_antibacterial_drugs.csv"
PROTEIN_DRUG_FILE = "data/protein_drug.csv"
SMILES_FILE = "data/kegg_drug_smiles_antibacterial.csv"

print("Loading disease_protein.csv...")
disease_df = pd.read_csv(DISEASE_PROTEIN_FILE)
disease_df.columns = disease_df.columns.str.strip()

all_proteins = set(disease_df["uniprot_id"].unique())

print(f"Total unique disease proteins: {len(all_proteins)}\n")

# ================= STEP 1: GET ALL KEGG DRUGS =================
print("Fetching KEGG drug list...")
drug_response = requests.get("https://rest.kegg.jp/list/drug")
drug_lines = drug_response.text.strip().split("\n")
drug_ids = [line.split("\t")[0] for line in drug_lines]

print(f"Total KEGG drugs: {len(drug_ids)}\n")

protein_drug_pairs = []
valid_drugs = set()

# ================= STEP 2: BUILD PROTEIN-DRUG EDGES =================
for drug_id in drug_ids:

    print(f"Checking {drug_id}")

    link_url = f"https://rest.kegg.jp/link/genes/{drug_id}"
    link_resp = requests.get(link_url)

    if link_resp.status_code != 200:
        continue

    lines = link_resp.text.strip().split("\n")

    for line in lines:
        parts = line.split("\t")
        if len(parts) != 2:
            continue

        gene_entry = parts[1]  # example: eco:b0002

        # Convert KEGG gene → UniProt
        conv_url = f"https://rest.kegg.jp/conv/uniprot/{gene_entry}"
        conv_resp = requests.get(conv_url)

        if conv_resp.status_code != 200:
            continue

        conv_lines = conv_resp.text.strip().split("\n")

        for conv_line in conv_lines:
            conv_parts = conv_line.split("\t")
            if len(conv_parts) != 2:
                continue

            uniprot_id = conv_parts[1].replace("uniprot:", "").strip()

            if uniprot_id in all_proteins:
                protein_drug_pairs.append((uniprot_id, drug_id))
                valid_drugs.add(drug_id)

        time.sleep(0.03)

    time.sleep(0.1)

# ================= SAVE PROTEIN-DRUG =================
protein_drug_df = pd.DataFrame(
    protein_drug_pairs,
    columns=["uniprot_id", "kegg_drug_id"]
).drop_duplicates()

protein_drug_df.to_csv(PROTEIN_DRUG_FILE, index=False)

print("\nSaved protein_drug.csv")
print("Total edges:", len(protein_drug_df))

# ================= SAVE ANTIBACTERIAL DRUG LIST =================
antibiotic_df = pd.DataFrame(
    list(valid_drugs),
    columns=["kegg_drug_id"]
)

antibiotic_df.to_csv(ANTIBIOTIC_FILE, index=False)

print("\nSaved kegg_antibacterial_drugs.csv")
print("Total valid antibacterial drugs:", len(antibiotic_df))

# ================= FETCH SMILES =================
print("\nFetching SMILES...")

smiles_pairs = []

for drug_id in valid_drugs:

    get_url = f"https://rest.kegg.jp/get/{drug_id}"
    get_resp = requests.get(get_url)

    if get_resp.status_code != 200:
        continue

    for line in get_resp.text.split("\n"):
        if line.startswith("SMILES"):
            smiles = line.replace("SMILES", "").strip()
            smiles_pairs.append((drug_id, smiles))
            break

    time.sleep(0.1)

smiles_df = pd.DataFrame(
    smiles_pairs,
    columns=["kegg_drug_id", "SMILES"]
)

smiles_df.to_csv(SMILES_FILE, index=False)

print("\nSaved kegg_drug_smiles_antibacterial.csv")
print("\nDataset build complete.")