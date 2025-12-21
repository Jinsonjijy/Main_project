import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
import re

# ==================================================
# FILES
# ==================================================
CSV_FILE = "data/dataset.csv"
INPUT_FASTA = "data/protein.fasta"
OUTPUT_FASTA = "protein_final.fasta"

UNIPROT_COLUMN = "UniProtID"   # make sure this matches CSV exactly

# ==================================================
# PARAMETERS
# ==================================================
MIN_LEN = 50
MAX_LEN = 512

# ==================================================
# LOAD UniProt IDs FROM CSV
# ==================================================
df = pd.read_csv(CSV_FILE)

uniprot_ids = set(
    df[UNIPROT_COLUMN]
    .dropna()
    .astype(str)
    .str.strip()
)

print(f"✅ UniProt IDs found in CSV: {len(uniprot_ids)}")

# ==================================================
# CLEAN FASTA
# ==================================================
seen = set()
clean_records = []

total = 0
kept = 0

for record in SeqIO.parse(INPUT_FASTA, "fasta"):
    total += 1

    # ---------- Extract UniProt ID ----------
    # Handles: drugbank_target|P45059
    if "|" in record.id:
        uniprot_id = record.id.split("|")[-1]
    else:
        uniprot_id = record.id

    # ---------- Keep only proteins used in CSV ----------
    if uniprot_id not in uniprot_ids:
        continue

    # ---------- Remove isoforms ----------
    if "-" in uniprot_id:
        continue

    # ---------- Clean sequence ----------
    seq = str(record.seq).upper()
    seq = re.sub(r"[^ACDEFGHIKLMNPQRSTVWY]", "", seq)

    # ---------- Length filter ----------
    if not (MIN_LEN <= len(seq) <= MAX_LEN):
        continue

    # ---------- Remove duplicates ----------
    if uniprot_id in seen:
        continue

    seen.add(uniprot_id)

    # ---------- Save cleaned record ----------
    record.id = uniprot_id
    record.name = ""
    record.description = ""
    record.seq = Seq(seq)   # ✅ FIX IS HERE

    clean_records.append(record)
    kept += 1

# ==================================================
# WRITE OUTPUT
# ==================================================
SeqIO.write(clean_records, OUTPUT_FASTA, "fasta")

print("\n✅ FASTA CLEANING COMPLETED")
print("➡ Total proteins in FASTA :", total)
print("➡ Proteins kept           :", kept)
print("➡ Output FASTA            :", OUTPUT_FASTA)
