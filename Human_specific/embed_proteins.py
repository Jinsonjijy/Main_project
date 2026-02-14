import torch
import esm
from Bio import SeqIO
from tqdm import tqdm

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
FASTA_FILE = "/content/protein_cleaned.fasta"
OUTPUT_FILE = "/content/protein_embeddings.pt"

MIN_LEN = 30
MAX_LEN = 1022      # ESM safe max length
STRIDE = 512        # overlap for long proteins

# --------------------------------------------------
# DEVICE (GPU)
# --------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# --------------------------------------------------
# Load ESM-2 (150M) — STABLE ON COLAB
# --------------------------------------------------
model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
model.eval()
model.to(device)

batch_converter = alphabet.get_batch_converter()

# --------------------------------------------------
# SAFE GN extractor
# --------------------------------------------------
def extract_gene_symbol(desc):
    if " GN=" not in desc:
        return None
    return desc.split(" GN=")[1].split()[0]

# --------------------------------------------------
# EMBEDDING FUNCTION (CHUNKED & OOM-SAFE)
# --------------------------------------------------
def embed_protein(sequence):
    sequence = sequence.replace(" ", "").upper()
    L = len(sequence)

    if L < MIN_LEN:
        return None

    chunk_embeddings = []

    for i in range(0, L, STRIDE):
        chunk = sequence[i:i + MAX_LEN]
        if len(chunk) < MIN_LEN:
            continue

        data = [("protein", chunk)]
        _, _, tokens = batch_converter(data)
        tokens = tokens.to(device)

        with torch.inference_mode():
            outputs = model(tokens, repr_layers=[30])

        reps = outputs["representations"][30][0, 1:-1]
        chunk_embeddings.append(reps.mean(dim=0))

        del tokens, outputs
        torch.cuda.empty_cache()

    if not chunk_embeddings:
        return None

    return torch.stack(chunk_embeddings).mean(dim=0)

# --------------------------------------------------
# RUN EMBEDDING
# --------------------------------------------------
protein_embeddings = {}
skipped = 0

for record in tqdm(SeqIO.parse(FASTA_FILE, "fasta"), desc="Embedding proteins"):
    gene = extract_gene_symbol(record.description)

    if gene is None:
        skipped += 1
        continue

    emb = embed_protein(str(record.seq))
    if emb is not None:
        protein_embeddings[gene] = emb.cpu()

# --------------------------------------------------
# SAVE
# --------------------------------------------------
torch.save(protein_embeddings, OUTPUT_FILE)

print("\n✅ EMBEDDING COMPLETED SUCCESSFULLY")
print("🧬 Proteins embedded:", len(protein_embeddings))
print("⚠ Skipped (no GN):", skipped)
print("📐 Embedding dimension:", next(iter(protein_embeddings.values())).shape[0])
