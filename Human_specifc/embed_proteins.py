import torch
from Bio import SeqIO
from transformers import BertTokenizer, BertModel
from tqdm import tqdm

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
FASTA_FILE = "data/protein.fasta"
OUTPUT_FILE = "protein_embeddings.pt"

WINDOW = 510       # ProtBERT safe window
STRIDE = 256       # overlap
MIN_LEN = 30

# --------------------------------------------------
# DEVICE
# --------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------
# Load ProtBERT
# --------------------------------------------------
tokenizer = BertTokenizer.from_pretrained(
    "Rostlab/prot_bert",
    do_lower_case=False
)
model = BertModel.from_pretrained("Rostlab/prot_bert")
model.to(device)
model.eval()

# --------------------------------------------------
# Embedding function (NO truncation)
# --------------------------------------------------
def embed_protein(sequence):
    sequence = sequence.replace(" ", "").upper()
    embeddings = []

    for i in range(0, len(sequence), STRIDE):
        chunk = sequence[i:i + WINDOW]
        if len(chunk) < MIN_LEN:
            continue

        chunk = " ".join(list(chunk))
        inputs = tokenizer(chunk, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        emb = outputs.last_hidden_state.mean(dim=1)
        embeddings.append(emb)

    if len(embeddings) == 0:
        return None

    return torch.mean(torch.stack(embeddings), dim=0).squeeze(0)

# --------------------------------------------------
# Generate embeddings
# --------------------------------------------------
protein_embeddings = {}

for record in tqdm(list(SeqIO.parse(FASTA_FILE, "fasta"))):
    protein_id = record.id.strip()
    seq = str(record.seq)

    emb = embed_protein(seq)
    if emb is not None:
        protein_embeddings[protein_id] = emb.cpu()

# --------------------------------------------------
# Save
# --------------------------------------------------
torch.save(protein_embeddings, OUTPUT_FILE)

print(f"\n✅ Saved {len(protein_embeddings)} protein embeddings to {OUTPUT_FILE}")
print("Embedding dimension:", next(iter(protein_embeddings.values())).shape[0])
