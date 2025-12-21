import torch
from Bio import SeqIO
from transformers import BertTokenizer, BertModel

# --------------------------------------------------
# INPUT / OUTPUT
# --------------------------------------------------
FASTA_FILE = "data/protein.fasta"   # your cleaned FASTA
OUTPUT_FILE = "protein_embeddings.pt"

# --------------------------------------------------
# Load ProtBERT
# --------------------------------------------------
tokenizer = BertTokenizer.from_pretrained(
    "Rostlab/prot_bert",
    do_lower_case=False
)
model = BertModel.from_pretrained("Rostlab/prot_bert")
model.eval()

def embed_protein(seq):
    seq = " ".join(list(seq))
    inputs = tokenizer(
        seq,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze(0)

# --------------------------------------------------
# Generate embeddings
# --------------------------------------------------
embeddings = {}

for record in SeqIO.parse(FASTA_FILE, "fasta"):
    uniprot_id = record.id.split("|")[-1]
    print("Embedding:", uniprot_id)
    embeddings[uniprot_id] = embed_protein(str(record.seq))

# --------------------------------------------------
# Save
# --------------------------------------------------
torch.save(embeddings, OUTPUT_FILE)
print("\n✅ Saved protein embeddings to", OUTPUT_FILE)
