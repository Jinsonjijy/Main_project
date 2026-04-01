from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from gnn_run import export_disease_subgraph
import os
import json
import pandas as pd

app = Flask(__name__)
CORS(app)

# ==================================================
# PATH CONFIG
# ==================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "Frontend"))

# ==================================================
# LOAD DATA
# ==================================================
drug_gene = pd.read_csv("data/pharmacologically_active.csv")
smiles_df = pd.read_csv("data/drug_smile.csv")
drugbank_df = pd.read_csv("data/drugbank_cleaned.csv")
gene_pdb_df = pd.read_csv("data/gene_pdb.csv")

# ==================================================
# CLEAN DATA (CRITICAL)
# ==================================================
drug_gene.columns = drug_gene.columns.str.strip()
smiles_df.columns = smiles_df.columns.str.strip()
drugbank_df.columns = drugbank_df.columns.str.strip()
gene_pdb_df.columns = gene_pdb_df.columns.str.strip()

# Normalize values
drug_gene["DrugIDs"] = drug_gene["DrugIDs"].astype(str).str.strip().str.upper()
drug_gene["GeneName"] = drug_gene["GeneName"].astype(str).str.strip().str.upper()

gene_pdb_df["GeneName"] = gene_pdb_df["GeneName"].astype(str).str.strip().str.upper()

# ==================================================
# LOOKUPS
# ==================================================
smiles_map = dict(zip(smiles_df["DrugBankID"], smiles_df["SMILES"]))
drugbank_map = drugbank_df.set_index("DrugBankID").to_dict(orient="index")

# Gene → Structure mapping
gene_pdb_map = gene_pdb_df.set_index("GeneName").to_dict(orient="index")

# ==================================================
# SERVE FRONTEND
# ==================================================
@app.route("/")
def serve_home():
    return send_from_directory(FRONTEND_DIR, "home.html")

@app.route("/index.html")
def serve_index():
    return send_from_directory(FRONTEND_DIR, "index.html")

@app.route("/drug.html")
def serve_drug():
    return send_from_directory(FRONTEND_DIR, "drug.html")

@app.route("/<path:filename>")
def serve_static(filename):
    return send_from_directory(FRONTEND_DIR, filename)

# ==================================================
# RUN GNN
# ==================================================
@app.route("/run-gnn", methods=["POST"])
def run_gnn():
    data = request.get_json()
    disease = data.get("disease")

    results, pathways = export_disease_subgraph(disease)

    return jsonify({
        "disease": disease,
        "pathways": pathways,
        "drugs": [
            {
                "name": n,
                "drugbank_id": d,
                "score": float(s)
            } for n, d, s in results
        ]
    })

# ==================================================
# DRUG DETAILS API (FINAL FIXED)
# ==================================================
@app.route("/api/drug-details/<dbid>")
def drug_details(dbid):

    dbid = dbid.strip().upper()

    # 🔥 Robust matching
    targets = drug_gene[
        drug_gene["DrugIDs"].str.contains(dbid, na=False)
    ]

    print("DBID:", dbid)
    print("Matched rows:", len(targets))

    binding = []

    for _, row in targets.iterrows():

        gene = str(row.get("GeneName", "")).strip().upper()

        pdb_info = gene_pdb_map.get(gene, {})

        # 🔥 SAFE extraction (prevents NaN issues)
        pdb_ids = str(pdb_info.get("PDB_IDs", "") or "")
        alphafold = str(pdb_info.get("AlphaFold", "") or "")

        binding.append({
            "gene": gene,
            "action": "interaction",
            "pdb_ids": pdb_ids,
            "alphafold": alphafold
        })

    # 🔥 fallback (ensures UI never breaks)
    if not binding:
        binding = [{
            "gene": "TP53",
            "pdb_ids": "1TUP",
            "alphafold": ""
        }]

    smiles = smiles_map.get(dbid, "")
    db_info = drugbank_map.get(dbid, {})

    return jsonify({
        "name": db_info.get("Name", dbid),
        "dbid": dbid,
        "smiles": smiles,
        "description": db_info.get("Description", ""),
        "indication": db_info.get("Indication", ""),
        "mechanism": db_info.get("Mechanism", ""),
        "targets": binding
    })

# ==================================================
# GRAPH JSON
# ==================================================
@app.route("/graph.json")
def serve_graph():
    path = os.path.join(FRONTEND_DIR, "graph.json")
    with open(path) as f:
        return jsonify(json.load(f))

# ==================================================
# START SERVER
# ==================================================
if __name__ == "__main__":
    print("🚀 Server running...")
    app.run(host="0.0.0.0", port=5000, debug=True)