from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from gnn_run import export_disease_subgraph
import os
import json

app = Flask(__name__)
CORS(app)

# Path to Frontend folder
FRONTEND_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "Frontend")
)

# ==========================================
# Serve Frontend Pages
# ==========================================

@app.route("/")
def serve_home():
    return send_from_directory(FRONTEND_DIR, "home.html")

@app.route("/index.html")
def serve_index():
    return send_from_directory(FRONTEND_DIR, "index.html")

@app.route("/<path:filename>")
def serve_static_files(filename):
    return send_from_directory(FRONTEND_DIR, filename)


# ==========================================
# Run GNN
# ==========================================

@app.route("/run-gnn", methods=["POST"])
def run_gnn():

    data = request.get_json(force=True)
    disease = data.get("disease")

    if not disease:
        return jsonify({"error": "Disease name missing"}), 400

    try:
        results, pathways = export_disease_subgraph(disease)

        return jsonify({
            "status": "success",
            "disease": disease,
            "drugs": [
                {
                    "name": name,
                    "drugbank_id": dbid,
                    "score": score
                }
                for name, dbid, score in results
            ],
            "pathways": pathways
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==========================================
# Serve Generated JSON
# ==========================================

@app.route("/graph.json")
def serve_graph():
    with open(os.path.join(FRONTEND_DIR, "graph.json"), "r") as f:
        return jsonify(json.load(f))


@app.route("/results.json")
def serve_results():
    with open(os.path.join(FRONTEND_DIR, "results.json"), "r") as f:
        return jsonify(json.load(f))


# ==========================================
# Start Server
# ==========================================

if __name__ == "__main__":
    app.run(debug=True, port=5000)
