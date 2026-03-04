from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from gnn_run import export_disease_subgraph
import os
import json

app = Flask(__name__)
CORS(app)

# ==================================================
# PATH CONFIG
# ==================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "Frontend"))

print("Frontend directory:", FRONTEND_DIR)

# ==================================================
# SERVE FRONTEND
# ==================================================

@app.route("/")
def serve_home():
    return send_from_directory(FRONTEND_DIR, "home.html")

@app.route("/index.html")
def serve_index():
    return send_from_directory(FRONTEND_DIR, "index.html")

@app.route("/<path:filename>")
def serve_static(filename):
    return send_from_directory(FRONTEND_DIR, filename)


# ==================================================
# RUN GNN (MAIN API)
# ==================================================

@app.route("/run-gnn", methods=["GET", "POST"])
def run_gnn():

    # Allow GET for testing in browser
    if request.method == "GET":
        return jsonify({"message": "run-gnn endpoint is working. Use POST with JSON."})

    try:
        data = request.get_json(force=True)
        print("Incoming request:", data)

        disease = data.get("disease")

        if not disease:
            return jsonify({"error": "Disease name missing"}), 400

        results, pathways = export_disease_subgraph(disease)

        return jsonify({
            "status": "success",
            "disease": disease,
            "pathways": pathways,
            "drugs": [
                {
                    "name": name,
                    "drugbank_id": dbid,
                    "score": float(score)
                }
                for name, dbid, score in results
            ]
        })

    except Exception as e:
        print("ERROR:", str(e))
        return jsonify({"error": str(e)}), 500


# ==================================================
# OPTIONAL: SERVE GENERATED JSON FILES
# ==================================================

@app.route("/results.json")
def serve_results():
    path = os.path.join(FRONTEND_DIR, "results.json")
    if not os.path.exists(path):
        return jsonify({"error": "results.json not found"}), 404

    with open(path, "r") as f:
        return jsonify(json.load(f))


@app.route("/graph.json")
def serve_graph():
    path = os.path.join(FRONTEND_DIR, "graph.json")
    if not os.path.exists(path):
        return jsonify({"error": "graph.json not found"}), 404

    with open(path, "r") as f:
        return jsonify(json.load(f))


# ==================================================
# START SERVER
# ==================================================

if __name__ == "__main__":
    print("Starting Flask server...")
    app.run(host="0.0.0.0", port=5000, debug=True)