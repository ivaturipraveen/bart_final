from flask import Flask, request, jsonify
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

# Global variables to hold loaded data
index = None
texts = None
model = None

def load_faiss_and_data(embedding_file, faiss_index_file, data_file):
    """Load embeddings, FAISS index, and processed metadata."""
    # Load embeddings (optional, if you want to inspect them)
    embeddings = np.load(embedding_file)

    # Load FAISS index
    index = faiss.read_index(faiss_index_file)

    # Load metadata and texts
    with open(data_file, 'r') as f:
        data = json.load(f)

    return embeddings, index, data["texts"], data["metadata"]

def assistant_response(query, best_text):
    """Generate an assistant-like response for the query."""
    response_template = (
        "Here is the best result I found based on your query:\n\n"
        "Query: {query}\n\n"
        "Answer: {best_text}\n\n"
        "Let me know if you'd like further assistance or more details!"
    )
    return response_template.format(query=query, best_text=best_text)

@app.route("/search", methods=["POST"])
def search():
    """Endpoint to handle search queries."""
    global index, texts, model

    # Get the query from the request
    data = request.json
    query = data.get("query", "")

    if not query:
        return jsonify({"error": "Query is missing"}), 400

    # Generate embedding for the query
    query_embedding = model.encode([query]).astype('float32')

    # Search the FAISS index
    distances, indices = index.search(query_embedding, 1)  # Get only the best result

    # Extract the best match
    best_match_idx = indices[0][0]
    best_distance = distances[0][0]
    best_text = texts[best_match_idx]

    # Generate assistant-like response
    response = assistant_response(query, best_text)
    return jsonify({"response": response, "distance": best_distance})

if __name__ == "__main__":
    # File paths
    embedding_file = "/combined_embeddings.npy"
    faiss_index_file = "/combined_faiss.index"
    data_file = "/processed_data.json"

    # Load data and index
    _, index, texts, _ = load_faiss_and_data(
        embedding_file, faiss_index_file, data_file
    )

    # Initialize SentenceTransformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Start the Flask app
    app.run(host="0.0.0.0", port=5000)
