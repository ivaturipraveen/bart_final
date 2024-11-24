from flask import Flask, request, jsonify
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder

# Initialize Flask app
app = Flask(__name__)

# Load FAISS index and related data
def load_faiss_and_data(embedding_file, faiss_index_file, data_file):
    """Load embeddings, FAISS index, and processed metadata."""
    embeddings = np.load(embedding_file)
    index = faiss.read_index(faiss_index_file)
    with open(data_file, 'r') as f:
        data = json.load(f)
    return embeddings, index, data["texts"], data["metadata"]

def rerank_results(query, top_texts):
    """Rerank results using a CrossEncoder model."""
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    scores = cross_encoder.predict([(query, text) for text in top_texts])
    best_idx = np.argmax(scores)
    return top_texts[best_idx], scores[best_idx], scores

def query_faiss_index_with_reranking(query, index, texts, model, top_k=5):
    """Query the FAISS index and rerank results."""
    query_embedding = model.encode([query]).astype('float32')
    distances, indices = index.search(query_embedding, top_k)

    # Collect top results
    top_texts = [texts[idx] for idx in indices[0]]

    # Rerank results using CrossEncoder
    best_text, best_score, all_scores = rerank_results(query, top_texts)

    return {
        "query": query,
        "faiss_results": [{"text": text, "score": score} for text, score in zip(top_texts, all_scores)],
        "best_result": {"text": best_text, "score": best_score}
    }

# Load models and data on startup
@app.before_first_request
def load_resources():
    global index, texts, metadata, model
    embedding_file = "/combined_embeddings.npy"
    faiss_index_file = "/combined_faiss.index"
    data_file = "/processed_data.json"

    # Load FAISS index and related data
    _, index, texts, metadata = load_faiss_and_data(embedding_file, faiss_index_file, data_file)

    # Load SentenceTransformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')

# Define API route
@app.route('/query', methods=['POST'])
def handle_query():
    data = request.json
    if 'query' not in data:
        return jsonify({"error": "Query field is required"}), 400

    query = data['query']
    top_k = data.get('top_k', 5)  # Default to 5 results if not specified

    # Perform FAISS query with reranking
    response = query_faiss_index_with_reranking(query, index, texts, model, top_k)

    return jsonify(response)

# Run the Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
