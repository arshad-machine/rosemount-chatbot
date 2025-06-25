from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import tensorflow_hub as hub
import faiss
import numpy as np
import pickle

app = Flask(__name__)
CORS(app)

# Load chunks
with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

# Load Universal Sentence Encoder
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
embeddings = embed(chunks).numpy()

# FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/chat', methods=['POST'])
def chat():
    user_msg = request.json.get("message", "")
    query_vec = embed([user_msg]).numpy()
    distances, indices = index.search(query_vec, 3)
    answer = chunks[indices[0][0]]
    return jsonify({"response": answer})

if __name__ == '__main__':
    app.run()