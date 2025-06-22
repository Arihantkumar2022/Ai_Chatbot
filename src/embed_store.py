# embed_store.py

import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer
from pathlib import Path

# === Constants ===
CHUNKS_PATH = "data/chunked_document.txt"
INDEX_PATH = "data/faiss_index.bin"
EMBEDDINGS_PATH = "data/embeddings.pkl"
MODEL_NAME = "all-MiniLM-L6-v2"

# === Load chunks ===
if not os.path.exists(CHUNKS_PATH):
    raise FileNotFoundError("Chunks file not found.")

with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
    chunks = [line.strip() for line in f if line.strip()]

# === Generate embeddings ===
model = SentenceTransformer(MODEL_NAME)
embeddings = model.encode(chunks, show_progress_bar=True)

# === Store embeddings in FAISS index ===
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# === Save index and embeddings ===
os.makedirs("data", exist_ok=True)

faiss.write_index(index, INDEX_PATH)
with open(EMBEDDINGS_PATH, "wb") as f:
    pickle.dump(embeddings, f)

print(f"✅ Stored {len(chunks)} chunks in FAISS index.")
