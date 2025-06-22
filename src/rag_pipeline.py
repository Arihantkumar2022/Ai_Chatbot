# rag_pipeline.py

import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import ollama  # 🦙 LLaMA via Ollama

# === Constants ===
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
LLM_NAME = 'llama3'  # Ollama model name

# === Load Embedding Model ===
embedding_model = SentenceTransformer(MODEL_NAME)

# === Load Document Chunks ===
with open("data/chunked_document.txt", "r", encoding="utf-8") as f:
    chunks = [line.strip() for line in f if line.strip()]

# === Load FAISS Index ===
index = faiss.read_index("data/faiss_index.bin")

# === Load Embeddings ===
with open("data/embeddings.pkl", "rb") as f:
    embeddings = pickle.load(f)

# === Retriever ===
def search_chunks(query, top_k=5):
    query_embedding = embedding_model.encode([query])
    query_vector = np.array(query_embedding).astype("float32")
    D, I = index.search(query_vector, top_k)
    return [chunks[i] for i in I[0] if i < len(chunks)]

# === Prompt Constructor ===
def build_prompt(query, relevant_chunks):
    context = "\n\n".join(relevant_chunks)
    return f"You are a helpful assistant. Use the following context to answer the question.\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"

# === Generator using Ollama ===
def generate_answer(prompt, model=LLM_NAME):
    response = ollama.chat(model=model, messages=[
        {"role": "user", "content": prompt}
    ])
    return response['message']['content']
