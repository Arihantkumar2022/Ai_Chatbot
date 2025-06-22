# rag_pipeline.py

import faiss
import pickle
import os
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# === Paths ===
EMBEDDINGS_PATH = "data/faiss_index"
CHUNKS_PATH = "data/chunked_document.txt"
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"  # Replace with your preferred LLM

# === Load resources ===
print("\n🔍 Loading FAISS index and chunks...")
index = faiss.read_index(os.path.join(EMBEDDINGS_PATH, "index.faiss"))
with open(os.path.join(EMBEDDINGS_PATH, "chunks.pkl"), "rb") as f:
    chunks = pickle.load(f)

embed_model = SentenceTransformer("all-MiniLM-L6-v2")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
model = model.to("cuda" if torch.cuda.is_available() else "cpu")

# === Functions ===
def search_chunks(query: str, top_k: int = 3):
    query_embedding = embed_model.encode([query])
    D, I = index.search(query_embedding, top_k)
    return [chunks[i] for i in I[0]]

def build_prompt(query: str, context_chunks):
    context = "\n---\n".join(context_chunks)
    prompt = f"""
You are a helpful assistant. Use the context below to answer the user's query.

Context:
{context}

Query:
{query}

Answer:
"""
    return prompt

def generate_answer(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=300, do_sample=False)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# === Test ===
if __name__ == "__main__":
    user_query = input("\n💬 Enter your question: ")
    retrieved = search_chunks(user_query)
    print("\n Retrieved Context Chunks:")
    for i, chunk in enumerate(retrieved):
        print(f"[{i+1}] {chunk[:200]}...\n")

    prompt = build_prompt(user_query, retrieved)
    print("\n Generating answer...")
    answer = generate_answer(prompt)
    print("\n Answer:\n", answer.split("Answer:")[-1].strip())
