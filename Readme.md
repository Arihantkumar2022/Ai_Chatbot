AI Chatbot with RAG Pipeline 

- Overview
This project is part of the Junior AI Engineer assignment by  Amlgo Labs . It implements a Retrieval-Augmented Generation (RAG) pipeline using an open-source instruction-tuned LLM and vector database to power a document-aware chatbot.

 

-Architecture & Workflow

1.  Document Preprocessing : Clean and split raw text into manageable chunks.
2.  Embedding Generation : Use `all-MiniLM-L6-v2` to convert chunks into dense vectors.
3.  Vector Store Indexing : Store these vectors in FAISS for fast semantic search.
4.  Retriever : At query time, perform similarity search over FAISS to retrieve top-k relevant chunks.
5.  Prompt Construction : Inject retrieved context and user query into a prompt.
6.  Generator : Use `llama3` model (via Ollama) to generate factual answers.
7.  Streamlit UI : Display chatbot interaction with real-time streaming response.

 

-  Folder Structure
```
project/
├── data/                       Raw and processed document files
│   ├── AI Training Document.pdf
│   ├── chunk_id_map.pkl
│   ├── chunked_document.txt
│   ├── embeddings.pkl
│   ├── faiss_index
│   ├── faiss_index.bin
│   ├── preprocess.py
│   └── raw_document.txt
├── __pycache__/                Cached compiled Python files
│   ├── rag_pipeline.cpython-310.pyc
│   └── rag_pipeline.cpython-313.pyc
├── chunker.py
├── embed_store.py
├── rag_pipeline.py
├── streamlit_app.py
├── requirements.txt            Project dependencies
└── README.md                   Project summary and instructions
```

 

-  Setup & Installation

1.  Install Dependencies 
```bash
pip install -r requirements.txt
```

2.  Start Ollama Server & Pull Model 
```bash
ollama run llama3
```

3.  Run Preprocessing Script 
```bash
python data/preprocess.py
```

4.  Launch Streamlit Chatbot 
```bash
streamlit run streamlit_app.py
```

 

-  Model & Embedding Info
-  LLM Generator : `llama3` via Ollama
-  Embedding Model : `all-MiniLM-L6-v2`
-  Vector DB : FAISS (flat index)

 

-  Prompt Format
```
You are a helpful assistant. Answer based only on the context below.

Context:
<chunk1>
<chunk2>
...

Question: <user_question>
Answer:
```

 

-  Sample Queries
1. What are the user's obligations in the document?
2. When can the agreement be terminated?
3. What data privacy rights does the user have?
4. Who governs the terms of the policy?
5. What actions violate the agreement?

 

-  Demo Video / Screenshots
https://drive.google.com/drive/folders/1xcJW6NS_iLMKBls0EISr2Hr_nEtccgMJ?usp=drive_link
 

-  Notes on Limitations
- The model can hallucinate if context is insufficient
- Slow response possible for long contexts
- Answer quality depends on chunk retrieval accuracy
