# streamlit_app.py

import streamlit as st
from rag_pipeline import search_chunks, build_prompt, generate_answer, MODEL_NAME, chunks
import time

# === App Config ===
st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("🤖 AI Chatbot with RAG Pipeline")

# === Sidebar ===
st.sidebar.markdown("## ℹ️ System Info")
st.sidebar.markdown(f"**Model:** `{MODEL_NAME}`")
st.sidebar.markdown(f"**Chunks Loaded:** `{len(chunks)}`")

if st.sidebar.button("🔄 Clear Chat"):
    st.session_state.messages = []
    st.rerun()

# === Session State ===
if "messages" not in st.session_state:
    st.session_state.messages = []

# === User Input ===
user_input = st.chat_input("Ask something from the document...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Retrieval
    retrieved_chunks = search_chunks(user_input)
    prompt = build_prompt(user_input, retrieved_chunks)

    # Streaming Answer
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        answer = generate_answer(prompt)
        final_answer = answer.split("Answer:")[-1].strip()

        stream_text = ""
        for sentence in final_answer.split(". "):
            stream_text += sentence.strip() + ". "
            response_placeholder.markdown(stream_text + "▌")
            time.sleep(0.1)
        response_placeholder.markdown(stream_text)

    st.session_state.messages.append({"role": "assistant", "content": final_answer})

    # Show Retrieved Context
    with st.expander("📚 Source Chunks", expanded=False):
        for i, chunk in enumerate(retrieved_chunks):
            st.markdown(f"**Chunk {i+1}:**\n{chunk[:500]}...")

# === Chat History ===
if st.session_state.messages:
    st.divider()
    st.markdown("### 🧾 Chat History")
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
