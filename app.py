# app.py

import streamlit as st
from langchain_groq import ChatGroq
from src.rag_assistant import (
    setup_environment,
    load_and_chunk_publications,
    get_vector_db,
    get_rag_response
)
import os

# --- 1. INITIALIZATION ---
# Initialize environment and LLM
setup_environment()
llm = ChatGroq(model="llama3-8b-8192", temperature=0.7)

# Load and process documents only once
@st.cache_resource
def load_database():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_file_path = os.path.join(script_dir, 'src', 'project_1_publications.json')

    chunked_docs = load_and_chunk_publications(json_file_path)
    collection, embedding_model = get_vector_db(chunked_docs)
    return collection, embedding_model

collection, embedding_model = load_database()

# --- 2. STREAMLIT UI ---
st.title("📚 RAG Research Assistant")
st.write("Ask any question about the provided AI/ML research papers.")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if prompt := st.chat_input("What is your question?"):
    # Add user message to history and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display AI response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = get_rag_response(prompt, llm, collection, embedding_model)
            st.markdown(response)

    # Add AI response to history
    st.session_state.messages.append({"role": "assistant", "content": response})