# src/rag_assistant.py

import os
import json
import chromadb
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from .logger import rag_logger

# --- 1. SETUP and LOADING ---
def setup_environment():
    """Loads environment variables."""
    load_dotenv()

def load_and_chunk_publications(file_path):
    """Loads publications from a JSON file and splits them into chunks."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        documents = [f"Title: {item.get('title', '')}\n\n{item.get('publication_description', '')}" for item in data]

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunked_docs = text_splitter.create_documents(documents)
        return chunked_docs
    except Exception as e:
        rag_logger.error(f"Error in load and chunk publications : {e}")
        raise e

# --- 2. EMBEDDING and VECTOR DB ---
def get_vector_db(chunked_docs):
    """Creates embeddings and stores them in ChromaDB."""
    try:
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        client = chromadb.PersistentClient(path="./research_db")
        collection = client.get_or_create_collection(name="ml_publications", metadata={"hnsw:space": "cosine"})

        # Check if the database is already populated
        if collection.count() == 0:
            print("Database is empty. Populating with new embeddings...")
            documents_to_embed = [doc.page_content for doc in chunked_docs]
            batch_size = 100
            for i in range(0, len(documents_to_embed), batch_size):
                batch_docs = documents_to_embed[i:i + batch_size]
                embeddings = embedding_model.embed_documents(batch_docs)
                ids = [f"doc_{i+j}" for j in range(len(batch_docs))]
                collection.add(embeddings=embeddings, documents=batch_docs, ids=ids)
            print("Embeddings stored successfully.")
        else:
            print("Database already populated.")

        return collection, embedding_model
    except Exception as e:
        rag_logger.error(f"Error in getting vector DB : {e}")
        raise e

# --- 3. RAG PIPELINE ---
def get_rag_response(query, llm, collection, embedding_model, past_messages):
    """Processes a query, retrieves context, and generates an answer."""
    try:
        prompt_template = """
        You are a helpful AI research assistant. 
        Answer the user's question based ONLY on the provided research context and previous conversation. 
        If the answer is not present in the context, clearly state: "I cannot find the information in the provided documents."
        Relate your answer to the previous conversation where possible, and if appropriate, ask a follow-up question to gather more relevant details.
        
        PREVIOUS CONVERSATION:
        {past_messages}
        
        CONTEXT:
        {context}
        
        QUESTION:
        {question}
        
        ANSWER:
        """

        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question", "past_messages"])
        rag_chain = LLMChain(llm=llm, prompt=prompt)

        # 1. Retrieve relevant context
        query_embedding = embedding_model.embed_query(query)
        results = collection.query(query_embeddings=[query_embedding], n_results=3)
        context = "\n\n".join(results['documents'][0])

        # 2. Generate answer
        response = rag_chain.run(context=context, question=query, past_messages=past_messages)
        return response
    except Exception as e:
        rag_logger.error(f"Error in getting rag responses: {e}")
        return f"Error occurred: {e}"