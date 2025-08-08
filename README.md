# RAG Research Assistant 
This project is a Retrieval-Augmented Generation (RAG) based AI assistant designed to answer questions about a collection of AI and Machine Learning research publications. It leverages a local vector database to store document embeddings and uses the Groq API for fast language model inference. The application features both a command-line interface and a user-friendly web UI built with Streamlit.

This project was developed as part of the Agentic AI Developer Certification Program by Ready Tensor.

Features
RAG Pipeline: Answers questions using context retrieved from a specialized document set, reducing hallucinations and providing source-grounded answers.

Local Vector Store: Uses ChromaDB to store document embeddings locally, ensuring data privacy and fast retrieval.

High-Performance LLM: Powered by the Groq API using the Llama 3 model for near-instantaneous responses.

Dual Interface: Interacts with the assistant via a simple command-line interface or a clean, web-based UI built with Streamlit.

Modular Code: The codebase is refactored into a reusable  module for easy integration and extension.

Tech Stack
Framework: LangChain

LLM: Groq (Llama 3 8B)

Embeddings: HuggingFace sentence-transformers/all-MiniLM-L6-v2

Vector Database: ChromaDB

UI: Streamlit

Environment Management: python-dotenv

Getting Started
Follow these instructions to set up and run the project on your local machine.

1. Prerequisites
Python 3.9+

A Groq API Key (get one for free at console.groq.com)

2. Installation & Setup
Clone the repository:

git clone https://github.com/your-username/RAG-ASSISTANT.git
cd RAG-ASSISTANT


Create and activate a virtual environment:

# For Windows
python -m venv myenv
myenv\Scripts\activate

# For macOS/Linux
python3 -m venv myenv
source myenv/bin/activate


Install the required dependencies:

pip install -r requirements.txt


Configure your environment variables:

Create a copy of the example environment file:

# In the 'src' directory
cp .env.example .env


Open src/.env and add your Groq API key:

GROQ_API_KEY="your-actual-api-key-here"


3. Running the Application
You can run the RAG Assistant in two ways:

A) Streamlit Web UI (Recommended)

To launch the user-friendly web interface, run the following command from the project's root directory:

streamlit run app.py


This will open the application in your web browser. The first time you run it, it will take a few minutes to download the embedding model and build the vector database.

B) Command-Line Interface

To chat with the assistant directly in your terminal, run this command from the project's root directory:

python src/rag_assistant.py
