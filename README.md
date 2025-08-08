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

SAMPLE Inputs and Outputs:
📚 RAG Research Assistant
Ask any question about the provided AI/ML research papers.

Q)who is mahatma gandhi?

A)I cannot find the information in the provided documents.

Q)what is vae

A)Based on the provided context, a VAE (Variational Auto-Encoder) is a class of generative models designed to encode data into a compressed latent space and then decode it to reconstruct the original input. The architecture of a VAE consists of two main components: the encoder and the decoder. The encoder compresses the input data into a latent space representation, while the latent space represents the compressed data as a probability distribution, typically Gaussian. VAEs differ from traditional auto-encoders in that they have probabilistic encoding in the latent space, whereas traditional auto-encoders have deterministic encoding.

Q)how vae is used

A)Based on the provided context, VAEs are generative models designed to encode data into a compressed latent space and then decode it to reconstruct the original input. According to the context, VAEs have several applications, including:

Compression: VAEs can be used to compress data into a lower-dimensional representation, making it more efficient for storage and transmission.
Reconstruction: VAEs can be used to reconstruct the original input from the latent space representation.
Generation: VAEs can be used to generate new samples by sampling from the latent space distribution.
The context does not provide specific examples of how VAEs are used, but it highlights the differences between VAEs and traditional auto-encoders. To gather more relevant details, I would like to ask a follow-up question:

Could you please provide more context or specific examples of how VAEs are used in practice, such as in image generation, data compression, or anomaly detection?

Q)I am using it in data compression

A)Based on the provided context, it seems that you are using the Variational Auto-Encoder (VAE) for data compression. The context highlights the advantages of VAEs over traditional compression methods, including adaptability, feature learning, and the ability to capture the essence of complex data in a compact latent representation.

According to the context, VAEs can be used for compression, and it provides an example of using VAEs for efficient data compression using the MNIST dataset. The context also mentions that the principles of VAE applications extend to various real-world datasets, including image processing, tabular data, time series analysis, and more.

In your specific use case, you mentioned using VAEs for data compression. The context suggests that VAEs can be a powerful solution for data compression, particularly in scenarios where traditional methods may not be optimized for the type of data or application.

To gather more relevant details, I would like to ask a follow-up question:

Could you please provide more information on your specific use case, such as the type of data you are compressing, the size of the dataset, and the desired compression ratio? This would help me provide more tailored guidance and insights on how to effectively use VAEs for data compression in your context.

Q)where i use vae can you remind me

A)Based on our previous conversation, I can remind you that you mentioned using the Variational Auto-Encoder (VAE) for data compression. You also mentioned that you are using VAEs for data compression, and I provided some general information on how VAEs can be used for data compression.

To specifically answer your question, you used VAEs for data compression.
