
# Streamlit LLM Application

## Project Overview
An interactive PDF Question Answering application developed using Streamlit, LangChain, ChromaDB, and Ollama.

## Objectives
- Develop an interactive LLM web application.
- Upload and process PDF documents.
- Extract text from PDF files.
- Store document embeddings in ChromaDB.
- Retrieve relevant document chunks using semantic search.
- Generate answers using Ollama.

## Technologies Used
- Python
- Streamlit
- LangChain
- ChromaDB
- Hugging Face Embeddings
- Ollama (Llama 3.2)

## Features
- PDF document upload
- PDF filename and page count display
- Customizable chunk size and overlap
- Semantic search with Top-5 retrieval
- AI-generated answers
- Chat interface and conversation history
- Retrieved document chunks display

## How to Run

1. Install the required libraries:
   pip install -r requirements.txt

2. Ensure Ollama is running and llama3.2:latest is available.

3. Run the application:
   streamlit run app.py

4. Open http://localhost:8501 in your browser.
