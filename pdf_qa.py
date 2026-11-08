
import os
import re

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM


# Initialize embedding model
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# Initialize Ollama LLM
def get_llm():
    return OllamaLLM(model="llama3.2:latest")


# Extract text from PDF
def load_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    return documents


# Split PDF into chunks
def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = splitter.split_documents(documents)
    return chunks


# Create ChromaDB collection
def create_vector_store(
    chunks,
    persist_directory="chroma_db",
    collection_name="pdf_collection"
):
    embeddings = get_embeddings()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=persist_directory
    )

    return vectorstore


# Retrieve Top-5 relevant chunks
def retrieve_chunks(vectorstore, question):
    results = vectorstore.similarity_search(
        question,
        k=5
    )
    return results


# Generate answer using Ollama
def generate_answer(question, retrieved_chunks):
    llm = get_llm()

    context = "\n\n".join(
        [doc.page_content for doc in retrieved_chunks]
    )

    prompt = f"""
You are a helpful PDF question answering assistant.

Answer the question using only the information provided
in the context.

If the answer is not available in the context,
say that the information is not available in the PDF.

Context:
{context}

Question:
{question}

Answer:
"""

    response = llm.invoke(prompt)
    return response

# Generate answer without retrieval
def generate_answer_without_retrieval(question):
    llm = get_llm()

    prompt = f"""
Answer the following question using your general knowledge.

Question: {question}

Answer:
"""

    response = llm.invoke(prompt)
    return response
