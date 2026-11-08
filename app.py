
import streamlit as st
import os
import uuid
import csv

from pdf_qa import (
    load_pdf,
    split_documents,
    create_vector_store,
    retrieve_chunks,
    generate_answer,
    generate_answer_without_retrieval
)

def save_results(question, chunks, answer, answer_without_retrieval):
    os.makedirs("results", exist_ok=True)

    file_path = "results/comparison.csv"
    file_exists = os.path.exists(file_path)

    with open(file_path, "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow([
                "Question",
                "Retrieved Chunks",
                "Answer With Retrieval",
                "Answer Without Retrieval"
            ])

        writer.writerow([
            question,
            "\n\n".join(chunk.page_content for chunk in chunks),
            answer,
            answer_without_retrieval
        ])


st.set_page_config(
    page_title="PDF Question Answering",
    page_icon="📚",
    layout="wide"
)

st.title("📚 PDF Question Answering System")
st.write("Upload a PDF and ask questions based on its content.")

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

if "page_count" not in st.session_state:
    st.session_state.page_count = 0


with st.sidebar:
    st.header("⚙️ Settings")
    st.write("Upload your PDF document below.")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"]
    )

    chunk_size = st.slider(
        "Chunk Size",
        min_value=300,
        max_value=1500,
        value=1000,
        step=100
    )

    chunk_overlap = st.slider(
        "Chunk Overlap",
        min_value=0,
        max_value=300,
        value=200,
        step=50
    )

    process_button = st.button(
        "Process PDF",
        use_container_width=True
    )

    if uploaded_file and process_button:
        os.makedirs("dataset/uploaded_files", exist_ok=True)

        pdf_path = os.path.join(
            "dataset/uploaded_files",
            uploaded_file.name
        )

        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        with st.spinner("Processing PDF..."):
            documents = load_pdf(pdf_path)

            chunks = split_documents(
                documents,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )

            collection_name = "pdf_" + uuid.uuid4().hex[:12]

            vectorstore = create_vector_store(
                chunks,
                persist_directory="chroma_db",
                collection_name=collection_name
            )

            st.session_state.vectorstore = vectorstore
            st.session_state.pdf_name = uploaded_file.name
            st.session_state.page_count = len(documents)
            st.session_state.chat_history = []

        st.success("PDF processed successfully!")


if st.session_state.vectorstore is not None:
    st.subheader("📄 Uploaded Document")

    col1, col2 = st.columns(2)

    col1.metric("PDF Filename", st.session_state.pdf_name)
    col2.metric("Total Pages", st.session_state.page_count)

    st.divider()
    st.subheader("💬 Ask Questions")

    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(chat["question"])

        with st.chat_message("assistant"):
            st.write(chat["answer"])

            with st.expander("View Retrieved Chunks"):
                for i, chunk in enumerate(chat["chunks"]):
                    st.write(f"**Chunk {i+1}**")
                    st.write(chunk.page_content)
                    st.divider()

            with st.expander("Answer Without Retrieval"):
                st.write(chat["answer_without_retrieval"])

    question = st.chat_input("Ask something about your PDF...")

    if question:
        with st.chat_message("user"):
            st.write(question)

        with st.spinner("Generating answers..."):
            retrieved_chunks = retrieve_chunks(
                st.session_state.vectorstore,
                question
            )

            answer = generate_answer(
                question,
                retrieved_chunks
            )

            answer_without_retrieval = generate_answer_without_retrieval(
                question
            )

            save_results(
                question,
                retrieved_chunks,
                answer,
                answer_without_retrieval
            )

        with st.chat_message("assistant"):
            st.write("**Answer With Retrieval:**")
            st.write(answer)

            with st.expander("View Retrieved Chunks"):
                for i, chunk in enumerate(retrieved_chunks):
                    st.write(f"**Chunk {i+1}**")
                    st.write(chunk.page_content)
                    st.divider()

            with st.expander("Answer Without Retrieval"):
                st.write(answer_without_retrieval)

        st.session_state.chat_history.append({
            "question": question,
            "answer": answer,
            "answer_without_retrieval": answer_without_retrieval,
            "chunks": retrieved_chunks
        })
