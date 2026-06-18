from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv

from llms.gemini_llm import GeminiLLM
from llms.llama_cpp_llm import LlamaCppLLM
from utils.chunker import chunk_text
from utils.pdf_loader import load_pdf
from utils.vectordb import VectorDB

import os

load_dotenv()


st.set_page_config(
    page_title="RAG Chatbot",
    layout="wide",
)

st.title("📚 RAG Chatbot")


llm_choice = st.sidebar.selectbox(
    "Select LLM",
    [
        "Gemini",
        "Llama.cpp",
    ],
)

uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"],
)

if "vectordb" not in st.session_state:
    st.session_state.vectordb = VectorDB()

if uploaded_file:
    temp_pdf = Path("temp.pdf")

    with open(
        temp_pdf,
        "wb",
    ) as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("Processing document..."):
        text = load_pdf(str(temp_pdf))

        chunks = chunk_text(text)

        st.session_state.vectordb.add_documents(chunks)

    st.success("Document indexed.")

question = st.text_input("Ask a question")

if question:
    context = st.session_state.vectordb.search(question)

    if llm_choice == "Gemini":
        llm: Any = GeminiLLM(
            api_key=os.getenv(
                "GOOGLE_API_KEY",
                "",
            )
        )

    else:
        llm = LlamaCppLLM(model_path=("models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"))

    answer = llm.generate(
        context=context,
        question=question,
    )

    st.markdown("### Answer")
    st.write(answer)
