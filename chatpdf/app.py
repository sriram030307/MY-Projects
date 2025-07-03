
import streamlit as st
from pdf_reader import extract_text
from qa_model import ask_question

st.title("📄 ChatPDF - Ask Questions to Your PDF")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")
if uploaded_file:
    text = extract_text(uploaded_file)
    st.success("PDF Loaded!")

    query = st.text_input("Ask a question about the PDF:")
    if query:
        answer = ask_question(text, query)
        st.write("🧠 Answer:", answer)
