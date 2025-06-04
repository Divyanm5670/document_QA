import streamlit as st
from utils.ingest import ingest_file
from utils.qa_engine import ask_question, load_vectorstore
import os
import uuid

st.set_page_config(page_title="📄 Local Document Q&A", layout="wide")
st.title("📚 Document Question and Answer ")

os.makedirs("docs", exist_ok=True)

if "file_indexed" not in st.session_state:
    st.session_state.file_indexed = False
if "question" not in st.session_state:
    st.session_state.question = ""
if "answer" not in st.session_state:
    st.session_state.answer = ""

uploaded_file = st.file_uploader("Upload a document", type=["pdf", "txt", "docx"])

if uploaded_file and not st.session_state.file_indexed:
    unique_filename = f"{uuid.uuid4()}_{uploaded_file.name}"
    file_path = os.path.join("docs", unique_filename)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    with st.spinner("Indexing document..."):
        ingest_file(file_path)
        load_vectorstore()  # Load after indexing!

    st.success("Indexed! Now ask your questions.")
    st.session_state.file_indexed = True

st.write("### Ask a question about your document:")
col1, col2 = st.columns([8, 1])
with col1:
    question_input = st.text_input(" ", value=st.session_state.question, label_visibility="collapsed", key="question_box")
with col2:
    if st.button("❌", help="Clear question and answer"):
        st.session_state.question = ""
        st.session_state.answer = ""
        st.stop()

st.session_state.question = question_input

if st.session_state.question.strip():
    with st.spinner("Searching and answering..."):
        st.session_state.answer = ask_question(st.session_state.question)

if st.session_state.answer:
    st.markdown(f"**Answer:** {st.session_state.answer}")
