import os
import pickle
import re
import fitz  # PyMuPDF
import docx
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

model = SentenceTransformer('all-MiniLM-L6-v2')

def load_text(file_path):
    if file_path.endswith(".pdf"):
        doc = fitz.open(file_path)
        text = "\n".join([page.get_text() for page in doc])
    elif file_path.endswith(".docx"):
        doc = docx.Document(file_path)
        text = "\n".join([para.text for para in doc.paragraphs])
    elif file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        raise ValueError("Unsupported file type.")
    return clean_text(text)

def clean_text(text):
    lines = text.split('\n')
    clean_lines = []
    for line in lines:
        line = line.strip()
        if re.match(r'^\d+$', line):  # Skip lines with just numbers
            continue
        if line.lower().startswith("references") or line.lower().startswith("abstract"):
            continue
        if len(line) < 10:
            continue
        clean_lines.append(line)
    return "\n".join(clean_lines)

def split_text(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " "]
    )
    return text_splitter.split_text(text)

def create_vectorstore(text_chunks, index_dir="vectorstore"):
    embeddings = model.encode(text_chunks, show_progress_bar=True, convert_to_numpy=True)

    os.makedirs(index_dir, exist_ok=True)
    # Save both embeddings and text chunks in one pickle file
    with open(os.path.join(index_dir, "index.pkl"), "wb") as f:
        pickle.dump({"chunks": text_chunks, "embeddings": embeddings}, f)

def ingest_file(file_path):
    text = load_text(file_path)
    chunks = split_text(text)
    create_vectorstore(chunks)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python ingest.py <file_path>")
    else:
        ingest_file(sys.argv[1])

