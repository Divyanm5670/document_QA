import pickle
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity

# Load models once
model = SentenceTransformer('all-MiniLM-L6-v2')
qa_model = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

def load_vectorstore(index_dir="vectorstore"):
    """
    Load embeddings and corresponding texts from disk.
    """
    with open(os.path.join(index_dir, "index.pkl"), "rb") as f:
        data = pickle.load(f)
    # data is expected to be a dict with keys: "chunks" and "embeddings"
    chunks = data["chunks"]
    embeddings = np.array(data["embeddings"])
    return embeddings, chunks

def retrieve_context(query, embeddings, chunks, top_k=5):
    """
    Retrieve top_k relevant text chunks based on cosine similarity.
    """
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    # Join top relevant chunks as context
    context = "\n".join([chunks[i] for i in top_indices])
    return context

def ask_question(query):
    embeddings, chunks = load_vectorstore()
    context = retrieve_context(query, embeddings, chunks)
    
    result = qa_model({
        "context": context,
        "question": query
    })
    return result["answer"]

if __name__ == "__main__":
    while True:
        question = input("\nAsk a question (or type 'exit'): ")
        if question.lower() == "exit":
            break
        print("Answer:", ask_question(question))

