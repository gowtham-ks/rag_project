import os
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline

# Multi-LLM selection via environment variable
LLM_MODEL = os.environ.get("LLM_MODEL", "distilgpt2")  # default

# Load documents
def load_documents(folder="docs"):
    docs, filenames = [], []
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        text = ""
        if filename.endswith(".txt"):
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
        elif filename.endswith(".pdf"):
            reader = PdfReader(path)
            for page in reader.pages:
                text += page.extract_text()
        if text.strip():
            docs.append(text)
            filenames.append(filename)
    return docs, filenames

documents, filenames = load_documents()
print(f"Loaded {len(documents)} documents.")

# Embeddings and FAISS
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = embedding_model.encode(documents)
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))
print("FAISS index created with embeddings.")

# Retrieve top-k
def retrieve(query, k=3):
    query_vec = embedding_model.encode([query])
    distances, indices = index.search(np.array(query_vec), k)
    results, sources = [], []
    for i in indices[0]:
        results.append(documents[i])
        sources.append(filenames[i])
    return results, sources

# Summarization
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
def summarize_documents(docs):
    summaries = []
    for doc in docs:
        chunk_size = 500
        chunks = [doc[i:i+chunk_size] for i in range(0, len(doc), chunk_size)]
        doc_summary = ""
        for chunk in chunks:
            doc_summary += summarizer(chunk, max_length=150, min_length=40, do_sample=False)[0]['summary_text'] + " "
        summaries.append(doc_summary.strip())
    return summaries

# Text generation
generator = pipeline("text-generation", model=LLM_MODEL)

# Generate answer
def answer(query):
    top_docs, sources = retrieve(query)
    summarized_docs = summarize_documents(top_docs)
    context = " ".join(summarized_docs)
    prompt = f"Answer the question using this context:\n{context}\nQuestion: {query}\nAnswer:"
    result = generator(prompt, max_length=250, do_sample=True)[0]['generated_text']
    return result, sources

# Interactive Q&A
if __name__ == "__main__":
    print("Enhanced RAG Assistant Ready! Type 'exit' to quit.")
    print(f"Using LLM: {LLM_MODEL}")
    while True:
        q = input("Ask a question: ")
        if q.lower() == "exit":
            break
        ans, src = answer(q)
        print("\nAnswer:\n", ans)
        print("\nSources:", ", ".join(src))
        print("-"*50)
