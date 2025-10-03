# Enhanced RAG Project — Free Model Q&A Assistant

## 📌 Overview
This project demonstrates a **Retrieval-Augmented Generation (RAG) assistant** using **only free and open-source models**.  
The assistant can read multiple documents (TXT and PDF), retrieve relevant context, summarize it, and generate answers to user queries.

---

## ✨ Features
- 📂 Reads TXT and PDF documents  
- 🔍 Embeddings via `all-MiniLM-L6-v2`  
- 📡 Fast retrieval using FAISS  
- 📝 Multi-document summarization with `sshleifer/distilbart-cnn-12-6`  
- 🤖 Answer generation with `distilgpt2`  
- 💬 Interactive Q&A with source highlighting  
- ✅ Licensed under MIT License  

---

## 📂 Project Structure
```
rag_project/
├── docs/                  # Folder containing sample TXT/PDF documents
│   ├── sample1.txt
│   ├── sample2.txt
│   └── sample3.pdf
├── main.py                # Main RAG assistant code
├── README.md              # Project documentation
└── LICENSE                # MIT License
```

---

## ⚙️ Setup Instructions

### 1. Install Dependencies
Make sure you have Python 3.11+ installed, then run:
```bash
pip install faiss-cpu sentence-transformers transformers PyPDF2 fpdf
```

### 2. Add Documents
- Place your TXT and PDF files inside the `docs/` folder.  
- Ensure PDFs are **text-based** (not scanned images).  

### 3. Run the Assistant
Navigate to the project folder and run:
```bash
python main.py
```

---

## 🖥️ Example Usage
```
Ask a question: What is Python used for?

Answer:
Python is a high-level programming language widely used for web development, data science, AI, and automation.

Sources: sample1.txt
```

---

## 📜 License
This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## 📝 Notes
- ✅ 100% Free & Offline: All models used are open-source.  
- 📘 Designed for **educational use and Ready Tensor submission**.  
