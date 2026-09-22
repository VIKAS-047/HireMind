# HireMind

HireMind is an AI assistant that helps students prepare for internships and job interviews. It uses RAG (Retrieval-Augmented Generation) with LangChain, ChromaDB, and a local LLaMA3 model to answer questions from real interview experiences, resumes, and documents.

## What It Does

Add your PDF, DOCX, and JSON files to the `data/` folder, then ask questions like:

- "What were the interview questions for Google SDE roles?"
- "Share the candidate experience for data analyst roles at EY."
- "Any tips for cracking intern interviews at Microsoft?"

Each answer is returned in a structured format:

- Company
- Role
- Interview Questions
- Experience Summary
- Tips

## Features

- Semantic search using ChromaDB and HuggingFace embeddings
- PDF, DOCX, and JSON ingestion with LangChain and Unstructured
- RAG pipeline powered by Ollama (LLaMA3), running locally
- Structured, easy-to-read answers
- Simple Streamlit interface

## Tech Stack

| Layer     | Tool                                 |
|-----------|--------------------------------------|
| LLM       | Ollama (LLaMA3)                      |
| Embedding | HuggingFace (`intfloat/e5-small-v2`) |
| RAG       | LangChain                            |
| Vector DB | ChromaDB                             |
| Frontend  | Streamlit                            |
| File I/O  | `pymupdf`, `unstructured`, `docx`    |

## Project Structure

```
.
├── app.py      # Main Streamlit app
├── a.ipynb     # Experimental notebook
├── req.txt     # Dependencies
├── .env        # Environment variables
└── data/       # Input documents (PDF, DOCX, JSON)
```

## Getting Started

**1. Clone the repository**

```bash
git clone https://github.com/VIKAS-047/HireMind.git
cd HireMind
```

**2. Install Ollama and pull the model**

```bash
ollama pull llama3
```

**3. Install dependencies**

```bash
pip install -r req.txt
```

**4. Add your documents**

Create a `data/` folder in the project root and put your PDF, DOCX, and JSON files in it.

**5. Run the app**

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal (usually `http://localhost:8501`) and start asking questions.
