# HireMind

HireMind is an intelligent assistant that helps students prepare for internships and job interviews by providing insights from real interview experiences, structured Q&A, and tailored advice—powered by RAG (Retrieval-Augmented Generation) using LangChain, LLMs, and vector search.

---

## 1. What It Does

HireMind intelligently analyzes resumes, JSON interview logs, and documents (PDF/DOCX) from your `data/` folder to answer questions like:
- *"What were the interview questions for Google SDE roles?"*
- *"Share the candidate experience for data analyst roles at EY."*
- *"Any tips for cracking intern interviews at Microsoft?"*

It returns structured results including:
- **Company**
- **Role**
- **Interview Questions**
- **Experience Summary**
- **Tips**

---

## 2. Features

- **Smart Retrieval:** Powered by Chroma DB and HuggingFace embeddings (`e5-small-v2`).
- **Multi-Format Ingestion:** Parsers for PDF, DOCX, and JSON files using LangChain and PyMuPDF.
- **Local RAG Pipeline:** Generates answer context using Ollama (LLaMA3).
- **Structured Career Guidance:** Automatically formats answers for readability.
- **Interactive Web Interface:** Streamlit UI for seamless querying.

---

## 3. Tech Stack

| Layer | Tooling |
| :--- | :--- |
| **LLM** | Ollama (LLaMA3) |
| **Embeddings** | HuggingFace (`intfloat/e5-small-v2`) |
| **Framework** | LangChain |
| **Vector DB** | ChromaDB |
| **Frontend** | Streamlit |
| **File I/O** | PyMuPDF (`fitz`), Unstructured, `python-docx` |

---

## 4. Project Structure

```text
HireMind/
├── app.py                      # Main Streamlit app
├── data/                       # Input documents (PDFs, DOCX, JSON)
├── notebooks/                  # Experimental & data preprocessing notebooks
│   └── data_preprocessing.ipynb
├── .env                        # Local environment variables (ignored by Git)
├── .gitignore                  # Git ignore rules
├── README.md                   # Project documentation
└── requirements.txt            # Required dependencies