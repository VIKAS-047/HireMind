from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma 
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import glob
import os
import pandas as pd
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
import streamlit as st
import json
import fitz  
import re
from langchain_core.documents import Document
import fitz  

class FitzPDFLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load(self):
        try:
            documents = []
            doc = fitz.open(self.file_path)
            full_text = ""
            for page_num in range(len(doc)):
                page = doc[page_num]
                full_text += page.get_text("text") + "\n"
            doc.close()
            if full_text.strip():
                documents.append(Document(
                    page_content=full_text.strip(),
                    metadata={"source": self.file_path}
                ))          
            return documents
        except Exception as e:
            print(f"❌ Error loading {self.file_path} with fitz: {e}")
            return []

@st.cache_resource
def f():
    base_path = os.path.join(os.getcwd(), "data")  
    pdf_paths = glob.glob(os.path.join(base_path, "**", "*.pdf"), recursive=True)
    docx_paths = glob.glob(os.path.join(base_path, "**", "*.docx"), recursive=True)
    json_paths = glob.glob(os.path.join(base_path, "**", "*.json"), recursive=True)
    all_documents = []
    docs_json=[]
    for path in json_paths:
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                data = json.load(f)
                if isinstance(data, dict) and "interviews" in data:
                    company_name = data.get("metadata", {}).get("company", "Unknown")
                    for interview in data["interviews"]:
                        interview_text = json.dumps(interview, indent=2)
                        docs_json.append(Document(page_content=interview_text, metadata={"source": path, "company": company_name}))
                else:
                    doc_text = json.dumps(data, indent=2)
                    all_documents.append(Document(page_content=doc_text, metadata={"source": path}))
        except Exception as e:
            print(f"❌ Error loading JSON {path}: {e}")

    for path in pdf_paths:
        try:
            loader = FitzPDFLoader(path)
            docs = loader.load()
            all_documents.extend(docs)
        except Exception as e:
            print(f"Error loading {path}: {e}")

    for path in docx_paths:
        try:
            loader = UnstructuredFileLoader(path)
            docs = loader.load()
            all_documents.extend(docs)
        except Exception as e:
            print(f"❌ Error loading DOCX {path}: {e}")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    docs_pdf= text_splitter.split_documents(all_documents)
    docs=docs_pdf+docs_json
    huggingface_embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/e5-small-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    db = Chroma.from_documents(documents=docs, embedding=huggingface_embeddings)

    ret = db.as_retriever(search_kwargs={"k": 10})
    llm = Ollama(model="llama3")

    prompt = ChatPromptTemplate.from_messages([
        ("system", 
     "You are an expert career assistant that helps students prepare for internships and job interviews. "
     "Your responses must be strictly based on the given context. "
     "Present interview questions, experiences, and tips clearly and in a well-structured format."),
    ("human", 
    """
    CONTEXT:
    {context}

    QUESTION:
    {input}

    Format your answer like this:
    - Company:
    - Role:
    - Interview Questions (with rounds if available):
    - Candidate Experience Summary:
    - Tips:
    """)
    ])

    doc_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(ret, doc_chain)

st.title("InternGenie")
query = st.text_input("Your smart assistant for landing the right internship.")

if query:
    rag_chain = f()
    result = rag_chain.invoke({"input": query})
    st.markdown("### 💡 Answer:")
    st.write(result["answer"])