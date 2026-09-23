import glob
import os
import json
import fitz  # PyMuPDF
import streamlit as st

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

class FitzPDFLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load(self):
        documents = []
        try:
            doc = fitz.open(self.file_path)
            for page_num in range(len(doc)):
                text = doc[page_num].get_text("text")
                if text.strip():
                    documents.append(Document(
                        page_content=text.strip(),
                        metadata={"source": self.file_path, "page": page_num + 1}
                    ))
            doc.close()
        except Exception as e:
            st.error(f"Error loading PDF {self.file_path}: {e}")
        return documents

@st.cache_resource(show_spinner="Indexing documents and loading LLM...")
def initialize_rag_chain():
    base_path = os.path.join(os.getcwd(), "data")  
    pdf_paths = glob.glob(os.path.join(base_path, "**", "*.pdf"), recursive=True)
    docx_paths = glob.glob(os.path.join(base_path, "**", "*.docx"), recursive=True)
    json_paths = glob.glob(os.path.join(base_path, "**", "*.json"), recursive=True)

    all_documents = []
    docs_json = []

    # Fixed shadowing issue by renaming file handler to json_file
    for path in json_paths:
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as json_file:
                data = json.load(json_file)
                if isinstance(data, dict) and "interviews" in data:
                    company_name = data.get("metadata", {}).get("company", "Unknown")
                    for interview in data["interviews"]:
                        interview_text = json.dumps(interview, indent=2)
                        docs_json.append(Document(
                            page_content=interview_text, 
                            metadata={"source": path, "company": company_name}
                        ))
                else:
                    doc_text = json.dumps(data, indent=2)
                    all_documents.append(Document(page_content=doc_text, metadata={"source": path}))
        except Exception as e:
            print(f"❌ Error loading JSON {path}: {e}")

    for path in pdf_paths:
        loader = FitzPDFLoader(path)
        all_documents.extend(loader.load())

    for path in docx_paths:
        try:
            loader = UnstructuredFileLoader(path)
            all_documents.extend(loader.load())
        except Exception as e:
            print(f"❌ Error loading DOCX {path}: {e}")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    docs_pdf = text_splitter.split_documents(all_documents)
    docs = docs_pdf + docs_json

    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/e5-small-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    # Persist vector database locally so it doesn't need to rebuild every session
    persist_dir = os.path.join(os.getcwd(), "chroma_db")
    db = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory=persist_dir)

    retriever = db.as_retriever(search_kwargs={"k": 10})
    llm = OllamaLLM(model="llama3")

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
    return create_retrieval_chain(retriever, doc_chain)

# Streamlit UI
st.title("InternGenie")
query = st.text_input("Your smart assistant for landing the right internship.")

if query:
    with st.spinner("Searching and generating response..."):
        rag_chain = initialize_rag_chain()
        result = rag_chain.invoke({"input": query})
        st.markdown("### 💡 Answer:")
        st.write(result["answer"])