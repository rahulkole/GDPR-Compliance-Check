import os
import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader
from dotenv import load_dotenv
import glob
from embedding import Embedder
from typing import List

from langchain_community.vectorstores import Chroma

#langchain imports
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from chunking import CustomChunking
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document

load_dotenv()

kb_path = "./data/chroma_gdpr"

legal_chunker = CustomChunking(chunk_size = 1200, chunk_overlap = 300)

def build_kb(folderpath):
    """Loads GDPR and EU Digital Law PDFs, chunks them,
       saves to the vector store"""

    pdf_files = glob.glob(f"{folderpath}/*.pdf")
    if not pdf_files:
        raise FileNotFoundError(f"No PDFs found in {folderpath}")

    all_docs = []
    for file in pdf_files:
        print(f"Loading {file}")
        loader = PyMuPDFLoader(file)  # Pass **file**, not folder
        all_docs.extend(loader.load())

    #CHUNKING
    chunks = legal_chunker.split_documents(all_docs)

    #EMBEDDING
    embeddings= Embedder("all-MiniLM-L6-v2")
    
    #VECTOR STORE
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory= kb_path
    )
    
    print(f"Knowledge Base built with {len(chunks)} chunks.")
    return vectorstore 


def load_kb():
    """
    Loads an existing GDPR knowledge base from disk.
    No re-embedding.
    """

    embeddings = Embedder("all-MiniLM-L6-v2")

    return Chroma(
        persist_directory=kb_path,
        embedding_function=embeddings
    )


###Knowledge base done, now User Document Utility side

def extract_tfpdf(file_path: str) -> str:   #extract text from pdf
    """
    Simple PDF text extractor.
    """
    reader = PdfReader(file_path)
    text = ""

    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"

    return text

def etfurl(url):
    """Extracts raw text from any URL using BeautifulSoup."""
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        # Remove script and style elements
        for script_or_style in soup(['script', 'style']):
            script_or_style.extract()
        return soup.get_text(separator=' ')
    except Exception as e:
        return f"Error scraping URL: {e}"



###Q/A 

def retrieval_chain(vectorstore: Chroma):
    """
    Creates a GDPR compliance analysis chain using Groq LLM.
    """

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0
    )

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are a GDPR compliance expert.

Use ONLY the provided GDPR context.
Do not hallucinate.

GDPR CONTEXT:
{context}

QUESTION:
{question}

Your task:
- Identify potential GDPR compliance gaps
- Reference relevant GDPR Articles where applicable
- Be concise and structured
"""
    )

    def chain(query_dict):
        query = query_dict.get("query", "")

        # Retrieve relevant GDPR chunks
        retrieved_docs = vectorstore.similarity_search(query, k=3)

        # Build context
        context = "\n\n".join(doc.page_content for doc in retrieved_docs)

        # Prompt LLM
        formatted_prompt = prompt.format(
            context=context,
            question=query
        )

        response = llm.invoke(formatted_prompt)

        return {
            "result": response.content,
            "source_documents": retrieved_docs
        }

    return chain