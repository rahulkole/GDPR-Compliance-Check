# AI-Powered GDPR Compliance Advisor

Analyze company privacy policies and identify potential GDPR compliance gaps using AI-driven Retrieval-Augmented Generation (RAG).

---

## Overview

This project implements a **Retrieval-Augmented Generation (RAG)** system to analyze privacy policy documents (PDFs or website URLs). It extracts, chunks, and embeds the text, searches relevant GDPR articles from a local Chroma vector database, and generates context-aware compliance insights using a Groq-hosted LLaMA 3.1 model.

It is designed to help companies, legal teams, and auditors quickly spot **potential GDPR gaps** in privacy policies, all via a Streamlit web interface.

---

## Key Features

- Upload **Privacy Policy PDFs** or provide a **website URL**
- Automatic text extraction and structured legal chunking
- Semantic search over official GDPR texts
- AI-generated compliance gap analysis with references to GDPR articles
- A web interface using Streamlit for easy exploration
- Stores and queries vector embeddings using Chroma

---

## Tech Stack

- **Frontend:** Streamlit  
- **Vector Database:** Chroma  
- **Embeddings:** Sentence-Transformers (`all-MiniLM-L6-v2`)  
- **LLM:** Groq (LLaMA 3.1)  
- **RAG & Processing:** LangChain  
- **Legal Text Chunking:** Custom hybrid chunking module  
- **Web Scraping:** BeautifulSoup for URL policies  
- **PDF Extraction:** PyPDF / PyMuPDF  

---
## Installation

1. **Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/gdpr-compliance-advisor.git
cd gdpr-compliance-advisor
```

2. **Create a virtual environment (optional but recommended)**
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**  
Create a `.env` file in the project root:
```env
GROQ_API_KEY=your_groq_api_key_here
```

> **Do not commit `.env`**. `.gitignore` already excludes it.
>  Don't push your API keys unless you're feeling generous enough to pay for entire internet's API requests haha. 

## Usage

1. **Run the Streamlit app**
```bash
streamlit run app.py
```

2. **Upload or provide a URL**  
   - Upload a company privacy policy PDF  
   - Or paste a URL of the policy page

3. **Run Compliance Check**  
   - Retrieves relevant GDPR articles  
   - Generates an AI-driven compliance analysis  
   - Displays **source references** and summary analysis  

---

## How It Works

1. **Document Processing**  
   - Extracts PDF or web page text  
   - Splits into meaningful sections using **RecursiveCharacterTextSplitter** with legal document structure detection (ARTICLE headers, Section markers) and sliding window overlap for context preservation

2. **Embedding & Vector Store**  
   - Embeds text using **Sentence-Transformers**  
   - Stores embeddings in **Chroma** for semantic retrieval

3. **RAG-based Compliance Analysis**  
   - **Groq LLaMA 3.1** retrieves relevant GDPR context  
   - Generates structured insights highlighting compliance gaps with citations

4. **Streamlit Interface**  
   - Upload PDF / URL  
   - Run analysis  
   - View AI-generated summary + retrieved GDPR references

---

## References

- [GDPR Official Text](https://gdpr-info.eu/)  
- [LangChain Documentation](https://www.langchain.com/)  
- [Chroma Vector Database](https://www.trychroma.com/)  
- [Groq AI LLM](https://www.groq.com/)
