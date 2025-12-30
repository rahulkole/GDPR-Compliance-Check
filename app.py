import os
import streamlit as st
from utils import extract_tfpdf, load_kb, etfurl, retrieval_chain

# --- Streamlit Page Setup ---
st.set_page_config(page_title="GDPR Compliance Check", page_icon="⚖️")
st.title("GDPR Compliance Advisor ⚖️")
st.markdown("Upload your Company Privacy Policy or paste a URL to get a free GDPR compliance analysis.")

# --- Load Knowledge Base ---
kb_vecstore = load_kb()

# Sidebar
with st.sidebar:
    st.header("Knowledge Base")
    st.info("GDPR knowledge base loaded from disk.")

# --- Initialize session state ---
if "user_txt" not in st.session_state:
    st.session_state.user_txt = ""

# --- PDF Upload ---
up_file = st.file_uploader("Upload Company Privacy Policy PDF", type=["pdf"])
if up_file:
    os.makedirs("data/temp", exist_ok=True)  # Ensure folder exists
    pdf_path = "data/temp/upload.pdf"
    with open(pdf_path, "wb") as f:
        f.write(up_file.getbuffer())
    st.session_state.user_txt = extract_tfpdf(pdf_path)
    st.success("PDF text extracted successfully!")

# --- URL Input ---
url = st.text_input("Or Paste your Company Policy URL")
if url and st.button("Analyze URL"):
    with st.spinner("Scraping and processing URL..."):
        text = etfurl(url)
        if text.startswith("Error scraping"):
            st.error(text)
        else:
            st.session_state.user_txt = text
            st.success("URL text extracted successfully!")

# --- ANALYSIS ---
if st.session_state.user_txt:
    if st.button("Run Compliance Check"):
        user_txt = st.session_state.user_txt
        with st.spinner("Analyzing for GDPR compliance..."):
            try:
                st.write("Creating QA chain...")
                qa_chain = retrieval_chain(kb_vecstore)

                st.write("Processing your policy...")
                # Only use first 2000 chars to avoid too long prompts
                question = f"Analyze this privacy policy for potential GDPR compliance gaps: {user_txt[:2000]}"

                st.write("Generating analysis (this may take a moment)...")
                result = qa_chain({"query": question})

                st.success("Analysis done!")
                st.header("Compliance Analysis")
                st.write(result["result"])

                st.divider()
                st.header("Retrieved GDPR Context")
                for i, doc in enumerate(result["source_documents"], 1):
                    st.subheader(f"GDPR Reference {i}")
                    st.caption(doc.metadata.get("section_header", ""))
                    st.write(doc.page_content[:800] + "...")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.info("Please check your Groq API key or internet connection.")
