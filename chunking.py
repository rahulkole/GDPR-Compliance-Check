import re
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class CustomChunking:
    """
    Implements Hybrid Legal Chunking:
    1. Document Structure (Splits by Headers like 'ARTICLE', 'Section').
    2. Sliding Window (Overlap to keep context).
    3. Metadata Enrichment (Attaches Section Headers to every chunk).
    """
    
    def __init__(self, chunk_size=1200, chunk_overlap=300):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Regex to find Legal Headers (Crucial for preserving meaning)
        self.section_pattern = re.compile(
            r'^(ARTICLE\s+\d+|Section\s+\w+|\d+\.\d+|\(\d+\))', 
            flags=re.IGNORECASE
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        final_chunks = []
        
        for doc in documents:
            text = doc.page_content
            
            # Step 1: Find split points based on structure (Headers)
            split_indices = [0]
            for match in self.section_pattern.finditer(text):
                split_indices.append(match.start())
            split_indices.append(len(text))
            split_indices = sorted(list(set(split_indices)))
            
            # Step 2: Create Section Objects
            sections = []
            for i in range(len(split_indices) - 1):
                start = split_indices[i]
                end = split_indices[i+1]
                section_text = text[start:end].strip()
                
                # Extract the Header for metadata
                first_line = section_text.split('\n')[0][:50] # Truncate long headers
                
                sections.append({
                    "text": section_text,
                    "metadata": {
                        "source": doc.metadata.get("source"),
                        "section_header": first_line
                    }
                })
            
            # Step 3: Recursive Split (Sliding Window) within Sections
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", " ", ""]
            )
            
            # Step 4: Enrich Metadata
            for section in sections:
                section_chunks = text_splitter.create_documents(
                    texts=[section["text"]], 
                    metadatas=[section["metadata"]]
                )
                
                for chunk in section_chunks:
                    header = section["metadata"]["section_header"]
                    # Add a description so the retriever knows the context
                    chunk.metadata["description"] = f"Part of {header}"
                    final_chunks.append(chunk)
                    
        print(f"Hybrid Chunking: Processed {len(final_chunks)} context-aware chunks.")
        return final_chunks