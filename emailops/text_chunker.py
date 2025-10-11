from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class ChunkConfig:
    chunk_size: int = 1500
    chunk_overlap: int = 100
    min_chunk_size: int = 50
    progressive_scaling: bool = True
    respect_sentences: bool = True
    respect_paragraphs: bool = True
    max_chunks: Optional[int] = None
    encoding: str = "utf-8"

class TextChunker:
    def __init__(self, config: ChunkConfig):
        self.config = config

    def chunk_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        if not text:
            return []
        
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.config.chunk_size
            chunk_text = text[start:end]
            
            chunk_metadata = metadata.copy() if metadata else {}
            chunk_metadata["chunk_index"] = len(chunks)
            
            chunks.append({
                "text": chunk_text,
                "metadata": chunk_metadata,
            })
            
            start += self.config.chunk_size - self.config.chunk_overlap
            
        return chunks


def prepare_index_units(
    text: str,
    doc_id: str,
    doc_path: str,
    subject: str = "",
    date: Optional[str] = None,
    chunk_size: int = 1500,
    chunk_overlap: int = 100,
) -> List[Dict[str, Any]]:
    """
    Prepare text for indexing by splitting it into chunks with metadata.
    
    This function is used by email_indexer.py to create indexable units from
    conversation text and attachments.
    
    Args:
        text: The text content to chunk
        doc_id: Base document identifier (e.g., "conv_id::conversation" or "conv_id::att1")
        doc_path: Path to the source document
        subject: Email subject or document title
        date: Optional date information
        chunk_size: Maximum size of each chunk in characters
        chunk_overlap: Number of characters to overlap between chunks
        
    Returns:
        List of chunk dictionaries, each containing:
        - id: Unique identifier for the chunk (doc_id::chunk{N})
        - text: The chunk text content
        - path: Path to source document
        - subject: Subject/title
        - date: Date information (if provided)
    """
    if not text or not text.strip():
        return []
    
    chunks: List[Dict[str, Any]] = []
    start = 0
    chunk_index = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end]
        
        # Create chunk with required fields
        chunk: Dict[str, Any] = {
            "id": f"{doc_id}::chunk{chunk_index}" if chunk_index > 0 else doc_id,
            "text": chunk_text,
            "path": doc_path,
            "subject": subject,
        }
        
        if date:
            chunk["date"] = date
            
        chunks.append(chunk)
        chunk_index += 1
        
        # Move to next chunk with overlap
        start += chunk_size - chunk_overlap
    
    return chunks
