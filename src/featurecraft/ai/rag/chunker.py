"""Document chunking for RAG retrieval."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal


@dataclass
class Chunk:
    """Document chunk with metadata.
    
    Attributes:
        text: Chunk text content
        metadata: Chunk metadata (source, page, chunk_id, etc.)
        chunk_id: Unique chunk identifier
    """
    
    text: str
    metadata: dict[str, Any]
    chunk_id: str = ""
    
    def __post_init__(self):
        """Generate chunk ID if not provided."""
        if not self.chunk_id:
            import hashlib
            self.chunk_id = hashlib.md5(self.text.encode()).hexdigest()[:12]


class Chunker:
    """Document chunker with configurable strategies.
    
    Supports fixed-size, sentence-based, and semantic chunking strategies.
    
    Example:
        >>> chunker = Chunker(strategy="fixed", chunk_size=512, overlap=64)
        >>> chunks = chunker.chunk(document_text)
    """
    
    def __init__(
        self,
        strategy: Literal["fixed", "sentence", "semantic"] = "fixed",
        chunk_size: int = 512,
        overlap: int = 64,
        min_chunk_size: int = 50,
    ):
        """Initialize chunker.
        
        Args:
            strategy: Chunking strategy (fixed, sentence, semantic)
            chunk_size: Target chunk size in characters
            overlap: Overlap between chunks
            min_chunk_size: Minimum chunk size (discard smaller chunks)
        """
        self.strategy = strategy
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size
    
    def chunk(self, text: str, metadata: dict[str, Any] | None = None) -> list[Chunk]:
        """Chunk document into smaller pieces.
        
        Args:
            text: Document text to chunk
            metadata: Base metadata to attach to all chunks
            
        Returns:
            List of Chunk objects
        """
        metadata = metadata or {}
        
        if self.strategy == "fixed":
            return self._chunk_fixed(text, metadata)
        elif self.strategy == "sentence":
            return self._chunk_sentence(text, metadata)
        elif self.strategy == "semantic":
            return self._chunk_semantic(text, metadata)
        else:
            raise ValueError(f"Unknown chunking strategy: {self.strategy}")
    
    def _chunk_fixed(self, text: str, metadata: dict[str, Any]) -> list[Chunk]:
        """Fixed-size chunking with overlap."""
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end].strip()
            
            if len(chunk_text) >= self.min_chunk_size:
                chunk_metadata = {
                    **metadata,
                    "start_pos": start,
                    "end_pos": end,
                    "chunk_strategy": "fixed",
                }
                chunks.append(Chunk(text=chunk_text, metadata=chunk_metadata))
            
            # Move start forward with overlap
            start = end - self.overlap
            if start >= len(text):
                break
        
        return chunks
    
    def _chunk_sentence(self, text: str, metadata: dict[str, Any]) -> list[Chunk]:
        """Sentence-based chunking (respects sentence boundaries)."""
        # Simple sentence splitting (can be improved with spaCy/nltk)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            sentence_len = len(sentence)
            
            # If adding this sentence exceeds chunk_size, finalize current chunk
            if current_length + sentence_len > self.chunk_size and current_chunk:
                chunk_text = " ".join(current_chunk)
                if len(chunk_text) >= self.min_chunk_size:
                    chunk_metadata = {
                        **metadata,
                        "sentence_count": len(current_chunk),
                        "chunk_strategy": "sentence",
                    }
                    chunks.append(Chunk(text=chunk_text, metadata=chunk_metadata))
                
                # Keep overlap sentences
                overlap_sentences = current_chunk[-(self.overlap // 50):]  # Rough overlap
                current_chunk = overlap_sentences + [sentence]
                current_length = sum(len(s) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_length += sentence_len
        
        # Add remaining chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            if len(chunk_text) >= self.min_chunk_size:
                chunk_metadata = {
                    **metadata,
                    "sentence_count": len(current_chunk),
                    "chunk_strategy": "sentence",
                }
                chunks.append(Chunk(text=chunk_text, metadata=chunk_metadata))
        
        return chunks
    
    def _chunk_semantic(self, text: str, metadata: dict[str, Any]) -> list[Chunk]:
        """Semantic chunking (groups by topic similarity).
        
        Note: This is a simplified version. For production, use
        models like sentence-transformers with clustering.
        """
        # Fallback to sentence-based for now
        # In production, this would use embeddings + clustering
        return self._chunk_sentence(text, metadata)


def chunk_document(
    text: str,
    strategy: Literal["fixed", "sentence", "semantic"] = "fixed",
    chunk_size: int = 512,
    overlap: int = 64,
    metadata: dict[str, Any] | None = None,
) -> list[Chunk]:
    """Chunk document (convenience function).
    
    Args:
        text: Document text
        strategy: Chunking strategy
        chunk_size: Target chunk size
        overlap: Overlap between chunks
        metadata: Base metadata
        
    Returns:
        List of Chunk objects
        
    Example:
        >>> chunks = chunk_document(
        ...     text=doc_text,
        ...     strategy="sentence",
        ...     chunk_size=512
        ... )
    """
    chunker = Chunker(
        strategy=strategy,
        chunk_size=chunk_size,
        overlap=overlap,
    )
    return chunker.chunk(text, metadata)

