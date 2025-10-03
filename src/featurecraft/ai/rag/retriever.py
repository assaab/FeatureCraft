"""High-level RAG retriever with caching and PII redaction."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Literal

from .chunker import Chunk, Chunker
from .embedder import Embedder, get_embedder
from .index import RAGIndex, SearchResult
from .loader import DocumentLoader
from ...logging import get_logger

logger = get_logger(__name__)


class RAGRetriever:
    """High-level RAG retriever with knowledge base management.
    
    Features:
    - Automatic knowledge base building from directories
    - Caching with TTL
    - PII redaction in retrieved contexts
    - Multi-source retrieval (schemas, stats, ontologies, past runs)
    
    Example:
        >>> retriever = RAGRetriever(
        ...     embedder="sentence_transformers",
        ...     knowledge_dirs=["knowledge_base/", "artifacts/"]
        ... )
        >>> context = retriever.retrieve(
        ...     query="How to handle high-cardinality categoricals?",
        ...     k=5
        ... )
    """
    
    def __init__(
        self,
        embedder: Embedder | str = "sentence_transformers",
        knowledge_dirs: list[str] | None = None,
        index_path: str | None = None,
        cache_ttl_hours: int = 24,
        enable_pii_redaction: bool = True,
        enable_bm25: bool = True,
    ):
        """Initialize RAG retriever.
        
        Args:
            embedder: Embedder instance or provider name
            knowledge_dirs: Directories to load documents from
            index_path: Path to save/load index
            cache_ttl_hours: Cache TTL in hours
            enable_pii_redaction: Enable PII redaction
            enable_bm25: Enable BM25 keyword search
        """
        # Get embedder
        if isinstance(embedder, str):
            self.embedder = get_embedder(embedder)
        else:
            self.embedder = embedder
        
        self.knowledge_dirs = knowledge_dirs or []
        self.index_path = index_path or ".cache/rag_index"
        self.cache_ttl_hours = cache_ttl_hours
        self.enable_pii_redaction = enable_pii_redaction
        self.enable_bm25 = enable_bm25
        
        # Index
        self.index: RAGIndex | None = None
        self._index_hash: str = ""
        
        # Load or build index
        self._load_or_build_index()
    
    def _load_or_build_index(self) -> None:
        """Load index from cache or build from scratch."""
        index_path = Path(self.index_path)
        
        # Compute hash of knowledge directories
        current_hash = self._compute_knowledge_hash()
        
        # Check if cached index exists and is up-to-date
        hash_file = index_path / "index_hash.txt"
        if index_path.exists() and hash_file.exists():
            with open(hash_file, "r") as f:
                cached_hash = f.read().strip()
            
            if cached_hash == current_hash:
                logger.info("Loading RAG index from cache...")
                try:
                    self.index = RAGIndex.load(str(index_path), self.embedder)
                    self._index_hash = cached_hash
                    return
                except Exception as e:
                    logger.warning(f"Failed to load cached index: {e}. Rebuilding...")
        
        # Build index from scratch
        logger.info("Building RAG index from knowledge base...")
        self.rebuild_index()
    
    def rebuild_index(self) -> None:
        """Rebuild index from knowledge directories."""
        # Load documents
        loader = DocumentLoader()
        all_chunks = []
        
        for directory in self.knowledge_dirs:
            if Path(directory).exists():
                chunks = loader.load_directory(directory)
                all_chunks.extend(chunks)
        
        if not all_chunks:
            logger.warning("No documents found in knowledge directories.")
            self.index = RAGIndex(self.embedder, enable_bm25=self.enable_bm25)
            return
        
        # Build index
        self.index = RAGIndex(self.embedder, enable_bm25=self.enable_bm25)
        self.index.add_chunks(all_chunks)
        
        # Save index
        self.save_index()
    
    def save_index(self) -> None:
        """Save index to disk."""
        if self.index is None:
            return
        
        index_path = Path(self.index_path)
        self.index.save(str(index_path))
        
        # Save hash
        current_hash = self._compute_knowledge_hash()
        with open(index_path / "index_hash.txt", "w") as f:
            f.write(current_hash)
        
        self._index_hash = current_hash
    
    def retrieve(
        self,
        query: str,
        k: int = 5,
        mode: Literal["vector", "bm25", "hybrid"] = "hybrid",
        filters: dict[str, Any] | None = None,
        redact_pii: bool | None = None,
    ) -> str:
        """Retrieve relevant context for query.
        
        Args:
            query: Query string
            k: Number of results to retrieve
            mode: Search mode (vector, bm25, hybrid)
            filters: Metadata filters
            redact_pii: Override PII redaction setting
            
        Returns:
            Concatenated context string
        """
        if self.index is None or not self.index.chunks:
            logger.warning("RAG index is empty. Returning empty context.")
            return ""
        
        # Search
        results = self.index.search(query, k=k, mode=mode, filters=filters)
        
        if not results:
            return ""
        
        # Build context
        context_parts = []
        for i, result in enumerate(results):
            chunk_text = result.chunk.text
            
            # Redact PII if enabled
            if redact_pii is None:
                redact_pii = self.enable_pii_redaction
            
            if redact_pii:
                chunk_text = self._redact_pii(chunk_text)
            
            # Format chunk
            source = result.chunk.metadata.get("source", "unknown")
            context_parts.append(
                f"[Source: {source}, Rank: {i+1}, Score: {result.score:.3f}]\n{chunk_text}\n"
            )
        
        return "\n".join(context_parts)
    
    def retrieve_results(
        self,
        query: str,
        k: int = 5,
        mode: Literal["vector", "bm25", "hybrid"] = "hybrid",
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Retrieve search results (without formatting).
        
        Args:
            query: Query string
            k: Number of results
            mode: Search mode
            filters: Metadata filters
            
        Returns:
            List of SearchResult objects
        """
        if self.index is None:
            return []
        
        return self.index.search(query, k=k, mode=mode, filters=filters)
    
    def add_documents(
        self,
        texts: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        source: str = "user_documents",
    ) -> None:
        """Add documents to index dynamically.
        
        Args:
            texts: List of document texts
            metadatas: List of metadata dicts (one per document)
            source: Source name for metadata
        """
        if self.index is None:
            self.index = RAGIndex(self.embedder, enable_bm25=self.enable_bm25)
        
        # Create chunks
        chunker = Chunker(strategy="sentence", chunk_size=512, overlap=64)
        chunks = []
        
        for i, text in enumerate(texts):
            metadata = metadatas[i] if metadatas else {}
            metadata["source"] = source
            
            doc_chunks = chunker.chunk(text, metadata)
            chunks.extend(doc_chunks)
        
        # Add to index
        self.index.add_chunks(chunks)
        logger.info(f"Added {len(chunks)} chunks to index")
    
    def _compute_knowledge_hash(self) -> str:
        """Compute hash of knowledge directories for cache invalidation."""
        hash_inputs = []
        
        for directory in self.knowledge_dirs:
            if Path(directory).exists():
                # Hash directory contents
                for file_path in sorted(Path(directory).rglob("*")):
                    if file_path.is_file():
                        hash_inputs.append(str(file_path))
                        hash_inputs.append(str(file_path.stat().st_mtime))
        
        hash_str = "|".join(hash_inputs)
        return hashlib.md5(hash_str.encode()).hexdigest()
    
    @staticmethod
    def _redact_pii(text: str) -> str:
        """Redact PII from text.
        
        Redacts:
        - Email addresses
        - Phone numbers
        - Social security numbers
        - Credit card numbers
        """
        # Email addresses
        text = re.sub(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "[EMAIL]",
            text
        )
        
        # Phone numbers
        text = re.sub(
            r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b',
            "[PHONE]",
            text
        )
        
        # SSN
        text = re.sub(
            r'\b\d{3}-\d{2}-\d{4}\b',
            "[SSN]",
            text
        )
        
        # Credit cards (simple pattern)
        text = re.sub(
            r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',
            "[CC]",
            text
        )
        
        return text

