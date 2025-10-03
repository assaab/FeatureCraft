"""Vector and keyword indexing for RAG retrieval."""

from __future__ import annotations

import json
import pickle
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .chunker import Chunk
from .embedder import Embedder
from ...logging import get_logger

logger = get_logger(__name__)


@dataclass
class SearchResult:
    """Search result with relevance score.
    
    Attributes:
        chunk: Retrieved chunk
        score: Relevance score (higher = more relevant)
        rank: Result rank (0 = top result)
    """
    
    chunk: Chunk
    score: float
    rank: int


class RAGIndex:
    """Hybrid vector + keyword index for RAG retrieval.
    
    Supports:
    - Dense vector search (cosine similarity)
    - Sparse keyword search (BM25)
    - Hybrid search (weighted combination)
    
    Example:
        >>> index = RAGIndex(embedder=embedder)
        >>> index.add_chunks(chunks)
        >>> results = index.search("customer churn prediction", k=5)
    """
    
    def __init__(
        self,
        embedder: Embedder,
        enable_bm25: bool = True,
    ):
        """Initialize RAG index.
        
        Args:
            embedder: Embedder for vector search
            enable_bm25: Enable BM25 keyword search
        """
        self.embedder = embedder
        self.enable_bm25 = enable_bm25
        
        # Vector storage
        self.chunks: list[Chunk] = []
        self.embeddings: np.ndarray | None = None
        
        # BM25 storage
        self.tokenized_chunks: list[list[str]] = []
        self.doc_freqs: Counter = Counter()
        self.idf_cache: dict[str, float] = {}
        self.avg_doc_len: float = 0.0
    
    def add_chunks(self, chunks: list[Chunk]) -> None:
        """Add chunks to index.
        
        Args:
            chunks: List of chunks to add
        """
        if not chunks:
            return
        
        logger.info(f"Indexing {len(chunks)} chunks...")
        
        # Add to storage
        self.chunks.extend(chunks)
        
        # Embed chunks
        texts = [c.text for c in chunks]
        new_embeddings = self.embedder.embed(texts)
        
        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])
        
        # Build BM25 index
        if self.enable_bm25:
            self._index_bm25(chunks)
        
        logger.info(f"✓ Indexed {len(self.chunks)} total chunks")
    
    def search(
        self,
        query: str,
        k: int = 5,
        mode: Literal["vector", "bm25", "hybrid"] = "hybrid",
        vector_weight: float = 0.7,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search index for relevant chunks.
        
        Args:
            query: Query string
            k: Number of results to return
            mode: Search mode (vector, bm25, hybrid)
            vector_weight: Weight for vector search in hybrid mode (0-1)
            filters: Metadata filters (e.g., {"source": "past_runs"})
            
        Returns:
            List of SearchResult objects ordered by relevance
        """
        if not self.chunks:
            logger.warning("Index is empty. Returning no results.")
            return []
        
        # Apply filters
        candidate_indices = self._apply_filters(filters) if filters else list(range(len(self.chunks)))
        
        if not candidate_indices:
            return []
        
        # Search
        if mode == "vector":
            scores = self._search_vector(query, candidate_indices)
        elif mode == "bm25":
            scores = self._search_bm25(query, candidate_indices)
        elif mode == "hybrid":
            vector_scores = self._search_vector(query, candidate_indices)
            bm25_scores = self._search_bm25(query, candidate_indices)
            
            # Normalize scores
            vector_scores = self._normalize_scores(vector_scores)
            bm25_scores = self._normalize_scores(bm25_scores)
            
            # Combine
            scores = vector_weight * vector_scores + (1 - vector_weight) * bm25_scores
        else:
            raise ValueError(f"Unknown search mode: {mode}")
        
        # Get top k
        top_k_indices = np.argsort(scores)[::-1][:k]
        
        results = [
            SearchResult(
                chunk=self.chunks[candidate_indices[idx]],
                score=float(scores[idx]),
                rank=rank,
            )
            for rank, idx in enumerate(top_k_indices)
        ]
        
        return results
    
    def _search_vector(self, query: str, candidate_indices: list[int]) -> np.ndarray:
        """Vector search using cosine similarity."""
        # Embed query
        query_embedding = self.embedder.embed([query])[0]
        
        # Get candidate embeddings
        candidate_embeddings = self.embeddings[candidate_indices]
        
        # Compute cosine similarity
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        candidate_norms = candidate_embeddings / (
            np.linalg.norm(candidate_embeddings, axis=1, keepdims=True) + 1e-8
        )
        
        scores = candidate_norms @ query_norm
        return scores
    
    def _search_bm25(self, query: str, candidate_indices: list[int]) -> np.ndarray:
        """BM25 keyword search."""
        if not self.enable_bm25 or not self.tokenized_chunks:
            return np.zeros(len(candidate_indices))
        
        query_tokens = self._tokenize(query)
        scores = np.zeros(len(candidate_indices))
        
        k1 = 1.5  # BM25 parameter
        b = 0.75  # BM25 parameter
        
        for i, idx in enumerate(candidate_indices):
            doc_tokens = self.tokenized_chunks[idx]
            doc_len = len(doc_tokens)
            
            score = 0.0
            for token in query_tokens:
                if token in doc_tokens:
                    tf = doc_tokens.count(token)
                    idf = self.idf_cache.get(token, 0.0)
                    
                    # BM25 formula
                    numerator = tf * (k1 + 1)
                    denominator = tf + k1 * (1 - b + b * (doc_len / self.avg_doc_len))
                    score += idf * (numerator / denominator)
            
            scores[i] = score
        
        return scores
    
    def _index_bm25(self, chunks: list[Chunk]) -> None:
        """Build BM25 index for chunks."""
        for chunk in chunks:
            tokens = self._tokenize(chunk.text)
            self.tokenized_chunks.append(tokens)
            
            # Update document frequencies
            unique_tokens = set(tokens)
            for token in unique_tokens:
                self.doc_freqs[token] += 1
        
        # Compute IDF
        n_docs = len(self.tokenized_chunks)
        for token, df in self.doc_freqs.items():
            self.idf_cache[token] = np.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
        
        # Compute average document length
        doc_lengths = [len(tokens) for tokens in self.tokenized_chunks]
        self.avg_doc_len = np.mean(doc_lengths) if doc_lengths else 0.0
    
    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple tokenization (can be improved with proper tokenizer)."""
        # Lowercase + split on non-alphanumeric
        import re
        tokens = re.findall(r'\w+', text.lower())
        return tokens
    
    def _apply_filters(self, filters: dict[str, Any]) -> list[int]:
        """Apply metadata filters to get candidate indices."""
        indices = []
        for i, chunk in enumerate(self.chunks):
            match = True
            for key, value in filters.items():
                if chunk.metadata.get(key) != value:
                    match = False
                    break
            if match:
                indices.append(i)
        return indices
    
    @staticmethod
    def _normalize_scores(scores: np.ndarray) -> np.ndarray:
        """Min-max normalize scores to [0, 1]."""
        min_score = scores.min()
        max_score = scores.max()
        
        if max_score - min_score < 1e-8:
            return np.ones_like(scores)
        
        return (scores - min_score) / (max_score - min_score)
    
    def save(self, path: str) -> None:
        """Save index to disk.
        
        Args:
            path: Directory path to save index
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save chunks
        with open(path / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)
        
        # Save embeddings
        if self.embeddings is not None:
            np.save(path / "embeddings.npy", self.embeddings)
        
        # Save BM25 data
        if self.enable_bm25:
            with open(path / "bm25.json", "w") as f:
                json.dump({
                    "tokenized_chunks": self.tokenized_chunks,
                    "doc_freqs": dict(self.doc_freqs),
                    "idf_cache": self.idf_cache,
                    "avg_doc_len": self.avg_doc_len,
                }, f)
        
        logger.info(f"✓ Saved index to {path}")
    
    @classmethod
    def load(cls, path: str, embedder: Embedder) -> RAGIndex:
        """Load index from disk.
        
        Args:
            path: Directory path to load index from
            embedder: Embedder instance
            
        Returns:
            Loaded RAGIndex
        """
        path = Path(path)
        
        index = cls(embedder=embedder, enable_bm25=False)
        
        # Load chunks
        with open(path / "chunks.pkl", "rb") as f:
            index.chunks = pickle.load(f)
        
        # Load embeddings
        embeddings_path = path / "embeddings.npy"
        if embeddings_path.exists():
            index.embeddings = np.load(embeddings_path)
        
        # Load BM25 data
        bm25_path = path / "bm25.json"
        if bm25_path.exists():
            with open(bm25_path, "r") as f:
                bm25_data = json.load(f)
            
            index.enable_bm25 = True
            index.tokenized_chunks = bm25_data["tokenized_chunks"]
            index.doc_freqs = Counter(bm25_data["doc_freqs"])
            index.idf_cache = bm25_data["idf_cache"]
            index.avg_doc_len = bm25_data["avg_doc_len"]
        
        logger.info(f"✓ Loaded index from {path} ({len(index.chunks)} chunks)")
        return index


def build_index(
    chunks: list[Chunk],
    embedder: Embedder,
    enable_bm25: bool = True,
) -> RAGIndex:
    """Build RAG index from chunks (convenience function).
    
    Args:
        chunks: List of chunks to index
        embedder: Embedder for vector search
        enable_bm25: Enable BM25 keyword search
        
    Returns:
        Built RAGIndex
        
    Example:
        >>> index = build_index(chunks, embedder=embedder)
        >>> results = index.search("customer churn", k=5)
    """
    index = RAGIndex(embedder=embedder, enable_bm25=enable_bm25)
    index.add_chunks(chunks)
    return index


def search_index(
    index: RAGIndex,
    query: str,
    k: int = 5,
    mode: Literal["vector", "bm25", "hybrid"] = "hybrid",
) -> list[SearchResult]:
    """Search index (convenience function).
    
    Args:
        index: RAG index
        query: Query string
        k: Number of results
        mode: Search mode
        
    Returns:
        List of SearchResult objects
    """
    return index.search(query, k=k, mode=mode)

