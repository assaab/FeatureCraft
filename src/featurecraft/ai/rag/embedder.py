"""Embedding providers for RAG retrieval."""

from __future__ import annotations

import hashlib
import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from ...logging import get_logger

logger = get_logger(__name__)


class Embedder(ABC):
    """Base class for embedding providers."""
    
    def __init__(self, cache_dir: str | None = None):
        """Initialize embedder.
        
        Args:
            cache_dir: Directory for embedding cache
        """
        self.cache_dir = cache_dir or ".cache/embeddings"
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
    
    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed texts into vectors.
        
        Args:
            texts: List of text strings
            
        Returns:
            Array of shape (len(texts), embedding_dim)
        """
        pass
    
    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Get embedding dimension."""
        pass
    
    def embed_with_cache(self, texts: list[str], cache_key: str | None = None) -> np.ndarray:
        """Embed texts with caching.
        
        Args:
            texts: List of text strings
            cache_key: Cache key (auto-generated if None)
            
        Returns:
            Array of embeddings
        """
        # Generate cache key
        if cache_key is None:
            cache_key = hashlib.md5(json.dumps(texts).encode()).hexdigest()
        
        cache_path = Path(self.cache_dir) / f"{cache_key}.npy"
        
        # Check cache
        if cache_path.exists():
            logger.debug(f"Loading embeddings from cache: {cache_path}")
            return np.load(cache_path)
        
        # Compute embeddings
        embeddings = self.embed(texts)
        
        # Save to cache
        np.save(cache_path, embeddings)
        logger.debug(f"Saved embeddings to cache: {cache_path}")
        
        return embeddings


class OpenAIEmbedder(Embedder):
    """OpenAI embeddings provider."""
    
    MODELS = {
        "text-embedding-3-small": {"dim": 1536, "cost_per_1m": 0.020},
        "text-embedding-3-large": {"dim": 3072, "cost_per_1m": 0.130},
        "text-embedding-ada-002": {"dim": 1536, "cost_per_1m": 0.100},
    }
    
    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        cache_dir: str | None = None,
    ):
        """Initialize OpenAI embedder.
        
        Args:
            model: OpenAI embedding model name
            api_key: OpenAI API key (or use OPENAI_API_KEY env var)
            cache_dir: Cache directory
        """
        super().__init__(cache_dir)
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY env var.")
        
        if model not in self.MODELS:
            raise ValueError(f"Unknown model: {model}. Available: {list(self.MODELS.keys())}")
    
    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed texts using OpenAI API."""
        try:
            import openai
        except ImportError:
            raise ImportError("OpenAI package required: pip install openai>=1.0.0")
        
        client = openai.OpenAI(api_key=self.api_key)
        
        # Batch embed (OpenAI supports up to 2048 texts per request)
        batch_size = 2048
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = client.embeddings.create(
                model=self.model,
                input=batch,
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
        
        return np.array(all_embeddings, dtype=np.float32)
    
    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension."""
        return self.MODELS[self.model]["dim"]


class SentenceTransformerEmbedder(Embedder):
    """Sentence Transformers embeddings provider (local, free)."""
    
    def __init__(
        self,
        model: str = "all-MiniLM-L6-v2",
        cache_dir: str | None = None,
        device: str | None = None,
    ):
        """Initialize Sentence Transformer embedder.
        
        Args:
            model: Sentence Transformer model name
            cache_dir: Cache directory
            device: Device (cpu, cuda, mps)
        """
        super().__init__(cache_dir)
        self.model_name = model
        self.device = device
        self._model = None
    
    @property
    def model(self):
        """Lazy load Sentence Transformer model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError(
                    "sentence-transformers required: pip install sentence-transformers"
                )
            
            self._model = SentenceTransformer(self.model_name, device=self.device)
            logger.info(f"Loaded Sentence Transformer model: {self.model_name}")
        
        return self._model
    
    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed texts using Sentence Transformers."""
        embeddings = self.model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return embeddings
    
    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension."""
        return self.model.get_sentence_embedding_dimension()


class MockEmbedder(Embedder):
    """Mock embedder for testing (returns random embeddings)."""
    
    def __init__(self, embedding_dim: int = 128, cache_dir: str | None = None):
        """Initialize mock embedder.
        
        Args:
            embedding_dim: Embedding dimension
            cache_dir: Cache directory
        """
        super().__init__(cache_dir)
        self._embedding_dim = embedding_dim
        self.rng = np.random.RandomState(42)
    
    def embed(self, texts: list[str]) -> np.ndarray:
        """Return random embeddings."""
        return self.rng.randn(len(texts), self._embedding_dim).astype(np.float32)
    
    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension."""
        return self._embedding_dim


def get_embedder(
    provider: str = "sentence_transformers",
    model: str | None = None,
    cache_dir: str | None = None,
    **kwargs,
) -> Embedder:
    """Get embedder by provider name.
    
    Args:
        provider: Provider name (openai, sentence_transformers, mock)
        model: Model name (provider-specific)
        cache_dir: Cache directory
        **kwargs: Additional provider-specific parameters
        
    Returns:
        Initialized embedder
        
    Example:
        >>> embedder = get_embedder("sentence_transformers", model="all-MiniLM-L6-v2")
        >>> embeddings = embedder.embed(["hello", "world"])
    """
    if provider == "openai":
        return OpenAIEmbedder(
            model=model or "text-embedding-3-small",
            cache_dir=cache_dir,
            **kwargs,
        )
    elif provider == "sentence_transformers":
        return SentenceTransformerEmbedder(
            model=model or "all-MiniLM-L6-v2",
            cache_dir=cache_dir,
            **kwargs,
        )
    elif provider == "mock":
        return MockEmbedder(cache_dir=cache_dir, **kwargs)
    else:
        raise ValueError(
            f"Unknown embedder provider: {provider}. "
            f"Available: openai, sentence_transformers, mock"
        )

