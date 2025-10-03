"""Document loaders for RAG knowledge base."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from .chunker import Chunk, Chunker
from ...logging import get_logger

logger = get_logger(__name__)


class DocumentLoader:
    """Loader for various document types.
    
    Supports:
    - Dataset schemas (JSON/CSV)
    - Dataset statistics (JSON)
    - Domain ontologies (JSON/YAML)
    - Past feature engineering runs (JSON)
    - Text documents (TXT/MD)
    
    Example:
        >>> loader = DocumentLoader()
        >>> chunks = loader.load_directory("knowledge_base/", source="domain_knowledge")
    """
    
    def __init__(
        self,
        chunker: Chunker | None = None,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
    ):
        """Initialize document loader.
        
        Args:
            chunker: Chunker instance (creates default if None)
            chunk_size: Target chunk size
            chunk_overlap: Overlap between chunks
        """
        self.chunker = chunker or Chunker(
            strategy="sentence",
            chunk_size=chunk_size,
            overlap=chunk_overlap,
        )
    
    def load_directory(
        self,
        directory: str,
        source: str = "documents",
        file_patterns: list[str] | None = None,
    ) -> list[Chunk]:
        """Load all documents from directory.
        
        Args:
            directory: Directory path
            source: Source name for metadata
            file_patterns: File patterns to include (e.g., ["*.json", "*.txt"])
            
        Returns:
            List of Chunk objects
        """
        directory = Path(directory)
        
        if not directory.exists():
            logger.warning(f"Directory not found: {directory}")
            return []
        
        patterns = file_patterns or ["*.json", "*.txt", "*.md", "*.csv", "*.yaml", "*.yml"]
        
        all_chunks = []
        for pattern in patterns:
            for file_path in directory.rglob(pattern):
                try:
                    chunks = self.load_file(str(file_path), source=source)
                    all_chunks.extend(chunks)
                except Exception as e:
                    logger.warning(f"Failed to load {file_path}: {e}")
        
        logger.info(f"✓ Loaded {len(all_chunks)} chunks from {directory}")
        return all_chunks
    
    def load_file(
        self,
        file_path: str,
        source: str | None = None,
    ) -> list[Chunk]:
        """Load single file.
        
        Args:
            file_path: Path to file
            source: Source name for metadata
            
        Returns:
            List of Chunk objects
        """
        file_path = Path(file_path)
        source = source or file_path.stem
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Determine file type
        ext = file_path.suffix.lower()
        
        if ext == ".json":
            return self._load_json(file_path, source)
        elif ext in [".txt", ".md"]:
            return self._load_text(file_path, source)
        elif ext == ".csv":
            return self._load_csv(file_path, source)
        elif ext in [".yaml", ".yml"]:
            return self._load_yaml(file_path, source)
        else:
            logger.warning(f"Unsupported file type: {ext}")
            return []
    
    def _load_json(self, file_path: Path, source: str) -> list[Chunk]:
        """Load JSON file."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Convert JSON to text representation
        text = self._json_to_text(data, file_path.stem)
        
        metadata = {
            "source": source,
            "file_path": str(file_path),
            "file_type": "json",
        }
        
        return self.chunker.chunk(text, metadata)
    
    def _load_text(self, file_path: Path, source: str) -> list[Chunk]:
        """Load text file."""
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        
        metadata = {
            "source": source,
            "file_path": str(file_path),
            "file_type": "text",
        }
        
        return self.chunker.chunk(text, metadata)
    
    def _load_csv(self, file_path: Path, source: str) -> list[Chunk]:
        """Load CSV file (converts to text representation)."""
        df = pd.read_csv(file_path)
        
        # Convert to text
        text = f"CSV file: {file_path.stem}\n\n"
        text += f"Shape: {df.shape[0]} rows, {df.shape[1]} columns\n\n"
        text += f"Columns: {', '.join(df.columns)}\n\n"
        text += f"Sample data:\n{df.head(10).to_string()}\n\n"
        
        if df.shape[0] > 10:
            text += f"Statistics:\n{df.describe().to_string()}\n"
        
        metadata = {
            "source": source,
            "file_path": str(file_path),
            "file_type": "csv",
            "n_rows": df.shape[0],
            "n_cols": df.shape[1],
        }
        
        return self.chunker.chunk(text, metadata)
    
    def _load_yaml(self, file_path: Path, source: str) -> list[Chunk]:
        """Load YAML file."""
        try:
            import yaml
        except ImportError:
            logger.warning("PyYAML not installed. Skipping YAML file.")
            return []
        
        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        
        # Convert to text
        text = self._json_to_text(data, file_path.stem)
        
        metadata = {
            "source": source,
            "file_path": str(file_path),
            "file_type": "yaml",
        }
        
        return self.chunker.chunk(text, metadata)
    
    @staticmethod
    def _json_to_text(data: Any, title: str = "") -> str:
        """Convert JSON/dict to human-readable text."""
        lines = []
        
        if title:
            lines.append(f"Document: {title}\n")
        
        def format_value(v: Any, indent: int = 0) -> str:
            """Format value with indentation."""
            prefix = "  " * indent
            
            if isinstance(v, dict):
                items = []
                for k, val in v.items():
                    items.append(f"{prefix}{k}: {format_value(val, indent + 1)}")
                return "\n".join(items)
            elif isinstance(v, list):
                if len(v) == 0:
                    return "[]"
                elif len(v) <= 5:
                    return ", ".join(str(item) for item in v)
                else:
                    return f"[{len(v)} items: {v[0]}, {v[1]}, ...]"
            else:
                return str(v)
        
        if isinstance(data, dict):
            lines.append(format_value(data))
        else:
            lines.append(json.dumps(data, indent=2))
        
        return "\n".join(lines)
    
    def load_past_runs(
        self,
        runs_dir: str,
        max_runs: int = 50,
    ) -> list[Chunk]:
        """Load past feature engineering runs.
        
        Args:
            runs_dir: Directory containing past run artifacts
            max_runs: Maximum number of runs to load
            
        Returns:
            List of Chunk objects
        """
        runs_dir = Path(runs_dir)
        
        if not runs_dir.exists():
            logger.warning(f"Runs directory not found: {runs_dir}")
            return []
        
        all_chunks = []
        run_count = 0
        
        # Look for metadata.json or metrics.json files
        for metadata_file in runs_dir.rglob("metadata.json"):
            if run_count >= max_runs:
                break
            
            try:
                with open(metadata_file, "r") as f:
                    run_data = json.load(f)
                
                # Create text summary
                text = f"Past Run: {metadata_file.parent.name}\n\n"
                text += json.dumps(run_data, indent=2)
                
                metadata = {
                    "source": "past_runs",
                    "run_id": metadata_file.parent.name,
                    "file_path": str(metadata_file),
                }
                
                chunks = self.chunker.chunk(text, metadata)
                all_chunks.extend(chunks)
                run_count += 1
                
            except Exception as e:
                logger.warning(f"Failed to load run {metadata_file}: {e}")
        
        logger.info(f"✓ Loaded {len(all_chunks)} chunks from {run_count} past runs")
        return all_chunks


def load_documents(
    directory: str,
    source: str = "documents",
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[Chunk]:
    """Load documents from directory (convenience function).
    
    Args:
        directory: Directory path
        source: Source name
        chunk_size: Target chunk size
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of Chunk objects
        
    Example:
        >>> chunks = load_documents("knowledge_base/", source="domain_knowledge")
    """
    loader = DocumentLoader(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return loader.load_directory(directory, source=source)

