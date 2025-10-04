"""Secure input/output utilities for FeatureCraft pipelines.

This module provides safe loading/saving functions with security checks.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Optional

import joblib

from .exceptions import SecurityError, ExportError
from .logging import get_logger

logger = get_logger(__name__)


def load_pipeline(
    pipeline_path: str,
    verify_checksum: bool = True,
    metadata_path: Optional[str] = None
) -> Any:
    """Load a FeatureCraft pipeline with optional checksum verification.
    
    Args:
        pipeline_path: Path to pipeline.joblib file
        verify_checksum: If True, verify SHA256 checksum against metadata
        metadata_path: Optional path to metadata.json (default: same dir as pipeline)
        
    Returns:
        Loaded sklearn Pipeline object
        
    Raises:
        SecurityError: If checksum verification fails
        FileNotFoundError: If pipeline or metadata file not found
        
    Security Warning:
        Pipelines use pickle serialization. Only load from trusted sources.
        Checksum verification helps detect tampering but cannot prevent
        malicious code execution if the original file was malicious.
        
    Example:
        >>> pipeline = load_pipeline("artifacts/pipeline.joblib")
        >>> # Or skip checksum verification (not recommended):
        >>> pipeline = load_pipeline("artifacts/pipeline.joblib", verify_checksum=False)
    """
    pipeline_path_obj = Path(pipeline_path)
    
    if not pipeline_path_obj.exists():
        raise FileNotFoundError(f"Pipeline file not found: {pipeline_path}")
    
    # Read pipeline file
    with open(pipeline_path_obj, "rb") as f:
        pipeline_bytes = f.read()
    
    # Verify checksum if requested
    if verify_checksum:
        _verify_pipeline_checksum(
            pipeline_bytes,
            pipeline_path_obj,
            metadata_path
        )
    else:
        logger.warning(
            "Loading pipeline without checksum verification. "
            "Only load from trusted sources!"
        )
    
    # Load pipeline
    try:
        pipeline = joblib.loads(pipeline_bytes)
        logger.info(f"Successfully loaded pipeline from {pipeline_path}")
        return pipeline
    except Exception as e:
        raise ExportError(
            f"Failed to deserialize pipeline: {e}",
            pipeline_path=str(pipeline_path)
        ) from e


def _verify_pipeline_checksum(
    pipeline_bytes: bytes,
    pipeline_path: Path,
    metadata_path: Optional[str] = None
) -> None:
    """Verify pipeline checksum against metadata.
    
    Args:
        pipeline_bytes: Raw pipeline file bytes
        pipeline_path: Path to pipeline file
        metadata_path: Optional metadata path
        
    Raises:
        SecurityError: If checksum doesn't match or metadata missing
    """
    # Compute actual checksum
    actual_checksum = hashlib.sha256(pipeline_bytes).hexdigest()
    
    # Try to load expected checksum from metadata.json
    if metadata_path is None:
        metadata_path = pipeline_path.parent / "metadata.json"
    else:
        metadata_path = Path(metadata_path)
    
    if not metadata_path.exists():
        # Try .sha256 file as fallback
        sha256_path = pipeline_path.with_suffix(".sha256")
        if sha256_path.exists():
            with open(sha256_path, encoding="utf-8") as f:
                checksum_line = f.read().strip()
                # Format: "checksum  filename"
                expected_checksum = checksum_line.split()[0]
        else:
            raise SecurityError(
                "Cannot verify checksum: no metadata.json or .sha256 file found. "
                "Set verify_checksum=False to skip (not recommended).",
                pipeline_path=str(pipeline_path),
                metadata_path=str(metadata_path)
            )
    else:
        with open(metadata_path, encoding="utf-8") as f:
            metadata = json.load(f)
        
        expected_checksum = metadata.get("pipeline_checksum_sha256")
        if not expected_checksum:
            raise SecurityError(
                "Metadata exists but missing 'pipeline_checksum_sha256' field. "
                "This pipeline may have been exported with an older version.",
                metadata_path=str(metadata_path)
            )
    
    # Compare checksums
    if actual_checksum != expected_checksum:
        raise SecurityError(
            "Pipeline checksum mismatch! File may be corrupted or tampered with. "
            "Do NOT load this pipeline unless you trust the source.",
            expected=expected_checksum,
            actual=actual_checksum,
            pipeline_path=str(pipeline_path)
        )
    
    logger.info(f"✓ Pipeline checksum verified: {actual_checksum[:16]}...")


def load_pipeline_metadata(metadata_path: str) -> dict:
    """Load pipeline metadata from JSON file.
    
    Args:
        metadata_path: Path to metadata.json
        
    Returns:
        Metadata dictionary
        
    Raises:
        FileNotFoundError: If metadata file doesn't exist
    """
    metadata_path_obj = Path(metadata_path)
    
    if not metadata_path_obj.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    with open(metadata_path_obj, encoding="utf-8") as f:
        return json.load(f)

