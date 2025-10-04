"""Custom exception hierarchy for FeatureCraft.

This module provides structured exceptions with error codes for better
debugging and user support.
"""

from typing import Any, Dict, Optional


class FeatureCraftError(Exception):
    """Base exception for all FeatureCraft errors.
    
    Attributes:
        message: Human-readable error message
        code: Error code for documentation/support (e.g., "FC_001")
        context: Additional context dictionary for debugging
    """
    
    def __init__(
        self,
        message: str,
        code: str = "FC_UNKNOWN",
        **context: Any
    ):
        super().__init__(message)
        self.message = message
        self.code = code
        self.context = context
    
    def __str__(self) -> str:
        ctx_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
        if ctx_str:
            return f"[{self.code}] {self.message} ({ctx_str})"
        return f"[{self.code}] {self.message}"


class InputValidationError(FeatureCraftError):
    """Raised when input data is invalid or malformed.
    
    Examples:
        - Missing required columns
        - Incompatible data types
        - Invalid value ranges
    """
    
    def __init__(self, message: str, **context: Any):
        super().__init__(message, code="FC_INPUT_INVALID", **context)


class PipelineNotFittedError(FeatureCraftError):
    """Raised when attempting operations on an unfitted pipeline.
    
    Examples:
        - Calling transform() before fit()
        - Calling export() before fit()
    """
    
    def __init__(self, message: str, **context: Any):
        super().__init__(message, code="FC_NOT_FITTED", **context)


class ConfigurationError(FeatureCraftError):
    """Raised when configuration is invalid or inconsistent.
    
    Examples:
        - Invalid parameter combinations
        - Missing required configuration
        - Type mismatches in config
    """
    
    def __init__(self, message: str, **context: Any):
        super().__init__(message, code="FC_CONFIG_INVALID", **context)


class SchemaValidationError(FeatureCraftError):
    """Raised when data schema validation fails.
    
    Examples:
        - Column mismatch between train and test
        - Type mismatch in transform
        - Missing features in production data
    """
    
    def __init__(self, message: str, **context: Any):
        super().__init__(message, code="FC_SCHEMA_MISMATCH", **context)


class SecurityError(FeatureCraftError):
    """Raised when security violations are detected.
    
    Examples:
        - Path traversal attempts
        - Checksum mismatches in loaded pipelines
        - Untrusted file operations
    """
    
    def __init__(self, message: str, **context: Any):
        super().__init__(message, code="FC_SECURITY", **context)


class FeatureEngineeringError(FeatureCraftError):
    """Raised when feature engineering operations fail.
    
    Examples:
        - Encoding failures
        - Scaling errors
        - Transformation exceptions
    """
    
    def __init__(self, message: str, **context: Any):
        super().__init__(message, code="FC_TRANSFORM_FAILED", **context)


class DataQualityError(FeatureCraftError):
    """Raised when data quality issues prevent processing.
    
    Examples:
        - Too many missing values
        - Constant features
        - Invalid target distribution
    """
    
    def __init__(self, message: str, **context: Any):
        super().__init__(message, code="FC_DATA_QUALITY", **context)


class ExportError(FeatureCraftError):
    """Raised when pipeline export/serialization fails.
    
    Examples:
        - File write permissions
        - Serialization errors
        - Missing metadata
    """
    
    def __init__(self, message: str, **context: Any):
        super().__init__(message, code="FC_EXPORT_FAILED", **context)

