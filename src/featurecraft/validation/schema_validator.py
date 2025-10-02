"""DataFrame schema validation using pandera for FeatureCraft pipelines."""

from __future__ import annotations

import json
import os
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from ..logging import get_logger
from ..utils import safe_import

logger = get_logger(__name__)

# Optional pandera import
pandera = safe_import("pandera")


class SchemaValidator(BaseEstimator, TransformerMixin):
    """Validate DataFrame schema to detect drift and type errors.
    
    This transformer learns a pandera DataFrameSchema from training data during fit()
    and validates new data during transform(). It can detect:
    - Column presence/absence
    - Type mismatches
    - Out-of-range values (for numerics)
    - Unknown categories (for categoricals)
    - Nullable violations
    
    The learned schema is serialized to disk alongside model artifacts for production use.
    
    Args:
        enabled: Whether validation is enabled (if False, acts as pass-through)
        coerce: Attempt to coerce types to match schema (True = soft validation)
        strict: Strict mode - fail on any schema violation
        schema_path: Path to save/load schema JSON (None = auto-generate in artifacts dir)
        check_column_order: Validate that column order matches training
        infer_categorical_from_object: Treat low-cardinality object columns as categorical
        max_categories_for_checks: Max unique values to store for categorical validation
        
    Attributes:
        schema_: Learned pandera DataFrameSchema (if pandera available)
        schema_dict_: Dict representation of schema for serialization
        columns_: List of expected column names
        dtypes_: Dict of column→dtype
        
    Example:
        >>> from featurecraft.validation import SchemaValidator
        >>> validator = SchemaValidator(enabled=True, coerce=True)
        >>> validator.fit(X_train)
        >>> # Validate new data
        >>> X_test_validated = validator.transform(X_test)
        >>> # Export schema
        >>> validator.export_schema("artifacts/schema.json")
        
    Notes:
        - If pandera is not installed, validator acts as pass-through with warnings
        - Schema is automatically saved during pipeline export via FeatureCraft
        - For production, always enable strict=False and coerce=True for robustness
    """
    
    def __init__(
        self,
        enabled: bool = True,
        coerce: bool = True,
        strict: bool = False,
        schema_path: Optional[str] = None,
        check_column_order: bool = False,
        infer_categorical_from_object: bool = True,
        max_categories_for_checks: int = 100,
    ) -> None:
        """Initialize schema validator."""
        self.enabled = enabled
        self.coerce = coerce
        self.strict = strict
        self.schema_path = schema_path
        self.check_column_order = check_column_order
        self.infer_categorical_from_object = infer_categorical_from_object
        self.max_categories_for_checks = max_categories_for_checks
        
        # Fitted state
        self.schema_: Optional[Any] = None  # pandera.DataFrameSchema if available
        self.schema_dict_: dict[str, Any] = {}
        self.columns_: list[str] = []
        self.dtypes_: dict[str, str] = {}
        self._pandera_available: bool = pandera is not None
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "SchemaValidator":
        """Learn schema from training data.
        
        Args:
            X: Training DataFrame
            y: Target (ignored, for sklearn compatibility)
            
        Returns:
            Self
            
        Raises:
            TypeError: If X is not a DataFrame
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"X must be a pandas DataFrame, got {type(X).__name__}")
        
        if not self.enabled:
            logger.debug("SchemaValidator disabled, skipping fit")
            return self
            
        if not self._pandera_available:
            logger.warning(
                "pandera not installed. Schema validation disabled. "
                "Install with: pip install 'featurecraft[schema]'"
            )
            self.columns_ = list(X.columns)
            self.dtypes_ = {col: str(X[col].dtype) for col in X.columns}
            return self
        
        # Learn schema from training data
        self.columns_ = list(X.columns)
        self.dtypes_ = {}
        column_schemas = {}
        
        for col in X.columns:
            col_series = X[col]
            dtype = col_series.dtype
            nullable = col_series.isna().any()
            
            # Store dtype for fallback
            self.dtypes_[col] = str(dtype)
            
            # Infer pandera Column schema
            if pd.api.types.is_numeric_dtype(dtype):
                # Numeric column - store min/max for range validation
                non_null = col_series.dropna()
                if len(non_null) > 0:
                    col_min = float(non_null.min())
                    col_max = float(non_null.max())
                    # Allow 10% padding for robustness
                    range_padding = abs(col_max - col_min) * 0.1
                    column_schemas[col] = pandera.Column(
                        dtype,
                        nullable=nullable,
                        coerce=self.coerce,
                        checks=[
                            pandera.Check.greater_than_or_equal_to(
                                col_min - range_padding,
                                ignore_na=True,
                                error=f"Column '{col}' has values below expected range"
                            ),
                            pandera.Check.less_than_or_equal_to(
                                col_max + range_padding,
                                ignore_na=True,
                                error=f"Column '{col}' has values above expected range"
                            ),
                        ]
                    )
                else:
                    column_schemas[col] = pandera.Column(dtype, nullable=True, coerce=self.coerce)
                    
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                # Datetime column
                column_schemas[col] = pandera.Column(dtype, nullable=nullable, coerce=self.coerce)
                
            else:
                # Object/string/categorical - check for category validation
                unique_count = col_series.nunique(dropna=True)
                if (self.infer_categorical_from_object and 
                    unique_count <= self.max_categories_for_checks and
                    unique_count > 0):
                    # Store allowed categories for validation
                    allowed_categories = set(col_series.dropna().unique())
                    column_schemas[col] = pandera.Column(
                        str,
                        nullable=nullable,
                        coerce=self.coerce,
                        checks=[
                            pandera.Check.isin(
                                allowed_categories,
                                ignore_na=True,
                                error=f"Column '{col}' contains unknown categories"
                            )
                        ] if not self.coerce else []  # Skip category checks if coercing
                    )
                else:
                    # High-cardinality or unknown - just check type
                    column_schemas[col] = pandera.Column(
                        str, nullable=nullable, coerce=self.coerce
                    )
        
        # Build DataFrameSchema
        self.schema_ = pandera.DataFrameSchema(
            columns=column_schemas,
            strict=self.check_column_order,
            coerce=self.coerce,
        )
        
        # Serialize schema to dict for export
        try:
            self.schema_dict_ = self.schema_.to_json()
        except Exception as e:
            logger.warning(f"Failed to serialize schema to JSON: {e}")
            self.schema_dict_ = {
                "columns": list(self.columns_),
                "dtypes": self.dtypes_,
            }
        
        logger.info(f"Learned schema for {len(self.columns_)} columns")
        return self
    
    def transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Validate DataFrame against learned schema.
        
        Args:
            X: DataFrame to validate
            y: Target (ignored, for sklearn compatibility)
            
        Returns:
            Validated (and possibly coerced) DataFrame
            
        Raises:
            ValueError: If validation fails in strict mode
            TypeError: If X is not a DataFrame
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"X must be a pandas DataFrame, got {type(X).__name__}")
        
        if not self.enabled:
            return X
        
        if not self._pandera_available:
            # Fallback validation without pandera
            return self._validate_fallback(X)
        
        if self.schema_ is None:
            raise RuntimeError("SchemaValidator not fitted. Call fit() first.")
        
        try:
            # Validate using pandera
            validated_df = self.schema_.validate(X, lazy=not self.strict)
            logger.debug(f"Schema validation passed for {len(X)} rows")
            return validated_df
            
        except Exception as e:
            error_msg = self._format_validation_error(e, X)
            
            if self.strict:
                raise ValueError(f"Schema validation failed:\n{error_msg}") from e
            else:
                logger.warning(f"Schema validation issues detected:\n{error_msg}")
                # Return original DataFrame with warning
                return X
    
    def _validate_fallback(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fallback validation without pandera (basic checks only).
        
        Args:
            X: DataFrame to validate
            
        Returns:
            Original DataFrame (with warnings if issues detected)
        """
        issues = []
        
        # Check columns
        missing_cols = set(self.columns_) - set(X.columns)
        extra_cols = set(X.columns) - set(self.columns_)
        
        if missing_cols:
            issues.append(f"Missing columns: {sorted(missing_cols)}")
        if extra_cols:
            issues.append(f"Extra columns: {sorted(extra_cols)}")
        
        # Check dtypes for common columns
        for col in set(self.columns_) & set(X.columns):
            expected_dtype = self.dtypes_.get(col, "unknown")
            actual_dtype = str(X[col].dtype)
            if expected_dtype != actual_dtype:
                issues.append(f"Column '{col}' dtype mismatch: expected {expected_dtype}, got {actual_dtype}")
        
        if issues:
            warning_msg = "Schema validation issues (pandera not available):\n" + "\n".join(f"  - {i}" for i in issues)
            if self.strict:
                raise ValueError(warning_msg)
            else:
                logger.warning(warning_msg)
        
        return X
    
    def _format_validation_error(self, error: Exception, X: pd.DataFrame) -> str:
        """Format pandera validation error into actionable message.
        
        Args:
            error: Pandera validation exception
            X: DataFrame that failed validation
            
        Returns:
            Formatted error message with suggestions
        """
        error_str = str(error)
        
        # Extract specific column issues if available
        suggestions = []
        suggestions.append("Schema validation failed. Common issues:")
        suggestions.append("  1. Column name mismatch (check spelling, case, spaces)")
        suggestions.append("  2. Type mismatch (e.g., string vs numeric)")
        suggestions.append("  3. Missing columns in new data")
        suggestions.append("  4. Unknown categories in categorical columns")
        suggestions.append("  5. Out-of-range values in numeric columns")
        suggestions.append("\nTo fix:")
        suggestions.append("  - Set schema_coerce=True to auto-convert types")
        suggestions.append("  - Set validate_schema=False to disable validation")
        suggestions.append("  - Inspect schema with validator.schema_dict_")
        
        return error_str + "\n\n" + "\n".join(suggestions)
    
    def export_schema(self, path: str) -> None:
        """Export learned schema to JSON file.
        
        Args:
            path: Output file path (*.json)
            
        Example:
            >>> validator.fit(X_train)
            >>> validator.export_schema("artifacts/schema.json")
        """
        if not self.schema_dict_:
            logger.warning("No schema to export (validator not fitted or disabled)")
            return
        
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.schema_dict_, f, indent=2, default=str)
        
        logger.info(f"Schema exported to {path}")
    
    def load_schema(self, path: str) -> "SchemaValidator":
        """Load schema from JSON file.
        
        Args:
            path: Input file path (*.json)
            
        Returns:
            Self
            
        Example:
            >>> validator = SchemaValidator()
            >>> validator.load_schema("artifacts/schema.json")
            >>> validated_df = validator.transform(X_test)
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Schema file not found: {path}")
        
        with open(path, "r", encoding="utf-8") as f:
            self.schema_dict_ = json.load(f)
        
        # Reconstruct schema from dict
        if self._pandera_available and self.schema_dict_:
            try:
                self.schema_ = pandera.DataFrameSchema.from_json(self.schema_dict_)
                self.columns_ = list(self.schema_.columns.keys())
                self.dtypes_ = {col: str(col_schema.dtype) for col, col_schema in self.schema_.columns.items()}
            except Exception as e:
                logger.warning(f"Failed to reconstruct pandera schema from JSON: {e}")
                # Fallback to basic dict
                self.columns_ = self.schema_dict_.get("columns", [])
                self.dtypes_ = self.schema_dict_.get("dtypes", {})
        
        logger.info(f"Schema loaded from {path}")
        return self
    
    def get_feature_names_out(self, input_features: Optional[list[str]] = None) -> list[str]:
        """Get output feature names (pass-through for schema validator).
        
        Args:
            input_features: Input feature names (unused, for sklearn compatibility)
            
        Returns:
            List of column names (same as input)
        """
        return self.columns_ if self.columns_ else (input_features or [])

