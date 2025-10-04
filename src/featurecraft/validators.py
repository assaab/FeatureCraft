"""Validation utilities for FeatureCraft."""

from __future__ import annotations

import pandas as pd


def validate_input_frame(df: pd.DataFrame, target: str) -> None:
    """Validate input DataFrame and target column.
    
    Args:
        df: Input DataFrame to validate
        target: Target column name
        
    Raises:
        ValueError: If DataFrame is invalid (empty, missing target, duplicates, non-string columns)
    """
    if df.empty:
        raise ValueError("Input DataFrame is empty.")
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not found in DataFrame columns.")
    if df.columns.duplicated().any():
        raise ValueError("DataFrame contains duplicated column names.")
    
    # Validate that all column names are strings
    non_string_cols = [col for col in df.columns if not isinstance(col, str)]
    if non_string_cols:
        raise ValueError(
            f"All column names must be strings. Found non-string columns: {non_string_cols[:5]}"
            f"{' ...' if len(non_string_cols) > 5 else ''}"
        )


def leak_prone_columns(df: pd.DataFrame, target: str) -> list[str]:
    """Identify potentially leaky columns."""
    bad = []
    lower_cols = {c.lower(): c for c in df.columns}
    for key in ["target", "label", "outcome", "result"]:
        if key in lower_cols and lower_cols[key] != target:
            bad.append(lower_cols[key])
    return bad
