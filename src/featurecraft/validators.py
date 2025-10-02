"""Validation utilities for FeatureCraft."""

from __future__ import annotations

import pandas as pd


def validate_input_frame(df: pd.DataFrame, target: str) -> None:
    """Validate input DataFrame and target column."""
    if df.empty:
        raise ValueError("Input DataFrame is empty.")
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not found in DataFrame columns.")
    if df.columns.duplicated().any():
        raise ValueError("DataFrame contains duplicated column names.")


def leak_prone_columns(df: pd.DataFrame, target: str) -> list[str]:
    """Identify potentially leaky columns."""
    bad = []
    lower_cols = {c.lower(): c for c in df.columns}
    for key in ["target", "label", "outcome", "result"]:
        if key in lower_cols and lower_cols[key] != target:
            bad.append(lower_cols[key])
    return bad
