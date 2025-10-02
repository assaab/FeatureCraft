"""Utility functions for FeatureCraft."""

from __future__ import annotations

import base64
import importlib
import io
import warnings
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from scipy import sparse

from .leakage import LeakageGuardMixin, ensure_no_target_in_transform


def safe_import(module: str) -> Any | None:
    """Safely import a module, returning None if import fails."""
    try:
        return importlib.import_module(module)
    except Exception:
        return None


def is_numeric_series(s: pd.Series) -> bool:
    """Check if a series is numeric."""
    return pd.api.types.is_numeric_dtype(s)


def is_datetime_series(s: pd.Series) -> bool:
    """Check if a series is datetime."""
    return pd.api.types.is_datetime64_any_dtype(s) or pd.api.types.is_datetime64_dtype(s)


def is_text_candidate(s: pd.Series) -> bool:
    """Check if a series is likely text based on string length."""
    if pd.api.types.is_string_dtype(s) or s.dtype == "object":
        # Heuristic: average string length
        try:
            lengths = s.dropna().astype(str).map(len)
            return lengths.mean() >= 15
        except Exception:
            return False
    return False


def calc_cardinality(s: pd.Series) -> int:
    """Calculate cardinality (number of unique values)."""
    return int(s.nunique(dropna=True))


def calc_missing_rate(s: pd.Series) -> float:
    """Calculate missing rate."""
    n = len(s)
    if n == 0:
        return 0.0
    return float(s.isna().sum() / n)


def calc_skewness(s: pd.Series) -> float | None:
    """Calculate skewness of numeric series."""
    if not is_numeric_series(s):
        return None
    try:
        return float(s.dropna().skew())
    except Exception:
        return None


def calc_outlier_share(s: pd.Series) -> float | None:
    """Calculate share of outliers using IQR method."""
    if not is_numeric_series(s):
        return None
    x = s.dropna().astype(float)
    if x.empty:
        return 0.0
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    if iqr == 0:
        return 0.0
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = ((x < lower) | (x > upper)).sum()
    return float(outliers / len(x))


def fig_to_base64(fig: Figure) -> str:
    """Convert matplotlib figure to base64 string."""
    buf = io.BytesIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig.tight_layout()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def ensure_list(x: Any | Iterable[Any]) -> list:
    """Ensure input is a list."""
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)
    return [x]


def to_csr_matrix(data: list[dict[str, float]], n_features: int) -> sparse.csr_matrix:
    """Build a CSR from a list of {index: value} dicts, where index is in [0, n_features)."""
    indptr = [0]
    indices: list[int] = []
    values: list[float] = []
    for row in data:
        for idx, val in row.items():
            indices.append(int(idx))
            values.append(float(val))
        indptr.append(len(indices))
    return sparse.csr_matrix((values, indices, indptr), shape=(len(data), n_features), dtype=float)


def sample_df(
    df: pd.DataFrame,
    n: int | None = None,
    frac: float | None = None,
    stratify_by: str | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Sample DataFrame with optional stratification.
    
    Args:
        df: DataFrame to sample
        n: Number of samples (mutually exclusive with frac)
        frac: Fraction of samples (mutually exclusive with n)
        stratify_by: Column name for stratified sampling
        random_state: Random seed
        
    Returns:
        Sampled DataFrame
    """
    if n is None and frac is None:
        return df
    
    if n is not None and len(df) <= n:
        return df
    
    if frac is not None and frac >= 1.0:
        return df
    
    try:
        if stratify_by and stratify_by in df.columns:
            # Stratified sampling
            from sklearn.model_selection import train_test_split
            
            sample_size = n if n is not None else int(len(df) * frac)
            _, sampled = train_test_split(
                df,
                train_size=len(df) - sample_size,
                stratify=df[stratify_by],
                random_state=random_state,
            )
            return sampled
        else:
            # Random sampling
            return df.sample(n=n, frac=frac, random_state=random_state)
    except Exception as e:
        # Fallback to simple random sampling
        warnings.warn(f"Stratified sampling failed: {e}. Using random sampling.")
        return df.sample(n=n, frac=frac, random_state=random_state)


def safe_feature_names_out(transformer: Any, input_features: list[str] | None = None) -> list[str]:
    """Safely extract feature names from transformer with fallbacks.
    
    Args:
        transformer: Sklearn transformer
        input_features: Input feature names
        
    Returns:
        List of output feature names
    """
    # Try get_feature_names_out (sklearn 1.0+)
    if hasattr(transformer, "get_feature_names_out"):
        try:
            names = transformer.get_feature_names_out(input_features)
            return [str(n) for n in names]
        except Exception:
            pass
    
    # Try get_feature_names (older sklearn)
    if hasattr(transformer, "get_feature_names"):
        try:
            names = transformer.get_feature_names(input_features)
            return [str(n) for n in names]
        except Exception:
            pass
    
    # Fallback: use input features if available
    if input_features:
        return input_features
    
    # Ultimate fallback: generate generic names
    if hasattr(transformer, "n_features_out_"):
        n = transformer.n_features_out_
        return [f"feature_{i}" for i in range(n)]
    
    return ["feature_0"]


__all__ = [
    "LeakageGuardMixin",
    "ensure_no_target_in_transform",
    "safe_import",
    "is_numeric_series",
    "is_datetime_series",
    "is_text_candidate",
    "calc_cardinality",
    "calc_missing_rate",
    "calc_skewness",
    "calc_outlier_share",
    "fig_to_base64",
    "ensure_list",
    "to_csr_matrix",
    "sample_df",
    "safe_feature_names_out",
]
