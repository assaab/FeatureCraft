"""Imputation utilities for FeatureCraft."""

from __future__ import annotations

# Enable experimental IterativeImputer (required before importing IterativeImputer)
# This import has side effects that enable the imputer; must be imported before use
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer

from .config import FeatureCraftConfig


def choose_numeric_imputer(
    missing_rate: float, n_features: int, n_rows: int, cfg: FeatureCraftConfig
) -> SimpleImputer | KNNImputer | IterativeImputer:
    """Choose appropriate numeric imputer based on missing rate and data size."""
    if missing_rate <= cfg.numeric_simple_impute_max:
        return SimpleImputer(strategy="median", add_indicator=True)
    if missing_rate <= cfg.numeric_advanced_impute_max:
        if n_features <= 100 and n_rows <= 200_000:
            return KNNImputer(n_neighbors=5)
        return IterativeImputer(max_iter=10, sample_posterior=False, random_state=cfg.random_state)
    return SimpleImputer(strategy="median", add_indicator=True)


def categorical_imputer(cfg: FeatureCraftConfig | None = None) -> SimpleImputer:
    """Get categorical imputer.
    
    Args:
        cfg: Configuration (optional, uses default if None)
        
    Returns:
        Configured SimpleImputer for categorical data
    """
    if cfg is None:
        cfg = FeatureCraftConfig()
    return SimpleImputer(strategy=cfg.categorical_impute_strategy)
