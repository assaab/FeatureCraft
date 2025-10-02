"""Class imbalance handling utilities for FeatureCraft."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from .logging import get_logger

logger = get_logger(__name__)

# Try importing imbalanced-learn
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline

    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False
    logger.debug("imbalanced-learn not installed. SMOTE features disabled.")


def detect_minority_ratio(y: pd.Series) -> float:
    """Detect minority class ratio in target variable.
    
    Args:
        y: Target variable series
        
    Returns:
        Ratio of minority class (0.0 to 1.0)
    """
    if y.isna().any():
        y = y.dropna()
    
    vc = y.value_counts()
    if len(vc) == 0:
        return 1.0
    
    minority_count = vc.min()
    total_count = vc.sum()
    
    return float(minority_count / total_count) if total_count > 0 else 1.0


def auto_resampler(
    y: pd.Series,
    threshold: float = 0.10,
    enabled: bool = True,
    random_state: int = 42,
    k_neighbors: int = 5,
    sampling_strategy: str = "auto",
) -> Optional[SMOTE]:
    """Create SMOTE resampler if minority class is below threshold.
    
    Args:
        y: Target variable
        threshold: Minority ratio threshold below which to apply SMOTE
        enabled: Whether SMOTE is enabled
        random_state: Random seed
        k_neighbors: Number of neighbors for SMOTE
        sampling_strategy: SMOTE sampling strategy
        
    Returns:
        SMOTE instance if applicable, None otherwise
    """
    if not enabled:
        logger.debug("SMOTE disabled via config")
        return None
    
    if not HAS_IMBLEARN:
        logger.warning(
            "SMOTE requested but imbalanced-learn not installed. "
            "Install with: pip install imbalanced-learn"
        )
        return None
    
    minority_ratio = detect_minority_ratio(y)
    
    if minority_ratio >= threshold:
        logger.debug(
            f"Minority ratio {minority_ratio:.2%} >= threshold {threshold:.2%}. "
            "SMOTE not needed."
        )
        return None
    
    logger.info(
        f"Minority ratio {minority_ratio:.2%} < threshold {threshold:.2%}. "
        "Creating SMOTE resampler."
    )
    
    # Adjust k_neighbors if needed
    min_class_count = int(y.value_counts().min())
    actual_k = min(k_neighbors, min_class_count - 1)
    
    if actual_k < 1:
        logger.warning(
            f"Minority class has only {min_class_count} samples. "
            "SMOTE requires at least 2 samples per class."
        )
        return None
    
    if actual_k != k_neighbors:
        logger.info(f"Adjusting SMOTE k_neighbors from {k_neighbors} to {actual_k}")
    
    try:
        smote = SMOTE(
            random_state=random_state,
            k_neighbors=actual_k,
            sampling_strategy=sampling_strategy,
        )
        return smote
    except Exception as e:
        logger.warning(f"Failed to create SMOTE resampler: {e}")
        return None


def smote_inside_cv(
    estimator: BaseEstimator,
    random_state: int = 42,
    k_neighbors: int = 5,
    sampling_strategy: str = "auto",
    enabled: bool = True,
) -> BaseEstimator:
    """Wrap estimator with SMOTE for use inside CV folds.
    
    This creates an imbalanced-learn Pipeline that applies SMOTE during fit
    but not during transform/predict, ensuring resampling happens within each CV fold.
    
    Args:
        estimator: Sklearn estimator to wrap
        random_state: Random seed
        k_neighbors: Number of neighbors for SMOTE
        sampling_strategy: SMOTE sampling strategy
        enabled: Whether SMOTE is enabled
        
    Returns:
        Pipeline with SMOTE + estimator if enabled, otherwise original estimator
    """
    if not enabled:
        return estimator
    
    if not HAS_IMBLEARN:
        logger.warning(
            "SMOTE requested but imbalanced-learn not installed. "
            "Using estimator without resampling."
        )
        return estimator
    
    try:
        smote = SMOTE(
            random_state=random_state,
            k_neighbors=k_neighbors,
            sampling_strategy=sampling_strategy,
        )
        
        # Use imblearn Pipeline which properly handles resampling
        pipeline = ImbPipeline(steps=[("smote", smote), ("estimator", estimator)])
        
        logger.info("Created SMOTE pipeline for CV-safe resampling")
        return pipeline
        
    except Exception as e:
        logger.warning(f"Failed to create SMOTE pipeline: {e}. Using original estimator.")
        return estimator


def get_class_weight_advisory(y: pd.Series, threshold: float = 0.20) -> Optional[str]:
    """Get class_weight advisory for imbalanced classification.
    
    Args:
        y: Target variable
        threshold: Minority ratio threshold
        
    Returns:
        Advisory string if imbalance detected, None otherwise
    """
    minority_ratio = detect_minority_ratio(y)
    
    if minority_ratio < threshold:
        return (
            f"Class imbalance detected (minority ratio: {minority_ratio:.2%}). "
            f"Consider using class_weight='balanced' in your estimator or enable SMOTE."
        )
    
    return None

