"""SHAP explainability utilities for FeatureCraft."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from .config import FeatureCraftConfig
from .logging import get_logger

logger = get_logger(__name__)

# Try importing SHAP
try:
    import shap

    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    logger.debug("SHAP not installed. Explainability features disabled.")


def safe_shap_summary(
    estimator: BaseEstimator,
    X: pd.DataFrame,
    feature_names: list[str],
    max_samples: int = 100,
    cfg: Optional[FeatureCraftConfig] = None,
) -> Optional[Tuple[np.ndarray, str]]:
    """Generate SHAP summary with safe fallbacks.
    
    Args:
        estimator: Fitted sklearn estimator
        X: Feature matrix (post-transformation)
        feature_names: Names of features
        max_samples: Maximum samples for SHAP computation
        cfg: FeatureCraft configuration
        
    Returns:
        Tuple of (shap_values, base64_plot) if successful, None otherwise
    """
    if not HAS_SHAP:
        logger.warning(
            "SHAP requested but not installed. "
            "Install with: pip install shap"
        )
        return None
    
    if cfg is None:
        from .config import FeatureCraftConfig
        cfg = FeatureCraftConfig()
    
    # Sample data if too large
    if len(X) > max_samples:
        logger.info(f"Sampling {max_samples} rows for SHAP computation")
        rng = np.random.default_rng(cfg.random_state)
        sample_idx = rng.choice(len(X), size=max_samples, replace=False)
        X_sample = X.iloc[sample_idx] if isinstance(X, pd.DataFrame) else X[sample_idx]
    else:
        X_sample = X
    
    try:
        # Detect model type and create appropriate explainer
        model_type = type(estimator).__name__.lower()
        
        if any(keyword in model_type for keyword in ["tree", "forest", "boost", "xgb", "lgb"]):
            explainer = shap.TreeExplainer(estimator)
            logger.info("Using TreeExplainer for SHAP values")
        elif any(keyword in model_type for keyword in ["linear", "logistic", "ridge", "lasso"]):
            explainer = shap.LinearExplainer(estimator, X_sample)
            logger.info("Using LinearExplainer for SHAP values")
        else:
            # Fallback to KernelExplainer (slower)
            logger.info("Using KernelExplainer for SHAP values (may be slow)")
            background = shap.sample(X_sample, min(100, len(X_sample)), random_state=cfg.random_state)
            explainer = shap.KernelExplainer(estimator.predict, background)
        
        # Compute SHAP values
        shap_values = explainer.shap_values(X_sample)
        
        # Handle multi-output case (classification with >2 classes)
        if isinstance(shap_values, list):
            # Use first class for summary
            shap_values = shap_values[0]
        
        # Generate summary plot
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create summary plot
        if len(feature_names) <= 20:
            shap.summary_plot(
                shap_values,
                X_sample,
                feature_names=feature_names,
                show=False,
                max_display=20,
            )
        else:
            # Show top 20 by mean absolute SHAP
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            top_idx = np.argsort(mean_abs_shap)[-20:]
            
            shap.summary_plot(
                shap_values[:, top_idx],
                X_sample[:, top_idx] if isinstance(X_sample, np.ndarray) else X_sample.iloc[:, top_idx],
                feature_names=[feature_names[i] for i in top_idx],
                show=False,
                max_display=20,
            )
        
        # Convert to base64
        from .utils import fig_to_base64
        b64 = fig_to_base64(plt.gcf())
        plt.close()
        
        logger.info("SHAP summary generated successfully")
        return shap_values, b64
        
    except Exception as e:
        logger.warning(f"SHAP computation failed: {e}")
        return None


def get_feature_importance_from_shap(
    shap_values: np.ndarray, feature_names: list[str], top_n: int = 20
) -> pd.DataFrame:
    """Extract feature importance from SHAP values.
    
    Args:
        shap_values: SHAP values array
        feature_names: Feature names
        top_n: Number of top features to return
        
    Returns:
        DataFrame with feature importance rankings
    """
    # Calculate mean absolute SHAP value for each feature
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": mean_abs_shap,
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values("importance", ascending=False)
    
    # Return top N
    return importance_df.head(top_n).reset_index(drop=True)

