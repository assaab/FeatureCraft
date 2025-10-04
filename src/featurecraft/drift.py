"""Distribution drift detection utilities for FeatureCraft."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from .config import FeatureCraftConfig
from .logging import get_logger
from .utils import is_numeric_series

logger = get_logger(__name__)


def psi_categorical(reference: pd.Series, current: pd.Series) -> float:
    """Calculate Population Stability Index for categorical features.
    
    PSI = sum((actual% - expected%) * ln(actual% / expected%))
    
    Args:
        reference: Reference distribution (training data)
        current: Current distribution (new data)
        
    Returns:
        PSI value (0 = no drift, >0.25 = significant drift)
    """
    ref_clean = reference.dropna()
    curr_clean = current.dropna()
    
    # Explicit check for empty distributions - prevent division by zero and invalid PSI
    if len(ref_clean) == 0 or len(curr_clean) == 0:
        logger.warning(
            "Cannot compute PSI: one or both distributions are empty. "
            f"Reference size: {len(ref_clean)}, Current size: {len(curr_clean)}. "
            "Returning PSI = 0.0"
        )
        return 0.0
    
    # Get proportions
    ref_props = ref_clean.value_counts(normalize=True)
    curr_props = curr_clean.value_counts(normalize=True)
    
    # Combine all categories
    all_cats = set(ref_props.index) | set(curr_props.index)
    
    psi_sum = 0.0
    epsilon = 1e-10  # Small value to avoid log(0)
    
    for cat in all_cats:
        ref_prop = ref_props.get(cat, epsilon)
        curr_prop = curr_props.get(cat, epsilon)
        
        # Add small epsilon to avoid zero division
        ref_prop = max(ref_prop, epsilon)
        curr_prop = max(curr_prop, epsilon)
        
        psi_sum += (curr_prop - ref_prop) * np.log(curr_prop / ref_prop)
    
    return abs(psi_sum)


def ks_numeric(reference: pd.Series, current: pd.Series) -> float:
    """Calculate Kolmogorov-Smirnov statistic for numeric features.
    
    KS = max|CDF_ref(x) - CDF_curr(x)|
    
    Args:
        reference: Reference distribution
        current: Current distribution
        
    Returns:
        KS statistic (0 = no drift, >0.1 = significant drift)
    """
    from scipy import stats
    
    ref_clean = reference.dropna().astype(float)
    curr_clean = current.dropna().astype(float)
    
    if len(ref_clean) < 2 or len(curr_clean) < 2:
        return 0.0
    
    try:
        ks_stat, _ = stats.ks_2samp(ref_clean, curr_clean)
        return float(ks_stat)
    except Exception as e:
        logger.warning(f"KS test failed: {e}")
        return 0.0


class DriftDetector:
    """Detect distribution drift between reference and current datasets."""
    
    def __init__(self, cfg: FeatureCraftConfig):
        """Initialize drift detector.
        
        Args:
            cfg: FeatureCraft configuration with drift thresholds
        """
        self.cfg = cfg
    
    def detect(
        self, reference_df: pd.DataFrame, current_df: pd.DataFrame
    ) -> Dict[str, Tuple[float, str]]:
        """Detect drift for all columns.
        
        Args:
            reference_df: Reference dataset (training data)
            current_df: Current dataset (new data)
            
        Returns:
            Dict mapping column name to (drift_score, severity)
            Severity: "OK", "WARN", "CRITICAL"
        """
        drift_results = {}
        
        for col in reference_df.columns:
            if col not in current_df.columns:
                logger.debug(f"Column '{col}' not in current data, skipping drift check")
                continue
            
            ref_series = reference_df[col]
            curr_series = current_df[col]
            
            # Skip if both are completely null
            if ref_series.isna().all() or curr_series.isna().all():
                continue
            
            # Detect drift based on column type
            if is_numeric_series(ref_series) and is_numeric_series(curr_series):
                drift_score = ks_numeric(ref_series, curr_series)
                severity = self._assess_severity_ks(drift_score)
            else:
                drift_score = psi_categorical(
                    ref_series.astype(str), curr_series.astype(str)
                )
                severity = self._assess_severity_psi(drift_score)
            
            drift_results[col] = (drift_score, severity)
        
        return drift_results
    
    def _assess_severity_psi(self, psi: float) -> str:
        """Assess drift severity for PSI.
        
        PSI interpretation:
        - < 0.1: No significant drift
        - 0.1 - 0.25: Moderate drift
        - > 0.25: Significant drift
        """
        threshold = self.cfg.drift_psi_threshold
        
        if psi < 0.1:
            return "OK"
        elif psi < threshold:
            return "WARN"
        else:
            return "CRITICAL"
    
    def _assess_severity_ks(self, ks: float) -> str:
        """Assess drift severity for KS statistic.
        
        KS interpretation:
        - < 0.05: No significant drift
        - 0.05 - 0.1: Moderate drift
        - > 0.1: Significant drift
        """
        threshold = self.cfg.drift_ks_threshold
        
        if ks < 0.05:
            return "OK"
        elif ks < threshold:
            return "WARN"
        else:
            return "CRITICAL"


def summarize_drift_report(
    drift_results: Dict[str, Tuple[float, str]]
) -> Dict[str, Any]:
    """Summarize drift detection results.
    
    Args:
        drift_results: Results from DriftDetector.detect()
        
    Returns:
        Summary dict with counts and critical columns
    """
    summary = {
        "total_columns": len(drift_results),
        "ok_count": 0,
        "warn_count": 0,
        "critical_count": 0,
        "critical_columns": [],
        "warn_columns": [],
    }
    
    for col, (score, severity) in drift_results.items():
        if severity == "OK":
            summary["ok_count"] += 1
        elif severity == "WARN":
            summary["warn_count"] += 1
            summary["warn_columns"].append((col, score))
        elif severity == "CRITICAL":
            summary["critical_count"] += 1
            summary["critical_columns"].append((col, score))
    
    return summary

