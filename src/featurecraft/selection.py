"""Feature selection utilities for FeatureCraft."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression

from .logging import get_logger

logger = get_logger(__name__)


def prune_correlated(df: pd.DataFrame, threshold: float = 0.95) -> list[str]:
    """Prune highly correlated features."""
    corr = df.corr(numeric_only=True).abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return to_drop


def compute_vif_drop(df: pd.DataFrame, threshold: float = 10.0) -> list[str]:
    """Iteratively drop features with VIF > threshold using sklearn LinearRegression (no intercept)."""
    cols = list(df.columns)
    dropped: list[str] = []
    while True:
        vmax = 0.0
        worst = None
        for _i, c in enumerate(cols):
            X = df[cols].drop(columns=[c]).fillna(0.0).values
            y = df[c].fillna(0.0).values
            if X.shape[1] == 0:
                continue
            reg = LinearRegression()
            reg.fit(X, y)
            r2 = reg.score(X, y)
            vif = 1.0 / max(1e-6, (1.0 - r2))
            if vif > vmax:
                vmax, worst = vif, c
        if vmax > threshold and worst is not None:
            dropped.append(worst)
            cols.remove(worst)
        else:
            break
    return dropped


class WOEIVSelector(BaseEstimator, TransformerMixin):
    """Weight of Evidence / Information Value based feature selector for binary classification.
    
    This selector uses Information Value (IV) scores from WoE encoding to rank and select features.
    Features with IV below the threshold are dropped.
    
    IV Interpretation:
    - < 0.02: Not predictive
    - 0.02 - 0.1: Weak predictive power
    - 0.1 - 0.3: Medium predictive power
    - 0.3 - 0.5: Strong predictive power
    - > 0.5: Suspicious (check for leakage)
    
    Args:
        threshold: Minimum IV threshold (features below this are dropped)
        smoothing: Smoothing factor for WoE computation
        random_state: Random seed
        
    Attributes:
        iv_scores_: Dict mapping feature name to IV score
        selected_features_: List of selected feature names
        dropped_features_: List of dropped feature names
        
    Example:
        >>> from featurecraft.selection import WOEIVSelector
        >>> selector = WOEIVSelector(threshold=0.02)
        >>> selector.fit(X_train, y_train)
        >>> X_train_selected = selector.transform(X_train)
        >>> print(f"Kept {len(selector.selected_features_)} / {X_train.shape[1]} features")
        >>> print(f"IV scores: {selector.iv_scores_}")
        
    Notes:
        - Only works for binary classification tasks
        - Requires categorical features (converts to string internally)
        - Uses WoEEncoder internally to compute IV scores
    """
    
    def __init__(
        self,
        threshold: float = 0.02,
        smoothing: float = 0.5,
        random_state: int = 42,
    ) -> None:
        """Initialize WoE/IV selector."""
        self.threshold = float(threshold)
        self.smoothing = float(smoothing)
        self.random_state = int(random_state)
        
        # Fitted state
        self.iv_scores_: dict[str, float] = {}
        self.selected_features_: list[str] = []
        self.dropped_features_: list[str] = []
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "WOEIVSelector":
        """Fit selector by computing IV scores for all features.
        
        Args:
            X: Training features
            y: Training target (must be binary)
            
        Returns:
            Self
            
        Raises:
            ValueError: If y is not binary or if X is not a DataFrame
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"X must be a pandas DataFrame, got {type(X).__name__}")
        if not isinstance(y, pd.Series):
            raise TypeError(f"y must be a pandas Series, got {type(y).__name__}")
        
        # Check binary target
        nunique = y.nunique()
        if nunique != 2:
            raise ValueError(
                f"WOEIVSelector requires binary target (2 unique values), got {nunique}. "
                "This selector is only applicable for binary classification."
            )
        
        # Compute IV scores using WoEEncoder
        from .encoders import WoEEncoder
        
        encoder = WoEEncoder(
            cols=list(X.columns),
            smoothing=self.smoothing,
            random_state=self.random_state,
        )
        encoder.fit(X, y)
        self.iv_scores_ = encoder.get_iv_scores()
        
        # Select features based on threshold
        self.selected_features_ = [
            col for col, iv in self.iv_scores_.items() if iv >= self.threshold
        ]
        self.dropped_features_ = [
            col for col, iv in self.iv_scores_.items() if iv < self.threshold
        ]
        
        if not self.selected_features_:
            logger.warning(
                f"WOEIVSelector: No features passed IV threshold {self.threshold}. "
                "Keeping all features to avoid empty output."
            )
            self.selected_features_ = list(X.columns)
            self.dropped_features_ = []
        
        logger.info(
            f"WOEIVSelector: Kept {len(self.selected_features_)} / {len(X.columns)} features "
            f"with IV >= {self.threshold}"
        )
        if self.dropped_features_:
            logger.debug(f"Dropped features (low IV): {self.dropped_features_}")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform by selecting features with sufficient IV.
        
        Args:
            X: Features to transform
            
        Returns:
            DataFrame with selected features only
        """
        if not self.selected_features_:
            raise RuntimeError("WOEIVSelector not fitted. Call fit() first.")
        
        return X[self.selected_features_].copy()
    
    def get_feature_names_out(self, input_features: Optional[list[str]] = None) -> list[str]:
        """Get output feature names.
        
        Args:
            input_features: Input feature names (unused, returns selected features)
            
        Returns:
            List of selected feature names
        """
        return self.selected_features_
    
    def get_support(self, indices: bool = False):
        """Get mask or indices of selected features (sklearn compatibility).
        
        Args:
            indices: If True, return indices; if False, return boolean mask
            
        Returns:
            Boolean mask or integer indices of selected features
        """
        if not self.iv_scores_:
            raise RuntimeError("WOEIVSelector not fitted. Call fit() first.")
        
        all_features = list(self.iv_scores_.keys())
        mask = [feat in self.selected_features_ for feat in all_features]
        
        if indices:
            return np.where(mask)[0]
        else:
            return np.array(mask)
