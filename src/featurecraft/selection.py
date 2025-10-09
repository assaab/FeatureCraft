"""Feature selection utilities for FeatureCraft.

This module provides comprehensive feature selection methods including:
- Filter Methods: Correlation, Mutual Information, Chi², VIF
- Wrapper Methods: RFE, Forward/Backward Selection
- Embedded Methods: Lasso (L1), Tree-based importance
- Advanced Methods: Boruta algorithm

These selectors help reduce dimensionality and improve model performance
by identifying the most predictive features for a given task.
"""

from __future__ import annotations

from typing import List, Literal, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso, LassoCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import (
    chi2,
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression,
    RFE,
)
from sklearn.metrics import get_scorer

from .logging import get_logger
from .types import TaskType

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


class MutualInfoSelector(BaseEstimator, TransformerMixin):
    """Mutual Information based feature selector for classification and regression.
    
    This selector uses mutual information scores to rank features and select top K.
    Mutual information measures the dependency between features and target, capturing
    both linear and non-linear relationships.
    
    Args:
        k: Number of top features to select (default: 50)
        task: Task type ('classification' or 'regression')
        random_state: Random seed
        
    Attributes:
        mi_scores_: Dict mapping feature name to MI score
        selected_features_: List of selected feature names (top K)
        dropped_features_: List of dropped feature names
        
    Example:
        >>> from featurecraft.selection import MutualInfoSelector
        >>> selector = MutualInfoSelector(k=30, task='classification')
        >>> selector.fit(X_train, y_train)
        >>> X_train_selected = selector.transform(X_train)
        >>> print(f"Kept {len(selector.selected_features_)} / {X_train.shape[1]} features")
        >>> print(f"Top 5 MI scores: {sorted(selector.mi_scores_.items(), key=lambda x: -x[1])[:5]}")
        
    Notes:
        - Works for both classification and regression
        - Captures non-linear relationships (unlike correlation)
        - Computationally expensive for high-dimensional data
        - Automatically handles discrete and continuous features
    """
    
    def __init__(
        self,
        k: int = 50,
        task: str = "classification",
        random_state: int = 42,
    ) -> None:
        """Initialize Mutual Information selector."""
        self.k = int(k)
        self.task = task
        self.random_state = int(random_state)
        
        # Fitted state
        self.mi_scores_: dict[str, float] = {}
        self.selected_features_: list[str] = []
        self.dropped_features_: list[str] = []
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MutualInfoSelector":
        """Fit selector by computing MI scores for all features.
        
        Args:
            X: Training features
            y: Training target
            
        Returns:
            Self
            
        Raises:
            ValueError: If X is not a DataFrame or y is not a Series
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"X must be a pandas DataFrame, got {type(X).__name__}")
        if not isinstance(y, pd.Series):
            raise TypeError(f"y must be a pandas Series, got {type(y).__name__}")
        
        # Handle empty or single-feature case
        if X.shape[1] == 0:
            logger.warning("MutualInfoSelector: Empty feature set, nothing to select")
            self.selected_features_ = []
            self.dropped_features_ = []
            self.mi_scores_ = {}
            return self
        
        if X.shape[1] <= self.k:
            logger.info(
                f"MutualInfoSelector: Feature count ({X.shape[1]}) <= k ({self.k}), "
                "keeping all features"
            )
            self.selected_features_ = list(X.columns)
            self.dropped_features_ = []
            self.mi_scores_ = {col: 1.0 for col in X.columns}
            return self
        
        # Convert to numpy and handle missing values
        X_array = X.fillna(0).values
        y_array = y.values
        
        # Compute mutual information scores
        try:
            if self.task.lower() in {"classification", "binary", "multiclass"}:
                mi_scores = mutual_info_classif(
                    X_array, 
                    y_array,
                    random_state=self.random_state,
                    n_neighbors=min(3, len(X) - 1)  # Prevent errors with small datasets
                )
            else:  # regression
                mi_scores = mutual_info_regression(
                    X_array, 
                    y_array,
                    random_state=self.random_state,
                    n_neighbors=min(3, len(X) - 1)
                )
        except Exception as e:
            logger.error(f"MutualInfoSelector: MI computation failed: {e}. Keeping all features.")
            self.selected_features_ = list(X.columns)
            self.dropped_features_ = []
            self.mi_scores_ = {col: 0.0 for col in X.columns}
            return self
        
        # Map scores to feature names
        self.mi_scores_ = {col: float(score) for col, score in zip(X.columns, mi_scores)}
        
        # Select top K features by MI score
        sorted_features = sorted(
            self.mi_scores_.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        self.selected_features_ = [feat for feat, _ in sorted_features[:self.k]]
        self.dropped_features_ = [feat for feat, _ in sorted_features[self.k:]]
        
        logger.info(
            f"MutualInfoSelector: Selected top {len(self.selected_features_)} / {len(X.columns)} features "
            f"by mutual information"
        )
        if self.dropped_features_:
            logger.debug(f"Dropped {len(self.dropped_features_)} features with low MI scores")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform by selecting top K features.
        
        Args:
            X: Features to transform
            
        Returns:
            DataFrame with selected features only
        """
        if not hasattr(self, 'mi_scores_'):
            raise RuntimeError("MutualInfoSelector not fitted. Call fit() first.")
        
        if not self.selected_features_:
            logger.warning("No features selected, returning empty DataFrame")
            return pd.DataFrame(index=X.index)
        
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
        if not self.mi_scores_:
            raise RuntimeError("MutualInfoSelector not fitted. Call fit() first.")
        
        all_features = list(self.mi_scores_.keys())
        mask = [feat in self.selected_features_ for feat in all_features]
        
        if indices:
            return np.where(mask)[0]
        else:
            return np.array(mask)


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


class Chi2Selector(BaseEstimator, TransformerMixin):
    """Chi-squared (χ²) statistical test for feature selection.
    
    Uses chi-squared test to measure dependency between each non-negative feature
    and the target class. Higher χ² scores indicate stronger association.
    
    Note: All features must be non-negative. Apply transformations if needed.
    
    Parameters
    ----------
    k : int or 'all', optional
        Number of top features to select. If 'all', selects all features. Default: 50
    alpha : float, optional
        If provided, select features with p-value < alpha. Overrides k. Default: None
    
    Attributes
    ----------
    chi2_scores_ : dict
        Chi-squared scores for each feature
    p_values_ : dict
        P-values for each feature
    selected_features_ : list
        Names of selected features
    dropped_features_ : list
        Names of dropped features
        
    Examples
    --------
    >>> from featurecraft.selection import Chi2Selector
    >>> selector = Chi2Selector(k=30)
    >>> selector.fit(X_train, y_train)
    >>> X_selected = selector.transform(X_train)
    >>> print(f"Top features: {selector.selected_features_[:5]}")
    
    Notes
    -----
    - Only works for classification tasks
    - Features must be non-negative (use MinMaxScaler if needed)
    - Fast and effective for categorical/count features
    """
    
    def __init__(
        self,
        k: Union[int, str] = 50,
        alpha: Optional[float] = None,
    ):
        self.k = k
        self.alpha = alpha
        self.chi2_scores_: dict = {}
        self.p_values_: dict = {}
        self.selected_features_: list = []
        self.dropped_features_: list = []
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fit chi-squared selector.
        
        Parameters
        ----------
        X : pd.DataFrame
            Training features (must be non-negative)
        y : pd.Series
            Training target (categorical)
            
        Returns
        -------
        self
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"X must be DataFrame, got {type(X).__name__}")
        if not isinstance(y, pd.Series):
            raise TypeError(f"y must be Series, got {type(y).__name__}")
        
        # Ensure non-negative features
        X_array = X.fillna(0).values
        if np.any(X_array < 0):
            logger.warning(
                "Chi2Selector: Negative values detected. Chi² requires non-negative features. "
                "Setting negative values to 0."
            )
            X_array = np.clip(X_array, 0, None)
        
        y_array = y.values
        
        # Compute chi-squared scores
        try:
            scores, pvals = chi2(X_array, y_array)
        except Exception as e:
            logger.error(f"Chi2Selector: Failed to compute χ² scores: {e}")
            self.selected_features_ = list(X.columns)
            self.dropped_features_ = []
            return self
        
        # Store scores
        self.chi2_scores_ = {col: float(score) for col, score in zip(X.columns, scores)}
        self.p_values_ = {col: float(pval) for col, pval in zip(X.columns, pvals)}
        
        # Select features
        if self.alpha is not None:
            # Select by p-value threshold
            self.selected_features_ = [
                col for col, pval in self.p_values_.items() if pval < self.alpha
            ]
            self.dropped_features_ = [
                col for col, pval in self.p_values_.items() if pval >= self.alpha
            ]
        else:
            # Select top k by score
            k_val = len(X.columns) if self.k == 'all' else min(int(self.k), len(X.columns))
            sorted_features = sorted(
                self.chi2_scores_.items(),
                key=lambda x: x[1],
                reverse=True
            )
            self.selected_features_ = [feat for feat, _ in sorted_features[:k_val]]
            self.dropped_features_ = [feat for feat, _ in sorted_features[k_val:]]
        
        if not self.selected_features_:
            logger.warning("Chi2Selector: No features selected. Keeping all features.")
            self.selected_features_ = list(X.columns)
            self.dropped_features_ = []
        
        logger.info(
            f"Chi2Selector: Selected {len(self.selected_features_)} / {len(X.columns)} features"
        )
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform by selecting features."""
        if not self.selected_features_:
            raise RuntimeError("Chi2Selector not fitted. Call fit() first.")
        return X[self.selected_features_].copy()
    
    def get_feature_names_out(self, input_features=None) -> list:
        """Get output feature names."""
        return self.selected_features_
    
    def get_support(self, indices: bool = False):
        """Get mask or indices of selected features."""
        if not self.chi2_scores_:
            raise RuntimeError("Chi2Selector not fitted.")
        all_features = list(self.chi2_scores_.keys())
        mask = [feat in self.selected_features_ for feat in all_features]
        return np.where(mask)[0] if indices else np.array(mask)


class RFESelector(BaseEstimator, TransformerMixin):
    """Recursive Feature Elimination (RFE) selector.
    
    Wrapper method that recursively removes features and builds a model on remaining
    features. Uses feature importance or coefficients to eliminate least important
    features at each step.
    
    Parameters
    ----------
    estimator : estimator object, optional
        Base estimator with fit method. If None, uses LogisticRegression for
        classification or LinearRegression for regression. Default: None
    n_features_to_select : int, optional
        Number of features to select. Default: 50
    step : int or float, optional
        Number/fraction of features to remove at each iteration. Default: 1
    task : str, optional
        Task type: 'classification' or 'regression'. Default: 'classification'
    
    Attributes
    ----------
    rfe_ : RFE
        Fitted RFE selector
    selected_features_ : list
        Names of selected features
    ranking_ : dict
        Feature rankings (1 = best)
        
    Examples
    --------
    >>> from featurecraft.selection import RFESelector
    >>> selector = RFESelector(n_features_to_select=30, step=5)
    >>> selector.fit(X_train, y_train)
    >>> X_selected = selector.transform(X_train)
    >>> print(f"Selected: {selector.selected_features_}")
    >>> print(f"Rankings: {selector.ranking_}")
    
    Notes
    -----
    - Computationally expensive for large datasets
    - Works with any estimator that exposes feature_importances_ or coef_
    - More accurate than filter methods but slower
    """
    
    def __init__(
        self,
        estimator=None,
        n_features_to_select: int = 50,
        step: Union[int, float] = 1,
        task: str = "classification",
    ):
        self.estimator = estimator
        self.n_features_to_select = n_features_to_select
        self.step = step
        self.task = task
        self.rfe_ = None
        self.selected_features_: list = []
        self.ranking_: dict = {}
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fit RFE selector.
        
        Parameters
        ----------
        X : pd.DataFrame
            Training features
        y : pd.Series
            Training target
            
        Returns
        -------
        self
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"X must be DataFrame, got {type(X).__name__}")
        if not isinstance(y, pd.Series):
            raise TypeError(f"y must be Series, got {type(y).__name__}")
        
        # Choose default estimator if not provided
        if self.estimator is None:
            if self.task.lower() in ['classification', 'binary', 'multiclass']:
                estimator = LogisticRegression(max_iter=1000, random_state=42)
            else:
                estimator = LinearRegression()
        else:
            estimator = clone(self.estimator)
        
        # Create RFE selector
        n_features = min(self.n_features_to_select, X.shape[1])
        self.rfe_ = RFE(
            estimator=estimator,
            n_features_to_select=n_features,
            step=self.step,
        )
        
        # Fit RFE
        try:
            X_array = X.fillna(0).values
            y_array = y.values
            self.rfe_.fit(X_array, y_array)
            
            # Extract selected features
            mask = self.rfe_.support_
            self.selected_features_ = X.columns[mask].tolist()
            
            # Extract rankings
            self.ranking_ = {
                col: int(rank) for col, rank in zip(X.columns, self.rfe_.ranking_)
            }
            
            logger.info(
                f"RFESelector: Selected {len(self.selected_features_)} / {len(X.columns)} features"
            )
        
        except Exception as e:
            logger.error(f"RFESelector: Fitting failed: {e}. Keeping all features.")
            self.selected_features_ = list(X.columns)
            self.ranking_ = {col: 1 for col in X.columns}
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform by selecting features."""
        if not self.selected_features_:
            raise RuntimeError("RFESelector not fitted. Call fit() first.")
        return X[self.selected_features_].copy()
    
    def get_feature_names_out(self, input_features=None) -> list:
        """Get output feature names."""
        return self.selected_features_
    
    def get_support(self, indices: bool = False):
        """Get mask or indices of selected features."""
        if self.rfe_ is None:
            raise RuntimeError("RFESelector not fitted.")
        return self.rfe_.get_support(indices=indices)


class SequentialFeatureSelector(BaseEstimator, TransformerMixin):
    """Sequential Feature Selection using forward or backward search.
    
    Wrapper method that adds (forward) or removes (backward) features one at a time
    based on cross-validated model performance.
    
    Parameters
    ----------
    estimator : estimator object, optional
        Base estimator. If None, uses LogisticRegression/LinearRegression. Default: None
    n_features_to_select : int, optional
        Number of features to select. Default: 50
    direction : str, optional
        'forward' or 'backward'. Default: 'forward'
    scoring : str, optional
        Scoring metric for CV. Default: 'accuracy' for classification, 'r2' for regression
    cv : int, optional
        Number of CV folds. Default: 3
    task : str, optional
        'classification' or 'regression'. Default: 'classification'
    
    Attributes
    ----------
    selected_features_ : list
        Names of selected features
    scores_ : dict
        CV scores at each step
        
    Examples
    --------
    >>> from featurecraft.selection import SequentialFeatureSelector
    >>> selector = SequentialFeatureSelector(n_features_to_select=20, direction='forward')
    >>> selector.fit(X_train, y_train)
    >>> X_selected = selector.transform(X_train)
    
    Notes
    -----
    - Very slow for large feature sets (O(n²))
    - Provides best performance among wrapper methods
    - Use with caution on high-dimensional data
    """
    
    def __init__(
        self,
        estimator=None,
        n_features_to_select: int = 50,
        direction: Literal['forward', 'backward'] = 'forward',
        scoring: Optional[str] = None,
        cv: int = 3,
        task: str = 'classification',
    ):
        self.estimator = estimator
        self.n_features_to_select = n_features_to_select
        self.direction = direction
        self.scoring = scoring
        self.cv = cv
        self.task = task
        self.selected_features_: list = []
        self.scores_: dict = {}
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fit sequential feature selector.
        
        Parameters
        ----------
        X : pd.DataFrame
            Training features
        y : pd.Series
            Training target
            
        Returns
        -------
        self
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"X must be DataFrame, got {type(X).__name__}")
        if not isinstance(y, pd.Series):
            raise TypeError(f"y must be Series, got {type(y).__name__}")
        
        # Choose default estimator
        if self.estimator is None:
            if self.task.lower() in ['classification', 'binary', 'multiclass']:
                estimator = LogisticRegression(max_iter=1000, random_state=42)
                scoring = self.scoring or 'accuracy'
            else:
                estimator = LinearRegression()
                scoring = self.scoring or 'r2'
        else:
            estimator = clone(self.estimator)
            scoring = self.scoring or 'accuracy'
        
        # Import SequentialFeatureSelector from sklearn
        try:
            from sklearn.feature_selection import SequentialFeatureSelector as SFS
            
            n_features = min(self.n_features_to_select, X.shape[1])
            sfs = SFS(
                estimator=estimator,
                n_features_to_select=n_features,
                direction=self.direction,
                scoring=scoring,
                cv=self.cv,
                n_jobs=-1,
            )
            
            X_array = X.fillna(0).values
            y_array = y.values
            sfs.fit(X_array, y_array)
            
            # Extract selected features
            mask = sfs.get_support()
            self.selected_features_ = X.columns[mask].tolist()
            
            logger.info(
                f"SequentialFeatureSelector ({self.direction}): "
                f"Selected {len(self.selected_features_)} / {len(X.columns)} features"
            )
        
        except ImportError:
            logger.error(
                "SequentialFeatureSelector requires sklearn >= 0.24. "
                "Falling back to all features."
            )
            self.selected_features_ = list(X.columns)
        except Exception as e:
            logger.error(f"SequentialFeatureSelector failed: {e}. Keeping all features.")
            self.selected_features_ = list(X.columns)
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform by selecting features."""
        if not self.selected_features_:
            raise RuntimeError("SequentialFeatureSelector not fitted.")
        return X[self.selected_features_].copy()
    
    def get_feature_names_out(self, input_features=None) -> list:
        """Get output feature names."""
        return self.selected_features_


class LassoSelector(BaseEstimator, TransformerMixin):
    """Feature selection using Lasso (L1) regularization.
    
    Embedded method that uses L1 penalty to shrink coefficients to zero,
    effectively performing feature selection during model training.
    
    Parameters
    ----------
    alpha : float or 'auto', optional
        L1 regularization strength. If 'auto', uses LassoCV. Default: 'auto'
    threshold : float, optional
        Coefficient magnitude threshold. Features with |coef| < threshold are dropped.
        Default: 1e-5
    task : str, optional
        'classification' or 'regression'. Default: 'regression'
    cv : int, optional
        Number of CV folds for alpha selection (when alpha='auto'). Default: 5
    
    Attributes
    ----------
    lasso_ : Lasso or LassoCV
        Fitted Lasso model
    coefficients_ : dict
        Feature coefficients
    selected_features_ : list
        Names of features with non-zero coefficients
        
    Examples
    --------
    >>> from featurecraft.selection import LassoSelector
    >>> selector = LassoSelector(alpha='auto')
    >>> selector.fit(X_train, y_train)
    >>> X_selected = selector.transform(X_train)
    >>> print(f"Non-zero coefficients: {len(selector.selected_features_)}")
    
    Notes
    -----
    - Fast and effective for high-dimensional data
    - Assumes features are scaled (standardized)
    - Works best with independent features
    - For classification, uses LogisticRegression with L1 penalty
    """
    
    def __init__(
        self,
        alpha: Union[float, str] = 'auto',
        threshold: float = 1e-5,
        task: str = 'regression',
        cv: int = 5,
    ):
        self.alpha = alpha
        self.threshold = threshold
        self.task = task
        self.cv = cv
        self.lasso_ = None
        self.coefficients_: dict = {}
        self.selected_features_: list = []
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fit Lasso selector.
        
        Parameters
        ----------
        X : pd.DataFrame
            Training features
        y : pd.Series
            Training target
            
        Returns
        -------
        self
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"X must be DataFrame, got {type(X).__name__}")
        if not isinstance(y, pd.Series):
            raise TypeError(f"y must be Series, got {type(y).__name__}")
        
        X_array = X.fillna(0).values
        y_array = y.values
        
        # Choose Lasso model
        try:
            if self.task.lower() in ['classification', 'binary', 'multiclass']:
                # Use LogisticRegression with L1 penalty
                if self.alpha == 'auto':
                    # Use LogisticRegressionCV
                    from sklearn.linear_model import LogisticRegressionCV
                    self.lasso_ = LogisticRegressionCV(
                        penalty='l1',
                        solver='saga',
                        cv=self.cv,
                        random_state=42,
                        max_iter=1000,
                        n_jobs=-1,
                    )
                else:
                    self.lasso_ = LogisticRegression(
                        penalty='l1',
                        C=1.0 / self.alpha,
                        solver='saga',
                        random_state=42,
                        max_iter=1000,
                    )
            else:
                # Use Lasso for regression
                if self.alpha == 'auto':
                    self.lasso_ = LassoCV(cv=self.cv, random_state=42, n_jobs=-1)
                else:
                    self.lasso_ = Lasso(alpha=self.alpha, random_state=42)
            
            # Fit model
            self.lasso_.fit(X_array, y_array)
            
            # Extract coefficients
            if hasattr(self.lasso_, 'coef_'):
                coefs = self.lasso_.coef_
                # Handle multi-class case (multiple coefficient arrays)
                if coefs.ndim == 2:
                    coefs = np.abs(coefs).mean(axis=0)
                self.coefficients_ = {
                    col: float(coef) for col, coef in zip(X.columns, coefs)
                }
            else:
                logger.warning("Lasso model has no coef_ attribute")
                self.coefficients_ = {col: 1.0 for col in X.columns}
            
            # Select features with non-zero coefficients
            self.selected_features_ = [
                col for col, coef in self.coefficients_.items()
                if abs(coef) > self.threshold
            ]
            
            if not self.selected_features_:
                logger.warning(
                    f"LassoSelector: No features passed threshold {self.threshold}. "
                    "Keeping all features."
                )
                self.selected_features_ = list(X.columns)
            
            logger.info(
                f"LassoSelector: Selected {len(self.selected_features_)} / {len(X.columns)} "
                f"features with |coef| > {self.threshold}"
            )
        
        except Exception as e:
            logger.error(f"LassoSelector failed: {e}. Keeping all features.")
            self.selected_features_ = list(X.columns)
            self.coefficients_ = {col: 1.0 for col in X.columns}
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform by selecting features."""
        if not self.selected_features_:
            raise RuntimeError("LassoSelector not fitted.")
        return X[self.selected_features_].copy()
    
    def get_feature_names_out(self, input_features=None) -> list:
        """Get output feature names."""
        return self.selected_features_


class TreeImportanceSelector(BaseEstimator, TransformerMixin):
    """Feature selection using tree-based feature importance.
    
    Embedded method that uses feature importances from tree-based models
    (Random Forest, Gradient Boosting, etc.) to select top features.
    
    Parameters
    ----------
    estimator : estimator object, optional
        Tree-based estimator. If None, uses RandomForest. Default: None
    threshold : float or str, optional
        Importance threshold. Options:
        - float: Minimum importance value
        - 'mean': Mean importance
        - 'median': Median importance
        - 'auto': Mean * 0.1
        Default: 'auto'
    n_features : int, optional
        Maximum number of features to select. Default: None (no limit)
    task : str, optional
        'classification' or 'regression'. Default: 'classification'
    
    Attributes
    ----------
    estimator_ : estimator
        Fitted tree model
    importances_ : dict
        Feature importances
    selected_features_ : list
        Names of selected features
        
    Examples
    --------
    >>> from featurecraft.selection import TreeImportanceSelector
    >>> selector = TreeImportanceSelector(threshold='mean')
    >>> selector.fit(X_train, y_train)
    >>> X_selected = selector.transform(X_train)
    >>> print(f"Top features: {sorted(selector.importances_.items(), key=lambda x: -x[1])[:10]}")
    
    Notes
    -----
    - Fast and handles non-linear relationships
    - Works well with mixed feature types
    - Biased toward high-cardinality features
    - Can use any tree-based model (RF, XGBoost, LightGBM)
    """
    
    def __init__(
        self,
        estimator=None,
        threshold: Union[float, str] = 'auto',
        n_features: Optional[int] = None,
        task: str = 'classification',
    ):
        self.estimator = estimator
        self.threshold = threshold
        self.n_features = n_features
        self.task = task
        self.estimator_ = None
        self.importances_: dict = {}
        self.selected_features_: list = []
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fit tree importance selector.
        
        Parameters
        ----------
        X : pd.DataFrame
            Training features
        y : pd.Series
            Training target
            
        Returns
        -------
        self
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"X must be DataFrame, got {type(X).__name__}")
        if not isinstance(y, pd.Series):
            raise TypeError(f"y must be Series, got {type(y).__name__}")
        
        # Choose default estimator
        if self.estimator is None:
            if self.task.lower() in ['classification', 'binary', 'multiclass']:
                self.estimator_ = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1,
                )
            else:
                self.estimator_ = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1,
                )
        else:
            self.estimator_ = clone(self.estimator)
        
        # Fit model
        try:
            X_array = X.fillna(0).values
            y_array = y.values
            self.estimator_.fit(X_array, y_array)
            
            # Extract feature importances
            if hasattr(self.estimator_, 'feature_importances_'):
                importances = self.estimator_.feature_importances_
                self.importances_ = {
                    col: float(imp) for col, imp in zip(X.columns, importances)
                }
            else:
                logger.warning("Estimator has no feature_importances_ attribute")
                self.importances_ = {col: 1.0 for col in X.columns}
            
            # Determine threshold
            imp_values = np.array(list(self.importances_.values()))
            if isinstance(self.threshold, str):
                if self.threshold == 'mean':
                    threshold_val = np.mean(imp_values)
                elif self.threshold == 'median':
                    threshold_val = np.median(imp_values)
                elif self.threshold == 'auto':
                    threshold_val = np.mean(imp_values) * 0.1
                else:
                    threshold_val = 0.0
            else:
                threshold_val = float(self.threshold)
            
            # Select features above threshold
            self.selected_features_ = [
                col for col, imp in self.importances_.items()
                if imp >= threshold_val
            ]
            
            # Limit to n_features if specified
            if self.n_features is not None and len(self.selected_features_) > self.n_features:
                sorted_features = sorted(
                    self.importances_.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                self.selected_features_ = [feat for feat, _ in sorted_features[:self.n_features]]
            
            if not self.selected_features_:
                logger.warning("TreeImportanceSelector: No features selected. Keeping all.")
                self.selected_features_ = list(X.columns)
            
            logger.info(
                f"TreeImportanceSelector: Selected {len(self.selected_features_)} / {len(X.columns)} "
                f"features with importance >= {threshold_val:.4f}"
            )
        
        except Exception as e:
            logger.error(f"TreeImportanceSelector failed: {e}. Keeping all features.")
            self.selected_features_ = list(X.columns)
            self.importances_ = {col: 1.0 for col in X.columns}
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform by selecting features."""
        if not self.selected_features_:
            raise RuntimeError("TreeImportanceSelector not fitted.")
        return X[self.selected_features_].copy()
    
    def get_feature_names_out(self, input_features=None) -> list:
        """Get output feature names."""
        return self.selected_features_


class BorutaSelector(BaseEstimator, TransformerMixin):
    """Boruta feature selection algorithm.
    
    All-relevant feature selection method that uses Random Forest to identify
    features that are significantly more important than random shadow features.
    
    The algorithm:
    1. Creates shadow features by permuting original features
    2. Trains Random Forest on combined dataset (original + shadow)
    3. Compares feature importance to max shadow importance
    4. Marks features as confirmed/rejected based on statistical tests
    5. Repeats until all features are confirmed/rejected or max iterations reached
    
    Parameters
    ----------
    estimator : estimator object, optional
        Tree-based estimator. If None, uses RandomForest. Default: None
    n_estimators : int, optional
        Number of trees in Random Forest. Default: 100
    max_iter : int, optional
        Maximum iterations for Boruta. Default: 100
    alpha : float, optional
        Significance level for statistical tests. Default: 0.05
    task : str, optional
        'classification' or 'regression'. Default: 'classification'
    random_state : int, optional
        Random seed. Default: 42
    
    Attributes
    ----------
    selected_features_ : list
        Confirmed important features
    tentative_features_ : list
        Features with uncertain importance
    rejected_features_ : list
        Confirmed unimportant features
    importances_ : dict
        Feature importances
    n_iterations_ : int
        Number of iterations run
        
    Examples
    --------
    >>> from featurecraft.selection import BorutaSelector
    >>> selector = BorutaSelector(max_iter=50)
    >>> selector.fit(X_train, y_train)
    >>> X_selected = selector.transform(X_train)
    >>> print(f"Confirmed: {len(selector.selected_features_)}")
    >>> print(f"Tentative: {len(selector.tentative_features_)}")
    >>> print(f"Rejected: {len(selector.rejected_features_)}")
    
    Notes
    -----
    - Robust all-relevant feature selection
    - Can identify truly important features
    - Computationally expensive (many RF iterations)
    - Better than single-shot importance methods
    
    References
    ----------
    Kursa, M. B., & Rudnicki, W. R. (2010). Feature selection with the Boruta package.
    Journal of Statistical Software, 36(11), 1-13.
    """
    
    def __init__(
        self,
        estimator=None,
        n_estimators: int = 100,
        max_iter: int = 100,
        alpha: float = 0.05,
        task: str = 'classification',
        random_state: int = 42,
    ):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.max_iter = max_iter
        self.alpha = alpha
        self.task = task
        self.random_state = random_state
        
        self.selected_features_: list = []
        self.tentative_features_: list = []
        self.rejected_features_: list = []
        self.importances_: dict = {}
        self.n_iterations_: int = 0
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fit Boruta selector.
        
        Parameters
        ----------
        X : pd.DataFrame
            Training features
        y : pd.Series
            Training target
            
        Returns
        -------
        self
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"X must be DataFrame, got {type(X).__name__}")
        if not isinstance(y, pd.Series):
            raise TypeError(f"y must be Series, got {type(y).__name__}")
        
        # Initialize
        X_work = X.fillna(0).copy()
        y_array = y.values
        n_features = X_work.shape[1]
        
        # Choose default estimator
        if self.estimator is None:
            if self.task.lower() in ['classification', 'binary', 'multiclass']:
                base_estimator = RandomForestClassifier(
                    n_estimators=self.n_estimators,
                    max_depth=7,
                    random_state=self.random_state,
                    n_jobs=-1,
                )
            else:
                base_estimator = RandomForestRegressor(
                    n_estimators=self.n_estimators,
                    max_depth=7,
                    random_state=self.random_state,
                    n_jobs=-1,
                )
        else:
            base_estimator = clone(self.estimator)
        
        # Track feature status: 0=tentative, 1=confirmed, -1=rejected
        feature_status = {col: 0 for col in X_work.columns}
        hit_counts = {col: 0 for col in X_work.columns}
        
        try:
            rng = np.random.RandomState(self.random_state)
            
            for iteration in range(self.max_iter):
                # Create shadow features
                X_shadow = X_work.apply(lambda col: rng.permutation(col.values), axis=0)
                X_shadow.columns = [f"shadow_{col}" for col in X_shadow.columns]
                
                # Combine original and shadow
                X_combined = pd.concat([X_work, X_shadow], axis=1)
                
                # Train model
                estimator = clone(base_estimator)
                estimator.fit(X_combined.values, y_array)
                
                # Get importances
                if hasattr(estimator, 'feature_importances_'):
                    importances = estimator.feature_importances_
                else:
                    logger.warning("Estimator has no feature_importances_")
                    break
                
                # Split importances
                n_orig = n_features
                orig_importances = importances[:n_orig]
                shadow_importances = importances[n_orig:]
                
                # Compute max shadow importance
                max_shadow_importance = np.max(shadow_importances)
                
                # Update hit counts for features exceeding max shadow
                for i, col in enumerate(X_work.columns):
                    if feature_status[col] == 0:  # Only consider tentative features
                        if orig_importances[i] > max_shadow_importance:
                            hit_counts[col] += 1
                        else:
                            hit_counts[col] -= 1
                
                # Statistical test: binomial test
                # If hit_count significantly > 0, confirm; if significantly < 0, reject
                from scipy import stats as scipy_stats
                
                for col in X_work.columns:
                    if feature_status[col] != 0:
                        continue
                    
                    # Binomial test: p(hit_count | p=0.5, n=iteration+1)
                    n_trials = iteration + 1
                    n_hits = (hit_counts[col] + n_trials) / 2  # Convert to actual hits
                    p_value = scipy_stats.binomtest(
                        int(n_hits),
                        n_trials,
                        0.5,
                        alternative='two-sided'
                    ).pvalue
                    
                    if p_value < self.alpha:
                        if hit_counts[col] > 0:
                            feature_status[col] = 1  # Confirmed
                        else:
                            feature_status[col] = -1  # Rejected
                
                # Check termination
                if all(status != 0 for status in feature_status.values()):
                    logger.info(f"Boruta: All features decided at iteration {iteration + 1}")
                    break
            
            self.n_iterations_ = iteration + 1
            
            # Extract final feature sets
            self.selected_features_ = [
                col for col, status in feature_status.items() if status == 1
            ]
            self.tentative_features_ = [
                col for col, status in feature_status.items() if status == 0
            ]
            self.rejected_features_ = [
                col for col, status in feature_status.items() if status == -1
            ]
            
            # Compute final importances
            estimator = clone(base_estimator)
            estimator.fit(X_work.values, y_array)
            if hasattr(estimator, 'feature_importances_'):
                self.importances_ = {
                    col: float(imp)
                    for col, imp in zip(X_work.columns, estimator.feature_importances_)
                }
            
            # Include tentative features in selected
            self.selected_features_.extend(self.tentative_features_)
            
            if not self.selected_features_:
                logger.warning("BorutaSelector: No features confirmed. Keeping all.")
                self.selected_features_ = list(X.columns)
            
            logger.info(
                f"BorutaSelector: Confirmed {len(self.selected_features_)} features, "
                f"Rejected {len(self.rejected_features_)} features "
                f"in {self.n_iterations_} iterations"
            )
        
        except Exception as e:
            logger.error(f"BorutaSelector failed: {e}. Keeping all features.")
            self.selected_features_ = list(X.columns)
            self.tentative_features_ = []
            self.rejected_features_ = []
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform by selecting confirmed features."""
        if not self.selected_features_:
            raise RuntimeError("BorutaSelector not fitted.")
        return X[self.selected_features_].copy()
    
    def get_feature_names_out(self, input_features=None) -> list:
        """Get output feature names."""
        return self.selected_features_
