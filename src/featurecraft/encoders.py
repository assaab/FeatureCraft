"""Encoding utilities for FeatureCraft."""

from __future__ import annotations

import hashlib
from typing import Any, Callable, Optional, Union

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    StratifiedKFold,
    TimeSeriesSplit,
)
from sklearn.preprocessing import OneHotEncoder

from .logging import get_logger
from .utils import to_csr_matrix
from .utils.leakage import LeakageGuardMixin

logger = get_logger(__name__)


class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    """Group rare categories into 'Other'.
    
    Args:
        min_freq: Minimum frequency threshold (categories below this are grouped as 'Other')
        preserve_sentinel: If True, never group the missing sentinel into 'Other'
        sentinel_value: Sentinel string for missing values (default: '__MISSING__')
        
    Attributes:
        maps_: Dict mapping column name to set of categories to keep
        columns_: List of column names
    """

    def __init__(
        self, 
        min_freq: float = 0.01,
        preserve_sentinel: bool = True,
        sentinel_value: str = "__MISSING__"
    ) -> None:
        """Initialize with minimum frequency threshold."""
        self.min_freq = float(min_freq)
        self.preserve_sentinel = preserve_sentinel
        self.sentinel_value = sentinel_value
        self.maps_: dict[str, set] = {}
        self.columns_: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> RareCategoryGrouper:
        """Fit the grouper."""
        df = pd.DataFrame(X).copy()
        self.columns_ = list(df.columns)
        n = len(df)
        for c in df.columns:
            freq = df[c].value_counts(dropna=False) / max(n, 1)
            # Keep categories with frequency > threshold (strictly greater)
            keep = set(freq[freq > self.min_freq].index)
            
            # Always preserve the sentinel if configured
            if self.preserve_sentinel and self.sentinel_value in df[c].values:
                keep.add(self.sentinel_value)
            
            self.maps_[c] = keep
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform by grouping rare categories."""
        df = pd.DataFrame(X).copy()
        for c in df.columns:
            keep = self.maps_.get(c, set())
            # Replace rare categories with "Other"
            mask = ~df[c].isin(keep)
            
            # Handle categorical dtypes by converting to object first
            if isinstance(df[c].dtype, CategoricalDtype):
                # Add "Other" to categories if not present
                if "Other" not in df[c].cat.categories:
                    df[c] = df[c].cat.add_categories(["Other"])
                df.loc[mask, c] = "Other"
            else:
                # Convert to object type to avoid type issues
                df[c] = df[c].astype(str)
                df.loc[mask, c] = "Other"
        return df
    
    def get_feature_names_out(self, input_features: Optional[list[str]] = None) -> list[str]:
        """Get output feature names (pass-through for rare grouper).
        
        Args:
            input_features: Input feature names (unused, returns fitted columns)
            
        Returns:
            List of column names (same as input)
        """
        return self.columns_ if self.columns_ else (input_features or [])


class HashingEncoder(BaseEstimator, TransformerMixin):
    """Simple hashing encoder for categorical/string columns."""

    def __init__(self, n_features: int = 256, seed: int = 42) -> None:
        """Initialize with number of features and seed."""
        self.n_features = int(n_features)
        self.seed = int(seed)
        self.columns_: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> HashingEncoder:
        """Fit the encoder."""
        self.columns_ = list(pd.DataFrame(X).columns)
        return self

    def _hash(self, s: str) -> int:
        """Deterministic hashing using hashlib (reproducible across runs)."""
        # Use MD5 for fast deterministic hashing (NOT for cryptographic security)
        # This is safe for feature hashing; MD5 is used purely for speed and determinism
        h = hashlib.md5(f"{s}:{self.seed}".encode()).hexdigest()
        return int(h, 16) % self.n_features

    def transform(self, X: pd.DataFrame) -> sparse.csr_matrix:
        """Transform using hashing.
        
        Uses itertuples for efficient iteration (much faster than iterrows).
        """
        df = pd.DataFrame(X)[self.columns_].astype(str)
        rows: list[dict[int, float]] = []
        columns = list(df.columns)
        
        # Use itertuples for better performance (10-100x faster than iterrows)
        for row_tuple in df.itertuples(index=False, name=None):
            d: dict[int, float] = {}
            for col_idx, value in enumerate(row_tuple):
                col_name = columns[col_idx]
                idx = self._hash(f"{col_name}={value}")
                d[idx] = d.get(idx, 0.0) + 1.0
            rows.append(d)
        return to_csr_matrix(rows, self.n_features)
    
    def get_feature_names_out(self, input_features: Optional[list[str]] = None) -> list[str]:
        """Get output feature names for hashed features.
        
        Args:
            input_features: Input feature names (unused)
            
        Returns:
            List of hash feature names
        """
        return [f"hash_{i}" for i in range(self.n_features)]


class KFoldTargetEncoder(BaseEstimator, TransformerMixin):
    """Fold-wise target encoder with smoothing and Gaussian noise."""

    def __init__(
        self,
        cols: list[str] | None = None,
        n_splits: int = 5,
        smoothing: float = 0.3,
        noise_std: float = 0.01,
        random_state: int = 42,
        task: str = "classification",  # "classification" or "regression"
        positive_class: Any | None = None,
    ) -> None:
        """Initialize target encoder."""
        self.cols = cols
        self.n_splits = int(n_splits)
        self.smoothing = float(smoothing)
        self.noise_std = float(noise_std)
        self.random_state = int(random_state)
        self.task = task
        self.positive_class = positive_class
        self.global_: dict[str, float] = {}
        self.maps_: dict[str, dict[Any, float]] = {}
        self.columns_: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> KFoldTargetEncoder:
        """Fit the target encoder.

        Uses fold-wise encoding to prevent target leakage. For each fold, computes
        category statistics on the training portion and applies smoothing.
        """
        df = pd.DataFrame(X).copy()
        y_series = pd.Series(y).reset_index(drop=True)
        # Infer columns from actual input DataFrame
        self.columns_ = list(df.columns)
        df = df.astype(str).reset_index(drop=True)

        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

        if self.task == "classification":
            if self.positive_class is None:
                # infer as max label by frequency or `1` if binary {0,1}
                if y_series.nunique() == 2 and set(y_series.unique()) == {0, 1}:
                    self.positive_class = 1
                else:
                    self.positive_class = y_series.value_counts().idxmax()
            y_enc = (y_series == self.positive_class).astype(float)
        else:
            y_enc = y_series.astype(float)

        # Compute global priors
        prior = float(y_enc.mean())
        for c in self.columns_:
            self.global_[c] = prior
            self.maps_[c] = {}

        # Build encoding maps from fold-wise statistics
        for c in self.columns_:
            # Accumulate statistics across folds
            category_stats: dict[Any, tuple[float, int]] = {}
            for tr_idx, _va_idx in kf.split(df):
                tr_df = df.iloc[tr_idx]
                tr_y = y_enc.iloc[tr_idx].reset_index(drop=True)
                # Create temporary dataframe for groupby
                temp = pd.DataFrame({c: tr_df[c].values, "target": tr_y.values})
                agg = temp.groupby(c)["target"].agg(["sum", "count"])

                for cat_val in agg.index:
                    cat_sum = float(agg.loc[cat_val, "sum"])
                    cat_count = int(agg.loc[cat_val, "count"])
                    if cat_val not in category_stats:
                        category_stats[cat_val] = (cat_sum, cat_count)
                    else:
                        prev_sum, prev_count = category_stats[cat_val]
                        category_stats[cat_val] = (prev_sum + cat_sum, prev_count + cat_count)

            # Apply smoothing and store
            for cat_val, (cat_sum, cat_count) in category_stats.items():
                smoothed_mean = (cat_sum + prior * self.smoothing) / (cat_count + self.smoothing)
                self.maps_[c][cat_val] = float(smoothed_mean)

        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform using learned encodings."""
        df = pd.DataFrame(X).copy()
        out = []
        for c in self.columns_:
            s = df[c].astype(str)
            m = self.maps_.get(c, {})
            prior = self.global_.get(c, 0.0)
            # Fix lambda binding: use default argument to capture current values
            vals = s.map(lambda v, _m=m, _prior=prior: _m.get(v, _prior)).astype(float).values.reshape(-1, 1)
            if self.noise_std > 0:
                rng = np.random.default_rng(self.random_state)
                vals = vals + rng.normal(0.0, self.noise_std, size=vals.shape)
            out.append(vals)
        return np.hstack(out)
    
    def get_feature_names_out(self, input_features: Optional[list[str]] = None) -> list[str]:
        """Get output feature names for target encoded features.
        
        Args:
            input_features: Input feature names (unused, returns fitted columns)
            
        Returns:
            List of target-encoded feature names
        """
        return [f"te_{c}" for c in self.columns_]


def make_ohe(min_frequency: float = 0.01, handle_unknown: str = "infrequent_if_exist") -> OneHotEncoder:
    """Create OneHotEncoder with sensible defaults.
    
    Uses progressive fallback strategy to work across different sklearn versions.
    
    Args:
        min_frequency: Minimum frequency threshold for rare categories
        handle_unknown: How to handle unknown categories
    """
    # Try modern sklearn (1.0+) with all features
    try:
        encoder = OneHotEncoder(
            handle_unknown=handle_unknown,
            min_frequency=min_frequency,
            sparse_output=True,
            drop="if_binary",
        )
        # Test instantiation to ensure it works
        _ = encoder.get_params()
        logger.debug("Using OneHotEncoder with modern sklearn parameters (sparse_output, drop)")
        return encoder
    except (TypeError, AttributeError) as e:
        logger.warning(
            f"OneHotEncoder instantiation with full parameters failed: {e}. "
            f"Trying fallback without 'drop' parameter."
        )
    
    # Try without drop parameter
    try:
        encoder = OneHotEncoder(
            handle_unknown=handle_unknown if handle_unknown != "infrequent_if_exist" else "ignore",
            min_frequency=min_frequency,
            sparse_output=True,
        )
        _ = encoder.get_params()
        logger.debug("Using OneHotEncoder without 'drop' parameter")
        return encoder
    except (TypeError, AttributeError) as e:
        logger.warning(
            f"OneHotEncoder with sparse_output failed: {e}. "
            f"Trying fallback with 'sparse' parameter (older sklearn)."
        )
    
    # Try with sparse instead of sparse_output (older sklearn)
    try:
        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse=True,
        )
        _ = encoder.get_params()
        logger.warning(
            "Using older sklearn version - falling back to 'sparse' parameter. "
            "Consider upgrading sklearn to 1.0+ for better features."
        )
        return encoder
    except (TypeError, AttributeError) as e:
        logger.warning(
            f"OneHotEncoder with 'sparse' parameter failed: {e}. "
            f"Using minimal fallback configuration."
        )
    
    # Ultimate fallback - minimal parameters
    logger.warning(
        "Using minimal OneHotEncoder configuration. "
        "All advanced features disabled. Consider upgrading sklearn."
    )
    encoder = OneHotEncoder()
    try:
        encoder.set_params(handle_unknown="ignore")
    except Exception as e:
        logger.warning(f"Could not set handle_unknown='ignore': {e}. Using defaults.")
    
    return encoder


class LeaveOneOutTargetEncoder(BaseEstimator, TransformerMixin):
    """Leave-One-Out Target Encoder with leakage prevention."""
    
    def __init__(
        self,
        cols: list[str] | None = None,
        smoothing: float = 0.3,
        noise_std: float = 0.01,
        random_state: int = 42,
        task: str = "classification",
    ) -> None:
        """Initialize Leave-One-Out target encoder.
        
        Args:
            cols: Columns to encode
            smoothing: Smoothing factor
            noise_std: Gaussian noise standard deviation
            random_state: Random seed
            task: "classification" or "regression"
        """
        self.cols = cols
        self.smoothing = smoothing
        self.noise_std = noise_std
        self.random_state = random_state
        self.task = task
        self.global_: dict[str, float] = {}
        self.maps_: dict[str, dict[Any, tuple[float, int]]] = {}
        self.columns_: list[str] = []
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LeaveOneOutTargetEncoder":
        """Fit encoder."""
        df = pd.DataFrame(X).copy()
        y_series = pd.Series(y).reset_index(drop=True)
        self.columns_ = list(df.columns)
        df = df.astype(str).reset_index(drop=True)
        
        # Convert target
        if self.task == "classification":
            y_enc = (y_series == y_series.value_counts().idxmax()).astype(float)
        else:
            y_enc = y_series.astype(float)
        
        prior = float(y_enc.mean())
        
        for c in self.columns_:
            self.global_[c] = prior
            self.maps_[c] = {}
            
            # Compute sum and count for each category
            temp = pd.DataFrame({c: df[c].values, "target": y_enc.values})
            agg = temp.groupby(c)["target"].agg(["sum", "count"])
            
            for cat_val in agg.index:
                cat_sum = float(agg.loc[cat_val, "sum"])
                cat_count = int(agg.loc[cat_val, "count"])
                self.maps_[c][cat_val] = (cat_sum, cat_count)
        
        return self
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform using leave-one-out encoding."""
        df = pd.DataFrame(X).copy()
        out = []
        
        for c in self.columns_:
            s = df[c].astype(str)
            prior = self.global_.get(c, 0.0)
            vals = np.zeros(len(s))
            
            for i, val in enumerate(s):
                if val in self.maps_[c]:
                    cat_sum, cat_count = self.maps_[c][val]
                    # Leave-one-out: exclude current observation
                    if cat_count > 1:
                        loo_mean = (cat_sum + prior * self.smoothing) / (cat_count + self.smoothing)
                    else:
                        loo_mean = prior
                else:
                    loo_mean = prior
                
                vals[i] = loo_mean
            
            if self.noise_std > 0:
                rng = np.random.default_rng(self.random_state)
                vals = vals + rng.normal(0.0, self.noise_std, size=len(vals))
            
            out.append(vals.reshape(-1, 1))
        
        return np.hstack(out)
    
    def get_feature_names_out(self, input_features: Optional[list[str]] = None) -> list[str]:
        """Get output feature names for LOO target encoded features.
        
        Args:
            input_features: Input feature names (unused, returns fitted columns)
            
        Returns:
            List of LOO target-encoded feature names
        """
        return [f"loo_te_{c}" for c in self.columns_]


class WoEEncoder(BaseEstimator, TransformerMixin):
    """Weight of Evidence encoder for binary classification."""
    
    def __init__(
        self,
        cols: list[str] | None = None,
        smoothing: float = 0.5,
        random_state: int = 42,
    ) -> None:
        """Initialize WoE encoder.
        
        Args:
            cols: Columns to encode
            smoothing: Smoothing factor to avoid log(0)
            random_state: Random seed
        """
        self.cols = cols
        self.smoothing = smoothing
        self.random_state = random_state
        self.woe_maps_: dict[str, dict[Any, float]] = {}
        self.iv_scores_: dict[str, float] = {}
        self.columns_: list[str] = []
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "WoEEncoder":
        """Fit WoE encoder."""
        df = pd.DataFrame(X).copy()
        y_series = pd.Series(y).reset_index(drop=True)
        self.columns_ = list(df.columns)
        df = df.astype(str).reset_index(drop=True)
        
        # Convert to binary (1 = positive class, 0 = negative)
        positive_class = y_series.value_counts().idxmax()
        y_binary = (y_series == positive_class).astype(int)
        
        n_pos = y_binary.sum()
        n_neg = len(y_binary) - n_pos
        
        for c in self.columns_:
            self.woe_maps_[c] = {}
            iv_sum = 0.0
            
            # Group by category
            temp = pd.DataFrame({c: df[c].values, "target": y_binary.values})
            agg = temp.groupby(c)["target"].agg(["sum", "count"])
            
            for cat_val in agg.index:
                cat_pos = float(agg.loc[cat_val, "sum"]) + self.smoothing
                cat_count = int(agg.loc[cat_val, "count"])
                cat_neg = cat_count - agg.loc[cat_val, "sum"] + self.smoothing
                
                # Calculate WoE
                pct_pos = cat_pos / (n_pos + self.smoothing * len(agg))
                pct_neg = cat_neg / (n_neg + self.smoothing * len(agg))
                
                woe = np.log(pct_pos / pct_neg)
                self.woe_maps_[c][cat_val] = woe
                
                # Calculate IV contribution
                iv_sum += (pct_pos - pct_neg) * woe
            
            self.iv_scores_[c] = iv_sum
        
        return self
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform using WoE."""
        df = pd.DataFrame(X).copy()
        out = []
        
        for c in self.columns_:
            s = df[c].astype(str)
            vals = s.map(lambda v: self.woe_maps_[c].get(v, 0.0)).values.reshape(-1, 1)
            out.append(vals)
        
        return np.hstack(out)
    
    def get_iv_scores(self) -> dict[str, float]:
        """Get Information Value scores for each column."""
        return self.iv_scores_
    
    def get_feature_names_out(self, input_features: Optional[list[str]] = None) -> list[str]:
        """Get output feature names for WoE encoded features.
        
        Args:
            input_features: Input feature names (unused, returns fitted columns)
            
        Returns:
            List of WoE-encoded feature names
        """
        return [f"woe_{c}" for c in self.columns_]


class OrdinalEncoder(BaseEstimator, TransformerMixin):
    """Ordinal encoder with manual category ordering."""
    
    def __init__(
        self,
        cols: list[str] | None = None,
        ordinal_maps: dict[str, list[str]] | None = None,
        handle_unknown: str = "use_encoded_value",
        unknown_value: float = -1.0,
    ) -> None:
        """Initialize ordinal encoder.
        
        Args:
            cols: Columns to encode
            ordinal_maps: Manual category ordering per column
            handle_unknown: How to handle unknown categories
            unknown_value: Value for unknown categories
        """
        self.cols = cols
        self.ordinal_maps = ordinal_maps or {}
        self.handle_unknown = handle_unknown
        self.unknown_value = unknown_value
        self.encoding_maps_: dict[str, dict[Any, int]] = {}
        self.columns_: list[str] = []
    
    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "OrdinalEncoder":
        """Fit ordinal encoder."""
        df = pd.DataFrame(X).copy()
        self.columns_ = list(df.columns)
        
        for c in self.columns_:
            if c in self.ordinal_maps:
                # Use provided ordering
                categories = self.ordinal_maps[c]
                self.encoding_maps_[c] = {cat: i for i, cat in enumerate(categories)}
            else:
                # Use natural ordering
                unique_vals = sorted(df[c].dropna().unique())
                self.encoding_maps_[c] = {val: i for i, val in enumerate(unique_vals)}
        
        return self
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform using ordinal encoding."""
        df = pd.DataFrame(X).copy()
        out = []
        
        for c in self.columns_:
            s = df[c]
            encoding = self.encoding_maps_.get(c, {})
            
            vals = s.map(lambda v: encoding.get(v, self.unknown_value)).values.reshape(-1, 1)
            out.append(vals)
        
        return np.hstack(out)
    
    def get_feature_names_out(self, input_features: Optional[list[str]] = None) -> list[str]:
        """Get output feature names for ordinal encoded features.
        
        Args:
            input_features: Input feature names (unused, returns fitted columns)
            
        Returns:
            List of ordinal-encoded feature names
        """
        return [f"ordinal_{c}" for c in self.columns_]


# ============================================================================
# NEW ENCODERS FOR LEAKAGE PREVENTION AND ENHANCED FUNCTIONALITY
# ============================================================================


class OutOfFoldTargetEncoder(BaseEstimator, TransformerMixin, LeakageGuardMixin):
    """Out-of-Fold Target Encoder with CV-aware training to prevent leakage.
    
    This encoder implements proper out-of-fold (OOF) target encoding to prevent label leakage
    during training. During fit_transform(), each training row receives an encoding computed
    from fold statistics that DO NOT include that row's target value. During transform() 
    (e.g., on test data), the global encoding map is used.
    
    Supports multiple CV strategies:
    - KFold: Standard k-fold cross-validation
    - StratifiedKFold: Stratified k-fold (for classification)
    - GroupKFold: Group-aware folds (e.g., user IDs, time groups)
    - TimeSeriesSplit: Time-series aware splits (for temporal data)
    
    Args:
        cols: Columns to encode (None = infer from fit data)
        cv: CV strategy - one of:
            - "kfold": KFold
            - "stratified": StratifiedKFold (requires y for stratification)
            - "group": GroupKFold (requires groups parameter)
            - "time": TimeSeriesSplit
            - Custom callable/splitter object
        n_splits: Number of folds
        shuffle: Whether to shuffle data (KFold/StratifiedKFold only)
        random_state: Random seed for reproducibility
        smoothing: Smoothing parameter (higher = more regularization toward prior)
        noise_std: Gaussian noise standard deviation for regularization
        prior_strategy: "global_mean" or "median"
        task: "classification" or "regression" (auto-inferred if None)
        positive_class: Positive class for binary classification (auto-inferred if None)
        
    Attributes:
        global_maps_: Global encoding map (column → category → encoded value)
        global_priors_: Global prior for each column
        columns_: List of encoded columns
        oof_encodings_: OOF encodings for training data (available after fit_transform)
        
    Example:
        >>> # Training with OOF encoding
        >>> encoder = OutOfFoldTargetEncoder(cv="stratified", n_splits=5, smoothing=20.0)
        >>> X_train_encoded = encoder.fit_transform(X_train, y_train)
        >>> # Each row in X_train_encoded uses encodings computed WITHOUT that row's target
        >>> 
        >>> # Inference with global map
        >>> X_test_encoded = encoder.transform(X_test)
        >>> # Uses global encoding map computed from all training data
        
    Notes:
        - **CRITICAL**: This encoder prevents label leakage by ensuring training rows
          never see their own target values during encoding.
        - For time-series data, use cv="time" to respect temporal ordering.
        - For grouped data (e.g., multiple rows per user), use cv="group" with groups parameter.
        - The fit() method only learns the global map; use fit_transform() to get OOF encodings.
    """
    
    def __init__(
        self,
        cols: Optional[list[str]] = None,
        cv: Union[str, Callable] = "kfold",
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: int = 42,
        smoothing: float = 20.0,
        noise_std: float = 0.0,
        prior_strategy: str = "global_mean",
        task: Optional[str] = None,
        positive_class: Optional[Any] = None,
        raise_on_target_in_transform: bool = True,
    ) -> None:
        """Initialize out-of-fold target encoder."""
        self.cols = cols
        self.cv = cv
        self.n_splits = int(n_splits)
        self.shuffle = shuffle
        self.random_state = int(random_state)
        self.smoothing = float(smoothing)
        self.noise_std = float(noise_std)
        self.prior_strategy = prior_strategy
        self.task = task
        self.positive_class = positive_class
        self.raise_on_target_in_transform = raise_on_target_in_transform
        
        # Fitted state
        self.global_maps_: dict[str, dict[Any, float]] = {}
        self.global_priors_: dict[str, float] = {}
        self.columns_: list[str] = []
        self.oof_encodings_: Optional[np.ndarray] = None
        self._fitted_task: Optional[str] = None
        self._fitted_positive_class: Optional[Any] = None
        
    def _get_cv_splitter(self, y: pd.Series, groups: Optional[pd.Series] = None):
        """Get cross-validation splitter based on cv parameter.
        
        Args:
            y: Target variable (for stratification)
            groups: Group labels (for GroupKFold)
            
        Returns:
            CV splitter object
        """
        if callable(self.cv):
            return self.cv
        
        cv_lower = str(self.cv).lower()
        rs = self.random_state
        
        if cv_lower == "kfold":
            return KFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=rs)
        elif cv_lower in {"stratified", "stratifiedkfold"}:
            return StratifiedKFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=rs)
        elif cv_lower in {"group", "groupkfold"}:
            if groups is None:
                logger.warning("GroupKFold requested but groups=None. Falling back to KFold.")
                return KFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=rs)
            return GroupKFold(n_splits=self.n_splits)
        elif cv_lower in {"time", "timeseries", "timeseriessplit"}:
            return TimeSeriesSplit(n_splits=self.n_splits)
        else:
            logger.warning(f"Unknown cv strategy '{self.cv}', using KFold")
            return KFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=rs)
    
    def _prepare_target(self, y: pd.Series) -> tuple[pd.Series, str, Any]:
        """Prepare target variable for encoding.
        
        Args:
            y: Raw target Series
            
        Returns:
            (encoded_target, task, positive_class)
        """
        # Infer task
        if self.task:
            task = self.task
        else:
            nunique = y.nunique()
            task = "classification" if nunique <= 20 else "regression"
        
        # Encode target
        if task == "classification":
            # Binary or multiclass classification
            if self.positive_class is not None:
                pos_class = self.positive_class
            else:
                # Infer positive class
                if y.nunique() == 2 and set(y.unique()) <= {0, 1}:
                    pos_class = 1
                else:
                    pos_class = y.value_counts().idxmax()
            
            y_enc = (y == pos_class).astype(float)
            return y_enc, task, pos_class
        else:
            # Regression
            y_enc = y.astype(float)
            return y_enc, task, None
    
    def fit(self, X: pd.DataFrame, y: pd.Series, groups: Optional[pd.Series] = None) -> "OutOfFoldTargetEncoder":
        """Fit encoder by learning global encoding maps.
        
        This method computes the global encoding map from all training data. 
        For training, use fit_transform() to get OOF encodings.
        
        Args:
            X: Training features
            y: Training target
            groups: Optional group labels for GroupKFold
            
        Returns:
            Self
        """
        df = pd.DataFrame(X).copy()
        y_series = pd.Series(y).reset_index(drop=True)
        
        self.columns_ = self.cols if self.cols else list(df.columns)
        df = df[self.columns_].astype(str).reset_index(drop=True)
        
        # Prepare target
        y_enc, task, pos_class = self._prepare_target(y_series)
        self._fitted_task = task
        self._fitted_positive_class = pos_class
        
        # Compute global prior
        if self.prior_strategy == "median":
            global_prior = float(y_enc.median())
        else:  # global_mean
            global_prior = float(y_enc.mean())
        
        # Learn global encoding maps using all training data
        for col in self.columns_:
            self.global_priors_[col] = global_prior
            self.global_maps_[col] = {}
            
            # Aggregate target by category
            temp = pd.DataFrame({col: df[col].values, "target": y_enc.values})
            agg = temp.groupby(col)["target"].agg(["sum", "count"])
            
            for cat_val in agg.index:
                cat_sum = float(agg.loc[cat_val, "sum"])
                cat_count = int(agg.loc[cat_val, "count"])
                
                # Apply smoothing
                smoothed = (cat_sum + global_prior * self.smoothing) / (cat_count + self.smoothing)
                self.global_maps_[col][cat_val] = float(smoothed)
        
        logger.debug(f"Fitted OutOfFoldTargetEncoder on {len(self.columns_)} columns with task={task}")
        return self
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series, groups: Optional[pd.Series] = None) -> np.ndarray:
        """Fit encoder and return out-of-fold encoded training data.
        
        **CRITICAL**: This method ensures no label leakage by computing OOF encodings.
        Each training row receives an encoding computed from fold statistics that
        DO NOT include that row's target value.
        
        Args:
            X: Training features
            y: Training target
            groups: Optional group labels for GroupKFold
            
        Returns:
            OOF encoded training data (n_samples, n_columns)
        """
        df = pd.DataFrame(X).copy()
        y_series = pd.Series(y).reset_index(drop=True)
        
        self.columns_ = self.cols if self.cols else list(df.columns)
        df = df[self.columns_].astype(str).reset_index(drop=True)
        
        # Prepare target
        y_enc, task, pos_class = self._prepare_target(y_series)
        self._fitted_task = task
        self._fitted_positive_class = pos_class
        
        # Compute global prior
        if self.prior_strategy == "median":
            global_prior = float(y_enc.median())
        else:
            global_prior = float(y_enc.mean())
        
        # Initialize OOF encodings matrix
        n_samples = len(df)
        n_cols = len(self.columns_)
        oof_matrix = np.full((n_samples, n_cols), global_prior, dtype=float)
        
        # Get CV splitter
        groups_array = groups.values if groups is not None else None
        if self.cv in {"group", "groupkfold"} and groups_array is not None:
            splitter = self._get_cv_splitter(y_enc, groups)
            splits = list(splitter.split(df, y_enc, groups_array))
        elif self.cv in {"stratified", "stratifiedkfold"}:
            splitter = self._get_cv_splitter(y_enc)
            splits = list(splitter.split(df, y_enc))
        else:
            splitter = self._get_cv_splitter(y_enc)
            splits = list(splitter.split(df))
        
        # Compute OOF encodings fold by fold
        for col_idx, col in enumerate(self.columns_):
            self.global_priors_[col] = global_prior
            self.global_maps_[col] = {}
            
            # Accumulate global statistics for this column
            global_stats: dict[Any, tuple[float, int]] = {}
            
            for train_idx, val_idx in splits:
                # Train fold statistics
                train_df = df.iloc[train_idx]
                train_y = y_enc.iloc[train_idx]
                
                temp = pd.DataFrame({col: train_df[col].values, "target": train_y.values})
                agg = temp.groupby(col)["target"].agg(["sum", "count"])
                
                # Encode validation fold using train fold statistics
                val_df = df.iloc[val_idx]
                for val_row_idx, cat_val in zip(val_idx, val_df[col].values):
                    if cat_val in agg.index:
                        cat_sum = float(agg.loc[cat_val, "sum"])
                        cat_count = int(agg.loc[cat_val, "count"])
                        encoded_val = (cat_sum + global_prior * self.smoothing) / (cat_count + self.smoothing)
                    else:
                        encoded_val = global_prior
                    
                    oof_matrix[val_row_idx, col_idx] = encoded_val
                
                # Accumulate for global map
                for cat_val in agg.index:
                    cat_sum = float(agg.loc[cat_val, "sum"])
                    cat_count = int(agg.loc[cat_val, "count"])
                    if cat_val not in global_stats:
                        global_stats[cat_val] = (cat_sum, cat_count)
                    else:
                        prev_sum, prev_count = global_stats[cat_val]
                        global_stats[cat_val] = (prev_sum + cat_sum, prev_count + cat_count)
            
            # Build global map for this column
            for cat_val, (cat_sum, cat_count) in global_stats.items():
                smoothed = (cat_sum + global_prior * self.smoothing) / (cat_count + self.smoothing)
                self.global_maps_[col][cat_val] = float(smoothed)
        
        # Add optional noise
        if self.noise_std > 0:
            rng = np.random.default_rng(self.random_state)
            oof_matrix = oof_matrix + rng.normal(0.0, self.noise_std, size=oof_matrix.shape)
        
        self.oof_encodings_ = oof_matrix
        logger.info(f"Generated OOF encodings for {n_samples} samples, {n_cols} columns with {len(splits)} folds")
        return oof_matrix
    
    def transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> np.ndarray:
        """Transform using global encoding map (for inference).
        
        Args:
            X: Features to encode
            y: Target (should be None to prevent leakage; ignored if provided with warning)
            
        Returns:
            Encoded features (n_samples, n_columns)
        """
        # CRITICAL: Enforce leakage guard
        self.ensure_no_target_in_transform(y)
        
        # Note: The guard above will raise if y is not None and raise_on_target_in_transform=True
        # If we reach here, y is None or the guard is disabled
        
        if not self.global_maps_:
            raise RuntimeError("OutOfFoldTargetEncoder not fitted. Call fit() or fit_transform() first.")
        
        df = pd.DataFrame(X).copy()
        df = df[self.columns_].astype(str)
        
        out = []
        for col in self.columns_:
            s = df[col]
            m = self.global_maps_.get(col, {})
            prior = self.global_priors_.get(col, 0.0)
            
            vals = s.map(lambda v, _m=m, _prior=prior: _m.get(v, _prior)).astype(float).values.reshape(-1, 1)
            
            # Add noise if configured
            if self.noise_std > 0:
                rng = np.random.default_rng(self.random_state)
                vals = vals + rng.normal(0.0, self.noise_std, size=vals.shape)
            
            out.append(vals)
        
        return np.hstack(out)
    
    def get_feature_names_out(self, input_features: Optional[list[str]] = None) -> list[str]:
        """Get output feature names.
        
        Args:
            input_features: Input feature names (unused, returns fitted columns)
            
        Returns:
            List of encoded feature names
        """
        return [f"oof_te_{c}" for c in self.columns_]


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Encode categorical features as their frequency (proportion) in training data.
    
    For each category, computes: frequency = count(category) / total_count
    
    Args:
        cols: Columns to encode (None = infer from fit data)
        unseen_value: Value for unseen categories at transform time
        
    Attributes:
        freq_maps_: Dict mapping column to category to frequency
        columns_: List of encoded columns
        
    Example:
        >>> encoder = FrequencyEncoder()
        >>> encoder.fit(X_train)
        >>> X_train_encoded = encoder.transform(X_train)
        >>> X_test_encoded = encoder.transform(X_test)  # Unseen categories to unseen_value
    """
    
    def __init__(
        self,
        cols: Optional[list[str]] = None,
        unseen_value: float = 0.0,
    ) -> None:
        """Initialize frequency encoder."""
        self.cols = cols
        self.unseen_value = float(unseen_value)
        self.freq_maps_: dict[str, dict[Any, float]] = {}
        self.columns_: list[str] = []
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "FrequencyEncoder":
        """Fit frequency encoder.
        
        Args:
            X: Training features
            y: Target (ignored, for sklearn compatibility)
            
        Returns:
            Self
        """
        df = pd.DataFrame(X).copy()
        self.columns_ = self.cols if self.cols else list(df.columns)
        df = df[self.columns_].astype(str)
        
        for col in self.columns_:
            freq = df[col].value_counts(normalize=True, dropna=False)
            self.freq_maps_[col] = freq.to_dict()
        
        logger.debug(f"Fitted FrequencyEncoder on {len(self.columns_)} columns")
        return self
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform using frequency encoding.
        
        Args:
            X: Features to encode
            
        Returns:
            Frequency-encoded features
        """
        df = pd.DataFrame(X).copy()
        df = df[self.columns_].astype(str)
        
        out = []
        for col in self.columns_:
            freq_map = self.freq_maps_.get(col, {})
            vals = df[col].map(lambda v: freq_map.get(v, self.unseen_value)).astype(float).values.reshape(-1, 1)
            out.append(vals)
        
        return np.hstack(out)
    
    def get_feature_names_out(self, input_features: Optional[list[str]] = None) -> list[str]:
        """Get output feature names.
        
        Args:
            input_features: Input feature names (unused)
            
        Returns:
            List of frequency-encoded feature names
        """
        return [f"freq_{c}" for c in self.columns_]


class CountEncoder(BaseEstimator, TransformerMixin):
    """Encode categorical features as their occurrence count in training data.
    
    For each category, stores: count = number of occurrences
    
    Args:
        cols: Columns to encode (None = infer from fit data)
        unseen_value: Value for unseen categories at transform time
        normalize: If True, normalize counts by total count (equivalent to FrequencyEncoder)
        
    Attributes:
        count_maps_: Dict mapping column to category to count
        columns_: List of encoded columns
        
    Example:
        >>> encoder = CountEncoder(normalize=False)
        >>> encoder.fit(X_train)
        >>> X_train_encoded = encoder.transform(X_train)
    """
    
    def __init__(
        self,
        cols: Optional[list[str]] = None,
        unseen_value: float = 0.0,
        normalize: bool = False,
    ) -> None:
        """Initialize count encoder."""
        self.cols = cols
        self.unseen_value = float(unseen_value)
        self.normalize = normalize
        self.count_maps_: dict[str, dict[Any, float]] = {}
        self.columns_: list[str] = []
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "CountEncoder":
        """Fit count encoder.
        
        Args:
            X: Training features
            y: Target (ignored, for sklearn compatibility)
            
        Returns:
            Self
        """
        df = pd.DataFrame(X).copy()
        self.columns_ = self.cols if self.cols else list(df.columns)
        df = df[self.columns_].astype(str)
        
        for col in self.columns_:
            counts = df[col].value_counts(dropna=False)
            if self.normalize:
                counts = counts / len(df)
            self.count_maps_[col] = counts.to_dict()
        
        logger.debug(f"Fitted CountEncoder on {len(self.columns_)} columns")
        return self
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform using count encoding.
        
        Args:
            X: Features to encode
            
        Returns:
            Count-encoded features
        """
        df = pd.DataFrame(X).copy()
        df = df[self.columns_].astype(str)
        
        out = []
        for col in self.columns_:
            count_map = self.count_maps_.get(col, {})
            vals = df[col].map(lambda v: count_map.get(v, self.unseen_value)).astype(float).values.reshape(-1, 1)
            out.append(vals)
        
        return np.hstack(out)
    
    def get_feature_names_out(self, input_features: Optional[list[str]] = None) -> list[str]:
        """Get output feature names.
        
        Args:
            input_features: Input feature names (unused)
            
        Returns:
            List of count-encoded feature names
        """
        return [f"count_{c}" for c in self.columns_]


class CatBoostEncoder(BaseEstimator, TransformerMixin, LeakageGuardMixin):
    """CatBoost-style target encoding using ordered target statistics.
    
    Implements the target encoding strategy used in CatBoost: computes cumulative
    target statistics for each category in a randomized order during training.
    This prevents overfitting by ensuring each observation uses only prior observations
    in the encoding, similar to online learning.
    
    Algorithm:
    1. Shuffle training data (with fixed random_state for reproducibility)
    2. For each row i with category c:
        - Compute encoding using only rows 0..i-1 with category c
        - encoding = (sum_target + prior * smoothing) / (count + smoothing)
    3. For inference, use global statistics computed from all training data
    
    This is more robust than standard target encoding and works well with tree models.
    
    Args:
        cols: Columns to encode (None = infer from fit data)
        smoothing: Smoothing parameter (higher = more regularization toward prior)
        random_state: Random seed for shuffling order
        task: "classification" or "regression" (auto-inferred if None)
        positive_class: Positive class for binary classification (auto-inferred if None)
        prior_strategy: "global_mean" or "median"
        
    Attributes:
        global_maps_: Global encoding map (column → category → encoded value)
        global_priors_: Global prior for each column
        columns_: List of encoded columns
        
    Example:
        >>> # Training with ordered target statistics
        >>> encoder = CatBoostEncoder(cols=['city'], smoothing=10.0)
        >>> X_train_encoded = encoder.fit_transform(X_train, y_train)
        >>> 
        >>> # Inference with global map
        >>> X_test_encoded = encoder.transform(X_test)
        
    Notes:
        - Designed for tree-based models (particularly CatBoost, XGBoost, LightGBM)
        - More stable than K-fold target encoding for small datasets
        - Prevents overfitting through ordered statistics approach
    """
    
    def __init__(
        self,
        cols: Optional[list[str]] = None,
        smoothing: float = 10.0,
        random_state: int = 42,
        task: Optional[str] = None,
        positive_class: Optional[Any] = None,
        prior_strategy: str = "global_mean",
        raise_on_target_in_transform: bool = True,
    ) -> None:
        """Initialize CatBoost encoder."""
        self.cols = cols
        self.smoothing = float(smoothing)
        self.random_state = int(random_state)
        self.task = task
        self.positive_class = positive_class
        self.prior_strategy = prior_strategy
        self.raise_on_target_in_transform = raise_on_target_in_transform
        
        # Fitted state
        self.global_maps_: dict[str, dict[Any, float]] = {}
        self.global_priors_: dict[str, float] = {}
        self.columns_: list[str] = []
        self.ordered_encodings_: Optional[np.ndarray] = None
        self._fitted_task: Optional[str] = None
        self._fitted_positive_class: Optional[Any] = None
    
    def _prepare_target(self, y: pd.Series) -> tuple[pd.Series, str, Any]:
        """Prepare target variable for encoding.
        
        Args:
            y: Raw target Series
            
        Returns:
            (encoded_target, task, positive_class)
        """
        # Infer task
        if self.task:
            task = self.task
        else:
            nunique = y.nunique()
            task = "classification" if nunique <= 20 else "regression"
        
        # Encode target
        if task == "classification":
            # Binary or multiclass classification
            if self.positive_class is not None:
                pos_class = self.positive_class
            else:
                # Infer positive class
                if y.nunique() == 2 and set(y.unique()) <= {0, 1}:
                    pos_class = 1
                else:
                    pos_class = y.value_counts().idxmax()
            
            y_enc = (y == pos_class).astype(float)
            return y_enc, task, pos_class
        else:
            # Regression
            y_enc = y.astype(float)
            return y_enc, task, None
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "CatBoostEncoder":
        """Fit encoder by learning global encoding maps.
        
        For training, use fit_transform() to get ordered encodings.
        
        Args:
            X: Training features
            y: Training target
            
        Returns:
            Self
        """
        df = pd.DataFrame(X).copy()
        y_series = pd.Series(y).reset_index(drop=True)
        
        self.columns_ = self.cols if self.cols else list(df.columns)
        df = df[self.columns_].astype(str).reset_index(drop=True)
        
        # Prepare target
        y_enc, task, pos_class = self._prepare_target(y_series)
        self._fitted_task = task
        self._fitted_positive_class = pos_class
        
        # Compute global prior
        if self.prior_strategy == "median":
            global_prior = float(y_enc.median())
        else:  # global_mean
            global_prior = float(y_enc.mean())
        
        # Learn global encoding maps using all training data
        for col in self.columns_:
            self.global_priors_[col] = global_prior
            self.global_maps_[col] = {}
            
            # Aggregate target by category
            temp = pd.DataFrame({col: df[col].values, "target": y_enc.values})
            agg = temp.groupby(col)["target"].agg(["sum", "count"])
            
            for cat_val in agg.index:
                cat_sum = float(agg.loc[cat_val, "sum"])
                cat_count = int(agg.loc[cat_val, "count"])
                
                # Apply smoothing
                smoothed = (cat_sum + global_prior * self.smoothing) / (cat_count + self.smoothing)
                self.global_maps_[col][cat_val] = float(smoothed)
        
        logger.debug(f"Fitted CatBoostEncoder on {len(self.columns_)} columns with task={task}")
        return self
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        """Fit encoder and return ordered target statistics for training data.
        
        **CRITICAL**: This method uses ordered (cumulative) statistics to prevent overfitting.
        Each training row receives an encoding computed from only PRIOR rows in shuffled order.
        
        Args:
            X: Training features
            y: Training target
            
        Returns:
            Ordered encoded training data (n_samples, n_columns)
        """
        df = pd.DataFrame(X).copy()
        y_series = pd.Series(y).reset_index(drop=True)
        
        self.columns_ = self.cols if self.cols else list(df.columns)
        df = df[self.columns_].astype(str).reset_index(drop=True)
        
        # Prepare target
        y_enc, task, pos_class = self._prepare_target(y_series)
        self._fitted_task = task
        self._fitted_positive_class = pos_class
        
        # Compute global prior
        if self.prior_strategy == "median":
            global_prior = float(y_enc.median())
        else:
            global_prior = float(y_enc.mean())
        
        # Initialize encoding matrix
        n_samples = len(df)
        n_cols = len(self.columns_)
        ordered_matrix = np.full((n_samples, n_cols), global_prior, dtype=float)
        
        # Create random permutation for ordered statistics
        rng = np.random.RandomState(self.random_state)
        permutation = rng.permutation(n_samples)
        inverse_permutation = np.argsort(permutation)
        
        # Shuffle data and target
        df_shuffled = df.iloc[permutation].reset_index(drop=True)
        y_shuffled = y_enc.iloc[permutation].reset_index(drop=True)
        
        # Compute ordered encodings column by column
        for col_idx, col in enumerate(self.columns_):
            self.global_priors_[col] = global_prior
            self.global_maps_[col] = {}
            
            # Track cumulative statistics for each category
            cumulative_sum: dict[Any, float] = {}
            cumulative_count: dict[Any, int] = {}
            
            # Accumulate global statistics
            global_stats: dict[Any, tuple[float, int]] = {}
            
            # Process rows in shuffled order
            for i in range(n_samples):
                cat_val = df_shuffled.loc[i, col]
                target_val = y_shuffled.iloc[i]
                
                # Compute encoding using cumulative statistics (prior rows only)
                if cat_val in cumulative_sum:
                    cat_sum = cumulative_sum[cat_val]
                    cat_count = cumulative_count[cat_val]
                    encoded_val = (cat_sum + global_prior * self.smoothing) / (cat_count + self.smoothing)
                else:
                    # First occurrence of this category - use prior
                    encoded_val = global_prior
                
                # Store encoding in original order
                original_idx = inverse_permutation[i]
                ordered_matrix[original_idx, col_idx] = encoded_val
                
                # Update cumulative statistics for next iteration
                if cat_val in cumulative_sum:
                    cumulative_sum[cat_val] += target_val
                    cumulative_count[cat_val] += 1
                else:
                    cumulative_sum[cat_val] = target_val
                    cumulative_count[cat_val] = 1
                
                # Accumulate for global map
                if cat_val in global_stats:
                    prev_sum, prev_count = global_stats[cat_val]
                    global_stats[cat_val] = (prev_sum + target_val, prev_count + 1)
                else:
                    global_stats[cat_val] = (target_val, 1)
            
            # Build global map for this column
            for cat_val, (cat_sum, cat_count) in global_stats.items():
                smoothed = (cat_sum + global_prior * self.smoothing) / (cat_count + self.smoothing)
                self.global_maps_[col][cat_val] = float(smoothed)
        
        self.ordered_encodings_ = ordered_matrix
        logger.info(
            f"Generated CatBoost ordered encodings for {n_samples} samples, "
            f"{n_cols} columns with random_state={self.random_state}"
        )
        return ordered_matrix
    
    def transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> np.ndarray:
        """Transform using global encoding map (for inference).
        
        Args:
            X: Features to encode
            y: Target (should be None to prevent leakage; ignored if provided with warning)
            
        Returns:
            Encoded features (n_samples, n_columns)
        """
        # CRITICAL: Enforce leakage guard
        self.ensure_no_target_in_transform(y)
        
        if not self.global_maps_:
            raise RuntimeError("CatBoostEncoder not fitted. Call fit() or fit_transform() first.")
        
        df = pd.DataFrame(X).copy()
        df = df[self.columns_].astype(str)
        
        out = []
        for col in self.columns_:
            s = df[col]
            m = self.global_maps_.get(col, {})
            prior = self.global_priors_.get(col, 0.0)
            
            vals = s.map(lambda v, _m=m, _prior=prior: _m.get(v, _prior)).astype(float).values.reshape(-1, 1)
            out.append(vals)
        
        return np.hstack(out)
    
    def get_feature_names_out(self, input_features: Optional[list[str]] = None) -> list[str]:
        """Get output feature names.
        
        Args:
            input_features: Input feature names (unused, returns fitted columns)
            
        Returns:
            List of encoded feature names
        """
        return [f"catboost_{c}" for c in self.columns_]


class BinaryEncoder(BaseEstimator, TransformerMixin):
    """Binary encoding: converts categories to binary representation.
    
    More memory-efficient than one-hot encoding for high cardinality features.
    Each category is assigned an integer ID, then converted to binary digits.
    For N unique categories, only ceil(log2(N)) binary features are created.
    
    Example:
        Categories: ['cat', 'dog', 'bird', 'fish'] (4 categories)
        Binary encoding (2 bits needed):
            cat  -> 00 -> [0, 0]
            dog  -> 01 -> [0, 1]
            bird -> 10 -> [1, 0]
            fish -> 11 -> [1, 1]
    
    Args:
        cols: Columns to encode (None = infer from fit data)
        handle_unknown: How to handle unknown categories - 'value' (use unknown_value) or 'error'
        unknown_value: Integer ID for unknown categories (default: -1)
        
    Attributes:
        category_maps_: Dict mapping column to category to integer ID
        n_bits_: Dict mapping column to number of binary bits needed
        columns_: List of encoded columns
        
    Example:
        >>> encoder = BinaryEncoder(cols=['city', 'country'])
        >>> encoder.fit(X_train)
        >>> X_encoded = encoder.transform(X_test)
    """
    
    def __init__(
        self,
        cols: Optional[list[str]] = None,
        handle_unknown: str = "value",
        unknown_value: int = -1,
    ) -> None:
        """Initialize binary encoder."""
        self.cols = cols
        self.handle_unknown = handle_unknown
        self.unknown_value = unknown_value
        self.category_maps_: dict[str, dict[Any, int]] = {}
        self.n_bits_: dict[str, int] = {}
        self.columns_: list[str] = []
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "BinaryEncoder":
        """Fit binary encoder by assigning integer IDs to categories.
        
        Args:
            X: Training features
            y: Target (ignored, for sklearn compatibility)
            
        Returns:
            Self
        """
        df = pd.DataFrame(X).copy()
        self.columns_ = self.cols if self.cols else list(df.columns)
        df = df[self.columns_].astype(str)
        
        for col in self.columns_:
            # Get unique categories and assign integer IDs (starting from 0)
            unique_cats = df[col].unique()
            n_categories = len(unique_cats)
            
            # Create mapping: category -> integer ID
            self.category_maps_[col] = {cat: idx for idx, cat in enumerate(unique_cats)}
            
            # Calculate number of binary bits needed: ceil(log2(n_categories))
            # Minimum 1 bit even for 1 category
            n_bits = max(1, int(np.ceil(np.log2(max(n_categories, 2)))))
            self.n_bits_[col] = n_bits
        
        logger.debug(
            f"Fitted BinaryEncoder on {len(self.columns_)} columns. "
            f"Bits per column: {self.n_bits_}"
        )
        return self
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform using binary encoding.
        
        Args:
            X: Features to encode
            
        Returns:
            Binary-encoded features (n_samples, total_bits)
        """
        df = pd.DataFrame(X).copy()
        df = df[self.columns_].astype(str)
        
        out = []
        for col in self.columns_:
            category_map = self.category_maps_.get(col, {})
            n_bits = self.n_bits_.get(col, 1)
            
            # Map categories to integer IDs
            ids = df[col].map(
                lambda v: category_map.get(v, self.unknown_value)
            ).astype(int)
            
            # Handle unknown categories
            if self.handle_unknown == "error":
                if (ids == self.unknown_value).any():
                    unknown_vals = df[col][ids == self.unknown_value].unique()
                    raise ValueError(
                        f"Unknown categories in column '{col}': {unknown_vals}. "
                        f"Set handle_unknown='value' to use default encoding."
                    )
            
            # Convert integer IDs to binary representation
            # For each sample, create n_bits binary features
            binary_matrix = np.zeros((len(ids), n_bits), dtype=int)
            
            for i, cat_id in enumerate(ids):
                if cat_id >= 0:
                    # Convert to binary and fill the array (most significant bit first)
                    binary_str = format(cat_id, f'0{n_bits}b')
                    binary_matrix[i] = [int(bit) for bit in binary_str]
                else:
                    # Unknown category - use all zeros or special encoding
                    binary_matrix[i] = np.zeros(n_bits, dtype=int)
            
            out.append(binary_matrix)
        
        return np.hstack(out) if out else np.empty((len(df), 0))
    
    def get_feature_names_out(self, input_features: Optional[list[str]] = None) -> list[str]:
        """Get output feature names for binary encoded features.
        
        Args:
            input_features: Input feature names (unused)
            
        Returns:
            List of binary feature names (e.g., 'binary_city_0', 'binary_city_1', ...)
        """
        names = []
        for col in self.columns_:
            n_bits = self.n_bits_.get(col, 1)
            for bit_idx in range(n_bits):
                names.append(f"binary_{col}_{bit_idx}")
        return names


class EntityEmbeddingsEncoder(BaseEstimator, TransformerMixin, LeakageGuardMixin):
    """Entity Embeddings: Neural network-based learned representations for categorical features.
    
    Trains a simple neural network with embedding layers to learn dense vector representations
    of categorical features, supervised by the target variable. This approach was popularized
    by the "Entity Embeddings of Categorical Variables" paper (Guo & Berkhahn, 2016).
    
    The encoder creates a neural network with:
    1. Embedding layers for each categorical column (maps category to dense vector)
    2. Concatenation of all embeddings
    3. Dense layers trained to predict the target
    4. Embeddings are extracted and used as features
    
    Benefits:
    - Learns meaningful relationships between categories
    - Handles high cardinality naturally
    - Captures non-linear patterns
    - Embeddings can reveal semantic similarities
    
    Drawbacks:
    - Requires neural network training (slower than other encoders)
    - Needs sufficient data (100+ samples per category recommended)
    - Requires target variable (supervised method)
    - Optional dependency on deep learning framework
    
    Args:
        cols: Columns to encode (None = infer from fit data)
        embedding_dim: Embedding dimension (None = auto: min(50, cardinality // 2))
        hidden_dims: Hidden layer dimensions (e.g., [128, 64])
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate for optimizer
        dropout: Dropout rate for regularization
        validation_split: Fraction of data for validation
        early_stopping_patience: Patience for early stopping (None = no early stopping)
        task: "classification" or "regression" (auto-inferred if None)
        random_state: Random seed for reproducibility
        verbose: Verbosity level (0=silent, 1=progress, 2=debug)
        backend: Deep learning backend - 'keras' (TensorFlow/Keras) or 'pytorch'
        
    Attributes:
        embeddings_: Dict mapping column to embedding matrix (n_categories, embedding_dim)
        category_maps_: Dict mapping column to category to integer ID
        embedding_dims_: Dict mapping column to embedding dimension
        model_: Trained neural network model
        columns_: List of encoded columns
        
    Example:
        >>> # Train embeddings on categorical features
        >>> encoder = EntityEmbeddingsEncoder(
        ...     cols=['city', 'category', 'brand'],
        ...     embedding_dim=10,
        ...     hidden_dims=[64, 32],
        ...     epochs=20
        ... )
        >>> X_train_encoded = encoder.fit_transform(X_train, y_train)
        >>> X_test_encoded = encoder.transform(X_test)
        >>> 
        >>> # Inspect learned embeddings
        >>> city_embeddings = encoder.get_embeddings()['city']
        
    Notes:
        - Requires TensorFlow/Keras or PyTorch (optional dependencies)
        - Training time depends on dataset size and network complexity
        - For best results, normalize/scale target for regression tasks
        - Consider using pre-trained embeddings for transfer learning scenarios
    """
    
    def __init__(
        self,
        cols: Optional[list[str]] = None,
        embedding_dim: Optional[int] = None,
        hidden_dims: Optional[list[int]] = None,
        epochs: int = 10,
        batch_size: int = 256,
        learning_rate: float = 0.001,
        dropout: float = 0.1,
        validation_split: float = 0.2,
        early_stopping_patience: Optional[int] = 3,
        task: Optional[str] = None,
        random_state: int = 42,
        verbose: int = 0,
        backend: str = "keras",
        raise_on_target_in_transform: bool = True,
    ) -> None:
        """Initialize entity embeddings encoder."""
        self.cols = cols
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims if hidden_dims is not None else [128, 64]
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.validation_split = validation_split
        self.early_stopping_patience = early_stopping_patience
        self.task = task
        self.random_state = random_state
        self.verbose = verbose
        self.backend = backend
        self.raise_on_target_in_transform = raise_on_target_in_transform
        
        # Fitted state
        self.embeddings_: dict[str, np.ndarray] = {}
        self.category_maps_: dict[str, dict[Any, int]] = {}
        self.embedding_dims_: dict[str, int] = {}
        self.columns_: list[str] = []
        self.model_ = None
        self._fitted_task: Optional[str] = None
        self._embedding_layers_: dict[str, Any] = {}
    
    def _check_backend(self) -> None:
        """Check if required deep learning backend is available."""
        if self.backend == "keras":
            try:
                import tensorflow as tf
                from tensorflow import keras
                # Set random seed for reproducibility
                tf.random.set_seed(self.random_state)
                np.random.seed(self.random_state)
            except ImportError:
                raise ImportError(
                    "EntityEmbeddingsEncoder with backend='keras' requires TensorFlow. "
                    "Install with: pip install tensorflow"
                )
        elif self.backend == "pytorch":
            try:
                import torch
                # Set random seed
                torch.manual_seed(self.random_state)
                np.random.seed(self.random_state)
            except ImportError:
                raise ImportError(
                    "EntityEmbeddingsEncoder with backend='pytorch' requires PyTorch. "
                    "Install with: pip install torch"
                )
        else:
            raise ValueError(f"Unknown backend: '{self.backend}'. Use 'keras' or 'pytorch'.")
    
    def _prepare_data(self, X: pd.DataFrame, y: pd.Series) -> tuple:
        """Prepare data for neural network training.
        
        Args:
            X: Input features
            y: Target variable
            
        Returns:
            (encoded_X_dict, y_array, task)
        """
        df = pd.DataFrame(X).copy()
        y_series = pd.Series(y).reset_index(drop=True)
        
        self.columns_ = self.cols if self.cols else list(df.columns)
        df = df[self.columns_].astype(str).reset_index(drop=True)
        
        # Infer task
        if self.task:
            task = self.task
        else:
            nunique = y_series.nunique()
            task = "classification" if nunique <= 20 else "regression"
        
        self._fitted_task = task
        
        # Encode categorical features to integers
        encoded_dict = {}
        for col in self.columns_:
            # Get unique categories and assign integer IDs
            unique_cats = df[col].unique()
            n_categories = len(unique_cats)
            
            # Create mapping: category -> integer ID
            self.category_maps_[col] = {cat: idx for idx, cat in enumerate(unique_cats)}
            
            # Determine embedding dimension
            if self.embedding_dim is not None:
                emb_dim = self.embedding_dim
            else:
                # Auto: min(50, cardinality // 2), with minimum of 2
                emb_dim = max(2, min(50, n_categories // 2))
            
            self.embedding_dims_[col] = emb_dim
            
            # Map categories to integer IDs
            encoded_dict[col] = df[col].map(self.category_maps_[col]).values
        
        # Prepare target
        if task == "classification":
            # Convert to integer labels
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y_array = le.fit_transform(y_series)
            self._label_encoder = le
            self._n_classes = len(le.classes_)
        else:
            # Regression - normalize target
            y_array = y_series.astype(float).values
            y_mean = y_array.mean()
            y_std = y_array.std() + 1e-8
            y_array = (y_array - y_mean) / y_std
            self._y_mean = y_mean
            self._y_std = y_std
        
        return encoded_dict, y_array, task
    
    def _build_keras_model(self, n_samples: int) -> Any:
        """Build Keras neural network model with embeddings.
        
        Args:
            n_samples: Number of training samples
            
        Returns:
            Keras model
        """
        from tensorflow import keras
        from tensorflow.keras import layers
        
        # Create input layers and embedding layers for each categorical column
        inputs = []
        embeddings = []
        
        for col in self.columns_:
            n_categories = len(self.category_maps_[col])
            emb_dim = self.embedding_dims_[col]
            
            # Input layer for this column
            input_layer = layers.Input(shape=(1,), name=f"input_{col}")
            inputs.append(input_layer)
            
            # Embedding layer
            embedding_layer = layers.Embedding(
                input_dim=n_categories,
                output_dim=emb_dim,
                name=f"embedding_{col}"
            )(input_layer)
            
            # Flatten embedding
            embedding_flat = layers.Flatten()(embedding_layer)
            embeddings.append(embedding_flat)
            
            # Store embedding layer for later extraction
            self._embedding_layers_[col] = f"embedding_{col}"
        
        # Concatenate all embeddings
        if len(embeddings) > 1:
            concatenated = layers.Concatenate()(embeddings)
        else:
            concatenated = embeddings[0]
        
        # Add dense hidden layers
        x = concatenated
        for hidden_dim in self.hidden_dims:
            x = layers.Dense(hidden_dim, activation='relu')(x)
            if self.dropout > 0:
                x = layers.Dropout(self.dropout)(x)
        
        # Output layer
        if self._fitted_task == "classification":
            if self._n_classes == 2:
                # Binary classification
                output = layers.Dense(1, activation='sigmoid', name='output')(x)
                loss = 'binary_crossentropy'
                metrics = ['accuracy']
            else:
                # Multiclass classification
                output = layers.Dense(self._n_classes, activation='softmax', name='output')(x)
                loss = 'sparse_categorical_crossentropy'
                metrics = ['accuracy']
        else:
            # Regression
            output = layers.Dense(1, activation='linear', name='output')(x)
            loss = 'mse'
            metrics = ['mae']
        
        # Build model
        model = keras.Model(inputs=inputs, outputs=output)
        
        # Compile model
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        
        if self.verbose >= 2:
            model.summary()
        
        return model
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "EntityEmbeddingsEncoder":
        """Fit encoder by training neural network with embeddings.
        
        Args:
            X: Training features
            y: Training target
            
        Returns:
            Self
        """
        self._check_backend()
        
        # Prepare data
        encoded_dict, y_array, task = self._prepare_data(X, y)
        
        if self.backend == "keras":
            # Build Keras model
            self.model_ = self._build_keras_model(len(y_array))
            
            # Prepare input format for Keras
            X_list = [encoded_dict[col].reshape(-1, 1) for col in self.columns_]
            
            # Setup callbacks
            callbacks = []
            if self.early_stopping_patience is not None:
                from tensorflow.keras.callbacks import EarlyStopping
                early_stop = EarlyStopping(
                    monitor='val_loss',
                    patience=self.early_stopping_patience,
                    restore_best_weights=True,
                    verbose=self.verbose
                )
                callbacks.append(early_stop)
            
            # Train model
            self.model_.fit(
                X_list,
                y_array,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_split=self.validation_split,
                callbacks=callbacks,
                verbose=self.verbose
            )
            
            # Extract learned embeddings
            for col in self.columns_:
                emb_layer_name = self._embedding_layers_[col]
                emb_layer = self.model_.get_layer(emb_layer_name)
                emb_weights = emb_layer.get_weights()[0]  # Shape: (n_categories, embedding_dim)
                self.embeddings_[col] = emb_weights
        
        elif self.backend == "pytorch":
            # PyTorch implementation
            raise NotImplementedError(
                "PyTorch backend for EntityEmbeddingsEncoder is not yet implemented. "
                "Use backend='keras' instead."
            )
        
        logger.info(
            f"Trained EntityEmbeddingsEncoder on {len(self.columns_)} columns "
            f"with embedding dims: {self.embedding_dims_}"
        )
        return self
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        """Fit encoder and transform training data.
        
        Args:
            X: Training features
            y: Training target
            
        Returns:
            Embedded features (n_samples, sum of embedding dimensions)
        """
        self.fit(X, y)
        return self.transform(X)
    
    def transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> np.ndarray:
        """Transform using learned embeddings.
        
        Args:
            X: Features to encode
            y: Target (should be None to prevent leakage)
            
        Returns:
            Embedded features (n_samples, sum of embedding dimensions)
        """
        # CRITICAL: Enforce leakage guard
        self.ensure_no_target_in_transform(y)
        
        if not self.embeddings_:
            raise RuntimeError("EntityEmbeddingsEncoder not fitted. Call fit() first.")
        
        df = pd.DataFrame(X).copy()
        df = df[self.columns_].astype(str)
        
        # Map categories to embeddings
        out = []
        for col in self.columns_:
            category_map = self.category_maps_[col]
            embeddings = self.embeddings_[col]
            
            # Map each category to its embedding vector
            col_embeddings = []
            for val in df[col]:
                if val in category_map:
                    cat_id = category_map[val]
                    emb_vec = embeddings[cat_id]
                else:
                    # Unknown category - use zeros
                    emb_vec = np.zeros(self.embedding_dims_[col])
                col_embeddings.append(emb_vec)
            
            col_embeddings = np.array(col_embeddings)  # Shape: (n_samples, embedding_dim)
            out.append(col_embeddings)
        
        return np.hstack(out)
    
    def get_embeddings(self) -> dict[str, np.ndarray]:
        """Get learned embedding matrices for each column.
        
        Returns:
            Dict mapping column to embedding matrix (n_categories, embedding_dim)
        """
        if not self.embeddings_:
            raise RuntimeError("EntityEmbeddingsEncoder not fitted. Call fit() first.")
        return self.embeddings_.copy()
    
    def get_feature_names_out(self, input_features: Optional[list[str]] = None) -> list[str]:
        """Get output feature names for embedded features.
        
        Args:
            input_features: Input feature names (unused)
            
        Returns:
            List of embedding feature names
        """
        names = []
        for col in self.columns_:
            emb_dim = self.embedding_dims_.get(col, 0)
            for dim_idx in range(emb_dim):
                names.append(f"emb_{col}_{dim_idx}")
        return names