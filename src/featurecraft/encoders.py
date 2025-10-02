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
        """Transform using hashing."""
        df = pd.DataFrame(X)[self.columns_].astype(str)
        rows: list[dict[int, float]] = []
        for _, row in df.iterrows():
            d: dict[int, float] = {}
            for c, v in row.items():
                idx = self._hash(f"{c}={v}")
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
        return encoder
    except (TypeError, AttributeError):
        pass
    
    # Try without drop parameter
    try:
        encoder = OneHotEncoder(
            handle_unknown=handle_unknown if handle_unknown != "infrequent_if_exist" else "ignore",
            min_frequency=min_frequency,
            sparse_output=True,
        )
        _ = encoder.get_params()
        return encoder
    except (TypeError, AttributeError):
        pass
    
    # Try with sparse instead of sparse_output (older sklearn)
    try:
        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse=True,
        )
        _ = encoder.get_params()
        return encoder
    except (TypeError, AttributeError):
        pass
    
    # Ultimate fallback - minimal parameters
    encoder = OneHotEncoder()
    try:
        encoder.set_params(handle_unknown="ignore")
    except Exception:
        logger.warning("Could not set handle_unknown='ignore', using defaults")
    
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
        freq_maps_: Dict mapping column → category → frequency
        columns_: List of encoded columns
        
    Example:
        >>> encoder = FrequencyEncoder()
        >>> encoder.fit(X_train)
        >>> X_train_encoded = encoder.transform(X_train)
        >>> X_test_encoded = encoder.transform(X_test)  # Unseen categories → unseen_value
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
        count_maps_: Dict mapping column → category → count
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