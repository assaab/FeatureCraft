"""Additional transformers for FeatureCraft."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PowerTransformer


class NumericConverter(BaseEstimator, TransformerMixin):
    """Convert columns to numeric, handling mixed types gracefully."""
    
    def __init__(self, columns: Sequence[str] | None = None) -> None:
        """Initialize with optional column list."""
        self.columns = columns
        self.columns_: list[str] = []
    
    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "NumericConverter":
        """Fit converter (just stores column names)."""
        df = pd.DataFrame(X)
        self.columns_ = list(self.columns) if self.columns else list(df.columns)
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform by converting to numeric with coercion."""
        df = pd.DataFrame(X).copy()
        for col in self.columns_:
            if col in df.columns:
                # Try to convert to numeric, replacing errors with NaN
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    
    def get_feature_names_out(self, input_features: Sequence[str] | None = None) -> list[str]:
        """Get output feature names."""
        return self.columns_


class DateTimeFeatures(BaseEstimator, TransformerMixin):
    """Extract datetime features."""

    def __init__(self, columns: Sequence[str]) -> None:
        """Initialize with column names."""
        # Store columns as-is for sklearn compatibility
        self.columns = columns
        self.out_columns_: list[str] = []

    def fit(self, X: pd.DataFrame, y=None) -> DateTimeFeatures:
        """Fit transformer."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform by extracting datetime features."""
        df = pd.DataFrame(X).copy()
        out = pd.DataFrame(index=df.index)
        # Convert to list for iteration
        cols = list(self.columns) if not isinstance(self.columns, list) else self.columns
        for c in cols:
            s = pd.to_datetime(df[c], errors="coerce")
            out[f"{c}_year"] = s.dt.year
            out[f"{c}_quarter"] = s.dt.quarter
            out[f"{c}_month"] = s.dt.month
            out[f"{c}_weekday"] = s.dt.weekday
            # Only extract hour if time component exists
            if s.dt.hour.notna().any():
                out[f"{c}_hour"] = s.dt.hour
            out[f"{c}_is_weekend"] = s.dt.weekday.isin([5, 6]).astype(int)
            # cyclic
            out[f"{c}_month_sin"] = np.sin(2 * np.pi * (s.dt.month.fillna(0) / 12))
            out[f"{c}_month_cos"] = np.cos(2 * np.pi * (s.dt.month.fillna(0) / 12))
            out[f"{c}_weekday_sin"] = np.sin(2 * np.pi * (s.dt.weekday.fillna(0) / 7))
            out[f"{c}_weekday_cos"] = np.cos(2 * np.pi * (s.dt.weekday.fillna(0) / 7))
            if s.dt.hour.notna().any():
                out[f"{c}_hour_sin"] = np.sin(2 * np.pi * (s.dt.hour.fillna(0) / 24))
                out[f"{c}_hour_cos"] = np.cos(2 * np.pi * (s.dt.hour.fillna(0) / 24))
        # Store output columns for feature name inference (must be done after all features created)
        if not self.out_columns_:  # Only set on first transform (fit_transform)
            self.out_columns_ = list(out.columns)
        return out

    def get_feature_names_out(self, input_features=None):
        """Get output feature names."""
        if self.out_columns_:
            return self.out_columns_
        # Fallback if transform hasn't been called yet
        return [f"dt_feat_{i}" for i in range(len(self.columns) * 11)]  # max possible features per column


class SkewedPowerTransformer(BaseEstimator, TransformerMixin):
    """Apply Yeo-Johnson to selected numeric columns based on skewness mask."""

    def __init__(self, columns: Sequence[str], skew_mask: Sequence[bool]) -> None:
        """Initialize with columns and skew mask."""
        # Store as-is for sklearn compatibility
        self.columns = columns
        self.skew_mask = skew_mask
        self.pt_: PowerTransformer | None = None
        self.cols_to_tx_: list[int] = []
        self.n_features_in_: int = 0

    def fit(self, X: pd.DataFrame, y=None) -> SkewedPowerTransformer:
        """Fit transformer."""
        # Handle both DataFrame and array inputs dynamically
        if isinstance(X, pd.DataFrame):
            df = X
            self.n_features_in_ = df.shape[1]
        else:
            # X is a numpy array - use actual shape, not expected columns
            X_arr = np.asarray(X)
            self.n_features_in_ = X_arr.shape[1]
            # Create DataFrame with generic column names
            df = pd.DataFrame(X_arr, columns=[f"col_{i}" for i in range(X_arr.shape[1])])
        
        # Convert mask to list
        mask = list(self.skew_mask) if not isinstance(self.skew_mask, list) else self.skew_mask
        
        # Only transform the first len(mask) columns that match the skew mask
        # (subsequent columns may be indicators added by imputer)
        n_original_cols = len(mask)
        self.cols_to_tx_ = [i for i, m in enumerate(mask) if m and i < df.shape[1]]
        
        if self.cols_to_tx_:
            self.pt_ = PowerTransformer(method="yeo-johnson")
            self.pt_.fit(df.iloc[:, self.cols_to_tx_])
        return self

    def transform(self, X) -> np.ndarray:
        """Transform data."""
        # Handle both DataFrame and array inputs dynamically
        if isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            X_arr = np.asarray(X)
            df = pd.DataFrame(X_arr, columns=[f"col_{i}" for i in range(X_arr.shape[1])])
        
        if self.pt_ and self.cols_to_tx_:
            tx = self.pt_.transform(df.iloc[:, self.cols_to_tx_])
            df.iloc[:, self.cols_to_tx_] = tx
        return df.values


class CategoricalMissingIndicator(BaseEstimator, TransformerMixin):
    """Add boolean missing indicators for categorical columns."""

    def __init__(self, columns: Sequence[str]) -> None:
        """Initialize with column names."""
        # Store as-is for sklearn compatibility
        self.columns = columns

    def fit(self, X, y=None) -> CategoricalMissingIndicator:
        """Fit transformer."""
        return self

    def transform(self, X) -> np.ndarray:
        """Transform by adding missing indicators."""
        # Handle both DataFrame and array inputs dynamically
        if isinstance(X, pd.DataFrame):
            df = X
        else:
            X_arr = np.asarray(X)
            # Create DataFrame with generic column names based on actual shape
            df = pd.DataFrame(X_arr, columns=[f"col_{i}" for i in range(X_arr.shape[1])])
        return df.isna().astype(int).values


class CategoricalCleaner(BaseEstimator, TransformerMixin):
    """Clean categorical columns with normalization and coercion."""
    
    def __init__(
        self,
        columns: Sequence[str] | None = None,
        lowercase: bool = True,
        strip_whitespace: bool = True,
        replace_empty: bool = True,
    ) -> None:
        """Initialize categorical cleaner.
        
        Args:
            columns: Columns to clean (None = all)
            lowercase: Convert to lowercase
            strip_whitespace: Strip leading/trailing whitespace
            replace_empty: Replace empty strings with NaN
        """
        self.columns = columns
        self.lowercase = lowercase
        self.strip_whitespace = strip_whitespace
        self.replace_empty = replace_empty
        self.columns_: list[str] = []
    
    def fit(self, X: pd.DataFrame, y=None) -> "CategoricalCleaner":
        """Fit cleaner."""
        df = pd.DataFrame(X)
        self.columns_ = list(self.columns) if self.columns else list(df.columns)
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Clean categorical data."""
        df = pd.DataFrame(X).copy()
        
        for col in self.columns_:
            if col not in df.columns:
                continue
            
            # Convert to string
            series = df[col].astype(str)
            
            # Strip whitespace
            if self.strip_whitespace:
                series = series.str.strip()
            
            # Lowercase
            if self.lowercase:
                series = series.str.lower()
            
            # Replace empty strings with NaN
            if self.replace_empty:
                series = series.replace(["", "nan", "none", "null"], np.nan)
            
            df[col] = series
        
        return df


class DimensionalityReducer(BaseEstimator, TransformerMixin):
    """Pluggable dimensionality reducer for numeric features."""
    
    def __init__(
        self,
        kind: str | None = None,
        max_components: int | None = None,
        variance: float | None = None,
        random_state: int = 42,
    ) -> None:
        """Initialize reducer.
        
        Args:
            kind: Reducer type: 'pca', 'svd', 'umap', or None
            max_components: Maximum number of components
            variance: Variance threshold for PCA (alternative to max_components)
            random_state: Random seed
        """
        self.kind = kind
        self.max_components = max_components
        self.variance = variance
        self.random_state = random_state
        self.reducer_ = None
        self.n_components_: int = 0
    
    def fit(self, X, y=None) -> "DimensionalityReducer":
        """Fit reducer."""
        if self.kind is None:
            return self
        
        X_arr = np.asarray(X)
        n_samples, n_features = X_arr.shape
        
        # Determine n_components
        if self.max_components:
            n_comp = min(self.max_components, n_features - 1, n_samples - 1)
        else:
            n_comp = min(n_features - 1, n_samples - 1)
        
        if n_comp < 1:
            return self
        
        self.n_components_ = n_comp
        
        try:
            if self.kind == "pca":
                from sklearn.decomposition import PCA
                
                if self.variance:
                    self.reducer_ = PCA(
                        n_components=self.variance, random_state=self.random_state
                    )
                else:
                    self.reducer_ = PCA(
                        n_components=n_comp, random_state=self.random_state
                    )
                self.reducer_.fit(X_arr)
                
            elif self.kind == "svd":
                from sklearn.decomposition import TruncatedSVD
                
                self.reducer_ = TruncatedSVD(
                    n_components=n_comp, random_state=self.random_state
                )
                self.reducer_.fit(X_arr)
                
            elif self.kind == "umap":
                try:
                    import umap
                    
                    self.reducer_ = umap.UMAP(
                        n_components=min(n_comp, 50),
                        random_state=self.random_state,
                    )
                    self.reducer_.fit(X_arr)
                except ImportError:
                    from .logging import get_logger
                    logger = get_logger(__name__)
                    logger.warning("UMAP not installed. Skipping dimensionality reduction.")
        
        except Exception as e:
            from .logging import get_logger
            logger = get_logger(__name__)
            logger.warning(f"Dimensionality reduction failed: {e}")
        
        return self
    
    def transform(self, X) -> np.ndarray:
        """Transform data."""
        if self.reducer_ is None:
            return np.asarray(X)
        
        X_arr = np.asarray(X)
        return self.reducer_.transform(X_arr)


class WinsorizerTransformer(BaseEstimator, TransformerMixin):
    """Clip extreme values based on percentiles (winsorization)."""
    
    def __init__(
        self,
        percentiles: tuple[float, float] = (0.01, 0.99),
        columns: list[str] | None = None,
    ) -> None:
        """Initialize winsorizer.
        
        Args:
            percentiles: Lower and upper percentiles for clipping
            columns: Columns to winsorize (None = all numeric)
        """
        self.percentiles = percentiles
        self.columns = columns
        self.lower_bounds_: dict[str, float] = {}
        self.upper_bounds_: dict[str, float] = {}
        self.columns_: list[str] = []
    
    def fit(self, X: pd.DataFrame, y=None) -> "WinsorizerTransformer":
        """Fit winsorizer by computing percentile bounds."""
        df = pd.DataFrame(X)
        self.columns_ = list(self.columns) if self.columns else list(df.select_dtypes(include=[np.number]).columns)
        
        for col in self.columns_:
            if col in df.columns:
                series = df[col].dropna()
                if len(series) > 0:
                    self.lower_bounds_[col] = series.quantile(self.percentiles[0])
                    self.upper_bounds_[col] = series.quantile(self.percentiles[1])
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply winsorization."""
        df = pd.DataFrame(X).copy()
        
        for col in self.columns_:
            if col in df.columns and col in self.lower_bounds_:
                df[col] = df[col].clip(
                    lower=self.lower_bounds_[col],
                    upper=self.upper_bounds_[col]
                )
        
        return df


class EnsureNumericOutput(BaseEstimator, TransformerMixin):
    """Final safety transformer to ensure all output is numeric."""
    
    def fit(self, X, y=None) -> "EnsureNumericOutput":
        """Fit (no-op)."""
        return self
    
    def transform(self, X):
        """Ensure all output is numeric, converting or raising error if not possible."""
        import scipy.sparse as sp
        from .logging import get_logger
        
        logger = get_logger(__name__)
        
        # Handle sparse matrices
        if sp.issparse(X):
            # Sparse matrices are already numeric
            return X
        
        # Handle numpy arrays
        if isinstance(X, np.ndarray):
            # Check if it's already numeric
            if np.issubdtype(X.dtype, np.number):
                return X
            
            # Array contains non-numeric data - try to diagnose and convert
            logger.warning("Output array contains non-numeric dtype. Attempting conversion...")
            try:
                # Flatten to check for problematic values
                flat = X.flatten()
                # Sample first few problematic values for error message
                problematic = [v for v in flat[:100] if isinstance(v, (str, bytes))]
                if problematic:
                    logger.error(f"Found non-numeric values in output: {problematic[:5]}")
                
                return X.astype(float)
            except (ValueError, TypeError) as e:
                raise ValueError(
                    f"Cannot convert output array to numeric. "
                    f"Contains non-numeric values. This indicates categorical columns were not properly encoded. "
                    f"Error: {e}"
                )
        
        # Handle DataFrames
        df = pd.DataFrame(X)
        
        # Check if all columns are numeric
        non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        
        if non_numeric_cols:
            logger.warning(f"Found {len(non_numeric_cols)} non-numeric columns in output: {non_numeric_cols[:10]}")
            
            # Try to convert non-numeric columns
            for col in non_numeric_cols:
                col_data = df[col]
                
                # Check what kind of data we have
                sample = col_data.dropna().head(10)
                if len(sample) > 0:
                    sample_values = sample.tolist()
                    logger.warning(f"Column '{col}' contains non-numeric data. Sample values: {sample_values}")
                
                # Try aggressive conversion
                try:
                    converted = pd.to_numeric(col_data, errors='coerce')
                    
                    # Check how many values were coerced to NaN
                    original_nan_count = col_data.isna().sum()
                    new_nan_count = converted.isna().sum()
                    coerced_count = new_nan_count - original_nan_count
                    
                    if coerced_count > 0:
                        logger.warning(
                            f"Column '{col}': Coerced {coerced_count} non-numeric values to NaN during conversion. "
                            f"This indicates the column should have been treated as categorical."
                        )
                    
                    df[col] = converted
                    
                except Exception as e:
                    # Identify the problematic values
                    unique_vals = col_data.unique()[:20]
                    raise ValueError(
                        f"Column '{col}' contains non-numeric data that cannot be converted. "
                        f"Sample unique values: {unique_vals}. "
                        f"This indicates a bug in categorical column detection. "
                        f"Error: {e}"
                    )
        
        return df.values
