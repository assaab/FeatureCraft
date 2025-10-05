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
    """Extract comprehensive datetime features with configurable feature groups.
    
    Feature Categories:
    - Basic Extraction: year, month, day, hour, minute, second, day_of_week, week_of_year, quarter, day_of_year
    - Cyclical Encoding: sin/cos transforms for month, day_of_week, hour, day_of_year (preserves cyclical patterns)
    - Boolean Flags: is_weekend, is_month_start, is_month_end, is_quarter_start, is_quarter_end, is_year_start, is_year_end
    - Seasonality: season (0=winter, 1=spring, 2=summer, 3=fall)
    - Business Logic: is_business_hour (9am-5pm weekdays), business_days_in_month
    - Relative Time: days_since_reference (requires reference_date parameter)
    """

    def __init__(
        self, 
        columns: Sequence[str],
        extract_basic: bool = True,
        extract_cyclical: bool = True,
        extract_boolean_flags: bool = True,
        extract_season: bool = True,
        extract_business: bool = True,
        extract_relative: bool = False,
        reference_date: pd.Timestamp | str | None = None,
        business_hour_start: int = 9,
        business_hour_end: int = 17,
    ) -> None:
        """Initialize datetime feature extractor.
        
        Args:
            columns: Column names to extract features from
            extract_basic: Extract year, month, day, hour, minute, second, etc.
            extract_cyclical: Extract sin/cos cyclical encodings
            extract_boolean_flags: Extract boolean flags (weekend, month_start, etc.)
            extract_season: Extract season feature
            extract_business: Extract business hour/day features
            extract_relative: Extract relative time features (requires reference_date)
            reference_date: Reference date for relative time features
            business_hour_start: Start hour for business hours (default 9am)
            business_hour_end: End hour for business hours (default 5pm)
        """
        # Store columns as-is for sklearn compatibility
        self.columns = columns
        self.extract_basic = extract_basic
        self.extract_cyclical = extract_cyclical
        self.extract_boolean_flags = extract_boolean_flags
        self.extract_season = extract_season
        self.extract_business = extract_business
        self.extract_relative = extract_relative
        self.reference_date = pd.to_datetime(reference_date) if reference_date else None
        self.business_hour_start = business_hour_start
        self.business_hour_end = business_hour_end
        self.out_columns_: list[str] = []
        self.has_time_component_: dict[str, bool] = {}

    def fit(self, X: pd.DataFrame, y=None) -> "DateTimeFeatures":
        """Fit transformer - detect if columns have time components."""
        df = pd.DataFrame(X)
        cols = list(self.columns) if not isinstance(self.columns, list) else self.columns
        
        for c in cols:
            if c in df.columns:
                s = pd.to_datetime(df[c], errors="coerce")
                # Check if time component exists (hour/minute/second are not all zeros)
                has_time = (
                    s.dt.hour.notna().any() and 
                    (s.dt.hour != 0).any() or 
                    (s.dt.minute != 0).any() or 
                    (s.dt.second != 0).any()
                )
                self.has_time_component_[c] = has_time
        
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform by extracting comprehensive datetime features."""
        df = pd.DataFrame(X).copy()
        out = pd.DataFrame(index=df.index)
        
        # Convert to list for iteration
        cols = list(self.columns) if not isinstance(self.columns, list) else self.columns
        
        for c in cols:
            s = pd.to_datetime(df[c], errors="coerce")
            has_time = self.has_time_component_.get(c, False)
            
            # ========== BASIC EXTRACTION ==========
            if self.extract_basic:
                out[f"{c}_year"] = s.dt.year
                out[f"{c}_month"] = s.dt.month
                out[f"{c}_day"] = s.dt.day
                out[f"{c}_day_of_week"] = s.dt.dayofweek  # Monday=0, Sunday=6
                out[f"{c}_day_of_year"] = s.dt.dayofyear
                out[f"{c}_week_of_year"] = s.dt.isocalendar().week
                out[f"{c}_quarter"] = s.dt.quarter
                
                # Only extract time components if they exist
                if has_time:
                    out[f"{c}_hour"] = s.dt.hour
                    out[f"{c}_minute"] = s.dt.minute
                    out[f"{c}_second"] = s.dt.second
            
            # ========== CYCLICAL ENCODING ==========
            if self.extract_cyclical:
                # Month (12 months)
                out[f"{c}_month_sin"] = np.sin(2 * np.pi * (s.dt.month.fillna(0) / 12))
                out[f"{c}_month_cos"] = np.cos(2 * np.pi * (s.dt.month.fillna(0) / 12))
                
                # Day of week (7 days)
                out[f"{c}_day_of_week_sin"] = np.sin(2 * np.pi * (s.dt.dayofweek.fillna(0) / 7))
                out[f"{c}_day_of_week_cos"] = np.cos(2 * np.pi * (s.dt.dayofweek.fillna(0) / 7))
                
                # Day of year (365 days)
                out[f"{c}_day_of_year_sin"] = np.sin(2 * np.pi * (s.dt.dayofyear.fillna(0) / 365.25))
                out[f"{c}_day_of_year_cos"] = np.cos(2 * np.pi * (s.dt.dayofyear.fillna(0) / 365.25))
                
                # Hour (24 hours) - only if time component exists
                if has_time:
                    out[f"{c}_hour_sin"] = np.sin(2 * np.pi * (s.dt.hour.fillna(0) / 24))
                    out[f"{c}_hour_cos"] = np.cos(2 * np.pi * (s.dt.hour.fillna(0) / 24))
            
            # ========== BOOLEAN FLAGS ==========
            if self.extract_boolean_flags:
                # Weekend
                out[f"{c}_is_weekend"] = s.dt.dayofweek.isin([5, 6]).astype(int)
                
                # Month boundaries
                out[f"{c}_is_month_start"] = s.dt.is_month_start.astype(int)
                out[f"{c}_is_month_end"] = s.dt.is_month_end.astype(int)
                
                # Quarter boundaries
                out[f"{c}_is_quarter_start"] = s.dt.is_quarter_start.astype(int)
                out[f"{c}_is_quarter_end"] = s.dt.is_quarter_end.astype(int)
                
                # Year boundaries
                out[f"{c}_is_year_start"] = s.dt.is_year_start.astype(int)
                out[f"{c}_is_year_end"] = s.dt.is_year_end.astype(int)
            
            # ========== SEASONALITY ==========
            if self.extract_season:
                # Northern hemisphere seasons (can be customized)
                # 0=Winter (Dec-Feb), 1=Spring (Mar-May), 2=Summer (Jun-Aug), 3=Fall (Sep-Nov)
                month = s.dt.month
                season = pd.Series(np.nan, index=s.index)
                season[month.isin([12, 1, 2])] = 0  # Winter
                season[month.isin([3, 4, 5])] = 1   # Spring
                season[month.isin([6, 7, 8])] = 2   # Summer
                season[month.isin([9, 10, 11])] = 3 # Fall
                out[f"{c}_season"] = season
            
            # ========== BUSINESS LOGIC ==========
            if self.extract_business:
                # Business hour (9am-5pm on weekdays by default)
                if has_time:
                    is_business_hour = (
                        (s.dt.dayofweek < 5) &  # Monday-Friday
                        (s.dt.hour >= self.business_hour_start) & 
                        (s.dt.hour < self.business_hour_end)
                    )
                    out[f"{c}_is_business_hour"] = is_business_hour.astype(int)
                
                # Business days in month
                # Using a vectorized approach for efficiency
                business_days_in_month = s.apply(
                    lambda x: np.busday_count(
                        x.replace(day=1).date(),
                        (x.replace(day=1) + pd.DateOffset(months=1)).date()
                    ) if pd.notna(x) else np.nan
                )
                out[f"{c}_business_days_in_month"] = business_days_in_month
            
            # ========== RELATIVE TIME ==========
            if self.extract_relative and self.reference_date is not None:
                # Days since reference date
                days_since = (s - self.reference_date).dt.days
                out[f"{c}_days_since_reference"] = days_since
                
                # Additional relative features
                out[f"{c}_weeks_since_reference"] = days_since / 7
                out[f"{c}_months_since_reference"] = (
                    (s.dt.year - self.reference_date.year) * 12 + 
                    (s.dt.month - self.reference_date.month)
                )
        
        # Store output columns for feature name inference (must be done after all features created)
        if not self.out_columns_:  # Only set on first transform (fit_transform)
            self.out_columns_ = list(out.columns)
        
        return out

    def get_feature_names_out(self, input_features=None):
        """Get output feature names."""
        if self.out_columns_:
            return self.out_columns_
        
        # Estimate based on enabled features (conservative estimate)
        cols = list(self.columns) if not isinstance(self.columns, list) else self.columns
        features_per_col = 0
        
        if self.extract_basic:
            features_per_col += 10  # year, month, day, day_of_week, day_of_year, week_of_year, quarter, hour, minute, second
        if self.extract_cyclical:
            features_per_col += 8   # month_sin/cos, day_of_week_sin/cos, day_of_year_sin/cos, hour_sin/cos
        if self.extract_boolean_flags:
            features_per_col += 7   # is_weekend, is_month_start/end, is_quarter_start/end, is_year_start/end
        if self.extract_season:
            features_per_col += 1   # season
        if self.extract_business:
            features_per_col += 2   # is_business_hour, business_days_in_month
        if self.extract_relative:
            features_per_col += 3   # days/weeks/months since reference
        
        return [f"dt_feat_{i}" for i in range(len(cols) * features_per_col)]


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
