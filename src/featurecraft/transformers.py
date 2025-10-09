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

    def fit(self, X: pd.DataFrame, y=None) -> "SkewedPowerTransformer":
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


class LogTransformer(BaseEstimator, TransformerMixin):
    """Apply log transform: log(x + shift).
    
    Suitable for right-skewed distributions with exponential growth patterns.
    Requires x > 0 after shift. Use log1p for data with zeros.
    """
    
    def __init__(
        self, 
        columns: Sequence[str] | None = None,
        shift: float = 1e-5,
        base: str = "natural",
    ) -> None:
        """Initialize log transformer.
        
        Args:
            columns: Columns to transform (None = all numeric)
            shift: Value to add before log to handle zeros: log(x + shift)
            base: Logarithm base - 'natural' (ln), '10', '2'
        """
        self.columns = columns
        self.shift = shift
        self.base = base
        self.columns_: list[str] = []
        self.min_values_: dict[str, float] = {}
    
    def fit(self, X: pd.DataFrame, y=None) -> "LogTransformer":
        """Fit transformer by validating data ranges."""
        df = pd.DataFrame(X)
        
        if self.columns is None:
            self.columns_ = list(df.select_dtypes(include=[np.number]).columns)
        else:
            self.columns_ = [c for c in self.columns if c in df.columns]
        
        # Check minimum values to ensure x + shift > 0
        for col in self.columns_:
            min_val = df[col].min()
            self.min_values_[col] = float(min_val)
            
            if min_val + self.shift <= 0:
                raise ValueError(
                    f"Column '{col}' has minimum value {min_val}, which with shift={self.shift} "
                    f"gives {min_val + self.shift} <= 0. Log requires positive values. "
                    f"Use log1p transform or increase shift parameter."
                )
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply log transform."""
        df = pd.DataFrame(X).copy()
        
        for col in self.columns_:
            if col not in df.columns:
                continue
            
            # Apply shift and log
            shifted = df[col] + self.shift
            
            if self.base == "natural":
                df[col] = np.log(shifted)
            elif self.base == "10":
                df[col] = np.log10(shifted)
            elif self.base == "2":
                df[col] = np.log2(shifted)
            else:
                raise ValueError(f"Unknown base: {self.base}. Use 'natural', '10', or '2'.")
        
        return df
    
    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Reverse the log transform."""
        df = pd.DataFrame(X).copy()
        
        for col in self.columns_:
            if col not in df.columns:
                continue
            
            if self.base == "natural":
                df[col] = np.exp(df[col]) - self.shift
            elif self.base == "10":
                df[col] = np.power(10, df[col]) - self.shift
            elif self.base == "2":
                df[col] = np.power(2, df[col]) - self.shift
        
        return df


class Log1pTransformer(BaseEstimator, TransformerMixin):
    """Apply log1p transform: log(1 + x).
    
    Handles zeros naturally without requiring a shift parameter.
    Suitable for count data and non-negative features with right skew.
    Requires x >= 0.
    """
    
    def __init__(self, columns: Sequence[str] | None = None) -> None:
        """Initialize log1p transformer.
        
        Args:
            columns: Columns to transform (None = all numeric)
        """
        self.columns = columns
        self.columns_: list[str] = []
        self.min_values_: dict[str, float] = {}
    
    def fit(self, X: pd.DataFrame, y=None) -> "Log1pTransformer":
        """Fit transformer by validating data ranges."""
        df = pd.DataFrame(X)
        
        if self.columns is None:
            self.columns_ = list(df.select_dtypes(include=[np.number]).columns)
        else:
            self.columns_ = [c for c in self.columns if c in df.columns]
        
        # Validate x >= 0
        for col in self.columns_:
            min_val = df[col].min()
            self.min_values_[col] = float(min_val)
            
            if min_val < 0:
                raise ValueError(
                    f"Column '{col}' has minimum value {min_val} < 0. "
                    f"log1p requires non-negative values. Consider using log with shift or yeo-johnson."
                )
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply log1p transform."""
        df = pd.DataFrame(X).copy()
        
        for col in self.columns_:
            if col not in df.columns:
                continue
            df[col] = np.log1p(df[col])
        
        return df
    
    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Reverse the log1p transform."""
        df = pd.DataFrame(X).copy()
        
        for col in self.columns_:
            if col not in df.columns:
                continue
            df[col] = np.expm1(df[col])
        
        return df


class SqrtTransformer(BaseEstimator, TransformerMixin):
    """Apply square root transform: √x.
    
    Suitable for moderate skewness and count data.
    Handles negatives based on strategy: abs, clip, or error.
    """
    
    def __init__(
        self, 
        columns: Sequence[str] | None = None,
        handle_negatives: str = "abs",
    ) -> None:
        """Initialize sqrt transformer.
        
        Args:
            columns: Columns to transform (None = all numeric)
            handle_negatives: Strategy for negative values:
                - 'abs': sqrt(abs(x)) * sign(x) (signed square root)
                - 'clip': Set negative values to 0
                - 'error': Raise error if negatives found
        """
        self.columns = columns
        self.handle_negatives = handle_negatives
        self.columns_: list[str] = []
        self.has_negatives_: dict[str, bool] = {}
    
    def fit(self, X: pd.DataFrame, y=None) -> "SqrtTransformer":
        """Fit transformer by checking for negative values."""
        df = pd.DataFrame(X)
        
        if self.columns is None:
            self.columns_ = list(df.select_dtypes(include=[np.number]).columns)
        else:
            self.columns_ = [c for c in self.columns if c in df.columns]
        
        for col in self.columns_:
            has_neg = (df[col] < 0).any()
            self.has_negatives_[col] = bool(has_neg)
            
            if has_neg and self.handle_negatives == "error":
                raise ValueError(
                    f"Column '{col}' contains negative values. "
                    f"sqrt requires non-negative values unless handle_negatives='abs' or 'clip'."
                )
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply square root transform."""
        df = pd.DataFrame(X).copy()
        
        for col in self.columns_:
            if col not in df.columns:
                continue
            
            if self.handle_negatives == "abs":
                # Signed square root: sqrt(abs(x)) * sign(x)
                df[col] = np.sqrt(np.abs(df[col])) * np.sign(df[col])
            elif self.handle_negatives == "clip":
                # Clip negatives to 0, then sqrt
                df[col] = np.sqrt(np.maximum(df[col], 0))
            else:  # error
                df[col] = np.sqrt(df[col])
        
        return df
    
    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Reverse the square root transform."""
        df = pd.DataFrame(X).copy()
        
        for col in self.columns_:
            if col not in df.columns:
                continue
            
            if self.handle_negatives == "abs":
                # Reverse signed square root: x² * sign(x)
                df[col] = np.square(df[col]) * np.sign(df[col])
            else:
                df[col] = np.square(df[col])
        
        return df


class ReciprocalTransformer(BaseEstimator, TransformerMixin):
    """Apply reciprocal transform: 1/x.
    
    Suitable for heavy right-tailed distributions.
    Adds small epsilon to prevent division by zero: 1/(x + epsilon).
    """
    
    def __init__(
        self, 
        columns: Sequence[str] | None = None,
        epsilon: float = 1e-10,
    ) -> None:
        """Initialize reciprocal transformer.
        
        Args:
            columns: Columns to transform (None = all numeric)
            epsilon: Small value to add to prevent division by zero: 1/(x + epsilon)
        """
        self.columns = columns
        self.epsilon = epsilon
        self.columns_: list[str] = []
    
    def fit(self, X: pd.DataFrame, y=None) -> "ReciprocalTransformer":
        """Fit transformer."""
        df = pd.DataFrame(X)
        
        if self.columns is None:
            self.columns_ = list(df.select_dtypes(include=[np.number]).columns)
        else:
            self.columns_ = [c for c in self.columns if c in df.columns]
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply reciprocal transform."""
        df = pd.DataFrame(X).copy()
        
        for col in self.columns_:
            if col not in df.columns:
                continue
            
            # Add epsilon to avoid division by zero
            df[col] = 1.0 / (df[col] + self.epsilon)
        
        return df
    
    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Reverse the reciprocal transform."""
        df = pd.DataFrame(X).copy()
        
        for col in self.columns_:
            if col not in df.columns:
                continue
            
            # Reverse: 1/y - epsilon
            df[col] = 1.0 / df[col] - self.epsilon
        
        return df


class BoxCoxTransformer(BaseEstimator, TransformerMixin):
    """Apply Box-Cox transform: (x^λ - 1) / λ.
    
    Optimal λ is estimated to maximize normality (Gaussian likelihood).
    Requires x > 0 (strictly positive values).
    For data with zeros or negatives, use Yeo-Johnson instead.
    """
    
    def __init__(
        self, 
        columns: Sequence[str] | None = None,
        lambda_value: float | None = None,
    ) -> None:
        """Initialize Box-Cox transformer.
        
        Args:
            columns: Columns to transform (None = all numeric)
            lambda_value: Fixed lambda value (None = optimize automatically)
        """
        self.columns = columns
        self.lambda_value = lambda_value
        self.columns_: list[str] = []
        self.lambda_values_: dict[str, float] = {}
        self.pt_: PowerTransformer | None = None
    
    def fit(self, X: pd.DataFrame, y=None) -> "BoxCoxTransformer":
        """Fit transformer by optimizing lambda."""
        df = pd.DataFrame(X)
        
        if self.columns is None:
            self.columns_ = list(df.select_dtypes(include=[np.number]).columns)
        else:
            self.columns_ = [c for c in self.columns if c in df.columns]
        
        # Validate x > 0 for all columns
        for col in self.columns_:
            min_val = df[col].min()
            if min_val <= 0:
                raise ValueError(
                    f"Column '{col}' has minimum value {min_val} <= 0. "
                    f"Box-Cox requires strictly positive values. Use Yeo-Johnson for data with zeros/negatives."
                )
        
        # Fit PowerTransformer with box-cox method
        self.pt_ = PowerTransformer(method="box-cox", standardize=False)
        self.pt_.fit(df[self.columns_])
        
        # Store learned lambda values
        if hasattr(self.pt_, 'lambdas_'):
            for i, col in enumerate(self.columns_):
                self.lambda_values_[col] = float(self.pt_.lambdas_[i])
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply Box-Cox transform."""
        df = pd.DataFrame(X).copy()
        
        if self.pt_ is None:
            return df
        
        # Transform
        transformed = self.pt_.transform(df[self.columns_])
        df[self.columns_] = transformed
        
        return df
    
    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Reverse the Box-Cox transform."""
        df = pd.DataFrame(X).copy()
        
        if self.pt_ is None:
            return df
        
        # Inverse transform
        inverse = self.pt_.inverse_transform(df[self.columns_])
        df[self.columns_] = inverse
        
        return df


class ExponentialTransformer(BaseEstimator, TransformerMixin):
    """Apply exponential transforms: e^x, x², x³, etc.
    
    Suitable for left-skewed distributions (rare use case).
    Use with caution as exponential transforms can create very large values.
    """
    
    def __init__(
        self, 
        columns: Sequence[str] | None = None,
        transform_type: str = "square",
        clip_max: float | None = None,
    ) -> None:
        """Initialize exponential transformer.
        
        Args:
            columns: Columns to transform (None = all numeric)
            transform_type: Type of transform - 'square' (x²), 'cube' (x³), 'exp' (e^x)
            clip_max: Maximum value to clip after transform (None = no clipping)
        """
        self.columns = columns
        self.transform_type = transform_type
        self.clip_max = clip_max
        self.columns_: list[str] = []
    
    def fit(self, X: pd.DataFrame, y=None) -> "ExponentialTransformer":
        """Fit transformer."""
        df = pd.DataFrame(X)
        
        if self.columns is None:
            self.columns_ = list(df.select_dtypes(include=[np.number]).columns)
        else:
            self.columns_ = [c for c in self.columns if c in df.columns]
        
        # Warn if exp transform might create very large values
        if self.transform_type == "exp":
            for col in self.columns_:
                max_val = df[col].max()
                if max_val > 10:
                    from .logging import get_logger
                    logger = get_logger(__name__)
                    logger.warning(
                        f"Column '{col}' has max value {max_val}. "
                        f"exp({max_val}) = {np.exp(max_val):.2e} may cause overflow. "
                        f"Consider using clip_max parameter or scaling before exponential transform."
                    )
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply exponential transform."""
        df = pd.DataFrame(X).copy()
        
        for col in self.columns_:
            if col not in df.columns:
                continue
            
            if self.transform_type == "square":
                df[col] = np.square(df[col])
            elif self.transform_type == "cube":
                df[col] = np.power(df[col], 3)
            elif self.transform_type == "exp":
                df[col] = np.exp(df[col])
            else:
                raise ValueError(
                    f"Unknown transform_type: '{self.transform_type}'. "
                    f"Use 'square', 'cube', or 'exp'."
                )
            
            # Optional clipping
            if self.clip_max is not None:
                df[col] = np.clip(df[col], -self.clip_max, self.clip_max)
        
        return df


class MathematicalTransformer(BaseEstimator, TransformerMixin):
    """Unified mathematical transformer with intelligent auto-selection.
    
    Automatically selects the optimal mathematical transform per column based on:
    - Data range (positivity, zeros, negatives)
    - Skewness magnitude and direction
    - Kurtosis (tail heaviness)
    - Data characteristics (count data, continuous, etc.)
    
    Supports: log, log1p, sqrt, box-cox, yeo-johnson, reciprocal, exponential, none
    """
    
    def __init__(
        self,
        columns: Sequence[str] | None = None,
        strategy: str = "auto",
        log_shift: float = 1e-5,
        sqrt_handle_negatives: str = "abs",
        reciprocal_epsilon: float = 1e-10,
        exponential_type: str = "square",
        boxcox_lambda: float | None = None,
        skew_threshold: float = 1.0,
    ) -> None:
        """Initialize mathematical transformer with unified interface.
        
        Args:
            columns: Columns to transform (None = all numeric)
            strategy: Transform strategy:
                - 'auto': Intelligently select best transform per column
                - 'log': log(x + shift)
                - 'log1p': log(1 + x)
                - 'sqrt': sqrt(x)
                - 'box_cox': Box-Cox with optimized lambda
                - 'yeo_johnson': Yeo-Johnson (handles negatives)
                - 'reciprocal': 1/(x + epsilon)
                - 'exponential': x², x³, or e^x
                - 'none': No transformation
            log_shift: Shift value for log transform
            sqrt_handle_negatives: How to handle negatives in sqrt (abs, clip, error)
            reciprocal_epsilon: Epsilon for reciprocal to prevent division by zero
            exponential_type: Type of exponential (square, cube, exp)
            boxcox_lambda: Fixed lambda for Box-Cox (None = optimize)
            skew_threshold: Skewness threshold for triggering transforms in auto mode
        """
        self.columns = columns
        self.strategy = strategy
        self.log_shift = log_shift
        self.sqrt_handle_negatives = sqrt_handle_negatives
        self.reciprocal_epsilon = reciprocal_epsilon
        self.exponential_type = exponential_type
        self.boxcox_lambda = boxcox_lambda
        self.skew_threshold = skew_threshold
        
        # Fitted attributes
        self.columns_: list[str] = []
        self.transformers_: dict[str, BaseEstimator] = {}
        self.strategies_: dict[str, str] = {}
        self.skewness_: dict[str, float] = {}
        self.data_stats_: dict[str, dict] = {}
    
    def fit(self, X: pd.DataFrame, y=None) -> "MathematicalTransformer":
        """Fit transformer by analyzing data and selecting optimal transforms.
        
        Args:
            X: Input DataFrame
            y: Target (unused, for sklearn compatibility)
            
        Returns:
            Self
        """
        df = pd.DataFrame(X)
        
        # Determine columns to transform
        if self.columns is None:
            self.columns_ = list(df.select_dtypes(include=[np.number]).columns)
        else:
            self.columns_ = [c for c in self.columns if c in df.columns]
        
        # Analyze each column and select transform
        for col in self.columns_:
            col_data = df[col].dropna()
            
            if len(col_data) == 0:
                # No valid data - skip transformation
                self.strategies_[col] = "none"
                self.transformers_[col] = None
                continue
            
            # Compute statistics for auto-selection
            stats = self._compute_column_stats(col_data)
            self.data_stats_[col] = stats
            self.skewness_[col] = stats['skewness']
            
            # Select strategy for this column
            if self.strategy == "auto":
                selected_strategy = self._auto_select_strategy(col_data, stats)
            else:
                selected_strategy = self.strategy
            
            self.strategies_[col] = selected_strategy
            
            # Create and fit transformer for this column
            transformer = self._create_transformer(col, selected_strategy)
            if transformer is not None:
                try:
                    transformer.fit(df[[col]], y)
                    self.transformers_[col] = transformer
                except Exception as e:
                    # Fallback to no transform if fitting fails
                    from .logging import get_logger
                    logger = get_logger(__name__)
                    logger.warning(
                        f"Failed to fit {selected_strategy} transform for column '{col}': {e}. "
                        f"Skipping transformation for this column."
                    )
                    self.strategies_[col] = "none"
                    self.transformers_[col] = None
            else:
                self.transformers_[col] = None
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply selected transforms to data.
        
        Args:
            X: Input DataFrame
            
        Returns:
            Transformed DataFrame
        """
        df = pd.DataFrame(X).copy()
        
        for col in self.columns_:
            if col not in df.columns:
                continue
            
            transformer = self.transformers_.get(col)
            if transformer is not None:
                try:
                    df[[col]] = transformer.transform(df[[col]])
                except Exception as e:
                    # Log warning but continue
                    from .logging import get_logger
                    logger = get_logger(__name__)
                    logger.warning(
                        f"Transform failed for column '{col}' with {self.strategies_[col]}: {e}. "
                        f"Leaving column unchanged."
                    )
        
        return df
    
    def _compute_column_stats(self, series: pd.Series) -> dict:
        """Compute comprehensive statistics for a column.
        
        Args:
            series: Column data (without NaNs)
            
        Returns:
            Dict with statistics
        """
        stats = {}
        
        # Basic statistics
        stats['min'] = float(series.min())
        stats['max'] = float(series.max())
        stats['mean'] = float(series.mean())
        stats['std'] = float(series.std())
        
        # Distribution shape
        try:
            stats['skewness'] = float(series.skew())
        except:
            stats['skewness'] = 0.0
        
        try:
            stats['kurtosis'] = float(series.kurtosis())
        except:
            stats['kurtosis'] = 0.0
        
        # Data characteristics
        stats['has_zeros'] = (series == 0).any()
        stats['has_negatives'] = (series < 0).any()
        stats['all_positive'] = (series > 0).all()
        stats['all_nonnegative'] = (series >= 0).all()
        
        # Count characteristics (useful for log1p)
        stats['all_integers'] = series.apply(lambda x: float(x).is_integer()).all()
        stats['looks_like_counts'] = stats['all_nonnegative'] and stats['all_integers']
        
        # Percentage of zeros (useful for log1p vs log)
        stats['zero_percentage'] = float((series == 0).mean())
        
        return stats
    
    def _auto_select_strategy(self, series: pd.Series, stats: dict) -> str:
        """Intelligently select the best transform strategy.
        
        Decision tree for transform selection:
        1. Check if transformation is needed (low skewness → none)
        2. Check data constraints (negatives, zeros)
        3. Select optimal transform based on distribution shape
        
        Args:
            series: Column data
            stats: Pre-computed statistics
            
        Returns:
            Strategy name
        """
        skew = abs(stats['skewness'])
        
        # RULE 1: Low skewness → No transform needed
        if skew < self.skew_threshold:
            return "none"
        
        # RULE 2: Negatives present → Yeo-Johnson (most flexible)
        if stats['has_negatives']:
            return "yeo_johnson"
        
        # RULE 3: Count data with zeros → log1p (natural choice)
        if stats['looks_like_counts'] and stats['has_zeros']:
            return "log1p"
        
        # RULE 4: Strictly positive data → Box-Cox or log
        if stats['all_positive']:
            # Box-Cox is more flexible but slower
            # Use log for very high skewness (>3), Box-Cox for moderate (1-3)
            if skew > 3.0:
                return "log"
            else:
                return "box_cox"
        
        # RULE 5: Non-negative with few zeros → log1p
        if stats['all_nonnegative']:
            if stats['zero_percentage'] < 0.1:  # Less than 10% zeros
                return "log1p"
            else:
                return "log1p"  # Still works with many zeros
        
        # RULE 6: Heavy right tail → reciprocal
        if stats['kurtosis'] > 10 and skew > 2:
            return "reciprocal"
        
        # RULE 7: Moderate skewness, non-negative → sqrt
        if stats['all_nonnegative'] and 1 <= skew < 3:
            return "sqrt"
        
        # RULE 8: Fallback to Yeo-Johnson (handles everything)
        return "yeo_johnson"
    
    def _create_transformer(self, col_name: str, strategy: str) -> BaseEstimator | None:
        """Create transformer instance for a specific strategy.
        
        Args:
            col_name: Column name
            strategy: Transform strategy
            
        Returns:
            Transformer instance or None
        """
        if strategy == "none":
            return None
        
        elif strategy == "log":
            return LogTransformer(
                columns=[col_name],
                shift=self.log_shift,
                base="natural",
            )
        
        elif strategy == "log1p":
            return Log1pTransformer(columns=[col_name])
        
        elif strategy == "sqrt":
            return SqrtTransformer(
                columns=[col_name],
                handle_negatives=self.sqrt_handle_negatives,
            )
        
        elif strategy == "box_cox":
            return BoxCoxTransformer(
                columns=[col_name],
                lambda_value=self.boxcox_lambda,
            )
        
        elif strategy == "yeo_johnson":
            # Use sklearn's PowerTransformer directly (wrapped for consistency)
            from sklearn.preprocessing import PowerTransformer as SKLearnPT
            # Create a simple wrapper
            class YeoJohnsonWrapper(BaseEstimator, TransformerMixin):
                def __init__(self, columns):
                    self.columns = columns
                    self.pt_ = SKLearnPT(method="yeo-johnson", standardize=False)
                
                def fit(self, X, y=None):
                    self.pt_.fit(X[self.columns])
                    return self
                
                def transform(self, X):
                    df = X.copy()
                    df[self.columns] = self.pt_.transform(X[self.columns])
                    return df
            
            return YeoJohnsonWrapper(columns=[col_name])
        
        elif strategy == "reciprocal":
            return ReciprocalTransformer(
                columns=[col_name],
                epsilon=self.reciprocal_epsilon,
            )
        
        elif strategy == "exponential":
            return ExponentialTransformer(
                columns=[col_name],
                transform_type=self.exponential_type,
            )
        
        else:
            from .logging import get_logger
            logger = get_logger(__name__)
            logger.warning(f"Unknown strategy '{strategy}' for column '{col_name}'. Skipping transformation.")
            return None
    
    def get_strategies(self) -> dict[str, str]:
        """Get selected strategies per column.
        
        Returns:
            Dict mapping column names to selected strategies
        """
        return self.strategies_.copy()
    
    def get_skewness(self) -> dict[str, float]:
        """Get computed skewness values per column.
        
        Returns:
            Dict mapping column names to skewness values
        """
        return self.skewness_.copy()
    
    def get_data_stats(self) -> dict[str, dict]:
        """Get comprehensive data statistics per column.
        
        Returns:
            Dict mapping column names to statistics dicts
        """
        return {k: v.copy() for k, v in self.data_stats_.items()}


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


class BinningTransformer(BaseEstimator, TransformerMixin):
    """Comprehensive binning/discretization transformer supporting multiple strategies.
    
    Converts continuous features into discrete bins using various strategies:
    - equal_width: Fixed-width intervals (uniform distribution)
    - equal_frequency: Quantile-based bins (equal sample counts)
    - kmeans: Cluster-based bins (data-driven boundaries)
    - decision_tree: Supervised binning (target-aware boundaries)
    - custom: User-defined bin edges
    
    This enables linear models to learn non-linear patterns and threshold effects.
    """
    
    def __init__(
        self,
        columns: list[str] | None = None,
        strategy: str = "equal_width",
        n_bins: int = 5,
        encode: str = "ordinal",
        custom_bins: dict[str, list[float]] | None = None,
        handle_unknown: str = "ignore",
        handle_invalid: str = "ignore",
        subsample: int | None = 200_000,
        random_state: int = 42,
    ) -> None:
        """Initialize binning transformer.
        
        Args:
            columns: Columns to bin (None = all numeric)
            strategy: Binning strategy - 'equal_width', 'equal_frequency', 'kmeans', 
                     'decision_tree', 'custom'
            n_bins: Number of bins (ignored for custom strategy)
            encode: Output encoding - 'ordinal' (0, 1, 2...) or 'onehot'
            custom_bins: Dict mapping column names to bin edges (for custom strategy)
            handle_unknown: How to handle values outside bin range - 'ignore' or 'error'
            handle_invalid: How to handle invalid values (inf, large values) - 'ignore' or 'error'
            subsample: Subsample size for kmeans/decision_tree (None = use all data)
            random_state: Random seed for kmeans/decision_tree
        """
        self.columns = columns
        self.strategy = strategy
        self.n_bins = n_bins
        self.encode = encode
        self.custom_bins = custom_bins
        self.handle_unknown = handle_unknown
        self.handle_invalid = handle_invalid
        self.subsample = subsample
        self.random_state = random_state
        
        # Fitted attributes
        self.columns_: list[str] = []
        self.bin_edges_: dict[str, np.ndarray] = {}
        self.bin_labels_: dict[str, list[str]] = {}
        self.n_bins_: dict[str, int] = {}
        
    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "BinningTransformer":
        """Fit binning transformer by computing bin edges.
        
        Args:
            X: Input DataFrame
            y: Target Series (required for decision_tree strategy)
            
        Returns:
            Self
        """
        df = pd.DataFrame(X)
        
        # Determine columns to bin
        if self.columns is None:
            # Auto-detect numeric columns
            self.columns_ = list(df.select_dtypes(include=[np.number]).columns)
        else:
            self.columns_ = [c for c in self.columns if c in df.columns]
        
        # Compute bin edges for each column
        for col in self.columns_:
            self.bin_edges_[col], self.n_bins_[col] = self._compute_bin_edges(
                df[col], col, y
            )
            # Generate labels for bins
            self.bin_labels_[col] = self._generate_bin_labels(col, self.n_bins_[col])
        
        return self
    
    def _compute_bin_edges(
        self, 
        series: pd.Series, 
        col_name: str,
        y: pd.Series | None = None
    ) -> tuple[np.ndarray, int]:
        """Compute bin edges for a single column based on strategy.
        
        Args:
            series: Input series to bin
            col_name: Column name (for custom bins lookup)
            y: Target series (for supervised binning)
            
        Returns:
            Tuple of (bin_edges, n_bins)
        """
        # Clean data: remove NaN and infinite values
        clean_data = series.replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(clean_data) == 0:
            # No valid data - return dummy edges
            return np.array([0, 1]), 1
        
        # Convert to numpy array
        X_col = clean_data.values
        
        # Subsample if needed (for expensive methods)
        if self.subsample and len(X_col) > self.subsample:
            if self.strategy in {'kmeans', 'decision_tree'}:
                rng = np.random.RandomState(self.random_state)
                indices = rng.choice(len(X_col), size=self.subsample, replace=False)
                X_col = X_col[indices]
                if y is not None:
                    # Get the corresponding y values for the subsampled indices
                    clean_indices = clean_data.index[indices]
                    y_sampled = y.loc[clean_indices]
                else:
                    y_sampled = None
            else:
                y_sampled = y
        else:
            # Use the indices of non-NaN values from the original series
            clean_indices = clean_data.index
            y_sampled = y.loc[clean_indices] if y is not None else None
        
        # Strategy-specific binning
        if self.strategy == "equal_width":
            # Fixed-width intervals
            min_val, max_val = X_col.min(), X_col.max()
            if min_val == max_val:
                # Constant column
                edges = np.array([min_val - 0.5, max_val + 0.5])
                return edges, 1
            edges = np.linspace(min_val, max_val, self.n_bins + 1)
            return edges, self.n_bins
        
        elif self.strategy == "equal_frequency":
            # Quantile-based bins
            quantiles = np.linspace(0, 1, self.n_bins + 1)
            edges = np.percentile(X_col, quantiles * 100)
            # Remove duplicate edges (happens with discrete/constant data)
            edges = np.unique(edges)
            actual_bins = len(edges) - 1
            return edges, actual_bins
        
        elif self.strategy == "kmeans":
            # Cluster-based binning
            from sklearn.cluster import KMeans
            
            # Reshape for sklearn
            X_reshaped = X_col.reshape(-1, 1)
            
            # Fit KMeans
            n_bins = min(self.n_bins, len(np.unique(X_col)))
            kmeans = KMeans(
                n_clusters=n_bins, 
                random_state=self.random_state,
                n_init=10
            )
            kmeans.fit(X_reshaped)
            
            # Use cluster centers as bin boundaries
            centers = np.sort(kmeans.cluster_centers_.flatten())
            
            # Create edges from centers
            min_val, max_val = X_col.min(), X_col.max()
            edges = [min_val]
            for i in range(len(centers) - 1):
                # Midpoint between adjacent centers
                edges.append((centers[i] + centers[i + 1]) / 2)
            edges.append(max_val)
            edges = np.array(edges)
            
            return edges, len(edges) - 1
        
        elif self.strategy == "decision_tree":
            # Supervised binning using decision tree splits
            if y_sampled is None:
                raise ValueError("decision_tree strategy requires y (target) to be provided")
            
            from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
            from sklearn.preprocessing import LabelEncoder
            
            # Determine if classification or regression
            if pd.api.types.is_numeric_dtype(y_sampled):
                # Check if it's actually classification with numeric labels
                n_unique = y_sampled.nunique()
                if n_unique <= 20:  # Heuristic: <= 20 unique values = classification
                    tree = DecisionTreeClassifier(
                        max_leaf_nodes=self.n_bins,
                        random_state=self.random_state
                    )
                else:
                    tree = DecisionTreeRegressor(
                        max_leaf_nodes=self.n_bins,
                        random_state=self.random_state
                    )
            else:
                # Categorical target - encode it
                le = LabelEncoder()
                y_encoded = le.fit_transform(y_sampled.astype(str))
                tree = DecisionTreeClassifier(
                    max_leaf_nodes=self.n_bins,
                    random_state=self.random_state
                )
                y_sampled = pd.Series(y_encoded)
            
            # Fit tree
            X_reshaped = X_col.reshape(-1, 1)
            tree.fit(X_reshaped, y_sampled)
            
            # Extract split thresholds from tree
            thresholds = []
            tree_obj = tree.tree_
            for node_id in range(tree_obj.node_count):
                if tree_obj.children_left[node_id] != tree_obj.children_right[node_id]:
                    # Internal node (has split)
                    thresholds.append(tree_obj.threshold[node_id])
            
            if not thresholds:
                # Tree didn't split - use equal width as fallback
                min_val, max_val = X_col.min(), X_col.max()
                edges = np.linspace(min_val, max_val, 3)
                return edges, 2
            
            # Create edges from thresholds
            thresholds = sorted(thresholds)
            edges = [X_col.min()] + thresholds + [X_col.max()]
            edges = np.unique(edges)
            
            return edges, len(edges) - 1
        
        elif self.strategy == "custom":
            # User-provided bin edges
            custom_bins_dict = self.custom_bins if self.custom_bins is not None else {}
            if col_name not in custom_bins_dict:
                raise ValueError(
                    f"Column '{col_name}' not found in custom_bins. "
                    f"Provide bin edges via custom_bins parameter."
                )
            edges = np.array(sorted(custom_bins_dict[col_name]))
            if len(edges) < 2:
                raise ValueError(f"custom_bins for '{col_name}' must have at least 2 edges")
            return edges, len(edges) - 1
        
        else:
            raise ValueError(
                f"Unknown binning strategy: '{self.strategy}'. "
                f"Choose from: equal_width, equal_frequency, kmeans, decision_tree, custom"
            )
    
    def _generate_bin_labels(self, col_name: str, n_bins: int) -> list[str]:
        """Generate human-readable labels for bins.
        
        Args:
            col_name: Column name
            n_bins: Number of bins
            
        Returns:
            List of bin labels
        """
        edges = self.bin_edges_[col_name]
        labels = []
        
        for i in range(n_bins):
            left = edges[i]
            right = edges[i + 1]
            
            # Format edges nicely
            if abs(left) < 1000 and abs(right) < 1000:
                label = f"{col_name}_[{left:.2f},{right:.2f})"
            else:
                label = f"{col_name}_[{left:.1e},{right:.1e})"
            
            labels.append(label)
        
        return labels
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform by binning continuous features.
        
        Args:
            X: Input DataFrame
            
        Returns:
            Transformed DataFrame with binned features added to original features
        """
        df = pd.DataFrame(X).copy()
        
        for col in self.columns_:
            if col not in df.columns:
                # Column missing in transform - skip or error based on handle_unknown
                if self.handle_unknown == "error":
                    raise ValueError(f"Column '{col}' not found in transform data")
                continue
            
            series = df[col]
            edges = self.bin_edges_[col]
            
            # Handle invalid values (inf, very large numbers)
            if self.handle_invalid == "ignore":
                series = series.replace([np.inf, -np.inf], np.nan)
            
            # Digitize: assign each value to a bin
            # Note: np.digitize uses right=False by default (i.e., [a, b) intervals)
            bin_indices = np.digitize(series.fillna(-np.inf), edges, right=False)
            
            # Adjust indices to be 0-based and handle out-of-range
            # digitize returns 0 for < edges[0] and len(edges) for >= edges[-1]
            bin_indices = bin_indices - 1  # Shift to 0-based
            bin_indices = np.clip(bin_indices, 0, len(edges) - 2)
            
            # Handle NaN values - assign to separate bin or keep as NaN
            is_nan = series.isna()
            bin_indices[is_nan] = -1  # Use -1 for NaN
            
            if self.encode == "ordinal":
                # Return ordinal encoding (0, 1, 2, ...)
                # Replace -1 (NaN) with actual NaN
                bin_values = bin_indices.astype(float)
                bin_values[bin_values == -1] = np.nan
                df[f"{col}_binned"] = bin_values
            
            elif self.encode == "onehot":
                # One-hot encoding of bins
                n_bins = self.n_bins_[col]
                for i in range(n_bins):
                    df[f"{col}_bin_{i}"] = (bin_indices == i).astype(int)
                
                # Add indicator for NaN
                df[f"{col}_bin_nan"] = is_nan.astype(int)
            
            else:
                raise ValueError(f"Unknown encode strategy: '{self.encode}'")
        
        return df
    
    def get_feature_names_out(self, input_features=None) -> list[str]:
        """Get output feature names."""
        names = []
        
        for col in self.columns_:
            if self.encode == "ordinal":
                names.append(f"{col}_binned")
            elif self.encode == "onehot":
                n_bins = self.n_bins_.get(col, self.n_bins)
                for i in range(n_bins):
                    names.append(f"{col}_bin_{i}")
                names.append(f"{col}_bin_nan")
        
        return names
    
    def get_bin_edges(self) -> dict[str, np.ndarray]:
        """Get computed bin edges for all columns.
        
        Returns:
            Dict mapping column names to bin edges
        """
        return self.bin_edges_.copy()
    
    def get_bin_labels(self) -> dict[str, list[str]]:
        """Get human-readable bin labels.
        
        Returns:
            Dict mapping column names to bin labels
        """
        return self.bin_labels_.copy()


class AutoBinningSelector(BaseEstimator, TransformerMixin):
    """Automatically select optimal binning strategy per column.
    
    Analyzes each numeric column and chooses the best binning strategy:
    - Skewed distributions → equal_frequency
    - Uniform distributions → equal_width
    - Target correlation + supervised → decision_tree
    - Complex patterns → kmeans
    """
    
    def __init__(
        self,
        columns: list[str] | None = None,
        n_bins: int = 5,
        encode: str = "ordinal",
        prefer_supervised: bool = True,
        skewness_threshold: float = 1.0,
        random_state: int = 42,
    ) -> None:
        """Initialize auto binning selector.
        
        Args:
            columns: Columns to bin (None = all numeric)
            n_bins: Number of bins per column
            encode: Output encoding - 'ordinal' or 'onehot'
            prefer_supervised: Use decision_tree if target correlation is high
            skewness_threshold: Skewness threshold for equal_frequency vs equal_width
            random_state: Random seed
        """
        self.columns = columns
        self.n_bins = n_bins
        self.encode = encode
        self.prefer_supervised = prefer_supervised
        self.skewness_threshold = skewness_threshold
        self.random_state = random_state
        
        # Fitted attributes
        self.transformers_: dict[str, BinningTransformer] = {}
        self.strategies_: dict[str, str] = {}
        self.columns_: list[str] = []
    
    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "AutoBinningSelector":
        """Fit by selecting optimal strategy per column.
        
        Args:
            X: Input DataFrame
            y: Target Series (optional, enables supervised binning)
            
        Returns:
            Self
        """
        df = pd.DataFrame(X)
        
        # Determine columns
        if self.columns is None:
            self.columns_ = list(df.select_dtypes(include=[np.number]).columns)
        else:
            self.columns_ = [c for c in self.columns if c in df.columns]
        
        # Select strategy for each column
        for col in self.columns_:
            strategy = self._select_strategy(df[col], y)
            self.strategies_[col] = strategy
            
            # Create and fit transformer for this column
            transformer = BinningTransformer(
                columns=[col],
                strategy=strategy,
                n_bins=self.n_bins,
                encode=self.encode,
                random_state=self.random_state,
            )
            transformer.fit(df[[col]], y)
            self.transformers_[col] = transformer
        
        return self
    
    def _select_strategy(self, series: pd.Series, y: pd.Series | None) -> str:
        """Select optimal binning strategy for a column.
        
        Args:
            series: Input series
            y: Target series (optional)
            
        Returns:
            Strategy name
        """
        clean_data = series.replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(clean_data) < 10:
            # Too few samples - use simple strategy
            return "equal_width"
        
        # Compute statistics
        try:
            skewness = abs(clean_data.skew())
        except:
            skewness = 0.0
        
        # Check target correlation if supervised mode
        if self.prefer_supervised and y is not None:
            try:
                # Compute correlation with target
                from scipy.stats import pearsonr, spearmanr
                
                # Align series and target
                aligned_x = series.loc[y.index].dropna()
                aligned_y = y.loc[aligned_x.index]
                
                if len(aligned_x) > 10:
                    # Use Spearman for robustness
                    corr, pval = spearmanr(aligned_x, aligned_y)
                    
                    # If strong correlation, use supervised binning
                    if abs(corr) > 0.3 and pval < 0.05:
                        return "decision_tree"
            except:
                pass
        
        # Skewness-based selection
        if skewness > self.skewness_threshold:
            return "equal_frequency"
        else:
            return "equal_width"
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform using column-specific strategies.
        
        Args:
            X: Input DataFrame
            
        Returns:
            Transformed DataFrame with binned features added
        """
        df = pd.DataFrame(X).copy()
        
        for col in self.columns_:
            if col in self.transformers_:
                # Each transformer returns the column plus its binned version
                transformed = self.transformers_[col].transform(df[[col]])
                # Add the binned columns to df
                for new_col in transformed.columns:
                    if new_col != col:  # Don't duplicate the original column
                        df[new_col] = transformed[new_col]
        
        return df
    
    def get_feature_names_out(self, input_features=None) -> list[str]:
        """Get output feature names."""
        names = []
        for col in self.columns_:
            if col in self.transformers_:
                names.extend(self.transformers_[col].get_feature_names_out())
        return names
    
    def get_strategies(self) -> dict[str, str]:
        """Get selected strategies per column.
        
        Returns:
            Dict mapping column names to selected strategies
        """
        return self.strategies_.copy()


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
