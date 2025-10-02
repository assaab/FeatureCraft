"""Time series utilities for FeatureCraft."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import TimeSeriesSplit

from .logging import get_logger

logger = get_logger(__name__)


def make_lags(df: pd.DataFrame, cols: list[str], lags: list[int]) -> pd.DataFrame:
    """Create lag features."""
    out = pd.DataFrame(index=df.index)
    for c in cols:
        for L in lags:
            out[f"{c}_lag{L}"] = df[c].shift(L)
    return out


def make_rolling(
    df: pd.DataFrame,
    cols: list[str],
    windows: list[int],
    funcs: Iterable[str] = ("mean", "std", "min", "max"),
) -> pd.DataFrame:
    """Create rolling window features."""
    out = pd.DataFrame(index=df.index)
    for c in cols:
        for w in windows:
            roll = df[c].rolling(window=w, min_periods=1)
            if "mean" in funcs:
                out[f"{c}_roll{w}_mean"] = roll.mean()
            if "std" in funcs:
                out[f"{c}_roll{w}_std"] = roll.std()
            if "min" in funcs:
                out[f"{c}_roll{w}_min"] = roll.min()
            if "max" in funcs:
                out[f"{c}_roll{w}_max"] = roll.max()
            out[f"{c}_roll{w}_count"] = roll.count()
    return out


def time_series_split(n_splits: int = 5) -> TimeSeriesSplit:
    """Create time series split."""
    return TimeSeriesSplit(n_splits=n_splits)


class FourierFeatures(BaseEstimator, TransformerMixin):
    """Generate Fourier features for time series periodicity."""
    
    def __init__(
        self,
        column: str,
        orders: list[int] = [3, 7],
        period: float = 365.25,
    ) -> None:
        """Initialize Fourier features.
        
        Args:
            column: Time column name
            orders: Fourier orders to generate
            period: Period length (e.g., 365.25 for yearly seasonality)
        """
        self.column = column
        self.orders = orders
        self.period = period
        self.feature_names_: list[str] = []
    
    def fit(self, X: pd.DataFrame, y=None) -> "FourierFeatures":
        """Fit transformer."""
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate Fourier features."""
        df = pd.DataFrame(X).copy()
        out = pd.DataFrame(index=df.index)
        
        if self.column not in df.columns:
            logger.warning(f"Column '{self.column}' not found for Fourier features")
            return out
        
        # Convert to numeric time index
        time_series = pd.to_datetime(df[self.column], errors='coerce')
        if time_series.isna().all():
            logger.warning(f"Cannot convert column '{self.column}' to datetime")
            return out
        
        # Calculate time index from first observation
        time_idx = (time_series - time_series.min()).dt.total_seconds() / (24 * 3600)
        
        # Generate Fourier terms
        for order in self.orders:
            freq = 2 * np.pi * order / self.period
            out[f"{self.column}_fourier_sin_{order}"] = np.sin(freq * time_idx)
            out[f"{self.column}_fourier_cos_{order}"] = np.cos(freq * time_idx)
        
        self.feature_names_ = list(out.columns)
        return out
    
    def get_feature_names_out(self, input_features=None):
        """Get output feature names."""
        return self.feature_names_ if self.feature_names_ else []


class HolidayFeatures(BaseEstimator, TransformerMixin):
    """Generate holiday features using holidays package."""
    
    def __init__(
        self,
        column: str,
        country_code: Optional[str] = None,
    ) -> None:
        """Initialize holiday features.
        
        Args:
            column: Date column name
            country_code: Country code for holidays (e.g., 'US', 'GB', 'FR')
        """
        self.column = column
        self.country_code = country_code
        self.holidays_ = None
        self.feature_names_: list[str] = []
    
    def fit(self, X: pd.DataFrame, y=None) -> "HolidayFeatures":
        """Fit transformer."""
        if self.country_code:
            try:
                import holidays
                
                # Get holidays for the country
                self.holidays_ = getattr(holidays, self.country_code, None)
                if self.holidays_ is None:
                    logger.warning(
                        f"Country code '{self.country_code}' not recognized. "
                        "Holiday features disabled."
                    )
            except ImportError:
                logger.warning(
                    "holidays package not installed. "
                    "Install with: pip install holidays"
                )
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate holiday features."""
        df = pd.DataFrame(X).copy()
        out = pd.DataFrame(index=df.index)
        
        if self.column not in df.columns:
            logger.warning(f"Column '{self.column}' not found for holiday features")
            return out
        
        if self.holidays_ is None:
            return out
        
        # Convert to datetime
        date_series = pd.to_datetime(df[self.column], errors='coerce')
        
        # Check if date is a holiday
        out[f"{self.column}_is_holiday"] = date_series.apply(
            lambda x: 1 if pd.notna(x) and x.date() in self.holidays_() else 0
        )
        
        # Days until next holiday
        def days_to_holiday(date):
            if pd.isna(date):
                return np.nan
            
            holiday_dates = sorted(self.holidays_().keys())
            future_holidays = [h for h in holiday_dates if h > date.date()]
            
            if future_holidays:
                return (future_holidays[0] - date.date()).days
            return np.nan
        
        out[f"{self.column}_days_to_holiday"] = date_series.apply(days_to_holiday)
        
        self.feature_names_ = list(out.columns)
        return out
    
    def get_feature_names_out(self, input_features=None):
        """Get output feature names."""
        return self.feature_names_ if self.feature_names_ else []
