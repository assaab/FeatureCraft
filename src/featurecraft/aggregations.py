"""Aggregation and group-based feature transformers for time-series and hierarchical data.

This module provides transformers for:
- GroupBy statistics (mean, sum, std, count, min, max per group)
- Rolling window features (moving averages, sums, etc.)
- Expanding window features (cumulative statistics)
- Lag features (previous values at various time steps)
- Rank features (percentile rankings within groups)

These are essential for:
- Customer transaction aggregations
- Time-series forecasting
- Hierarchical data modeling
- Historical pattern extraction
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .logging import get_logger

logger = get_logger(__name__)


class GroupByStatsTransformer(BaseEstimator, TransformerMixin):
    """Create group-level statistics for hierarchical data.
    
    Equivalent to SQL GROUP BY aggregations:
    SELECT group_col, AGG(value_col) FROM table GROUP BY group_col
    
    Examples:
        # Customer transaction aggregations
        >>> transformer = GroupByStatsTransformer(
        ...     group_cols=['customer_id'],
        ...     agg_cols=['amount'],
        ...     agg_functions=['mean', 'sum', 'std', 'count']
        ... )
        >>> X_transformed = transformer.fit_transform(X)
        # Creates: customer_id_amount_mean, customer_id_amount_sum, etc.
        
        # Store-level statistics
        >>> transformer = GroupByStatsTransformer(
        ...     group_cols=['store_id', 'category'],
        ...     agg_cols=['sales', 'quantity'],
        ...     agg_functions=['mean', 'max', 'min']
        ... )
    
    This enables models to leverage group-level patterns like:
    - Average customer spending
    - Total transaction count per user
    - Product sales volatility (std)
    - Store-category performance metrics
    """
    
    def __init__(
        self,
        group_cols: list[str] | str,
        agg_cols: list[str] | str | None = None,
        agg_functions: list[str] | str = "mean",
        add_count: bool = True,
        prefix: str | None = None,
        suffix: str | None = None,
        dropna: bool = True,
        fill_missing_groups: bool = True,
        missing_fill_value: float = 0.0,
    ) -> None:
        """Initialize GroupBy statistics transformer.
        
        Args:
            group_cols: Column(s) to group by (e.g., 'customer_id', ['store_id', 'category'])
            agg_cols: Column(s) to aggregate (None = all numeric columns)
            agg_functions: Aggregation function(s): 'mean', 'sum', 'std', 'min', 'max', 'median', 
                          'count', 'nunique', 'var', 'skew', 'kurt', or list of these
            add_count: Add group size (count) as a feature
            prefix: Prefix for output column names (default: auto-generated from group_cols)
            suffix: Suffix for output column names
            dropna: Drop NaN values before aggregation
            fill_missing_groups: Fill missing group values (e.g., new customers in test set)
            missing_fill_value: Value to use for missing groups (default: 0.0)
        """
        self.group_cols = [group_cols] if isinstance(group_cols, str) else list(group_cols)
        self.agg_cols = [agg_cols] if isinstance(agg_cols, str) else (list(agg_cols) if agg_cols else None)
        self.agg_functions = [agg_functions] if isinstance(agg_functions, str) else list(agg_functions)
        self.add_count = add_count
        self.prefix = prefix
        self.suffix = suffix
        self.dropna = dropna
        self.fill_missing_groups = fill_missing_groups
        self.missing_fill_value = missing_fill_value
        
        # Fitted attributes
        self.agg_cols_: list[str] = []
        self.group_stats_: pd.DataFrame | None = None
        self.feature_names_: list[str] = []
        
    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "GroupByStatsTransformer":
        """Compute group statistics from training data.
        
        Args:
            X: Training data
            y: Ignored, kept for sklearn compatibility
            
        Returns:
            self
        """
        df = pd.DataFrame(X).copy()
        
        # Validate group columns exist
        missing_cols = [col for col in self.group_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Group columns not found: {missing_cols}")
        
        # Auto-detect aggregation columns if not specified
        if self.agg_cols is None:
            # Use all numeric columns except group columns
            self.agg_cols_ = [col for col in df.select_dtypes(include=[np.number]).columns 
                             if col not in self.group_cols]
        else:
            self.agg_cols_ = self.agg_cols
            
        # Validate aggregation columns
        missing_agg_cols = [col for col in self.agg_cols_ if col not in df.columns]
        if missing_agg_cols:
            raise ValueError(f"Aggregation columns not found: {missing_agg_cols}")
        
        if not self.agg_cols_:
            logger.warning("No numeric columns found for aggregation")
            self.group_stats_ = pd.DataFrame()
            self.feature_names_ = []
            return self
        
        # Build aggregation dictionary
        agg_dict = {}
        for col in self.agg_cols_:
            agg_dict[col] = self.agg_functions.copy()
        
        # Compute group statistics
        if self.dropna:
            grouped = df.groupby(self.group_cols, dropna=True)
        else:
            grouped = df.groupby(self.group_cols, dropna=False)
        
        self.group_stats_ = grouped.agg(agg_dict)
        
        # Flatten multi-level column index
        self.group_stats_.columns = ['_'.join(col).strip('_') for col in self.group_stats_.columns.values]
        
        # Add count if requested
        if self.add_count:
            count_col = '_'.join(self.group_cols) + '_count'
            self.group_stats_[count_col] = grouped.size()
        
        # Reset index to make group columns regular columns
        self.group_stats_ = self.group_stats_.reset_index()
        
        # Generate feature names (excluding group columns)
        group_col_set = set(self.group_cols)
        self.feature_names_ = [col for col in self.group_stats_.columns if col not in group_col_set]
        
        # Apply prefix/suffix
        if self.prefix:
            self.feature_names_ = [f"{self.prefix}_{name}" for name in self.feature_names_]
            rename_dict = {old: new for old, new in zip(
                [col for col in self.group_stats_.columns if col not in group_col_set],
                self.feature_names_
            )}
            self.group_stats_ = self.group_stats_.rename(columns=rename_dict)
        
        if self.suffix:
            self.feature_names_ = [f"{name}_{self.suffix}" for name in self.feature_names_]
            rename_dict = {old: new for old, new in zip(
                [col for col in self.group_stats_.columns if col not in group_col_set],
                self.feature_names_
            )}
            self.group_stats_ = self.group_stats_.rename(columns=rename_dict)
        
        logger.info(f"Computed {len(self.feature_names_)} group statistics for {len(self.group_stats_)} groups")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add group statistics to data by merging on group columns.
        
        Args:
            X: Data to transform
            
        Returns:
            DataFrame with added group statistics
        """
        if self.group_stats_ is None or self.group_stats_.empty:
            logger.warning("No group statistics computed, returning original data")
            return pd.DataFrame(X)
        
        df = pd.DataFrame(X).copy()
        
        # Merge group statistics
        result = df.merge(
            self.group_stats_,
            on=self.group_cols,
            how='left',
            suffixes=('', '_groupby')
        )
        
        # Fill missing groups if requested
        if self.fill_missing_groups:
            for col in self.feature_names_:
                if col in result.columns:
                    result[col] = result[col].fillna(self.missing_fill_value)
        
        # Return only new features
        return result[self.feature_names_]
    
    def get_feature_names_out(self, input_features: Sequence[str] | None = None) -> list[str]:
        """Get output feature names.
        
        Returns:
            List of feature names
        """
        return self.feature_names_


class RollingWindowTransformer(BaseEstimator, TransformerMixin):
    """Create rolling window features for time-series data.
    
    Equivalent to SQL window functions:
    SELECT AVG(value) OVER (ORDER BY date ROWS 7 PRECEDING) FROM table
    
    Examples:
        # 7-day moving average for sales
        >>> transformer = RollingWindowTransformer(
        ...     columns=['sales'],
        ...     window_sizes=[7, 14, 28],
        ...     agg_functions=['mean', 'sum', 'std']
        ... )
        >>> X_transformed = transformer.fit_transform(X)
        # Creates: sales_roll_7_mean, sales_roll_7_sum, sales_roll_7_std, etc.
        
        # Store-specific rolling windows
        >>> transformer = RollingWindowTransformer(
        ...     columns=['sales', 'quantity'],
        ...     window_sizes=[3, 7],
        ...     agg_functions=['mean', 'max'],
        ...     group_by='store_id',
        ...     time_col='date'
        ... )
    
    This captures:
    - Short-term trends (3-day windows)
    - Medium-term patterns (7-day windows)
    - Long-term seasonality (28-day windows)
    - Volatility (rolling std)
    """
    
    def __init__(
        self,
        columns: list[str] | str,
        window_sizes: list[int] | int = 7,
        agg_functions: list[str] | str = "mean",
        group_by: str | list[str] | None = None,
        time_col: str | None = None,
        min_periods: int | None = None,
        center: bool = False,
        shift: int = 1,
    ) -> None:
        """Initialize rolling window transformer.
        
        Args:
            columns: Column(s) to compute rolling windows for
            window_sizes: Window size(s) in number of rows (e.g., 7 for 7-day windows)
            agg_functions: Aggregation function(s): 'mean', 'sum', 'std', 'min', 'max', 
                          'median', 'var', 'skew', 'kurt', or list
            group_by: Column(s) to group by (e.g., 'store_id' for store-specific windows)
            time_col: Time column for sorting (required if group_by is used)
            min_periods: Minimum observations required (default: window_size)
            center: Center the window (default: False = trailing window)
            shift: Shift window by N periods to avoid leakage (default: 1)
                  shift=1 means window ends at t-1 (doesn't include current row)
        """
        self.columns = [columns] if isinstance(columns, str) else list(columns)
        self.window_sizes = [window_sizes] if isinstance(window_sizes, int) else list(window_sizes)
        self.agg_functions = [agg_functions] if isinstance(agg_functions, str) else list(agg_functions)
        self.group_by = [group_by] if isinstance(group_by, str) else (list(group_by) if group_by else None)
        self.time_col = time_col
        self.min_periods = min_periods
        self.center = center
        self.shift = shift
        
        # Fitted attributes
        self.feature_names_: list[str] = []
        
    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "RollingWindowTransformer":
        """Fit transformer (just validates and generates feature names).
        
        Args:
            X: Training data
            y: Ignored
            
        Returns:
            self
        """
        df = pd.DataFrame(X)
        
        # Validate columns
        missing_cols = [col for col in self.columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Columns not found: {missing_cols}")
        
        if self.group_by:
            missing_group_cols = [col for col in self.group_by if col not in df.columns]
            if missing_group_cols:
                raise ValueError(f"Group columns not found: {missing_group_cols}")
            
            if self.time_col is None:
                logger.warning("time_col not specified with group_by. Assuming data is already sorted.")
            elif self.time_col not in df.columns:
                raise ValueError(f"Time column not found: {self.time_col}")
        
        # Generate feature names
        self.feature_names_ = []
        for col in self.columns:
            for window_size in self.window_sizes:
                for agg_func in self.agg_functions:
                    feature_name = f"{col}_roll_{window_size}_{agg_func}"
                    self.feature_names_.append(feature_name)
        
        logger.info(f"Will create {len(self.feature_names_)} rolling window features")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Compute rolling window features.
        
        Args:
            X: Data to transform
            
        Returns:
            DataFrame with rolling window features
        """
        df = pd.DataFrame(X).copy()
        
        # Sort by time if specified
        if self.time_col and self.time_col in df.columns:
            df = df.sort_values(self.time_col)
        
        result = pd.DataFrame(index=df.index)
        
        # Compute rolling windows
        if self.group_by:
            # Group-wise rolling windows
            for col in self.columns:
                for window_size in self.window_sizes:
                    min_periods = self.min_periods if self.min_periods is not None else window_size
                    
                    # Shift the column to avoid leakage
                    shifted_col = df.groupby(self.group_by)[col].shift(self.shift)
                    
                    for agg_func in self.agg_functions:
                        feature_name = f"{col}_roll_{window_size}_{agg_func}"
                        
                        rolling = shifted_col.groupby(df[self.group_by[0]] if len(self.group_by) == 1 else df[self.group_by].apply(tuple, axis=1)).rolling(
                            window=window_size,
                            min_periods=min_periods,
                            center=self.center
                        )
                        
                        if agg_func == 'mean':
                            result[feature_name] = rolling.mean().reset_index(level=0, drop=True)
                        elif agg_func == 'sum':
                            result[feature_name] = rolling.sum().reset_index(level=0, drop=True)
                        elif agg_func == 'std':
                            result[feature_name] = rolling.std().reset_index(level=0, drop=True)
                        elif agg_func == 'min':
                            result[feature_name] = rolling.min().reset_index(level=0, drop=True)
                        elif agg_func == 'max':
                            result[feature_name] = rolling.max().reset_index(level=0, drop=True)
                        elif agg_func == 'median':
                            result[feature_name] = rolling.median().reset_index(level=0, drop=True)
                        elif agg_func == 'var':
                            result[feature_name] = rolling.var().reset_index(level=0, drop=True)
                        elif agg_func == 'skew':
                            result[feature_name] = rolling.skew().reset_index(level=0, drop=True)
                        elif agg_func == 'kurt':
                            result[feature_name] = rolling.kurt().reset_index(level=0, drop=True)
                        else:
                            logger.warning(f"Unknown aggregation function: {agg_func}, skipping")
        else:
            # Global rolling windows
            for col in self.columns:
                for window_size in self.window_sizes:
                    min_periods = self.min_periods if self.min_periods is not None else window_size
                    
                    # Shift the column to avoid leakage
                    shifted_col = df[col].shift(self.shift)
                    
                    rolling = shifted_col.rolling(
                        window=window_size,
                        min_periods=min_periods,
                        center=self.center
                    )
                    
                    for agg_func in self.agg_functions:
                        feature_name = f"{col}_roll_{window_size}_{agg_func}"
                        
                        if agg_func == 'mean':
                            result[feature_name] = rolling.mean()
                        elif agg_func == 'sum':
                            result[feature_name] = rolling.sum()
                        elif agg_func == 'std':
                            result[feature_name] = rolling.std()
                        elif agg_func == 'min':
                            result[feature_name] = rolling.min()
                        elif agg_func == 'max':
                            result[feature_name] = rolling.max()
                        elif agg_func == 'median':
                            result[feature_name] = rolling.median()
                        elif agg_func == 'var':
                            result[feature_name] = rolling.var()
                        elif agg_func == 'skew':
                            result[feature_name] = rolling.skew()
                        elif agg_func == 'kurt':
                            result[feature_name] = rolling.kurt()
                        else:
                            logger.warning(f"Unknown aggregation function: {agg_func}, skipping")
        
        return result[self.feature_names_]
    
    def get_feature_names_out(self, input_features: Sequence[str] | None = None) -> list[str]:
        """Get output feature names."""
        return self.feature_names_


class ExpandingWindowTransformer(BaseEstimator, TransformerMixin):
    """Create expanding window features (cumulative statistics).
    
    Equivalent to SQL cumulative functions:
    SELECT SUM(value) OVER (ORDER BY date ROWS UNBOUNDED PRECEDING) FROM table
    
    Examples:
        # Cumulative sum and mean
        >>> transformer = ExpandingWindowTransformer(
        ...     columns=['sales', 'revenue'],
        ...     agg_functions=['sum', 'mean']
        ... )
        >>> X_transformed = transformer.fit_transform(X)
        # Creates: sales_expanding_sum, sales_expanding_mean, etc.
        
        # Customer lifetime value (cumulative revenue)
        >>> transformer = ExpandingWindowTransformer(
        ...     columns=['order_value'],
        ...     agg_functions=['sum', 'count'],
        ...     group_by='customer_id',
        ...     time_col='order_date'
        ... )
    
    This captures:
    - Running totals (cumulative sum)
    - Customer lifetime metrics
    - Sequential growth patterns
    - Historical context accumulation
    """
    
    def __init__(
        self,
        columns: list[str] | str,
        agg_functions: list[str] | str = "sum",
        group_by: str | list[str] | None = None,
        time_col: str | None = None,
        min_periods: int = 1,
        shift: int = 1,
    ) -> None:
        """Initialize expanding window transformer.
        
        Args:
            columns: Column(s) to compute expanding windows for
            agg_functions: Aggregation function(s): 'sum', 'mean', 'std', 'min', 'max', 
                          'median', 'var', 'count', or list
            group_by: Column(s) to group by (e.g., 'customer_id')
            time_col: Time column for sorting (required if group_by is used)
            min_periods: Minimum observations required (default: 1)
            shift: Shift window by N periods to avoid leakage (default: 1)
        """
        self.columns = [columns] if isinstance(columns, str) else list(columns)
        self.agg_functions = [agg_functions] if isinstance(agg_functions, str) else list(agg_functions)
        self.group_by = [group_by] if isinstance(group_by, str) else (list(group_by) if group_by else None)
        self.time_col = time_col
        self.min_periods = min_periods
        self.shift = shift
        
        # Fitted attributes
        self.feature_names_: list[str] = []
        
    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "ExpandingWindowTransformer":
        """Fit transformer (just validates and generates feature names)."""
        df = pd.DataFrame(X)
        
        # Validate columns
        missing_cols = [col for col in self.columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Columns not found: {missing_cols}")
        
        if self.group_by:
            missing_group_cols = [col for col in self.group_by if col not in df.columns]
            if missing_group_cols:
                raise ValueError(f"Group columns not found: {missing_group_cols}")
            
            if self.time_col and self.time_col not in df.columns:
                raise ValueError(f"Time column not found: {self.time_col}")
        
        # Generate feature names
        self.feature_names_ = []
        for col in self.columns:
            for agg_func in self.agg_functions:
                feature_name = f"{col}_expanding_{agg_func}"
                self.feature_names_.append(feature_name)
        
        logger.info(f"Will create {len(self.feature_names_)} expanding window features")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Compute expanding window features."""
        df = pd.DataFrame(X).copy()
        
        # Sort by time if specified
        if self.time_col and self.time_col in df.columns:
            df = df.sort_values(self.time_col)
        
        result = pd.DataFrame(index=df.index)
        
        # Compute expanding windows
        if self.group_by:
            # Group-wise expanding windows
            for col in self.columns:
                # Shift to avoid leakage
                shifted_col = df.groupby(self.group_by)[col].shift(self.shift)
                
                for agg_func in self.agg_functions:
                    feature_name = f"{col}_expanding_{agg_func}"
                    
                    if agg_func == 'count':
                        # For count, we just use cumulative count
                        result[feature_name] = df.groupby(self.group_by).cumcount()
                    else:
                        expanding = shifted_col.groupby(df[self.group_by[0]] if len(self.group_by) == 1 else df[self.group_by].apply(tuple, axis=1)).expanding(
                            min_periods=self.min_periods
                        )
                        
                        if agg_func == 'sum':
                            result[feature_name] = expanding.sum().reset_index(level=0, drop=True)
                        elif agg_func == 'mean':
                            result[feature_name] = expanding.mean().reset_index(level=0, drop=True)
                        elif agg_func == 'std':
                            result[feature_name] = expanding.std().reset_index(level=0, drop=True)
                        elif agg_func == 'min':
                            result[feature_name] = expanding.min().reset_index(level=0, drop=True)
                        elif agg_func == 'max':
                            result[feature_name] = expanding.max().reset_index(level=0, drop=True)
                        elif agg_func == 'median':
                            result[feature_name] = expanding.median().reset_index(level=0, drop=True)
                        elif agg_func == 'var':
                            result[feature_name] = expanding.var().reset_index(level=0, drop=True)
                        else:
                            logger.warning(f"Unknown aggregation function: {agg_func}, skipping")
        else:
            # Global expanding windows
            for col in self.columns:
                # Shift to avoid leakage
                shifted_col = df[col].shift(self.shift)
                
                for agg_func in self.agg_functions:
                    feature_name = f"{col}_expanding_{agg_func}"
                    
                    if agg_func == 'count':
                        result[feature_name] = range(len(df))
                    else:
                        expanding = shifted_col.expanding(min_periods=self.min_periods)
                        
                        if agg_func == 'sum':
                            result[feature_name] = expanding.sum()
                        elif agg_func == 'mean':
                            result[feature_name] = expanding.mean()
                        elif agg_func == 'std':
                            result[feature_name] = expanding.std()
                        elif agg_func == 'min':
                            result[feature_name] = expanding.min()
                        elif agg_func == 'max':
                            result[feature_name] = expanding.max()
                        elif agg_func == 'median':
                            result[feature_name] = expanding.median()
                        elif agg_func == 'var':
                            result[feature_name] = expanding.var()
                        else:
                            logger.warning(f"Unknown aggregation function: {agg_func}, skipping")
        
        return result[self.feature_names_]
    
    def get_feature_names_out(self, input_features: Sequence[str] | None = None) -> list[str]:
        """Get output feature names."""
        return self.feature_names_


class LagFeaturesTransformer(BaseEstimator, TransformerMixin):
    """Create lag features (previous values at various time steps).
    
    Equivalent to SQL LAG function:
    SELECT LAG(value, 1) OVER (ORDER BY date) FROM table
    
    Examples:
        # Sales from previous day and week
        >>> transformer = LagFeaturesTransformer(
        ...     columns=['sales'],
        ...     lags=[1, 7, 14, 28]
        ... )
        >>> X_transformed = transformer.fit_transform(X)
        # Creates: sales_lag_1, sales_lag_7, sales_lag_14, sales_lag_28
        
        # Store-specific lags
        >>> transformer = LagFeaturesTransformer(
        ...     columns=['sales', 'customers'],
        ...     lags=[1, 2, 3],
        ...     group_by='store_id',
        ...     time_col='date'
        ... )
    
    This captures:
    - Previous day values (lag=1)
    - Week-over-week patterns (lag=7)
    - Month-over-month patterns (lag=28)
    - Autoregressive relationships
    """
    
    def __init__(
        self,
        columns: list[str] | str,
        lags: list[int] | int = 1,
        group_by: str | list[str] | None = None,
        time_col: str | None = None,
        fill_value: float | None = None,
    ) -> None:
        """Initialize lag features transformer.
        
        Args:
            columns: Column(s) to create lags for
            lags: Lag period(s) in number of rows (e.g., [1, 7] for 1-day and 7-day lags)
            group_by: Column(s) to group by (e.g., 'store_id' for store-specific lags)
            time_col: Time column for sorting (required if group_by is used)
            fill_value: Value to fill missing lags (default: None = leave as NaN)
        """
        self.columns = [columns] if isinstance(columns, str) else list(columns)
        self.lags = [lags] if isinstance(lags, int) else list(lags)
        self.group_by = [group_by] if isinstance(group_by, str) else (list(group_by) if group_by else None)
        self.time_col = time_col
        self.fill_value = fill_value
        
        # Fitted attributes
        self.feature_names_: list[str] = []
        
    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "LagFeaturesTransformer":
        """Fit transformer (just validates and generates feature names)."""
        df = pd.DataFrame(X)
        
        # Validate columns
        missing_cols = [col for col in self.columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Columns not found: {missing_cols}")
        
        if self.group_by:
            missing_group_cols = [col for col in self.group_by if col not in df.columns]
            if missing_group_cols:
                raise ValueError(f"Group columns not found: {missing_group_cols}")
            
            if self.time_col and self.time_col not in df.columns:
                raise ValueError(f"Time column not found: {self.time_col}")
        
        # Generate feature names
        self.feature_names_ = []
        for col in self.columns:
            for lag in self.lags:
                feature_name = f"{col}_lag_{lag}"
                self.feature_names_.append(feature_name)
        
        logger.info(f"Will create {len(self.feature_names_)} lag features")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Compute lag features."""
        df = pd.DataFrame(X).copy()
        
        # Sort by time if specified
        if self.time_col and self.time_col in df.columns:
            df = df.sort_values(self.time_col)
        
        result = pd.DataFrame(index=df.index)
        
        # Compute lags
        if self.group_by:
            # Group-wise lags
            for col in self.columns:
                for lag in self.lags:
                    feature_name = f"{col}_lag_{lag}"
                    result[feature_name] = df.groupby(self.group_by)[col].shift(lag)
                    
                    if self.fill_value is not None:
                        result[feature_name] = result[feature_name].fillna(self.fill_value)
        else:
            # Global lags
            for col in self.columns:
                for lag in self.lags:
                    feature_name = f"{col}_lag_{lag}"
                    result[feature_name] = df[col].shift(lag)
                    
                    if self.fill_value is not None:
                        result[feature_name] = result[feature_name].fillna(self.fill_value)
        
        return result[self.feature_names_]
    
    def get_feature_names_out(self, input_features: Sequence[str] | None = None) -> list[str]:
        """Get output feature names."""
        return self.feature_names_


class RankFeaturesTransformer(BaseEstimator, TransformerMixin):
    """Create rank/percentile features within groups.
    
    Equivalent to SQL PERCENT_RANK:
    SELECT PERCENT_RANK() OVER (PARTITION BY group ORDER BY value) FROM table
    
    Examples:
        # Customer spending rank among all customers
        >>> transformer = RankFeaturesTransformer(
        ...     columns=['total_spent'],
        ...     method='percent'
        ... )
        >>> X_transformed = transformer.fit_transform(X)
        # Creates: total_spent_pct_rank (0.0 to 1.0)
        
        # Product rank within category
        >>> transformer = RankFeaturesTransformer(
        ...     columns=['price', 'sales'],
        ...     group_by='category',
        ...     method='percent',
        ...     ascending=False
        ... )
        # Creates: price_rank_in_category, sales_rank_in_category
    
    This captures:
    - Relative position within groups
    - Percentile rankings
    - Top/bottom performers
    - Competitive positioning
    """
    
    def __init__(
        self,
        columns: list[str] | str,
        group_by: str | list[str] | None = None,
        method: Literal['average', 'min', 'max', 'dense', 'percent'] = 'percent',
        ascending: bool = True,
        pct: bool = False,
    ) -> None:
        """Initialize rank features transformer.
        
        Args:
            columns: Column(s) to rank
            group_by: Column(s) to group by (None = global ranking)
            method: Ranking method:
                - 'average': average rank of tied values
                - 'min': minimum rank of tied values
                - 'max': maximum rank of tied values
                - 'dense': like min, but rank always increases by 1
                - 'percent': percentile rank (0.0 to 1.0)
            ascending: Rank in ascending order (True) or descending (False)
            pct: Convert rank to percentile (0.0 to 1.0) for non-percent methods
        """
        self.columns = [columns] if isinstance(columns, str) else list(columns)
        self.group_by = [group_by] if isinstance(group_by, str) else (list(group_by) if group_by else None)
        self.method = method
        self.ascending = ascending
        self.pct = pct
        
        # Fitted attributes
        self.feature_names_: list[str] = []
        
    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "RankFeaturesTransformer":
        """Fit transformer (just validates and generates feature names)."""
        df = pd.DataFrame(X)
        
        # Validate columns
        missing_cols = [col for col in self.columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Columns not found: {missing_cols}")
        
        if self.group_by:
            missing_group_cols = [col for col in self.group_by if col not in df.columns]
            if missing_group_cols:
                raise ValueError(f"Group columns not found: {missing_group_cols}")
        
        # Generate feature names
        self.feature_names_ = []
        for col in self.columns:
            if self.method == 'percent' or self.pct:
                feature_name = f"{col}_pct_rank"
            else:
                feature_name = f"{col}_rank"
            
            if self.group_by:
                group_suffix = '_'.join(self.group_by)
                feature_name += f"_in_{group_suffix}"
            
            self.feature_names_.append(feature_name)
        
        logger.info(f"Will create {len(self.feature_names_)} rank features")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Compute rank features."""
        df = pd.DataFrame(X).copy()
        
        result = pd.DataFrame(index=df.index)
        
        # Compute ranks
        if self.group_by:
            # Group-wise ranks
            for col, feature_name in zip(self.columns, self.feature_names_):
                result[feature_name] = df.groupby(self.group_by)[col].rank(
                    method=self.method if self.method != 'percent' else 'average',
                    ascending=self.ascending,
                    pct=(self.method == 'percent' or self.pct)
                )
        else:
            # Global ranks
            for col, feature_name in zip(self.columns, self.feature_names_):
                result[feature_name] = df[col].rank(
                    method=self.method if self.method != 'percent' else 'average',
                    ascending=self.ascending,
                    pct=(self.method == 'percent' or self.pct)
                )
        
        return result[self.feature_names_]
    
    def get_feature_names_out(self, input_features: Sequence[str] | None = None) -> list[str]:
        """Get output feature names."""
        return self.feature_names_

