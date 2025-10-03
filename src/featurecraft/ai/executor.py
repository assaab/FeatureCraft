"""Plan executor that translates feature specifications to actual transformations."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .schemas import FeaturePlan, FeatureSpec
from ..logging import get_logger

logger = get_logger(__name__)


# ============================================================================
# Custom Transformers for DSL Operations
# ============================================================================

class RollingAggregator(BaseEstimator, TransformerMixin):
    """Rolling window aggregation transformer."""
    
    def __init__(
        self,
        source_col: str,
        window: str,
        agg_func: str,
        key_col: str | None = None,
        time_col: str | None = None,
        output_name: str | None = None,
    ):
        """Initialize rolling aggregator.
        
        Args:
            source_col: Source column to aggregate
            window: Window size (e.g., "7d", "30d")
            agg_func: Aggregation function (mean, sum, std, min, max)
            key_col: Groupby key (e.g., customer_id)
            time_col: Time column for ordering
            output_name: Output feature name
        """
        self.source_col = source_col
        self.window = window
        self.agg_func = agg_func
        self.key_col = key_col
        self.time_col = time_col
        self.output_name = output_name or f"{source_col}_{agg_func}_{window}"
    
    def fit(self, X: pd.DataFrame, y=None):
        """Fit transformer (no-op for rolling aggregator)."""
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply rolling aggregation."""
        df = X.copy()
        
        if self.key_col and self.time_col:
            # Time-aware grouped rolling
            df_sorted = df.sort_values([self.key_col, self.time_col])
            
            result = (
                df_sorted
                .groupby(self.key_col)[self.source_col]
                .rolling(window=self.window, closed='left')
                .agg(self.agg_func)
                .reset_index(level=0, drop=True)
            )
            
            df[self.output_name] = result
        else:
            # Simple rolling (no grouping)
            df[self.output_name] = (
                df[self.source_col]
                .rolling(window=self.window)
                .agg(self.agg_func)
            )
        
        # Fill NaN with 0 or median
        df[self.output_name] = df[self.output_name].fillna(0)
        
        return df[[self.output_name]]


class LagTransformer(BaseEstimator, TransformerMixin):
    """Lag feature transformer."""
    
    def __init__(
        self,
        source_col: str,
        lag: int = 1,
        key_col: str | None = None,
        time_col: str | None = None,
        output_name: str | None = None,
    ):
        """Initialize lag transformer."""
        self.source_col = source_col
        self.lag = lag
        self.key_col = key_col
        self.time_col = time_col
        self.output_name = output_name or f"{source_col}_lag_{lag}"
    
    def fit(self, X: pd.DataFrame, y=None):
        """Fit transformer."""
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply lag transformation."""
        df = X.copy()
        
        if self.key_col and self.time_col:
            # Time-aware grouped lag
            df_sorted = df.sort_values([self.key_col, self.time_col])
            df[self.output_name] = (
                df_sorted
                .groupby(self.key_col)[self.source_col]
                .shift(self.lag)
            )
        else:
            # Simple lag
            df[self.output_name] = df[self.source_col].shift(self.lag)
        
        # Fill NaN with 0
        df[self.output_name] = df[self.output_name].fillna(0)
        
        return df[[self.output_name]]


class NUniqueTransformer(BaseEstimator, TransformerMixin):
    """Count unique values in window."""
    
    def __init__(
        self,
        source_col: str,
        window: str | None = None,
        key_col: str | None = None,
        time_col: str | None = None,
        output_name: str | None = None,
    ):
        """Initialize nunique transformer."""
        self.source_col = source_col
        self.window = window
        self.key_col = key_col
        self.time_col = time_col
        self.output_name = output_name or f"{source_col}_nunique"
    
    def fit(self, X: pd.DataFrame, y=None):
        """Fit transformer."""
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply nunique transformation."""
        df = X.copy()
        
        if self.key_col:
            # Grouped nunique
            df[self.output_name] = (
                df.groupby(self.key_col)[self.source_col]
                .transform("nunique")
            )
        else:
            # Global nunique (constant)
            df[self.output_name] = df[self.source_col].nunique()
        
        return df[[self.output_name]]


class RFMTransformer(BaseEstimator, TransformerMixin):
    """RFM (Recency, Frequency, Monetary) features for customer analytics."""
    
    def __init__(
        self,
        key_col: str,
        time_col: str,
        amount_col: str | None = None,
        reference_date: str | None = None,
    ):
        """Initialize RFM transformer."""
        self.key_col = key_col
        self.time_col = time_col
        self.amount_col = amount_col
        self.reference_date = reference_date
    
    def fit(self, X: pd.DataFrame, y=None):
        """Fit transformer."""
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate RFM features."""
        df = X.copy()
        
        # Reference date (max date or specified)
        if self.reference_date:
            ref_date = pd.to_datetime(self.reference_date)
        else:
            ref_date = pd.to_datetime(df[self.time_col]).max()
        
        # Recency: days since last transaction
        df[f"{self.key_col}_recency_days"] = (
            ref_date - pd.to_datetime(df.groupby(self.key_col)[self.time_col].transform("max"))
        ).dt.days
        
        # Frequency: number of transactions
        df[f"{self.key_col}_frequency"] = (
            df.groupby(self.key_col)[self.key_col].transform("count")
        )
        
        # Monetary: total/average amount
        if self.amount_col:
            df[f"{self.key_col}_monetary_total"] = (
                df.groupby(self.key_col)[self.amount_col].transform("sum")
            )
            df[f"{self.key_col}_monetary_avg"] = (
                df.groupby(self.key_col)[self.amount_col].transform("mean")
            )
        
        # Select RFM columns
        rfm_cols = [col for col in df.columns if col.startswith(f"{self.key_col}_")]
        return df[rfm_cols]


# ============================================================================
# Plan Executor
# ============================================================================

class PlanExecutor:
    """Executor that translates FeaturePlan to actual transformations.
    
    This class takes a validated FeaturePlan and executes it to generate
    features on a given dataset.
    
    Example:
        >>> executor = PlanExecutor()
        >>> df_transformed = executor.execute(plan, df)
    """
    
    def __init__(
        self,
        engine: Literal["pandas"] = "pandas",
        cache_intermediates: bool = False,
    ):
        """Initialize executor.
        
        Args:
            engine: Execution engine (pandas, spark, ray - only pandas supported now)
            cache_intermediates: Cache intermediate results for debugging
        """
        self.engine = engine
        self.cache_intermediates = cache_intermediates
        self._cache: dict[str, pd.DataFrame] = {}
        
        if engine != "pandas":
            raise NotImplementedError(f"Engine '{engine}' not yet supported. Use 'pandas'.")
    
    def execute(
        self,
        plan: FeaturePlan,
        df: pd.DataFrame,
        return_original: bool = False,
    ) -> pd.DataFrame:
        """Execute feature plan on DataFrame.
        
        Args:
            plan: Validated feature plan
            df: Input DataFrame
            return_original: If True, return df with added features. If False, return only new features.
            
        Returns:
            DataFrame with generated features
            
        Raises:
            ValueError: If feature spec cannot be executed
        """
        logger.info(f"Executing plan with {len(plan.candidates)} features")
        
        result_df = df.copy() if return_original else pd.DataFrame(index=df.index)
        
        for feat_spec in plan.candidates:
            try:
                logger.debug(f"Generating feature: {feat_spec.name}")
                feature_df = self._execute_feature(feat_spec, df)
                
                # Add to result
                result_df[feat_spec.name] = feature_df[feat_spec.name]
                
                # Cache if enabled
                if self.cache_intermediates:
                    self._cache[feat_spec.name] = feature_df
                    
            except Exception as e:
                logger.error(f"Failed to generate feature '{feat_spec.name}': {e}")
                # Add NaN column as fallback
                result_df[feat_spec.name] = np.nan
        
        logger.info(f"✓ Generated {len(plan.candidates)} features")
        return result_df
    
    def _execute_feature(
        self,
        spec: FeatureSpec,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Execute single feature specification.
        
        Args:
            spec: Feature specification
            df: Input DataFrame
            
        Returns:
            DataFrame with single generated feature
        """
        # Dispatch to appropriate executor based on type
        executors = {
            # Aggregations
            "rolling_mean": self._exec_rolling,
            "rolling_sum": self._exec_rolling,
            "rolling_std": self._exec_rolling,
            "rolling_min": self._exec_rolling,
            "rolling_max": self._exec_rolling,
            
            # Lags
            "lag": self._exec_lag,
            "diff": self._exec_diff,
            
            # Cardinality
            "nunique": self._exec_nunique,
            "count": self._exec_count,
            
            # Domain-specific
            "recency": self._exec_rfm,
            "frequency": self._exec_rfm,
            "monetary": self._exec_rfm,
            "rfm_score": self._exec_rfm,
            
            # Interactions
            "multiply": self._exec_interaction,
            "divide": self._exec_interaction,
            "add": self._exec_interaction,
            "subtract": self._exec_interaction,
            "ratio": self._exec_ratio,
        }
        
        executor_func = executors.get(spec.type)
        
        if executor_func:
            return executor_func(spec, df)
        else:
            # Fallback: create placeholder feature
            logger.warning(
                f"Feature type '{spec.type}' not implemented yet. Creating placeholder."
            )
            result = pd.DataFrame({spec.name: 0}, index=df.index)
            return result
    
    def _exec_rolling(self, spec: FeatureSpec, df: pd.DataFrame) -> pd.DataFrame:
        """Execute rolling aggregation."""
        # Extract aggregation function from type (e.g., "rolling_mean" -> "mean")
        agg_func = spec.type.replace("rolling_", "")
        
        transformer = RollingAggregator(
            source_col=spec.source_col,
            window=spec.window,
            agg_func=agg_func,
            key_col=spec.key_col,
            time_col=spec.time_col,
            output_name=spec.name,
        )
        
        return transformer.fit_transform(df)
    
    def _exec_lag(self, spec: FeatureSpec, df: pd.DataFrame) -> pd.DataFrame:
        """Execute lag transformation."""
        lag = spec.params.get("lag", 1)
        
        transformer = LagTransformer(
            source_col=spec.source_col,
            lag=lag,
            key_col=spec.key_col,
            time_col=spec.time_col,
            output_name=spec.name,
        )
        
        return transformer.fit_transform(df)
    
    def _exec_diff(self, spec: FeatureSpec, df: pd.DataFrame) -> pd.DataFrame:
        """Execute diff transformation."""
        result = df.copy()
        
        if spec.key_col:
            result[spec.name] = (
                result.groupby(spec.key_col)[spec.source_col].diff()
            )
        else:
            result[spec.name] = result[spec.source_col].diff()
        
        result[spec.name] = result[spec.name].fillna(0)
        return result[[spec.name]]
    
    def _exec_nunique(self, spec: FeatureSpec, df: pd.DataFrame) -> pd.DataFrame:
        """Execute nunique transformation."""
        transformer = NUniqueTransformer(
            source_col=spec.source_col,
            window=spec.window,
            key_col=spec.key_col,
            time_col=spec.time_col,
            output_name=spec.name,
        )
        
        return transformer.fit_transform(df)
    
    def _exec_count(self, spec: FeatureSpec, df: pd.DataFrame) -> pd.DataFrame:
        """Execute count transformation."""
        result = df.copy()
        
        if spec.key_col:
            result[spec.name] = (
                result.groupby(spec.key_col)[spec.source_col].transform("count")
            )
        else:
            result[spec.name] = len(df)
        
        return result[[spec.name]]
    
    def _exec_rfm(self, spec: FeatureSpec, df: pd.DataFrame) -> pd.DataFrame:
        """Execute RFM transformation."""
        # Infer columns from spec
        key_col = spec.key_col
        time_col = spec.time_col
        amount_col = spec.params.get("amount_col")
        
        transformer = RFMTransformer(
            key_col=key_col,
            time_col=time_col,
            amount_col=amount_col,
        )
        
        rfm_df = transformer.fit_transform(df)
        
        # Extract specific RFM component if specified
        if spec.type == "recency":
            return rfm_df[[f"{key_col}_recency_days"]].rename(columns={f"{key_col}_recency_days": spec.name})
        elif spec.type == "frequency":
            return rfm_df[[f"{key_col}_frequency"]].rename(columns={f"{key_col}_frequency": spec.name})
        elif spec.type == "monetary" and amount_col:
            return rfm_df[[f"{key_col}_monetary_total"]].rename(columns={f"{key_col}_monetary_total": spec.name})
        else:
            # Return all RFM features
            return rfm_df
    
    def _exec_interaction(self, spec: FeatureSpec, df: pd.DataFrame) -> pd.DataFrame:
        """Execute interaction features (multiply, add, etc.)."""
        result = df.copy()
        
        # spec.source_col should be a list of two columns
        if not isinstance(spec.source_col, list) or len(spec.source_col) != 2:
            raise ValueError(f"Interaction features require exactly 2 source columns, got {spec.source_col}")
        
        col1, col2 = spec.source_col
        
        if spec.type == "multiply":
            result[spec.name] = result[col1] * result[col2]
        elif spec.type == "add":
            result[spec.name] = result[col1] + result[col2]
        elif spec.type == "subtract":
            result[spec.name] = result[col1] - result[col2]
        elif spec.type == "divide":
            result[spec.name] = result[col1] / (result[col2] + 1e-8)  # Avoid division by zero
        
        return result[[spec.name]]
    
    def _exec_ratio(self, spec: FeatureSpec, df: pd.DataFrame) -> pd.DataFrame:
        """Execute ratio feature."""
        return self._exec_interaction(spec, df)


# ============================================================================
# Convenience Functions
# ============================================================================

def execute_plan(
    plan: FeaturePlan,
    df: pd.DataFrame,
    engine: Literal["pandas"] = "pandas",
    return_original: bool = False,
) -> pd.DataFrame:
    """Execute feature plan on DataFrame (convenience function).
    
    Args:
        plan: Validated feature plan
        df: Input DataFrame
        engine: Execution engine (only pandas supported now)
        return_original: If True, return df with added features
        
    Returns:
        DataFrame with generated features
        
    Example:
        >>> df_transformed = execute_plan(plan, train_df)
        >>> print(df_transformed.shape)
    """
    executor = PlanExecutor(engine=engine)
    return executor.execute(plan, df, return_original=return_original)

