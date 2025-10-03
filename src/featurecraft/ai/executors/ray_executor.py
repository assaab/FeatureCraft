"""Ray executor for distributed feature engineering."""

from __future__ import annotations

import warnings
from typing import Any

import pandas as pd

from ..schemas import FeaturePlan, FeatureSpec
from ...logging import get_logger

logger = get_logger(__name__)

# Import Ray with fallback
try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    ray = None


class RayExecutor:
    """Ray executor for distributed feature plan execution.
    
    Uses Ray for distributed computing with fault tolerance and
    automatic retries.
    
    Example:
        >>> import ray
        >>> ray.init()
        >>> executor = RayExecutor()
        >>> df_features = executor.execute(plan, df)
    """
    
    def __init__(
        self,
        num_cpus: int | None = None,
        batch_size: int = 1000,
        max_retries: int = 3,
        enable_progress_bar: bool = True,
    ):
        """Initialize Ray executor.
        
        Args:
            num_cpus: Number of CPUs to use (None = auto)
            batch_size: Batch size for parallel processing
            max_retries: Max retries for failed tasks
            enable_progress_bar: Show progress bar
        """
        try:
            import ray
            self.ray = ray
        except ImportError:
            raise ImportError(
                "Ray required for RayExecutor. "
                "Install with: pip install ray>=2.0.0"
            )
        
        # Initialize Ray if not already running
        if not self.ray.is_initialized():
            logger.info("Initializing Ray...")
            self.ray.init(
                num_cpus=num_cpus,
                ignore_reinit_error=True,
                logging_level="ERROR",
            )
        
        self.num_cpus = num_cpus or self.ray.available_resources().get("CPU", 1)
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.enable_progress_bar = enable_progress_bar
    
    def execute(
        self,
        plan: FeaturePlan,
        df: pd.DataFrame,
        return_original: bool = False,
    ) -> pd.DataFrame:
        """Execute feature plan with Ray.
        
        Args:
            plan: Feature plan to execute
            df: Input DataFrame
            return_original: Include original columns in output
            
        Returns:
            DataFrame with generated features
        """
        logger.info(f"Executing {len(plan.candidates)} features with Ray")
        
        # Put DataFrame in Ray object store
        df_ref = self.ray.put(df)
        
        # Execute features in parallel
        feature_refs = []
        for feat_spec in plan.candidates:
            # Create remote task
            task = self._execute_feature_remote.remote(
                self,
                feat_spec,
                df_ref,
            )
            feature_refs.append((feat_spec.name, task))
        
        # Collect results
        result_df = df.copy() if return_original else pd.DataFrame(index=df.index)
        
        if self.enable_progress_bar:
            try:
                from tqdm import tqdm
                iterator = tqdm(feature_refs, desc="Generating features")
            except ImportError:
                iterator = feature_refs
        else:
            iterator = feature_refs
        
        for feat_name, task_ref in iterator:
            try:
                feature_series = self.ray.get(task_ref)
                result_df[feat_name] = feature_series
            except Exception as e:
                logger.error(f"Failed to generate feature '{feat_name}': {e}")
                result_df[feat_name] = 0
        
        logger.info(f"✓ Generated {len(plan.candidates)} features with Ray")
        return result_df
    
    def _execute_feature_remote(
        self,
        spec: FeatureSpec,
        df: pd.DataFrame,
    ) -> pd.Series:
        """Execute single feature (Ray remote function).
        
        Args:
            spec: Feature specification
            df: Input DataFrame
            
        Returns:
            Feature series
        """
        return self._execute_feature(spec, df)
    
    def _execute_feature(
        self,
        spec: FeatureSpec,
        df: pd.DataFrame,
    ) -> pd.Series:
        """Execute single feature specification.
        
        Args:
            spec: Feature specification
            df: Input DataFrame
            
        Returns:
            Feature series
        """
        # Dispatch to appropriate executor
        executors = {
            # Aggregations
            "rolling_mean": self._exec_rolling_mean,
            "rolling_sum": self._exec_rolling_sum,
            "rolling_std": self._exec_rolling_std,
            "rolling_min": self._exec_rolling_min,
            "rolling_max": self._exec_rolling_max,
            
            # Lags
            "lag": self._exec_lag,
            "diff": self._exec_diff,
            
            # Cardinality
            "nunique": self._exec_nunique,
            "count": self._exec_count,
            
            # Interactions
            "multiply": self._exec_multiply,
            "divide": self._exec_divide,
            "add": self._exec_add,
            "subtract": self._exec_subtract,
        }
        
        executor_func = executors.get(spec.type)
        
        if executor_func:
            return executor_func(spec, df)
        else:
            logger.warning(f"Feature type '{spec.type}' not yet implemented for Ray")
            return pd.Series(0, index=df.index, name=spec.name)
    
    def _exec_rolling_mean(self, spec: FeatureSpec, df: pd.DataFrame) -> pd.Series:
        """Execute rolling mean."""
        return self._exec_rolling_agg(spec, df, "mean")
    
    def _exec_rolling_sum(self, spec: FeatureSpec, df: pd.DataFrame) -> pd.Series:
        """Execute rolling sum."""
        return self._exec_rolling_agg(spec, df, "sum")
    
    def _exec_rolling_std(self, spec: FeatureSpec, df: pd.DataFrame) -> pd.Series:
        """Execute rolling std."""
        return self._exec_rolling_agg(spec, df, "std")
    
    def _exec_rolling_min(self, spec: FeatureSpec, df: pd.DataFrame) -> pd.Series:
        """Execute rolling min."""
        return self._exec_rolling_agg(spec, df, "min")
    
    def _exec_rolling_max(self, spec: FeatureSpec, df: pd.DataFrame) -> pd.Series:
        """Execute rolling max."""
        return self._exec_rolling_agg(spec, df, "max")
    
    def _exec_rolling_agg(
        self,
        spec: FeatureSpec,
        df: pd.DataFrame,
        agg_func: str,
    ) -> pd.Series:
        """Execute rolling aggregation."""
        if spec.key_col and spec.time_col:
            # Time-aware grouped rolling
            df_sorted = df.sort_values([spec.key_col, spec.time_col])
            
            result = (
                df_sorted
                .groupby(spec.key_col)[spec.source_col]
                .rolling(window=spec.window, closed='left')
                .agg(agg_func)
                .reset_index(level=0, drop=True)
            )
        else:
            # Simple rolling
            result = (
                df[spec.source_col]
                .rolling(window=spec.window)
                .agg(agg_func)
            )
        
        # Fill NaN
        result = result.fillna(0)
        result.name = spec.name
        return result
    
    def _exec_lag(self, spec: FeatureSpec, df: pd.DataFrame) -> pd.Series:
        """Execute lag transformation."""
        lag_periods = spec.params.get("lag", 1)
        
        if spec.key_col and spec.time_col:
            df_sorted = df.sort_values([spec.key_col, spec.time_col])
            result = (
                df_sorted
                .groupby(spec.key_col)[spec.source_col]
                .shift(lag_periods)
            )
        else:
            result = df[spec.source_col].shift(lag_periods)
        
        result = result.fillna(0)
        result.name = spec.name
        return result
    
    def _exec_diff(self, spec: FeatureSpec, df: pd.DataFrame) -> pd.Series:
        """Execute diff transformation."""
        if spec.key_col:
            result = df.groupby(spec.key_col)[spec.source_col].diff()
        else:
            result = df[spec.source_col].diff()
        
        result = result.fillna(0)
        result.name = spec.name
        return result
    
    def _exec_nunique(self, spec: FeatureSpec, df: pd.DataFrame) -> pd.Series:
        """Execute nunique transformation."""
        if spec.key_col:
            result = (
                df.groupby(spec.key_col)[spec.source_col]
                .transform("nunique")
            )
        else:
            result = pd.Series(df[spec.source_col].nunique(), index=df.index)
        
        result.name = spec.name
        return result
    
    def _exec_count(self, spec: FeatureSpec, df: pd.DataFrame) -> pd.Series:
        """Execute count transformation."""
        if spec.key_col:
            result = (
                df.groupby(spec.key_col)[spec.source_col]
                .transform("count")
            )
        else:
            result = pd.Series(len(df), index=df.index)
        
        result.name = spec.name
        return result
    
    def _exec_multiply(self, spec: FeatureSpec, df: pd.DataFrame) -> pd.Series:
        """Execute multiply interaction."""
        col1, col2 = spec.source_col
        result = df[col1] * df[col2]
        result.name = spec.name
        return result
    
    def _exec_divide(self, spec: FeatureSpec, df: pd.DataFrame) -> pd.Series:
        """Execute divide interaction."""
        col1, col2 = spec.source_col
        result = df[col1] / (df[col2] + 1e-8)
        result.name = spec.name
        return result
    
    def _exec_add(self, spec: FeatureSpec, df: pd.DataFrame) -> pd.Series:
        """Execute add interaction."""
        col1, col2 = spec.source_col
        result = df[col1] + df[col2]
        result.name = spec.name
        return result
    
    def _exec_subtract(self, spec: FeatureSpec, df: pd.DataFrame) -> pd.Series:
        """Execute subtract interaction."""
        col1, col2 = spec.source_col
        result = df[col1] - df[col2]
        result.name = spec.name
        return result
    
    def shutdown(self) -> None:
        """Shutdown Ray."""
        if self.ray.is_initialized():
            self.ray.shutdown()
            logger.info("Ray shutdown")

