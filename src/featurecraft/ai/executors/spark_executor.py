"""Apache Spark executor for distributed feature engineering."""

from __future__ import annotations

import warnings
from typing import Any

import pandas as pd

from ..schemas import FeaturePlan, FeatureSpec
from ...logging import get_logger

logger = get_logger(__name__)


class SparkExecutor:
    """Apache Spark executor for distributed feature plan execution.
    
    Translates FeaturePlan operations to PySpark transformations for
    distributed execution on Spark clusters.
    
    Example:
        >>> from pyspark.sql import SparkSession
        >>> spark = SparkSession.builder.appName("FeatureCraft").getOrCreate()
        >>> executor = SparkExecutor(spark)
        >>> sdf_features = executor.execute(plan, sdf)
    """
    
    def __init__(
        self,
        spark_session: Any = None,
        cache_intermediate: bool = True,
        repartition: int | None = None,
    ):
        """Initialize Spark executor.
        
        Args:
            spark_session: SparkSession instance (creates one if None)
            cache_intermediate: Cache intermediate DataFrames
            repartition: Number of partitions (auto if None)
        """
        try:
            from pyspark.sql import SparkSession
            from pyspark.sql import functions as F
            from pyspark.sql import Window
            
            self.F = F
            self.Window = Window
        except ImportError:
            raise ImportError(
                "PySpark required for SparkExecutor. "
                "Install with: pip install pyspark>=3.3.0"
            )
        
        if spark_session is None:
            logger.info("Creating new SparkSession...")
            self.spark = SparkSession.builder \
                .appName("FeatureCraft-SparkExecutor") \
                .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
                .getOrCreate()
        else:
            self.spark = spark_session
        
        self.cache_intermediate = cache_intermediate
        self.repartition = repartition
        self._cache: dict[str, Any] = {}
    
    def execute(
        self,
        plan: FeaturePlan,
        sdf: Any,
        return_original: bool = False,
    ) -> Any:
        """Execute feature plan on Spark DataFrame.
        
        Args:
            plan: Feature plan to execute
            sdf: Spark DataFrame
            return_original: Include original columns in output
            
        Returns:
            Spark DataFrame with generated features
        """
        logger.info(f"Executing {len(plan.candidates)} features on Spark")
        
        # Repartition if specified
        if self.repartition:
            sdf = sdf.repartition(self.repartition)
        
        result_sdf = sdf if return_original else None
        
        for i, feat_spec in enumerate(plan.candidates):
            try:
                logger.debug(f"[{i+1}/{len(plan.candidates)}] Generating: {feat_spec.name}")
                
                feature_col = self._execute_feature(feat_spec, sdf)
                
                # Add to result
                if result_sdf is None:
                    result_sdf = sdf.select(sdf.columns[0]).withColumn(
                        feat_spec.name, feature_col
                    )
                else:
                    result_sdf = result_sdf.withColumn(feat_spec.name, feature_col)
                
                # Cache if enabled
                if self.cache_intermediate:
                    self._cache[feat_spec.name] = feature_col
                
            except Exception as e:
                logger.error(f"Failed to generate feature '{feat_spec.name}': {e}")
                # Add null column as fallback
                result_sdf = result_sdf.withColumn(feat_spec.name, self.F.lit(None))
        
        logger.info(f"✓ Generated {len(plan.candidates)} features on Spark")
        return result_sdf
    
    def _execute_feature(self, spec: FeatureSpec, sdf: Any) -> Any:
        """Execute single feature specification.
        
        Args:
            spec: Feature specification
            sdf: Spark DataFrame
            
        Returns:
            Column expression
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
            return executor_func(spec, sdf)
        else:
            logger.warning(f"Feature type '{spec.type}' not yet implemented for Spark")
            return self.F.lit(0)
    
    def _exec_rolling_mean(self, spec: FeatureSpec, sdf: Any) -> Any:
        """Execute rolling mean."""
        return self._exec_rolling_agg(spec, sdf, "mean")
    
    def _exec_rolling_sum(self, spec: FeatureSpec, sdf: Any) -> Any:
        """Execute rolling sum."""
        return self._exec_rolling_agg(spec, sdf, "sum")
    
    def _exec_rolling_std(self, spec: FeatureSpec, sdf: Any) -> Any:
        """Execute rolling std."""
        return self._exec_rolling_agg(spec, sdf, "stddev")
    
    def _exec_rolling_min(self, spec: FeatureSpec, sdf: Any) -> Any:
        """Execute rolling min."""
        return self._exec_rolling_agg(spec, sdf, "min")
    
    def _exec_rolling_max(self, spec: FeatureSpec, sdf: Any) -> Any:
        """Execute rolling max."""
        return self._exec_rolling_agg(spec, sdf, "max")
    
    def _exec_rolling_agg(self, spec: FeatureSpec, sdf: Any, agg_func: str) -> Any:
        """Execute rolling aggregation.
        
        Note: Spark doesn't natively support time-based rolling windows like pandas.
        This is a simplified implementation using row-based windows.
        """
        # Parse window (e.g., "30d" -> 30 rows approximation)
        window_rows = self._parse_window(spec.window)
        
        # Define window
        if spec.key_col and spec.time_col:
            # Partitioned window
            window_spec = (
                self.Window
                .partitionBy(spec.key_col)
                .orderBy(spec.time_col)
                .rowsBetween(-window_rows, -1)  # Exclude current row (time-safe)
            )
        else:
            # Global window
            window_spec = self.Window.rowsBetween(-window_rows, -1)
        
        # Apply aggregation
        col = sdf[spec.source_col]
        
        if agg_func == "mean":
            result = self.F.avg(col).over(window_spec)
        elif agg_func == "sum":
            result = self.F.sum(col).over(window_spec)
        elif agg_func == "stddev":
            result = self.F.stddev(col).over(window_spec)
        elif agg_func == "min":
            result = self.F.min(col).over(window_spec)
        elif agg_func == "max":
            result = self.F.max(col).over(window_spec)
        else:
            result = self.F.lit(0)
        
        # Fill nulls
        return self.F.coalesce(result, self.F.lit(0))
    
    def _exec_lag(self, spec: FeatureSpec, sdf: Any) -> Any:
        """Execute lag transformation."""
        lag_periods = spec.params.get("lag", 1)
        
        if spec.key_col and spec.time_col:
            window_spec = (
                self.Window
                .partitionBy(spec.key_col)
                .orderBy(spec.time_col)
            )
        else:
            window_spec = self.Window.orderBy(self.F.monotonically_increasing_id())
        
        result = self.F.lag(sdf[spec.source_col], lag_periods).over(window_spec)
        return self.F.coalesce(result, self.F.lit(0))
    
    def _exec_diff(self, spec: FeatureSpec, sdf: Any) -> Any:
        """Execute diff transformation."""
        if spec.key_col:
            window_spec = self.Window.partitionBy(spec.key_col).orderBy(spec.time_col)
        else:
            window_spec = self.Window.orderBy(self.F.monotonically_increasing_id())
        
        col = sdf[spec.source_col]
        lagged = self.F.lag(col, 1).over(window_spec)
        result = col - lagged
        return self.F.coalesce(result, self.F.lit(0))
    
    def _exec_nunique(self, spec: FeatureSpec, sdf: Any) -> Any:
        """Execute nunique transformation."""
        if spec.key_col:
            window_spec = self.Window.partitionBy(spec.key_col)
        else:
            window_spec = self.Window.partitionBy(self.F.lit(1))
        
        result = self.F.countDistinct(sdf[spec.source_col]).over(window_spec)
        return result
    
    def _exec_count(self, spec: FeatureSpec, sdf: Any) -> Any:
        """Execute count transformation."""
        if spec.key_col:
            window_spec = self.Window.partitionBy(spec.key_col)
        else:
            window_spec = self.Window.partitionBy(self.F.lit(1))
        
        result = self.F.count(sdf[spec.source_col]).over(window_spec)
        return result
    
    def _exec_multiply(self, spec: FeatureSpec, sdf: Any) -> Any:
        """Execute multiply interaction."""
        col1, col2 = spec.source_col
        return sdf[col1] * sdf[col2]
    
    def _exec_divide(self, spec: FeatureSpec, sdf: Any) -> Any:
        """Execute divide interaction."""
        col1, col2 = spec.source_col
        return sdf[col1] / (sdf[col2] + self.F.lit(1e-8))
    
    def _exec_add(self, spec: FeatureSpec, sdf: Any) -> Any:
        """Execute add interaction."""
        col1, col2 = spec.source_col
        return sdf[col1] + sdf[col2]
    
    def _exec_subtract(self, spec: FeatureSpec, sdf: Any) -> Any:
        """Execute subtract interaction."""
        col1, col2 = spec.source_col
        return sdf[col1] - sdf[col2]
    
    @staticmethod
    def _parse_window(window_str: str | None) -> int:
        """Parse window string to row count.
        
        This is a rough approximation since Spark doesn't support
        time-based windows like pandas.
        
        Args:
            window_str: Window string (e.g., "30d", "7d")
            
        Returns:
            Number of rows (approximation)
        """
        if window_str is None:
            return 10
        
        # Extract number
        import re
        match = re.search(r'(\d+)', window_str)
        if match:
            num = int(match.group(1))
            
            # Rough conversion (assumes daily data)
            if 'd' in window_str:
                return num
            elif 'w' in window_str:
                return num * 7
            elif 'm' in window_str:
                return num * 30
            elif 'y' in window_str:
                return num * 365
            else:
                return num
        
        return 10
    
    def to_pandas(self, sdf: Any) -> pd.DataFrame:
        """Convert Spark DataFrame to Pandas.
        
        Args:
            sdf: Spark DataFrame
            
        Returns:
            Pandas DataFrame
        """
        return sdf.toPandas()

