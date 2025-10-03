"""Pandas-based executor (single-machine, baseline implementation)."""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..executor import PlanExecutor
from ..schemas import FeaturePlan


class PandasExecutor:
    """Pandas-based executor for feature plans.
    
    This is a wrapper around the existing PlanExecutor for API consistency.
    
    Example:
        >>> executor = PandasExecutor()
        >>> df_features = executor.execute(plan, df)
    """
    
    def __init__(self, cache_intermediates: bool = False):
        """Initialize Pandas executor.
        
        Args:
            cache_intermediates: Cache intermediate results
        """
        self.executor = PlanExecutor(
            engine="pandas",
            cache_intermediates=cache_intermediates,
        )
    
    def execute(
        self,
        plan: FeaturePlan,
        df: pd.DataFrame,
        return_original: bool = False,
    ) -> pd.DataFrame:
        """Execute feature plan.
        
        Args:
            plan: Feature plan to execute
            df: Input DataFrame
            return_original: Include original columns in output
            
        Returns:
            DataFrame with generated features
        """
        return self.executor.execute(plan, df, return_original)

