"""FeatureCraft: automatic feature engineering and insights for tabular ML."""

from __future__ import annotations

from .config import FeatureCraftConfig
from .pipeline import AutoFeatureEngineer
from .report import ReportBuilder
from .settings import load_config, save_config
from .types import DatasetInsights, Issue
from .version import version

__all__ = [
    "AutoFeatureEngineer",
    "DatasetInsights",
    "FeatureCraftConfig",
    "Issue",
    "ReportBuilder",
    "load_config",
    "save_config",
    "version",
]
