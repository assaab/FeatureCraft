"""Type definitions for FeatureCraft."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import pandas as pd


class TaskType(str, Enum):
    """Task type enumeration."""

    CLASSIFICATION = "classification"
    REGRESSION = "regression"


@dataclass
class ColumnProfile:
    """Profile information for a single column."""

    name: str
    dtype: str
    cardinality: int | None = None
    missing_rate: float = 0.0
    skewness: float | None = None
    outlier_share: float | None = None
    is_numeric: bool = False
    is_categorical: bool = False
    is_text: bool = False
    is_datetime: bool = False
    is_ordinal: bool = False


@dataclass
class Issue:
    """Issue detected in dataset analysis."""

    severity: str  # "INFO", "WARN", "ERROR"
    message: str
    column: str | None = None
    code: str | None = None


@dataclass
class DatasetInsights:
    """Insights about a dataset."""

    n_rows: int
    n_cols: int
    task: TaskType
    target_name: str
    target_positive_class: Any | None = None
    target_class_balance: dict[Any, float] | None = None
    profiles: list[ColumnProfile] = field(default_factory=list)
    correlations: pd.DataFrame | None = None
    issues: list[Issue] = field(default_factory=list)
    figures: dict[str, str] = field(default_factory=dict)  # name -> base64 png
    summary: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineSummary:
    """Summary of a fitted pipeline."""

    feature_names: list[str]
    n_features_out: int
    steps: list[str]
    artifacts_path: str | None = None
