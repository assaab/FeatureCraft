"""Dataset profiling and insights generation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype

from .config import FeatureCraftConfig
from .logging import get_logger
from .types import ColumnProfile, DatasetInsights, Issue, TaskType
from .utils import (
    calc_cardinality,
    calc_missing_rate,
    calc_outlier_share,
    calc_skewness,
    is_datetime_series,
    is_numeric_series,
    is_text_candidate,
)

logger = get_logger(__name__)


def detect_task(y: pd.Series) -> TaskType:
    """Auto-detect task type from target variable."""
    if is_numeric_series(y) and y.nunique(dropna=True) > 15:
        return TaskType.REGRESSION
    return TaskType.CLASSIFICATION


def profile_columns(df: pd.DataFrame, cfg: FeatureCraftConfig) -> list[ColumnProfile]:
    """Profile each column in the DataFrame."""
    profiles: list[ColumnProfile] = []
    for c in df.columns:
        s = df[c]
        p = ColumnProfile(
            name=c,
            dtype=str(s.dtype),
            cardinality=calc_cardinality(s) if not is_numeric_series(s) else None,
            missing_rate=calc_missing_rate(s),
            skewness=calc_skewness(s),
            outlier_share=calc_outlier_share(s),
            is_numeric=is_numeric_series(s),
            is_categorical=(s.dtype == "object" or isinstance(s.dtype, CategoricalDtype))
            and not is_text_candidate(s),
            is_text=is_text_candidate(s),
            is_datetime=is_datetime_series(s),
        )
        profiles.append(p)
    return profiles


def compute_class_balance(y: pd.Series) -> dict[str, float] | None:
    """Compute class balance for classification targets."""
    if y.isna().any():
        y = y.dropna()
    vc = y.value_counts(normalize=True)
    try:
        return {str(k): float(v) for k, v in vc.items()}
    except Exception:
        return None


def correlation_matrix_numeric(df: pd.DataFrame, cfg: FeatureCraftConfig) -> pd.DataFrame:
    """Compute correlation matrix for numeric columns."""
    num_df = df.select_dtypes(include=[np.number]).copy()
    if num_df.shape[1] > cfg.max_corr_features:
        # Choose top by variance
        variances = num_df.var().sort_values(ascending=False)
        keep = variances.head(cfg.max_corr_features).index
        num_df = num_df[keep]
    # Impute quick median for corr plot
    num_df = num_df.fillna(num_df.median(numeric_only=True))
    return num_df.corr(numeric_only=True)


def summarize_issues(
    profiles: list[ColumnProfile],
    y: pd.Series,
    cfg: FeatureCraftConfig,
    task: TaskType,
    target_name: str,
) -> list[Issue]:
    """Summarize issues found in dataset."""
    issues: list[Issue] = []
    # Nulls
    for p in profiles:
        if p.missing_rate > 0.3:
            issues.append(
                Issue(
                    severity="WARN",
                    code="HIGH_NULL",
                    column=p.name,
                    message=f"{p.name} null_rate={p.missing_rate:.1%} (>30%) — consider drop or domain default.",
                )
            )
        elif p.missing_rate > 0:
            issues.append(
                Issue(
                    severity="INFO",
                    code="NULLS",
                    column=p.name,
                    message=f"{p.name} null_rate={p.missing_rate:.1%}.",
                )
            )
    # High cardinality
    for p in profiles:
        if p.is_categorical and p.cardinality is not None and p.cardinality > 50:
            issues.append(
                Issue(
                    severity="INFO",
                    code="HIGH_CARD",
                    column=p.name,
                    message=f"{p.name} cardinality={p.cardinality} (>50) — hashing/frequency encoding.",
                )
            )
    # Outliers
    for p in profiles:
        if p.is_numeric and (p.outlier_share or 0) > cfg.outlier_share_threshold:
            issues.append(
                Issue(
                    severity="INFO",
                    code="OUTLIERS",
                    column=p.name,
                    message=f"{p.name} outlier_share={(p.outlier_share or 0):.1%} (>5%) — consider RobustScaler/winsorize.",
                )
            )
    # Class imbalance
    if task == TaskType.CLASSIFICATION:
        bal = compute_class_balance(y)
        if bal:
            minority = min(bal.values())
            if minority < 0.2:
                issues.append(
                    Issue(
                        severity="INFO",
                        code="IMBALANCE",
                        column=target_name,
                        message=f"Minority class proportion={minority:.1%} (<20%) — use class_weight; <10% consider SMOTE in CV.",
                    )
                )
    return issues


def analyze_dataset(
    X: pd.DataFrame, y: pd.Series, target_name: str, cfg: FeatureCraftConfig
) -> DatasetInsights:
    """Analyze dataset and return insights."""
    task = detect_task(y)
    profiles = profile_columns(X, cfg)
    corr = correlation_matrix_numeric(X, cfg)
    issues = summarize_issues(profiles, y, cfg, task, target_name)
    bal = compute_class_balance(y) if task == TaskType.CLASSIFICATION else None
    insights = DatasetInsights(
        n_rows=int(X.shape[0]),
        n_cols=int(X.shape[1]),
        task=task,
        target_name=target_name,
        target_class_balance=bal,
        profiles=profiles,
        correlations=corr,
        issues=issues,
        summary={
            "n_numeric": int(sum(p.is_numeric for p in profiles)),
            "n_categorical": int(sum(p.is_categorical for p in profiles)),
            "n_text": int(sum(p.is_text for p in profiles)),
            "n_datetime": int(sum(p.is_datetime for p in profiles)),
        },
    )
    return insights
