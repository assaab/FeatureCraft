"""Core data types for FeatureCraft Agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from ..types import TaskType


@dataclass
class DatasetFingerprint:
    """Comprehensive dataset characteristics for strategy selection."""
    
    # === Basic Statistics ===
    n_rows: int
    n_cols: int
    n_features_original: int  # Excludes target
    target_name: str
    task_type: TaskType
    
    # === Column Type Counts ===
    n_numeric: int
    n_categorical: int
    n_text: int
    n_datetime: int
    n_id_like: int = 0
    n_constant: int = 0
    n_duplicate: int = 0
    
    # === Column Lists ===
    numeric_cols: List[str] = field(default_factory=list)
    categorical_cols: List[str] = field(default_factory=list)
    text_cols: List[str] = field(default_factory=list)
    datetime_cols: List[str] = field(default_factory=list)
    
    # === Data Quality Indicators ===
    missing_summary: Dict[str, float] = field(default_factory=dict)
    high_missing_cols: List[str] = field(default_factory=list)
    constant_cols: List[str] = field(default_factory=list)
    duplicate_col_groups: List[List[str]] = field(default_factory=list)
    
    # === Categorical Analysis ===
    cardinality_summary: Dict[str, int] = field(default_factory=dict)
    low_cardinality_cols: List[str] = field(default_factory=list)
    mid_cardinality_cols: List[str] = field(default_factory=list)
    high_cardinality_cols: List[str] = field(default_factory=list)
    ultra_high_cardinality_cols: List[str] = field(default_factory=list)
    rare_category_ratios: Dict[str, float] = field(default_factory=dict)
    
    # === Numeric Distribution Analysis ===
    skewness_summary: Dict[str, float] = field(default_factory=dict)
    kurtosis_summary: Dict[str, float] = field(default_factory=dict)
    heavily_skewed_cols: List[str] = field(default_factory=list)
    outlier_share_summary: Dict[str, float] = field(default_factory=dict)
    high_outlier_cols: List[str] = field(default_factory=list)
    near_zero_variance_cols: List[str] = field(default_factory=list)
    
    # === Target Analysis ===
    target_dtype: str = "unknown"
    target_nunique: int = 0
    target_missing_rate: float = 0.0
    class_balance: Optional[Dict[str, float]] = None
    minority_class_ratio: Optional[float] = None
    is_imbalanced: bool = False
    target_skewness: Optional[float] = None
    
    # === Correlation & Multicollinearity ===
    high_correlation_pairs: List[Tuple[str, str, float]] = field(default_factory=list)
    multicollinear_groups: List[List[str]] = field(default_factory=list)
    target_correlation_top: List[Tuple[str, float]] = field(default_factory=list)
    
    # === Mutual Information ===
    mutual_info_scores: Dict[str, float] = field(default_factory=dict)
    low_mi_cols: List[str] = field(default_factory=list)
    
    # === Time Series Diagnostics ===
    time_column: Optional[str] = None
    is_time_series: bool = False
    time_granularity: Optional[str] = None
    time_gaps: Optional[Dict[str, Any]] = None
    has_regular_intervals: bool = False
    
    # === Entity/Group Diagnostics ===
    entity_columns: List[str] = field(default_factory=list)
    entity_imbalance: Optional[Dict[str, float]] = None
    
    # === Text Column Diagnostics ===
    text_length_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    text_vocab_size: Dict[str, int] = field(default_factory=dict)
    
    # === Leakage Risk Signals ===
    leakage_risk_cols: List[str] = field(default_factory=list)
    id_like_with_high_mi: List[str] = field(default_factory=list)
    
    # === Computational Complexity Estimates ===
    estimated_feature_count_baseline: int = 0
    estimated_memory_gb: float = 0.0
    
    # === Metadata ===
    fingerprint_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    featurecraft_version: str = "1.0.0"


@dataclass
class EvaluationResult:
    """Result of a single pipeline evaluation."""
    
    pipeline_id: str
    cv_score_mean: float
    cv_score_std: float
    cv_scores: List[float]
    
    # All metrics
    metrics: Dict[str, float] = field(default_factory=dict)
    
    # Performance metrics
    fit_time_seconds: float = 0.0
    transform_time_seconds: float = 0.0
    total_time_seconds: float = 0.0
    
    # Feature info
    n_features_out: int = 0
    feature_names_out: Optional[List[str]] = None
    
    # Risk scores
    leakage_risk_score: float = 0.0
    
    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "pipeline_id": self.pipeline_id,
            "cv_score_mean": self.cv_score_mean,
            "cv_score_std": self.cv_score_std,
            "cv_scores": self.cv_scores,
            "metrics": self.metrics,
            "fit_time_seconds": self.fit_time_seconds,
            "transform_time_seconds": self.transform_time_seconds,
            "total_time_seconds": self.total_time_seconds,
            "n_features_out": self.n_features_out,
            "leakage_risk_score": self.leakage_risk_score,
            "timestamp": self.timestamp,
        }


@dataclass
class Candidate:
    """A candidate pipeline with its strategy and evaluation result."""
    
    strategy: Any  # FeatureStrategy from ai.advisor
    pipeline: Pipeline
    result: EvaluationResult
    
    @property
    def score(self) -> float:
        """Get mean CV score."""
        return self.result.cv_score_mean


@dataclass
class Baselines:
    """Baseline evaluation results."""
    
    raw: EvaluationResult
    auto: EvaluationResult


@dataclass
class AblationResults:
    """Results from ablation studies."""
    
    # Per-operation ablation: operation_name -> score_delta
    operation_impacts: Dict[str, float] = field(default_factory=dict)
    
    # Per-feature-family ablation
    family_impacts: Dict[str, float] = field(default_factory=dict)
    
    # Full results
    ablation_details: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class RunLedger:
    """Minimal state for reproducibility."""
    
    run_id: str
    dataset_hash: str
    target_name: str
    task_type: TaskType
    estimator_family: str
    random_seed: int
    cv_splits_hash: str
    config_snapshot: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "run_id": self.run_id,
            "dataset_hash": self.dataset_hash,
            "target_name": self.target_name,
            "task_type": self.task_type.value if hasattr(self.task_type, "value") else str(self.task_type),
            "estimator_family": self.estimator_family,
            "random_seed": self.random_seed,
            "cv_splits_hash": self.cv_splits_hash,
            "config_snapshot": self.config_snapshot,
            "timestamp": self.timestamp,
        }


@dataclass
class AgentResult:
    """Final result from agent execution."""
    
    # Core outputs
    run_id: str
    fingerprint: DatasetFingerprint
    best_strategy: Any  # FeatureStrategy
    best_pipeline: Pipeline
    best_result: EvaluationResult
    
    # Baselines
    baseline_raw: EvaluationResult
    baseline_auto: EvaluationResult
    
    # Analysis
    ablation_results: Optional[AblationResults] = None
    importance_scores: Optional[Dict[str, float]] = None
    shap_values: Optional[Any] = None
    leakage_report: Optional[Dict[str, Any]] = None
    
    # All candidates evaluated
    all_candidates: List[Candidate] = field(default_factory=list)
    
    # Ledger
    ledger: Optional[RunLedger] = None
    
    # Artifact paths
    artifact_dir: Optional[str] = None
    
    @property
    def best_score(self) -> float:
        """Get best CV score."""
        return self.best_result.cv_score_mean
    
    @property
    def improvement_pct(self) -> float:
        """Get improvement over baseline as percentage."""
        if self.baseline_auto.cv_score_mean == 0:
            return 0.0
        return (self.best_score / self.baseline_auto.cv_score_mean - 1) * 100
    
    def summary(self) -> str:
        """Generate summary string."""
        lines = [
            f"=== FeatureCraft Agent Result ===",
            f"Run ID: {self.run_id}",
            f"Task: {self.fingerprint.task_type}",
            f"Dataset: {self.fingerprint.n_rows} rows x {self.fingerprint.n_cols} cols",
            f"",
            f"Best Score: {self.best_score:.4f} +/- {self.best_result.cv_score_std:.4f}",
            f"Baseline (raw): {self.baseline_raw.cv_score_mean:.4f}",
            f"Baseline (auto): {self.baseline_auto.cv_score_mean:.4f}",
            f"Improvement: {self.improvement_pct:+.1f}%",
            f"",
            f"Features Out: {self.best_result.n_features_out}",
            f"Total Time: {self.best_result.total_time_seconds:.1f}s",
            f"",
            f"Artifacts: {self.artifact_dir}",
        ]
        return "\n".join(lines)
    
    def load_pipeline(self) -> Pipeline:
        """Load the best pipeline."""
        return self.best_pipeline

