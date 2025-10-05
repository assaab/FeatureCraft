"""FeatureCraft: automatic feature engineering and insights for tabular ML."""

from __future__ import annotations

from .cli import main
from .config import FeatureCraftConfig
from .explainability import (
    DecisionCategory,
    PipelineExplanation,
    PipelineExplainer,
    TransformationExplanation,
)
from .interactions import (
    ArithmeticInteractions,
    BinnedInteractions,
    CategoricalNumericInteractions,
    PolynomialInteractions,
    ProductInteractions,
    RatioFeatures,
    build_interaction_pipeline,
)
from .pipeline import AutoFeatureEngineer
from .report import ReportBuilder
from .settings import load_config, save_config
from .types import DatasetInsights, Issue
from .version import version

__all__ = [
    "ArithmeticInteractions",
    "AutoFeatureEngineer",
    "BinnedInteractions",
    "CategoricalNumericInteractions",
    "DatasetInsights",
    "DecisionCategory",
    "FeatureCraftConfig",
    "Issue",
    "PipelineExplanation",
    "PipelineExplainer",
    "PolynomialInteractions",
    "ProductInteractions",
    "RatioFeatures",
    "ReportBuilder",
    "TransformationExplanation",
    "build_interaction_pipeline",
    "load_config",
    "main",
    "save_config",
    "version",
]
