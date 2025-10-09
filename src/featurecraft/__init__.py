"""FeatureCraft: automatic feature engineering and insights for tabular ML."""

from __future__ import annotations

from .aggregations import (
    ExpandingWindowTransformer,
    GroupByStatsTransformer,
    LagFeaturesTransformer,
    RankFeaturesTransformer,
    RollingWindowTransformer,
)
from .cli import main
from .config import FeatureCraftConfig
from .encoders import (
    BinaryEncoder,
    CatBoostEncoder,
    CountEncoder,
    EntityEmbeddingsEncoder,
    FrequencyEncoder,
    HashingEncoder,
    KFoldTargetEncoder,
    LeaveOneOutTargetEncoder,
    OrdinalEncoder,
    OutOfFoldTargetEncoder,
    RareCategoryGrouper,
    WoEEncoder,
    make_ohe,
)
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
    "BinaryEncoder",
    "BinnedInteractions",
    "CatBoostEncoder",
    "CategoricalNumericInteractions",
    "CountEncoder",
    "DatasetInsights",
    "DecisionCategory",
    "EntityEmbeddingsEncoder",
    "ExpandingWindowTransformer",
    "FeatureCraftConfig",
    "FrequencyEncoder",
    "GroupByStatsTransformer",
    "HashingEncoder",
    "Issue",
    "KFoldTargetEncoder",
    "LagFeaturesTransformer",
    "LeaveOneOutTargetEncoder",
    "OrdinalEncoder",
    "OutOfFoldTargetEncoder",
    "PipelineExplanation",
    "PipelineExplainer",
    "PolynomialInteractions",
    "ProductInteractions",
    "RankFeaturesTransformer",
    "RareCategoryGrouper",
    "RatioFeatures",
    "ReportBuilder",
    "RollingWindowTransformer",
    "TransformationExplanation",
    "WoEEncoder",
    "build_interaction_pipeline",
    "load_config",
    "main",
    "make_ohe",
    "save_config",
    "version",
]
