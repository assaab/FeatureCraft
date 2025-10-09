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
from .clustering import (
    AdaptiveClusteringExtractor,
    ClusteringFeatureExtractor,
    MultiMethodClusteringExtractor,
    build_clustering_pipeline,
)
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
from .reducers import (
    AdaptiveDimensionalityReducer,
    DimensionalityReducer,
    MultiMethodDimensionalityReducer,
    build_dimensionality_reducer,
    build_reducer,
)
from .report import ReportBuilder
from .selection import (
    BorutaSelector,
    Chi2Selector,
    LassoSelector,
    MutualInfoSelector,
    RFESelector,
    SequentialFeatureSelector,
    TreeImportanceSelector,
    WOEIVSelector,
    compute_vif_drop,
    prune_correlated,
)
from .settings import load_config, save_config
from .statistical import (
    MissingValuePatternsTransformer,
    OutlierDetector,
    PercentileRankTransformer,
    QuantileTransformer,
    RowStatisticsTransformer,
    TargetBasedFeaturesTransformer,
    ZScoreTransformer,
    build_statistical_pipeline,
)
from .text import (
    NERFeatureExtractor,
    ReadabilityScoreExtractor,
    SentimentAnalyzer,
    TextPreprocessor,
    TextStatisticsExtractor,
    TopicModelingFeatures,
)
from .types import DatasetInsights, Issue
from .version import version

__all__ = [
    "AdaptiveClusteringExtractor",
    "AdaptiveDimensionalityReducer",
    "ArithmeticInteractions",
    "AutoFeatureEngineer",
    "BinaryEncoder",
    "BinnedInteractions",
    "BorutaSelector",
    "CatBoostEncoder",
    "CategoricalNumericInteractions",
    "Chi2Selector",
    "ClusteringFeatureExtractor",
    "CountEncoder",
    "DatasetInsights",
    "DecisionCategory",
    "DimensionalityReducer",
    "EntityEmbeddingsEncoder",
    "ExpandingWindowTransformer",
    "FeatureCraftConfig",
    "FrequencyEncoder",
    "GroupByStatsTransformer",
    "HashingEncoder",
    "Issue",
    "KFoldTargetEncoder",
    "LagFeaturesTransformer",
    "LassoSelector",
    "LeaveOneOutTargetEncoder",
    "MissingValuePatternsTransformer",
    "MultiMethodClusteringExtractor",
    "MultiMethodDimensionalityReducer",
    "MutualInfoSelector",
    "OrdinalEncoder",
    "OutOfFoldTargetEncoder",
    "OutlierDetector",
    "PercentileRankTransformer",
    "PipelineExplanation",
    "PipelineExplainer",
    "PolynomialInteractions",
    "ProductInteractions",
    "QuantileTransformer",
    "RFESelector",
    "RankFeaturesTransformer",
    "RareCategoryGrouper",
    "RatioFeatures",
    "ReportBuilder",
    "RollingWindowTransformer",
    "SequentialFeatureSelector",
    "TargetBasedFeaturesTransformer",
    "NERFeatureExtractor",
    "ReadabilityScoreExtractor",
    "SentimentAnalyzer",
    "TextPreprocessor",
    "TextStatisticsExtractor",
    "TopicModelingFeatures",
    "TransformationExplanation",
    "TreeImportanceSelector",
    "WOEIVSelector",
    "WoEEncoder",
    "ZScoreTransformer",
    "build_clustering_pipeline",
    "build_dimensionality_reducer",
    "build_interaction_pipeline",
    "build_reducer",
    "build_statistical_pipeline",
    "compute_vif_drop",
    "load_config",
    "main",
    "make_ohe",
    "prune_correlated",
    "save_config",
    "version",
]
