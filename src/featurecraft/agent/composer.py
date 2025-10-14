"""Composer Module: Converts strategies to sklearn pipelines."""

from __future__ import annotations

from typing import Any, List, Optional
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer

from ..ai.advisor import FeatureStrategy
from ..config import FeatureCraftConfig
from ..encoders import (
    FrequencyEncoder,
    KFoldTargetEncoder,
    HashingEncoder,
    make_ohe,
    RareCategoryGrouper,
)
from ..imputers import choose_numeric_imputer, categorical_imputer
from ..scalers import choose_scaler
from ..transformers import (
    SkewedPowerTransformer,
    DateTimeFeatures,
    EnsureNumericOutput,
)
from ..interactions import ArithmeticInteractions, PolynomialInteractions
from ..clustering import ClusteringFeatureExtractor
from ..statistical import RowStatisticsTransformer, OutlierDetector
from ..text import TextStatisticsExtractor
from ..logging import get_logger
from .config import AgentConfig
from .types import DatasetFingerprint

logger = get_logger(__name__)


class Composer:
    """Converts feature strategies to sklearn pipelines."""
    
    def __init__(self, config: AgentConfig):
        """Initialize composer.
        
        Args:
            config: Agent configuration
        """
        self.config = config
        self.fc_config = FeatureCraftConfig(
            random_state=config.random_seed,
            verbosity=1 if config.verbose else 0,
        )
    
    def build_pipeline(
        self,
        strategy: FeatureStrategy,
        X: pd.DataFrame,
        y: pd.Series,
        fingerprint: Optional[DatasetFingerprint] = None,
    ) -> Pipeline:
        """Build sklearn pipeline from strategy.
        
        Args:
            strategy: Feature engineering strategy
            X: Feature dataframe
            y: Target series
            fingerprint: Optional dataset fingerprint
            
        Returns:
            sklearn Pipeline
        """
        logger.info(f"Building pipeline for strategy: {strategy.reasoning}")
        
        steps = []
        
        # Identify column types
        numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
        categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        datetime_cols = [c for c in X.columns if pd.api.types.is_datetime64_any_dtype(X[c])]
        
        # === Step 1: Preprocessing ===
        transformers = []
        
        # Numeric preprocessing
        if len(numeric_cols) > 0:
            numeric_steps = []

            # Calculate missing rate for numeric columns
            numeric_missing_rate = X[numeric_cols].isnull().mean().mean()

            # Imputation
            imputer = choose_numeric_imputer(
                missing_rate=numeric_missing_rate,
                n_features=len(numeric_cols),
                n_rows=X.shape[0],
                cfg=FeatureCraftConfig(random_state=self.config.random_seed)
            )
            numeric_steps.append(("imputer", imputer))
            
            # Scaling (based on estimator family)
            scaler_type = self._select_scaler(self.config.estimator_family)
            if scaler_type != "none":
                numeric_steps.append(("scaler", choose_scaler(scaler_type)))
            
            numeric_pipeline = Pipeline(numeric_steps)
            transformers.append(("numeric", numeric_pipeline, numeric_cols))
        
        # Categorical preprocessing
        if len(categorical_cols) > 0 and fingerprint:
            cat_transformers = []
            
            # Low cardinality: OHE
            if len(fingerprint.low_cardinality_cols) > 0:
                low_card_steps = [
                    ("imputer", categorical_imputer()),
                    ("ohe", make_ohe()),
                ]
                cat_transformers.append((
                    "low_card",
                    Pipeline(low_card_steps),
                    fingerprint.low_cardinality_cols
                ))
            
            # Mid cardinality: Target encoding
            if len(fingerprint.mid_cardinality_cols) > 0 and strategy.use_target_encoding:
                mid_card_steps = [
                    ("imputer", categorical_imputer()),
                    ("target_enc", KFoldTargetEncoder(n_splits=5)),
                ]
                cat_transformers.append((
                    "mid_card",
                    Pipeline(mid_card_steps),
                    fingerprint.mid_cardinality_cols
                ))
            
            # High cardinality: Frequency encoding
            if len(fingerprint.high_cardinality_cols) > 0:
                high_card_steps = [
                    ("imputer", categorical_imputer()),
                    ("freq_enc", FrequencyEncoder()),
                ]
                cat_transformers.append((
                    "high_card",
                    Pipeline(high_card_steps),
                    fingerprint.high_cardinality_cols
                ))
            
            # Ultra high cardinality: Hashing
            if len(fingerprint.ultra_high_cardinality_cols) > 0:
                ultra_high_steps = [
                    ("imputer", categorical_imputer()),
                    ("hash_enc", HashingEncoder(n_components=256)),
                ]
                cat_transformers.append((
                    "ultra_high_card",
                    Pipeline(ultra_high_steps),
                    fingerprint.ultra_high_cardinality_cols
                ))
            
            if cat_transformers:
                transformers.extend(cat_transformers)
        
        # Datetime preprocessing
        if len(datetime_cols) > 0 and strategy.datetime_features:
            dt_pipeline = Pipeline([
                ("dt_features", DateTimeFeatures(
                    extract=strategy.datetime_features,
                    cyclical=True,
                ))
            ])
            transformers.append(("datetime", dt_pipeline, datetime_cols))
        
        if transformers:
            steps.append(("preprocessing", ColumnTransformer(
                transformers=transformers,
                remainder="drop",
                n_jobs=1,
            )))
        
        # === Step 2: Feature Engineering ===
        
        # Ensure numeric output after preprocessing
        steps.append(("ensure_numeric", EnsureNumericOutput()))
        
        # Statistical features
        if strategy.use_row_statistics and strategy.row_statistics_to_extract:
            steps.append(("row_stats", RowStatisticsTransformer(
                statistics=strategy.row_statistics_to_extract
            )))
        
        # Outlier detection
        if strategy.use_outlier_detection:
            steps.append(("outlier_detector", OutlierDetector(
                method="iqr",
            )))
        
        # Interactions
        if strategy.use_interactions and strategy.interaction_types:
            if "arithmetic" in strategy.interaction_types:
                steps.append(("arithmetic_interactions", ArithmeticInteractions(
                    operations=["multiply", "divide"],
                    top_k=min(strategy.max_interaction_features, 50),
                )))
            elif "polynomial" in strategy.interaction_types:
                steps.append(("polynomial_interactions", PolynomialInteractions(
                    degree=2,
                    include_bias=False,
                )))
        
        # Clustering
        if strategy.use_clustering and strategy.clustering_methods:
            for method in strategy.clustering_methods:
                steps.append((f"clustering_{method}", ClusteringFeatureExtractor(
                    method=method,
                    n_clusters=strategy.clustering_n_clusters,
                    extract_distance=True,
                )))
        
        # Build pipeline
        pipeline = Pipeline(steps)

        logger.info(f"Built pipeline with {len(steps)} steps")
        return pipeline

    def add_evaluation_estimator(self, pipeline: Pipeline, task_type: str) -> Pipeline:
        """Add a final estimator for pipeline evaluation purposes.

        Args:
            pipeline: Feature engineering pipeline
            task_type: Task type ('classification' or 'regression')

        Returns:
            Pipeline with final estimator added
        """
        from sklearn.dummy import DummyClassifier, DummyRegressor
        from sklearn.linear_model import LogisticRegression, LinearRegression

        # Create a copy of the steps to avoid modifying the original
        steps = list(pipeline.steps)

        if task_type == "classification":
            # For classification, use a simple logistic regression that supports predict_proba
            # This gives us a reasonable baseline for feature quality evaluation
            final_estimator = LogisticRegression(
                random_state=self.config.random_seed,
                max_iter=1000,
                solver='lbfgs'  # Good default for small datasets
            )
        else:
            # For regression, use linear regression
            final_estimator = LinearRegression()

        # Add the final estimator
        steps.append(("estimator", final_estimator))

        return Pipeline(steps)
    
    def strategy_to_config(
        self,
        strategy: FeatureStrategy,
        base_config: FeatureCraftConfig,
    ) -> FeatureCraftConfig:
        """Convert strategy to FeatureCraft config overrides.
        
        Args:
            strategy: Feature strategy
            base_config: Base configuration
            
        Returns:
            Modified configuration
        """
        config_dict = base_config.model_dump()
        
        # Encoding
        config_dict["use_target_encoding"] = strategy.use_target_encoding
        config_dict["use_frequency_encoding"] = strategy.use_frequency_encoding
        
        # Interactions
        if strategy.use_interactions:
            if "arithmetic" in strategy.interaction_types:
                config_dict["interactions_enabled"] = True
                config_dict["interactions_use_arithmetic"] = True
            if "polynomial" in strategy.interaction_types:
                config_dict["interactions_use_polynomial"] = True
        
        # Clustering
        config_dict["use_clustering_features"] = strategy.use_clustering
        
        # Text
        if strategy.text_strategy == "advanced":
            config_dict["text_extract_sentiment"] = True
        
        # Feature selection
        config_dict["apply_feature_selection"] = strategy.apply_feature_selection
        if strategy.target_n_features:
            config_dict["feature_selection_k"] = strategy.target_n_features
        
        return FeatureCraftConfig(**config_dict)
    
    def validate_pipeline(self, pipeline: Pipeline) -> bool:
        """Validate pipeline compatibility.
        
        Args:
            pipeline: sklearn Pipeline
            
        Returns:
            True if valid
        """
        try:
            # Check all steps have fit/transform
            for name, step in pipeline.steps:
                if not hasattr(step, "fit") or not hasattr(step, "transform"):
                    logger.warning(f"Step {name} missing fit/transform methods")
                    return False
            return True
        except Exception as e:
            logger.error(f"Pipeline validation failed: {e}")
            return False
    
    def _select_scaler(self, estimator_family: str) -> str:
        """Select scaler based on estimator family.
        
        Args:
            estimator_family: Estimator family
            
        Returns:
            Scaler type string
        """
        scaler_map = {
            "tree": "none",  # Trees don't need scaling
            "linear": "standard",
            "svm": "standard",
            "knn": "minmax",
            "nn": "minmax",
            "catboost": "none",
        }
        return scaler_map.get(estimator_family, "standard")

