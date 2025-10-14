"""Strategist Module: Heuristic-based strategy selection."""

from __future__ import annotations

from typing import List, Optional
from sklearn.model_selection import (
    BaseCrossValidator,
    KFold,
    StratifiedKFold,
    TimeSeriesSplit,
    GroupKFold,
)

from ..ai.advisor import FeatureStrategy
from ..config import FeatureCraftConfig
from ..logging import get_logger
from ..types import TaskType
from .config import AgentConfig, ComputeBudget
from .types import DatasetFingerprint

logger = get_logger(__name__)


class Strategist:
    """Strategy selection based on dataset characteristics."""
    
    def __init__(self, config: AgentConfig):
        """Initialize strategist.
        
        Args:
            config: Agent configuration
        """
        self.config = config
    
    def generate_initial_strategies(
        self,
        fingerprint: DatasetFingerprint,
        estimator_family: str,
        budget: ComputeBudget,
    ) -> List[FeatureStrategy]:
        """Generate initial candidate strategies.
        
        Args:
            fingerprint: Dataset fingerprint
            estimator_family: Estimator family (tree/linear/etc)
            budget: Compute budget
            
        Returns:
            List of candidate strategies
        """
        strategies = []
        
        # Generate base strategies
        for variant in ["conservative", "balanced", "aggressive"]:
            strategy = self._generate_strategy(
                fingerprint=fingerprint,
                estimator_family=estimator_family,
                variant=variant,
                budget=budget,
            )
            strategies.append(strategy)
        
        logger.info(f"Generated {len(strategies)} initial strategies")
        return strategies
    
    def select_cv_strategy(
        self,
        fingerprint: DatasetFingerprint,
    ) -> BaseCrossValidator:
        """Select appropriate CV strategy.
        
        Args:
            fingerprint: Dataset fingerprint
            
        Returns:
            Cross-validator instance
        """
        # Time series: use TimeSeriesSplit
        if fingerprint.is_time_series:
            logger.info("Using TimeSeriesSplit for time series data")
            return TimeSeriesSplit(
                n_splits=self.config.n_cv_folds,
                test_size=fingerprint.n_rows // (self.config.n_cv_folds + 1),
            )
        
        # Entity/group: use GroupKFold if entity column specified
        if self.config.entity_column and self.config.entity_column in fingerprint.entity_columns:
            logger.info(f"Using GroupKFold with entity column: {self.config.entity_column}")
            return GroupKFold(n_splits=self.config.n_cv_folds)
        
        # Classification: use StratifiedKFold
        if fingerprint.task_type == TaskType.CLASSIFICATION:
            logger.info("Using StratifiedKFold for classification")
            return StratifiedKFold(
                n_splits=self.config.n_cv_folds,
                shuffle=self.config.cv_shuffle,
                random_state=self.config.random_seed,
            )
        
        # Regression: use KFold
        logger.info("Using KFold for regression")
        return KFold(
            n_splits=self.config.n_cv_folds,
            shuffle=self.config.cv_shuffle,
            random_state=self.config.random_seed,
        )
    
    def _generate_strategy(
        self,
        fingerprint: DatasetFingerprint,
        estimator_family: str,
        variant: str,
        budget: ComputeBudget,
    ) -> FeatureStrategy:
        """Generate a single strategy.
        
        Args:
            fingerprint: Dataset fingerprint
            estimator_family: Estimator family
            variant: Strategy variant (conservative/balanced/aggressive)
            budget: Compute budget
            
        Returns:
            FeatureStrategy instance
        """
        # Base strategy
        strategy = FeatureStrategy()
        
        # === Encoding Strategy ===
        if len(fingerprint.mid_cardinality_cols) > 0:
            strategy.use_target_encoding = True
            strategy.encoding_priority = "target"
        elif len(fingerprint.high_cardinality_cols) > 0:
            strategy.use_frequency_encoding = True
            strategy.encoding_priority = "frequency"
        
        if len(fingerprint.ultra_high_cardinality_cols) > 0:
            strategy.encoding_priority = "hashing"
        
        # === Interactions ===
        if variant in ["balanced", "aggressive"]:
            if fingerprint.n_numeric <= 30 and estimator_family == "linear":
                strategy.use_interactions = True
                strategy.interaction_types = ["arithmetic"]
                
                if variant == "aggressive":
                    strategy.interaction_types.append("polynomial")
                    strategy.max_interaction_features = 100
                else:
                    strategy.max_interaction_features = 50
        
        # === Clustering ===
        if variant == "aggressive" and fingerprint.n_numeric >= 5:
            strategy.use_clustering = True
            strategy.clustering_methods = ["kmeans"]
            strategy.clustering_n_clusters = 5
        
        # === Statistical Features ===
        if variant in ["balanced", "aggressive"] and fingerprint.n_numeric >= 3:
            strategy.use_row_statistics = True
            strategy.row_statistics_to_extract = ["mean", "std"]
            
            if variant == "aggressive":
                strategy.row_statistics_to_extract.extend(["min", "max"])
        
        # === Outlier Detection ===
        if len(fingerprint.high_outlier_cols) > 0:
            strategy.use_outlier_detection = True
        
        # === Text Features ===
        if len(fingerprint.text_cols) > 0:
            if variant == "conservative":
                strategy.text_strategy = "basic"
                strategy.text_features_to_extract = ["length", "word_count"]
            elif variant == "balanced":
                strategy.text_strategy = "basic"
                strategy.text_features_to_extract = ["length", "word_count", "char_count"]
            else:
                strategy.text_strategy = "advanced"
                strategy.text_features_to_extract = ["length", "word_count", "char_count", "sentiment"]
        
        # === Datetime Features ===
        if len(fingerprint.datetime_cols) > 0:
            strategy.datetime_features = ["year", "month", "day", "dayofweek"]
            
            if variant in ["balanced", "aggressive"]:
                strategy.datetime_features.extend(["hour", "is_weekend"])
        
        # === Groupby Stats ===
        if len(fingerprint.entity_columns) > 0 and variant in ["balanced", "aggressive"]:
            strategy.use_groupby_stats = True
            strategy.groupby_columns = fingerprint.entity_columns[:2]  # Top 2
        
        # === Rolling/Lag Features ===
        if fingerprint.is_time_series and variant in ["balanced", "aggressive"]:
            strategy.use_rolling_features = True
            strategy.use_lag_features = True
        
        # === Domain Features ===
        # Detect domain heuristically (future enhancement)
        strategy.use_domain_features = False
        
        # === Feature Selection ===
        if variant == "aggressive" or fingerprint.estimated_feature_count_baseline > 100:
            strategy.apply_feature_selection = True
            strategy.selection_method = "mutual_info"
            
            # Target feature count based on dataset size
            if fingerprint.n_rows < 1000:
                strategy.target_n_features = 50
            elif fingerprint.n_rows < 10000:
                strategy.target_n_features = 100
            else:
                strategy.target_n_features = 200
        
        # === Reasoning ===
        strategy.reasoning = self._generate_reasoning(fingerprint, variant, estimator_family)
        strategy.estimated_feature_count = self._estimate_features(fingerprint, strategy)
        strategy.estimated_training_time = self._estimate_time(fingerprint, strategy)
        strategy.risk_level = self._assess_risk(fingerprint, strategy)
        
        return strategy
    
    def _generate_reasoning(
        self, fingerprint: DatasetFingerprint, variant: str, estimator_family: str
    ) -> str:
        """Generate human-readable reasoning."""
        lines = [
            f"Strategy: {variant.capitalize()}",
            f"Estimator: {estimator_family}",
            f"Dataset: {fingerprint.n_rows} rows, {fingerprint.n_numeric} numeric, "
            f"{fingerprint.n_categorical} categorical",
        ]
        
        if fingerprint.is_imbalanced:
            lines.append("[!] Dataset is imbalanced")
        
        if len(fingerprint.high_missing_cols) > 0:
            lines.append(f"[!] {len(fingerprint.high_missing_cols)} columns with high missingness")
        
        if len(fingerprint.leakage_risk_cols) > 0:
            lines.append(f"[!] {len(fingerprint.leakage_risk_cols)} columns with leakage risk")
        
        return " | ".join(lines)
    
    def _estimate_features(
        self, fingerprint: DatasetFingerprint, strategy: FeatureStrategy
    ) -> int:
        """Estimate output feature count."""
        count = fingerprint.n_numeric
        
        # Encoding
        for col in fingerprint.categorical_cols:
            card = fingerprint.cardinality_summary.get(col, 10)
            if card <= 10:
                count += card  # OHE
            else:
                count += 1  # Target/freq encoding
        
        # Interactions
        if strategy.use_interactions:
            if "polynomial" in strategy.interaction_types:
                count += fingerprint.n_numeric * (fingerprint.n_numeric + 1) // 2
            elif "arithmetic" in strategy.interaction_types:
                count += min(50, fingerprint.n_numeric * 2)
        
        # Clustering
        if strategy.use_clustering:
            count += len(strategy.clustering_methods) * (strategy.clustering_n_clusters + 1)
        
        # Statistical
        if strategy.use_row_statistics:
            count += len(strategy.row_statistics_to_extract)
        
        # Text
        if len(fingerprint.text_cols) > 0:
            count += len(fingerprint.text_cols) * len(strategy.text_features_to_extract)
        
        # Datetime
        if len(fingerprint.datetime_cols) > 0:
            count += len(fingerprint.datetime_cols) * len(strategy.datetime_features)
        
        return count
    
    def _estimate_time(
        self, fingerprint: DatasetFingerprint, strategy: FeatureStrategy
    ) -> str:
        """Estimate training time."""
        # Simple heuristic based on dataset size and feature count
        estimated_features = self._estimate_features(fingerprint, strategy)
        
        complexity_score = (
            fingerprint.n_rows / 10000
            * estimated_features / 100
        )
        
        if complexity_score < 1:
            return "fast (<1 min)"
        elif complexity_score < 5:
            return "moderate (1-5 min)"
        elif complexity_score < 15:
            return "slow (5-15 min)"
        else:
            return "very slow (>15 min)"
    
    def _assess_risk(
        self, fingerprint: DatasetFingerprint, strategy: FeatureStrategy
    ) -> str:
        """Assess risk level of strategy."""
        risk_score = 0
        
        # High feature count risk
        estimated_features = self._estimate_features(fingerprint, strategy)
        if estimated_features > 500:
            risk_score += 1
        
        # Leakage risk
        if len(fingerprint.leakage_risk_cols) > 0:
            risk_score += 2
        
        # High missing data
        if len(fingerprint.high_missing_cols) > fingerprint.n_cols * 0.2:
            risk_score += 1
        
        # Imbalanced data
        if fingerprint.is_imbalanced:
            risk_score += 1
        
        if risk_score == 0:
            return "low"
        elif risk_score <= 2:
            return "medium"
        else:
            return "high"

