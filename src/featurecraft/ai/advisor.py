"""AI-powered Feature Engineering Advisor using LLMs.

This module uses Large Language Models (OpenAI, Anthropic, or local models)
to intelligently analyze datasets and recommend optimal feature engineering strategies.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from rich.console import Console
from rich.panel import Panel

from ..config import FeatureCraftConfig
from ..logging import get_logger
from ..types import DatasetInsights, TaskType

logger = get_logger(__name__)
console = Console()


@dataclass
class FeatureStrategy:
    """Recommended feature engineering strategy."""
    
    # Core recommendations
    use_interactions: bool = False
    interaction_types: List[str] = None  # ['arithmetic', 'polynomial', 'ratios']
    max_interaction_features: int = 50
    
    # Encoding strategies
    use_target_encoding: bool = True
    use_frequency_encoding: bool = False
    encoding_priority: str = "target"  # 'target', 'frequency', 'hashing'
    
    # Text features (if applicable)
    text_strategy: str = "basic"  # 'basic', 'advanced', 'minimal'
    text_features_to_extract: List[str] = None
    
    # Datetime features (if applicable)
    datetime_features: List[str] = None
    
    # Clustering features (NEW)
    use_clustering: bool = False
    clustering_methods: List[str] = None  # ['kmeans', 'dbscan', 'gmm', 'hierarchical']
    clustering_n_clusters: int = 5
    
    # Statistical features (NEW)
    use_row_statistics: bool = False
    row_statistics_to_extract: List[str] = None  # ['mean', 'std', 'min', 'max', 'median']
    use_outlier_detection: bool = False
    use_percentile_ranking: bool = False
    
    # Aggregation features (NEW)
    use_groupby_stats: bool = False
    groupby_columns: List[str] = None
    use_rolling_features: bool = False
    use_lag_features: bool = False
    
    # Domain-specific features (NEW)
    use_domain_features: bool = False
    domain_type: Optional[str] = None  # 'finance', 'ecommerce', 'healthcare', 'geospatial'
    domain_features_to_extract: List[str] = None
    
    # Dimensionality reduction
    apply_feature_selection: bool = False
    target_n_features: Optional[int] = None
    selection_method: str = "mutual_info"  # 'mutual_info', 'tree_importance', 'correlation'
    
    # Reasoning
    reasoning: str = ""
    estimated_feature_count: int = 0
    estimated_training_time: str = "unknown"
    risk_level: str = "low"  # 'low', 'medium', 'high'
    
    # Specific config overrides
    config_overrides: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.interaction_types is None:
            self.interaction_types = []
        if self.text_features_to_extract is None:
            self.text_features_to_extract = []
        if self.datetime_features is None:
            self.datetime_features = []
        if self.clustering_methods is None:
            self.clustering_methods = []
        if self.row_statistics_to_extract is None:
            self.row_statistics_to_extract = []
        if self.groupby_columns is None:
            self.groupby_columns = []
        if self.domain_features_to_extract is None:
            self.domain_features_to_extract = []
        if self.config_overrides is None:
            self.config_overrides = {}


class AIFeatureAdvisor:
    """AI-powered advisor for intelligent feature engineering decisions.
    
    Uses LLMs (OpenAI GPT-4, Anthropic Claude, or local models) to analyze
    dataset characteristics and recommend optimal feature engineering strategies.
    
    Key Benefits:
    - Prevents feature explosion by being selective
    - Adapts to dataset size, complexity, and characteristics
    - Provides explainable recommendations
    - Reduces training time while maintaining performance
    
    Usage:
        advisor = AIFeatureAdvisor(api_key="your-key", model="gpt-4")
        strategy = advisor.recommend_strategy(X, y, insights)
        optimized_config = advisor.apply_strategy(base_config, strategy)
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        provider: str = "openai",  # 'openai', 'anthropic', 'local'
        temperature: float = 1.0,
        enable_ai: bool = True,
        require_client: bool = False,
    ):
        """Initialize AI Feature Advisor.
        
        Args:
            api_key: API key for LLM provider (reads from env if None)
            model: Model name (e.g., 'gpt-4', 'claude-3-sonnet')
            provider: LLM provider ('openai', 'anthropic', 'local')
            temperature: Temperature for LLM sampling (lower = more deterministic)
            enable_ai: Enable AI recommendations (if False, uses heuristics only)
            require_client: Raise error if client initialization fails (default: False)
        """
        self.model = model
        self.provider = provider.lower()
        self.temperature = temperature
        self.enable_ai = enable_ai
        self.require_client = require_client
        
        # Initialize LLM client
        self.client = None
        if enable_ai:
            self.client = self._initialize_client(api_key)
            if self.client is None and require_client:
                raise RuntimeError(
                    f"Failed to initialize {provider} client. "
                    "Please provide a valid API key or set require_client=False."
                )
    
    def _initialize_client(self, api_key: Optional[str]) -> Optional[Any]:
        """Initialize LLM client based on provider."""
        try:
            if self.provider == "openai":
                try:
                    import openai
                    key = api_key or os.getenv("OPENAI_API_KEY")
                    if not key:
                        logger.warning("OpenAI API key not found. AI recommendations disabled.")
                        return None
                    self.client = openai.OpenAI(api_key=key)
                    logger.info(f"✓ AI Advisor initialized with OpenAI ({self.model})")
                    return self.client
                except ImportError:
                    logger.warning("openai package not installed. Run: pip install openai")
                    return None
            
            elif self.provider == "anthropic":
                try:
                    import anthropic
                    key = api_key or os.getenv("ANTHROPIC_API_KEY")
                    if not key:
                        logger.warning("Anthropic API key not found. AI recommendations disabled.")
                        return None
                    self.client = anthropic.Anthropic(api_key=key)
                    logger.info(f"✓ AI Advisor initialized with Anthropic ({self.model})")
                    return self.client
                except ImportError:
                    logger.warning("anthropic package not installed. Run: pip install anthropic")
                    return None
            
            elif self.provider == "local":
                # Support for local LLMs (Ollama, LM Studio, etc.)
                logger.info("✓ AI Advisor initialized with local model")
                return "local"
            
            else:
                logger.warning(f"Unknown provider: {self.provider}")
                return None
        
        except Exception as e:
            logger.warning(f"Failed to initialize AI client: {e}")
            return None
    
    def recommend_strategy(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        insights: DatasetInsights,
        estimator_family: str = "tree",
        time_budget: Optional[str] = None,  # 'fast', 'balanced', 'thorough'
    ) -> FeatureStrategy:
        """Recommend optimal feature engineering strategy using AI.

        Args:
            X: Feature DataFrame
            y: Target Series
            insights: Dataset insights from analysis
            estimator_family: Estimator family ('tree', 'linear', 'svm', etc.)
            time_budget: Time budget constraint ('fast', 'balanced', 'thorough')

        Returns:
            FeatureStrategy with recommendations

        Raises:
            RuntimeError: If AI recommendation fails and fallback is disabled
            ValueError: If AI is not enabled or client is not available
        """
        # Check if AI is enabled and client is available
        if not self.enable_ai:
            raise ValueError("AI recommendations are disabled. Enable AI or use heuristic-based recommendations.")

        if self.client is None:
            raise ValueError("AI client is not available. Check API key and provider configuration.")

        # Try AI recommendation
        try:
            return self._ai_recommend(X, y, insights, estimator_family, time_budget)
        except Exception as e:
            # Log the error for debugging
            logger.error(f"AI recommendation failed: {e}")

            # Throw error instead of falling back to heuristics
            raise RuntimeError(
                f"AI feature engineering recommendation failed: {str(e)}. "
                "Please check your API key, model configuration, or use heuristic-based recommendations instead."
            )
    
    def _ai_recommend(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        insights: DatasetInsights,
        estimator_family: str,
        time_budget: Optional[str],
    ) -> FeatureStrategy:
        """Use AI to recommend feature engineering strategy."""
        # Prepare data summary for LLM
        data_summary = self._prepare_data_summary(X, y, insights, estimator_family, time_budget)
        
        # Construct prompt
        prompt = self._construct_recommendation_prompt(data_summary)
        
        # Call LLM
        if self.provider == "openai":
            response = self._call_openai(prompt)
        elif self.provider == "anthropic":
            response = self._call_anthropic(prompt)
        elif self.provider == "local":
            response = self._call_local(prompt)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
        
        # Parse response
        strategy = self._parse_ai_response(response, X, insights)
        
        return strategy
    
    def _heuristic_recommend(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        insights: DatasetInsights,
        estimator_family: str,
        time_budget: Optional[str],
    ) -> FeatureStrategy:
        """Heuristic-based recommendations (no AI required).
        
        Smart rules based on dataset characteristics:
        - Dataset size, feature count, complexity
        - Task type (classification vs regression)
        - Estimator family requirements
        - Time budget constraints
        """
        n_rows, n_cols = X.shape
        task = insights.task
        
        # Initialize strategy
        strategy = FeatureStrategy()
        config_overrides = {}
        reasoning_parts = []
        
        # Rule 1: Dataset size determines aggressiveness
        if n_rows < 1000:
            reasoning_parts.append(f"Small dataset ({n_rows} rows) - using conservative feature engineering to prevent overfitting")
            strategy.use_interactions = False
            strategy.apply_feature_selection = False
            config_overrides["interactions_enabled"] = False
            config_overrides["interactions_use_arithmetic"] = False
            config_overrides["interactions_use_polynomial"] = False
            config_overrides["interactions_use_ratios"] = False
            config_overrides["interactions_use_products"] = False
        
        elif n_rows < 10000:
            reasoning_parts.append(f"Medium dataset ({n_rows} rows) - using selective feature engineering with specific limits")
            strategy.use_interactions = (n_cols <= 20)  # Only if few features
            strategy.interaction_types = ['arithmetic'] if strategy.use_interactions else []
            strategy.max_interaction_features = 30
            config_overrides["interactions_enabled"] = strategy.use_interactions
            if strategy.use_interactions:
                config_overrides["interactions_use_arithmetic"] = True
                config_overrides["interactions_arithmetic_ops"] = ["multiply", "divide"]  # Specific ops
                config_overrides["interactions_max_arithmetic_pairs"] = min(15, n_cols * 2)
                config_overrides["interactions_use_polynomial"] = False
                config_overrides["interactions_use_ratios"] = True
                config_overrides["interactions_ratios_include_proportions"] = True
                config_overrides["interactions_ratios_include_log"] = False
                config_overrides["interactions_max_ratio_pairs"] = min(10, n_cols)
                config_overrides["interactions_use_products"] = False
                config_overrides["interactions_use_categorical_numeric"] = True
                config_overrides["interactions_cat_num_strategy"] = "group_stats"
                config_overrides["interactions_max_cat_num_pairs"] = 10
        
        else:  # Large dataset (>= 10000 rows)
            reasoning_parts.append(f"Large dataset ({n_rows} rows) - can afford more feature engineering with controlled limits")
            strategy.use_interactions = (n_cols <= 50)
            if strategy.use_interactions:
                strategy.interaction_types = ['arithmetic', 'ratios']
                if n_cols <= 15:  # Only add polynomial if very few features
                    strategy.interaction_types.append('polynomial')
                strategy.max_interaction_features = min(100, int(n_rows * 0.1))
                config_overrides["interactions_enabled"] = True
                config_overrides["interactions_use_arithmetic"] = True
                config_overrides["interactions_arithmetic_ops"] = ["multiply", "divide", "add"]
                config_overrides["interactions_max_arithmetic_pairs"] = min(50, n_cols * 3)
                config_overrides["interactions_use_polynomial"] = (n_cols <= 15)
                if n_cols <= 15:
                    config_overrides["interactions_polynomial_degree"] = 2
                    config_overrides["interactions_polynomial_max_features"] = min(10, n_cols)
                    config_overrides["interactions_polynomial_interaction_only"] = True
                config_overrides["interactions_use_ratios"] = True
                config_overrides["interactions_ratios_include_proportions"] = True
                config_overrides["interactions_ratios_include_log"] = False
                config_overrides["interactions_max_ratio_pairs"] = min(30, n_cols * 2)
                config_overrides["interactions_use_products"] = False  # Still too expensive
                config_overrides["interactions_use_categorical_numeric"] = True
                config_overrides["interactions_cat_num_strategy"] = "both"
                config_overrides["interactions_max_cat_num_pairs"] = min(20, n_cols)
        
        # Rule 2: High cardinality features - specific encoding strategy
        high_card_cols = [
            p.name for p in insights.profiles 
            if p.is_categorical and p.cardinality and p.cardinality > 50
        ]
        if high_card_cols:
            max_cardinality = max(p.cardinality for p in insights.profiles if p.is_categorical and p.cardinality)
            reasoning_parts.append(f"High cardinality features detected ({len(high_card_cols)} columns, max cardinality: {max_cardinality}) - using target encoding with appropriate smoothing")
            strategy.use_target_encoding = True
            strategy.encoding_priority = "target"
            config_overrides["use_target_encoding"] = True
            config_overrides["use_frequency_encoding"] = False
            config_overrides["use_count_encoding"] = False
            # Adjust smoothing based on cardinality
            if max_cardinality > 1000:
                config_overrides["te_smoothing"] = 30.0  # Higher smoothing for very high cardinality
                config_overrides["hashing_n_features_tabular"] = 256
            else:
                config_overrides["te_smoothing"] = 20.0
            config_overrides["te_noise"] = 0.01
            config_overrides["low_cardinality_max"] = 10
            config_overrides["mid_cardinality_max"] = 50
            config_overrides["rare_level_threshold"] = 0.01
        
        # Rule 3: Estimator family considerations
        if estimator_family.lower() in {"tree", "gbm"}:
            reasoning_parts.append(f"Tree-based estimator - skipping scaling, using auto transforms for heavily skewed features")
            config_overrides["scaler_tree"] = "none"
            # Trees handle interactions naturally, so be conservative
            if strategy.use_interactions:
                strategy.interaction_types = ['arithmetic']  # Most direct
                config_overrides["interactions_use_polynomial"] = False
            # Use auto transform for heavily skewed features (trees can benefit from log transforms)
            config_overrides["transform_strategy"] = "auto"
            config_overrides["skew_threshold"] = 2.0  # Higher threshold for trees
        
        elif estimator_family.lower() in {"linear", "svm"}:
            reasoning_parts.append(f"Linear/SVM estimator - enabling scaling, binning, and comprehensive transforms for non-linearity")
            config_overrides["scaler_linear"] = "standard"
            if strategy.use_interactions and n_cols <= 10:
                strategy.interaction_types = ['polynomial', 'arithmetic']
                config_overrides["interactions_use_polynomial"] = True
                config_overrides["interactions_polynomial_max_features"] = 8
            
            # BINNING: Critical for linear models to learn thresholds
            if n_cols <= 30:  # Only if manageable feature count
                reasoning_parts.append(f"Enabling binning for linear model - allows learning threshold effects and non-linear patterns")
                config_overrides["binning_enabled"] = True
                config_overrides["binning_strategy"] = "auto"
                config_overrides["binning_n_bins"] = 5
                config_overrides["binning_encode"] = "ordinal"
                config_overrides["binning_prefer_supervised"] = True
            
            # MATHEMATICAL TRANSFORMS: Normalize distributions for linear models
            reasoning_parts.append(f"Using auto mathematical transforms to normalize skewed distributions")
            config_overrides["transform_strategy"] = "auto"
            config_overrides["skew_threshold"] = 1.0  # Lower threshold for linear models
        
        # Rule 4: Text columns - cost-aware text feature extraction
        text_cols = [p.name for p in insights.profiles if p.is_text]
        if text_cols:
            if n_rows < 5000:
                reasoning_parts.append(f"Text columns ({len(text_cols)}) with small dataset - using lightweight text features only")
                strategy.text_strategy = "basic"
                strategy.text_features_to_extract = ['statistics']
                config_overrides["text_extract_statistics"] = True
                config_overrides["text_extract_linguistic"] = False
                config_overrides["text_extract_sentiment"] = False
                config_overrides["text_use_word_embeddings"] = False
                config_overrides["text_use_sentence_embeddings"] = False
                config_overrides["text_extract_ner"] = False
                config_overrides["text_use_topic_modeling"] = False
                config_overrides["tfidf_max_features"] = 1000
                config_overrides["ngram_range"] = (1, 1)
            elif n_rows < 50000:
                reasoning_parts.append(f"Text columns ({len(text_cols)}) with medium dataset - using moderate text features (statistics + sentiment)")
                strategy.text_strategy = "advanced"
                strategy.text_features_to_extract = ['statistics', 'sentiment']
                config_overrides["text_extract_statistics"] = True
                config_overrides["text_extract_linguistic"] = True
                config_overrides["text_extract_sentiment"] = True
                config_overrides["text_sentiment_method"] = "textblob"
                config_overrides["text_use_word_embeddings"] = False  # Still too expensive
                config_overrides["text_use_sentence_embeddings"] = False
                config_overrides["text_extract_ner"] = False
                config_overrides["text_use_topic_modeling"] = False
                config_overrides["tfidf_max_features"] = min(5000, n_rows // 2)
                config_overrides["ngram_range"] = (1, 2)
                config_overrides["svd_components_for_trees"] = min(100, n_rows // 100)
            else:
                reasoning_parts.append(f"Text columns ({len(text_cols)}) with large dataset - can afford comprehensive text features")
                strategy.text_strategy = "advanced"
                strategy.text_features_to_extract = ['statistics', 'sentiment', 'linguistic']
                config_overrides["text_extract_statistics"] = True
                config_overrides["text_extract_linguistic"] = True
                config_overrides["text_extract_sentiment"] = True
                config_overrides["text_sentiment_method"] = "textblob"
                config_overrides["text_use_word_embeddings"] = False  # Optional, user can enable
                config_overrides["text_use_sentence_embeddings"] = False  # Optional, user can enable
                config_overrides["text_extract_ner"] = False  # Optional, requires spaCy
                config_overrides["text_use_topic_modeling"] = False
                config_overrides["tfidf_max_features"] = min(20000, n_rows // 5)
                config_overrides["ngram_range"] = (1, 2)
                config_overrides["svd_components_for_trees"] = 200
        
        # Rule 5: Time budget constraints
        if time_budget == "fast":
            reasoning_parts.append("Fast time budget - disabling expensive operations")
            strategy.use_interactions = False
            strategy.text_strategy = "minimal"
            config_overrides["interactions_enabled"] = False
            config_overrides["text_extract_sentiment"] = False
            strategy.estimated_training_time = "< 1 minute"
        
        elif time_budget == "thorough":
            reasoning_parts.append("Thorough time budget - enabling comprehensive feature engineering")
            if n_rows > 5000 and n_cols < 30:
                strategy.use_interactions = True
                strategy.interaction_types = ['arithmetic', 'polynomial', 'ratios']
                config_overrides["interactions_enabled"] = True
                config_overrides["interactions_use_arithmetic"] = True
                config_overrides["interactions_use_polynomial"] = True
                config_overrides["interactions_use_ratios"] = True
            strategy.estimated_training_time = "5-15 minutes"
        
        else:  # balanced (default)
            strategy.estimated_training_time = "2-5 minutes"
        
        # Rule 6: Feature selection for high-dimensional data - specific methods
        if n_cols > 100 or (strategy.use_interactions and n_cols > 30):
            target_features = max(30, min(50, int(n_rows * 0.1), n_cols // 2))
            reasoning_parts.append(f"High-dimensional data ({n_cols} features) or interactions enabled - enabling feature selection to keep top {target_features} features")
            strategy.apply_feature_selection = True
            strategy.target_n_features = target_features
            strategy.selection_method = "mutual_info" if task == TaskType.CLASSIFICATION else "correlation"
            config_overrides["use_mi"] = (task == TaskType.CLASSIFICATION)
            if task == TaskType.CLASSIFICATION:
                config_overrides["mi_top_k"] = target_features
            config_overrides["corr_drop_threshold"] = 0.95
            config_overrides["vif_drop_threshold"] = 10.0
        
        # Rule 7: Class imbalance - SMOTE and WoE
        if insights.target_class_balance:
            minority = min(insights.target_class_balance.values())
            if minority < 0.05:
                reasoning_parts.append(f"Severe class imbalance ({minority:.1%}) - recommending SMOTE with conservative parameters and WoE encoding")
                config_overrides["use_smote"] = True
                config_overrides["smote_threshold"] = 0.05
                config_overrides["smote_k_neighbors"] = 3  # Conservative for severe imbalance
                config_overrides["smote_strategy"] = "auto"
                config_overrides["use_woe"] = True  # WoE works well with imbalanced binary classification
            elif minority < 0.15:
                reasoning_parts.append(f"Moderate class imbalance ({minority:.1%}) - recommending SMOTE and target encoding")
                config_overrides["use_smote"] = True
                config_overrides["smote_threshold"] = 0.15
                config_overrides["smote_k_neighbors"] = 5
                config_overrides["smote_strategy"] = "auto"
            elif minority < 0.30:
                reasoning_parts.append(f"Mild class imbalance ({minority:.1%}) - using class-aware encoding strategies")
                config_overrides["use_smote"] = False  # Not severe enough
                # Rely on proper CV and target encoding
        
        # Rule 8: Clustering features for segmentation and anomaly detection
        if n_rows >= 1000 and n_cols >= 5:
            numeric_count = len([p for p in insights.profiles if not p.is_categorical and not p.is_text])
            if numeric_count >= 5:
                # Clustering can be beneficial
                if task == TaskType.CLASSIFICATION and n_rows >= 5000:
                    reasoning_parts.append(f"Large dataset with {numeric_count} numeric features - enabling K-Means clustering for pattern discovery")
                    strategy.use_clustering = True
                    strategy.clustering_methods = ['kmeans']
                    strategy.clustering_n_clusters = min(10, max(3, int(np.sqrt(n_rows / 100))))
                    config_overrides["clustering_enabled"] = True
                    config_overrides["clustering_methods"] = ['kmeans']
                    config_overrides["clustering_n_clusters"] = strategy.clustering_n_clusters
                    config_overrides["clustering_extract_cluster_id"] = True
                    config_overrides["clustering_extract_distance"] = True
                    config_overrides["clustering_scale_features"] = True
                    estimated_features += strategy.clustering_n_clusters + 1  # cluster IDs + distance
        
        # Rule 9: Row statistics for wide datasets
        if n_cols >= 10:
            numeric_cols = [p.name for p in insights.profiles if not p.is_categorical and not p.is_text]
            if len(numeric_cols) >= 10:
                reasoning_parts.append(f"Wide dataset with {len(numeric_cols)} numeric columns - enabling row statistics to capture cross-feature patterns")
                strategy.use_row_statistics = True
                strategy.row_statistics_to_extract = ['mean', 'std', 'min', 'max']
                config_overrides["row_statistics_enabled"] = True
                config_overrides["row_statistics"] = ['mean', 'std', 'min', 'max']
                config_overrides["row_statistics_include_null_count"] = True
                estimated_features += len(strategy.row_statistics_to_extract) + 1  # stats + null count
        
        # Rule 10: Outlier detection for fraud/anomaly tasks
        if task == TaskType.CLASSIFICATION and n_rows >= 1000:
            # Check for keywords in target or column names that suggest fraud/anomaly detection
            is_anomaly_task = False
            if insights.target_name and any(keyword in insights.target_name.lower() 
                                           for keyword in ['fraud', 'anomaly', 'outlier', 'suspicious']):
                is_anomaly_task = True
            
            if is_anomaly_task:
                reasoning_parts.append("Detected fraud/anomaly detection task - enabling outlier detection features")
                strategy.use_outlier_detection = True
                config_overrides["outlier_detection_enabled"] = True
                config_overrides["outlier_detection_methods"] = ['iqr', 'zscore']
                config_overrides["outlier_detection_threshold"] = 1.5
                estimated_features += 2  # IQR flag + Z-score flag
        
        # Rule 11: GroupBy features for hierarchical/transactional data
        # Check for potential grouping columns (ID columns, categorical with moderate cardinality)
        potential_group_cols = [
            p.name for p in insights.profiles 
            if p.is_categorical and p.cardinality and 10 <= p.cardinality <= 10000
            and any(keyword in p.name.lower() for keyword in ['id', 'customer', 'user', 'store', 'product', 'category'])
        ]
        
        if potential_group_cols and n_rows >= 1000:
            reasoning_parts.append(f"Detected potential grouping columns ({', '.join(potential_group_cols[:2])}) - consider enabling groupby statistics for hierarchical patterns")
            # Don't enable by default (requires careful column selection), but mention it
            strategy.groupby_columns = potential_group_cols[:2]  # Store for reference
        
        # Rule 12: Domain-specific features detection
        # Finance detection
        finance_cols = [p.name for p in insights.profiles if any(
            keyword in p.name.lower() for keyword in ['price', 'close', 'open', 'high', 'low', 'volume', 'return']
        )]
        if len(finance_cols) >= 3 and n_rows >= 100:
            reasoning_parts.append(f"Detected finance-related columns ({', '.join(finance_cols[:3])}) - consider enabling technical indicators")
            strategy.use_domain_features = True
            strategy.domain_type = "finance"
            # Note: Don't enable by default as it requires specific column names
        
        # E-commerce detection
        ecommerce_cols = [p.name for p in insights.profiles if any(
            keyword in p.name.lower() for keyword in ['customer', 'purchase', 'transaction', 'amount', 'order']
        )]
        if len(ecommerce_cols) >= 3 and n_rows >= 100:
            if strategy.domain_type is None:  # Don't override finance
                reasoning_parts.append(f"Detected e-commerce columns ({', '.join(ecommerce_cols[:3])}) - consider RFM analysis if customer-level data")
                strategy.use_domain_features = True
                strategy.domain_type = "ecommerce"
        
        # Geospatial detection
        geo_cols = [p.name for p in insights.profiles if any(
            keyword in p.name.lower() for keyword in ['latitude', 'longitude', 'lat', 'lon', 'coord']
        )]
        if len(geo_cols) >= 2:
            if strategy.domain_type is None:
                reasoning_parts.append(f"Detected geospatial columns ({', '.join(geo_cols)}) - consider distance features")
                strategy.use_domain_features = True
                strategy.domain_type = "geospatial"
        
        # Estimate final feature count
        estimated_features = n_cols
        if strategy.use_interactions:
            # Rough estimate
            if 'arithmetic' in strategy.interaction_types:
                estimated_features += strategy.max_interaction_features // 2
            if 'polynomial' in strategy.interaction_types:
                estimated_features += min(20, n_cols * 2)
            if 'ratios' in strategy.interaction_types:
                estimated_features += strategy.max_interaction_features // 3
        
        strategy.estimated_feature_count = estimated_features
        
        # Risk assessment
        if estimated_features > n_rows * 0.5:
            strategy.risk_level = "high"
            reasoning_parts.append("⚠️ High risk of overfitting - features approaching sample size")
        elif estimated_features > n_rows * 0.2:
            strategy.risk_level = "medium"
        else:
            strategy.risk_level = "low"
        
        strategy.reasoning = "\n".join(reasoning_parts)
        strategy.config_overrides = config_overrides
        
        return strategy
    
    def _prepare_data_summary(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        insights: DatasetInsights,
        estimator_family: str,
        time_budget: Optional[str],
    ) -> Dict[str, Any]:
        """Prepare concise data summary for LLM."""
        n_rows, n_cols = X.shape
        
        summary = {
            "n_rows": n_rows,
            "n_features": n_cols,
            "task": insights.task.value,
            "estimator_family": estimator_family,
            "time_budget": time_budget or "balanced",
            "feature_types": {
                "numeric": insights.summary.get("n_numeric", 0),
                "categorical": insights.summary.get("n_categorical", 0),
                "text": insights.summary.get("n_text", 0),
                "datetime": insights.summary.get("n_datetime", 0),
            },
            "data_quality": {
                "has_missing": any(p.missing_rate > 0 for p in insights.profiles),
                "high_missing_cols": sum(1 for p in insights.profiles if p.missing_rate > 0.3),
                "high_cardinality_cols": sum(1 for p in insights.profiles if p.cardinality and p.cardinality > 50),
            },
            "complexity_indicators": {
                "has_outliers": any(p.outlier_share and p.outlier_share > 0.1 for p in insights.profiles),
                "has_skewed_features": sum(1 for p in insights.profiles if p.skewness and abs(p.skewness) > 2),
            }
        }
        
        # Add class imbalance info
        if insights.target_class_balance:
            minority = min(insights.target_class_balance.values())
            summary["class_imbalance"] = {
                "minority_class_ratio": minority,
                "is_imbalanced": minority < 0.2
            }
        
        return summary
    
    def _get_featurecraft_capabilities_context(self) -> str:
        """Get comprehensive FeatureCraft capabilities context for AI."""
        return """
FeatureCraft Capabilities & Configuration Parameters:

═══════════════════════════════════════════════════════════════
📊 INTERACTIONS (Feature Combination Engineering)
═══════════════════════════════════════════════════════════════

Arithmetic Interactions:
  • interactions_enabled: bool - Master switch for all interactions
  • interactions_use_arithmetic: bool - Enable arithmetic operations
  • interactions_arithmetic_ops: List[str] - Specific ops: ["add", "subtract", "multiply", "divide"]
  • interactions_max_arithmetic_pairs: int - Max pairs (1-1000, typical: 15-100)

Polynomial Features:
  • interactions_use_polynomial: bool - Enable polynomial expansion (x², x³, x₁.x₂)
  • interactions_polynomial_degree: int - Degree (2=quadratic, 3=cubic)
  • interactions_polynomial_interaction_only: bool - Only x₁.x₂, not x²
  • interactions_polynomial_max_features: int - Max input features (2-50, prevents explosion)

Ratio & Proportion Features:
  • interactions_use_ratios: bool - Enable A/B ratios
  • interactions_ratios_include_proportions: bool - Add A/(A+B) style
  • interactions_ratios_include_log: bool - Add log(A/B) ratios
  • interactions_max_ratio_pairs: int - Max pairs (1-500, typical: 15-50)

Multi-way Products:
  • interactions_use_products: bool - Enable A.B.C multi-way products
  • interactions_product_n_way: int - Number of features to multiply (2-5)
  • interactions_max_products: int - Max products to create (1-100)

Categorical x Numeric Interactions:
  • interactions_use_categorical_numeric: bool - Group statistics & deviations
  • interactions_cat_num_strategy: str - "group_stats", "deviation", or "both"
  • interactions_max_cat_num_pairs: int - Max pairs (1-200, typical: 10-30)

Domain-Specific:
  • interactions_specific_pairs: List[Tuple[str, str]] - Explicit pairs: [("age", "income")]
  • interactions_domain_formulas: Dict[str, str] - Custom formulas: {"bmi": "weight / (height ** 2)"}

═══════════════════════════════════════════════════════════════
🔤 ENCODING (Categorical Variable Transformation)
═══════════════════════════════════════════════════════════════

Target Encoding (Recommended for high cardinality):
  • use_target_encoding: bool - Out-of-fold target encoding
  • use_leave_one_out_te: bool - LOO instead of K-Fold
  • te_smoothing: float - Smoothing parameter (0-100, typical: 10-30)
  • te_noise: float - Regularization noise (0-1, typical: 0-0.05)
  • te_prior: str - "global_mean" or "median"

Other Encodings:
  • use_frequency_encoding: bool - Category → frequency count
  • use_count_encoding: bool - Category → occurrence count
  • use_woe: bool - Weight of Evidence (binary classification only)
  • use_ordinal: bool - Ordinal encoding for ordered categories
  • ordinal_maps: Dict[str, List[str]] - Manual ordering: {"size": ["S", "M", "L", "XL"]}

Hashing (for very high cardinality):
  • hashing_n_features_tabular: int - Hash features (8-8192, typical: 128-512)

Cardinality Thresholds:
  • low_cardinality_max: int - Max for one-hot encoding (1-1000, typical: 5-15)
  • mid_cardinality_max: int - Max for target encoding (1-10000, typical: 30-100)
  • rare_level_threshold: float - Group rare categories (0-1, typical: 0.01-0.05)

═══════════════════════════════════════════════════════════════
🔠 TEXT PROCESSING (NLP Feature Extraction)
═══════════════════════════════════════════════════════════════

Vectorization:
  • tfidf_max_features: int - Max TF-IDF features (100-1M, typical: 1000-20000)
  • ngram_range: Tuple[int, int] - N-gram range (typical: (1,1), (1,2), or (1,3))
  • text_use_hashing: bool - HashingVectorizer instead of TF-IDF
  • text_hashing_features: int - Hash features (1024-131072, typical: 4096-16384)
  • svd_components_for_trees: int - SVD reduction for trees (2-1000, typical: 50-200)

Basic Text Statistics (lightweight):
  • text_extract_statistics: bool - char_count, word_count, avg_word_length, etc.
  • text_extract_linguistic: bool - stopword_count, punctuation_count, uppercase_ratio

Sentiment Analysis (moderate cost):
  • text_extract_sentiment: bool - Polarity & subjectivity (TextBlob/VADER)
  • text_sentiment_method: str - "textblob" or "vader"

Word Embeddings (expensive):
  • text_use_word_embeddings: bool - Word2Vec, GloVe, FastText
  • text_embedding_method: str - "word2vec", "glove", "fasttext"
  • text_embedding_dims: int - Dimensions (50-300)
  • text_embedding_aggregation: str - "mean", "max", "sum"

Sentence Embeddings (very expensive, transformers):
  • text_use_sentence_embeddings: bool - BERT, SentenceTransformers
  • text_sentence_model: str - Model name (e.g., "all-MiniLM-L6-v2")
  • text_sentence_batch_size: int - Batch size (1-256, typical: 16-64)

Named Entity Recognition (expensive, requires spaCy):
  • text_extract_ner: bool - Extract entity counts (PERSON, ORG, GPE, etc.)
  • text_ner_model: str - "en_core_web_sm", "en_core_web_md", "en_core_web_lg"
  • text_ner_entity_types: List[str] - Entities to count

Topic Modeling (expensive):
  • text_use_topic_modeling: bool - LDA topic distributions
  • text_topic_n_topics: int - Number of topics (2-100, typical: 5-20)
  • text_topic_max_features: int - Vectorizer features (100-50000)

Readability Scores:
  • text_extract_readability: bool - Flesch-Kincaid, SMOG index, etc.
  • text_readability_metrics: List[str] - Metrics to compute

═══════════════════════════════════════════════════════════════
⚖️ SCALING & TRANSFORMS (Numeric Feature Normalization)
═══════════════════════════════════════════════════════════════

Estimator-Specific Scaling (CRITICAL for optimization):
  • scaler_linear: str - "standard", "minmax", "robust", "maxabs", "none"
  • scaler_svm: str - Same options (typically "standard")
  • scaler_tree: str - Same options (typically "none" - trees don't need scaling!)
  • scaler_knn: str - Same options (typically "minmax" or "standard")
  • scaler_nn: str - Same options (typically "minmax" or "standard")

Auto-Scaling:
  • scaler_robust_if_outliers: bool - Auto-use RobustScaler if outliers detected

Mathematical Transforms (NEW - Comprehensive):
  • transform_strategy: str - "auto", "log", "log1p", "sqrt", "box_cox", "yeo_johnson", "reciprocal", "exponential", "none"
  • transform_columns: List[str] | None - Specific columns to transform (None = auto-detect)
  • log_shift: float - Shift for log(x + shift) to handle zeros (typical: 1e-5)
  • sqrt_handle_negatives: str - "abs" (signed sqrt), "clip" (set to 0), "error"
  • reciprocal_epsilon: float - Prevent div by zero: 1/(x + eps) (typical: 1e-10)
  • exponential_transform_type: str - "square" (x²), "cube" (x³), "exp" (e^x)
  • boxcox_lambda: float | None - Fixed lambda for Box-Cox (None = optimize)
  • skew_threshold: float - Skewness threshold for auto transform selection (0+, typical: 0.5-2.0)
  
Transform Selection Guide:
  - "auto": Intelligently selects per column based on data (RECOMMENDED)
  - "log": Right-skewed exponential growth (requires x > 0)
  - "log1p": Handles zeros, good for count data (requires x ≥ 0)
  - "sqrt": Moderate skewness, count data (handles negatives with abs mode)
  - "box_cox": Optimizes λ for normality (requires x > 0, slower)
  - "yeo_johnson": Like Box-Cox but handles negatives (most flexible)
  - "reciprocal": Heavy right-tailed distributions (1/x)
  - "exponential": Left-skewed data (rare, use with caution)

Outlier Handling:
  • outlier_share_threshold: float - Trigger robust methods (0-1, typical: 0.03-0.10)
  • winsorize: bool - Clip extreme outliers
  • clip_percentiles: Tuple[float, float] - Clip range (typical: (0.01, 0.99))

═══════════════════════════════════════════════════════════════
📊 BINNING / DISCRETIZATION (NEW - Convert Continuous → Categorical)
═══════════════════════════════════════════════════════════════

Master Controls:
  • binning_enabled: bool - Enable binning/discretization
  • binning_columns: List[str] | None - Specific columns (None = auto-detect numeric)
  • binning_n_bins: int - Number of bins (2-20, typical: 3-7)
  • binning_encode: str - "ordinal" (0,1,2...) or "onehot"

Binning Strategies:
  • binning_strategy: str - "auto", "equal_width", "equal_frequency", "kmeans", "decision_tree", "custom"

Strategy Guide:
  - "auto": Intelligently selects best strategy per column (RECOMMENDED)
    → Uses decision_tree if target correlation strong
    → Uses equal_frequency for skewed distributions
    → Uses equal_width for uniform distributions
  - "equal_width": Fixed-width intervals (good for uniform data)
  - "equal_frequency": Quantile-based bins (good for skewed data, equal sample counts)
  - "kmeans": Cluster-based boundaries (data-driven, complex distributions)
  - "decision_tree": Supervised binning (target-aware splits, best for predictive power)
  - "custom": User-defined bin edges via binning_custom_bins

Auto Strategy Parameters:
  • binning_prefer_supervised: bool - Use decision_tree when correlation strong (default: True)
  • binning_skewness_threshold: float - Skewness threshold for equal_frequency vs equal_width (typical: 1.0)

Advanced Options:
  • binning_custom_bins: Dict[str, List[float]] - Custom edges per column: {"age": [0, 18, 35, 65, 100]}
  • binning_handle_unknown: str - "ignore" or "error" for out-of-range values
  • binning_subsample: int - Subsample for expensive methods (kmeans, decision_tree, typical: 200000)

Why Use Binning:
  - Linear models can learn threshold effects (e.g., "age > 65 → high risk")
  - Reduces overfitting on exact continuous values
  - Captures non-linear relationships in linear models
  - Can improve tree models by reducing splits on noisy continuous values
  - Enables categorical interactions with numeric features

Best For:
  - Linear models (logistic regression, linear regression) ✓✓✓
  - SVM models ✓✓
  - Neural networks ✓
  - Tree models (optional, may help reduce overfitting) ✓

Cost:
  - equal_width/equal_frequency: Very fast O(n log n)
  - kmeans: Moderate O(n * k * iterations)
  - decision_tree: Moderate-High O(n log n * n_bins)

═══════════════════════════════════════════════════════════════
🎯 FEATURE SELECTION (Dimensionality Reduction)
═══════════════════════════════════════════════════════════════

Correlation-Based:
  • corr_drop_threshold: float - Drop highly correlated (0-1, typical: 0.90-0.98)

Multicollinearity:
  • vif_drop_threshold: float - VIF threshold (1+, typical: 5-10)

Mutual Information:
  • use_mi: bool - Mutual information feature selection
  • mi_top_k: int - Keep top K features (1+, typical: 20-100)

Weight of Evidence (binary classification):
  • use_woe_selection: bool - WoE/IV-based selection
  • woe_iv_threshold: float - Min Information Value (0+, typical: 0.02-0.10)

═══════════════════════════════════════════════════════════════
📅 DATETIME FEATURES (Temporal Feature Engineering)
═══════════════════════════════════════════════════════════════

Basic Extraction:
  • dt_extract_basic: bool - year, month, day, hour, minute, second, day_of_week, week_of_year, quarter, day_of_year
  • dt_extract_cyclical: bool - Sin/cos encodings for cyclical patterns
  • dt_extract_boolean_flags: bool - is_weekend, is_month_start, is_month_end, etc.
  • dt_extract_season: bool - Season (0-3: winter, spring, summer, fall)
  • dt_extract_business: bool - is_business_hour, business_days_in_month

Time Series:
  • ts_default_lags: List[int] - Lag periods (e.g., [1, 7, 28])
  • ts_default_windows: List[int] - Rolling windows (e.g., [3, 7, 28])
  • use_fourier: bool - Fourier features for cyclical patterns
  • fourier_orders: List[int] - Orders (e.g., [3, 7] for daily/weekly)

═══════════════════════════════════════════════════════════════
⚖️ CLASS IMBALANCE HANDLING
═══════════════════════════════════════════════════════════════

SMOTE (oversampling):
  • use_smote: bool - Enable SMOTE oversampling
  • smote_threshold: float - Minority ratio trigger (0-1, typical: 0.05-0.20)
  • smote_k_neighbors: int - K neighbors (1-20, typical: 3-7)
  • smote_strategy: str - "auto", "minority", "all"

Undersampling:
  • use_undersample: bool - Random undersampling of majority class

═══════════════════════════════════════════════════════════════
🎲 MISSING VALUES & IMPUTATION
═══════════════════════════════════════════════════════════════

Numeric:
  • numeric_simple_impute_max: float - Threshold for simple imputation (0-1, typical: 0.05)
  • numeric_advanced_impute_max: float - Max for advanced imputation (0-1, typical: 0.30)

Categorical:
  • categorical_impute_strategy: str - "most_frequent" or "constant"
  • categorical_missing_indicator_min: float - Add missing indicator (0-1, typical: 0.05)
  • add_missing_indicators: bool - Binary flags for missingness

═══════════════════════════════════════════════════════════════
⚙️ CROSS-VALIDATION (for Target Encoding)
═══════════════════════════════════════════════════════════════

  • cv_n_splits: int - Number of folds (2-20, typical: 3-10)
  • cv_strategy: str - "kfold", "stratified", "group", "time"
  • cv_shuffle: bool - Shuffle data in KFold

═══════════════════════════════════════════════════════════════
🏭 CLUSTERING-BASED FEATURES (NEW - Unsupervised Pattern Discovery)
═══════════════════════════════════════════════════════════════

Master Controls:
  • clustering_enabled: bool - Enable clustering feature extraction
  • clustering_methods: List[str] - Methods: ["kmeans", "dbscan", "gmm", "hierarchical"]
  • clustering_n_clusters: int - Number of clusters (2-50, typical: 3-10)
  • clustering_columns: List[str] | None - Specific columns (None = all numeric)

Feature Extraction Options:
  • clustering_extract_cluster_id: bool - Cluster membership labels
  • clustering_extract_distance: bool - Distances to cluster centroids
  • clustering_extract_probabilities: bool - Soft assignments (GMM only)
  • clustering_extract_outlier_flag: bool - Outlier detection (DBSCAN)
  • clustering_extract_density: bool - Local density metrics
  • clustering_scale_features: bool - Auto-scale before clustering (recommended)

Algorithm-Specific Parameters:
  K-Means:
    • kmeans_init: str - "k-means++" or "random"
    • kmeans_max_iter: int - Max iterations (100-1000)
    • kmeans_n_init: int - Number of initializations (1-20)
  
  DBSCAN:
    • dbscan_eps: float - Neighborhood radius (0.1-10.0, typical: 0.3-1.0)
    • dbscan_min_samples: int - Min points for core (2-20, typical: 3-10)
    • dbscan_metric: str - Distance metric ("euclidean", "manhattan", "cosine")
  
  Gaussian Mixture:
    • gmm_covariance_type: str - "full", "tied", "diag", "spherical"
    • gmm_max_iter: int - Max EM iterations (50-500)
    • gmm_n_init: int - Number of initializations (1-10)
  
  Hierarchical:
    • hierarchical_linkage: str - "ward", "complete", "average", "single"
    • hierarchical_distance_threshold: float | None - Cut threshold

Use Cases:
  - Customer segmentation (K-Means, GMM)
  - Anomaly/fraud detection (DBSCAN outlier flags)
  - Pattern discovery in unlabeled data
  - Feature engineering for tree/linear models
  - Multi-modal distribution capture (GMM)
  - Hierarchical taxonomies (Hierarchical)

Best For:
  - Medium-large datasets (>1K rows recommended)
  - Numeric features with meaningful distances
  - Unsupervised pattern discovery
  - Tree models (cluster IDs as categorical splits)
  - Linear models (cluster distances as features)

Cost:
  - K-Means: Fast O(n * k * iterations), typical: <1s for 10K rows
  - DBSCAN: Moderate O(n log n), typical: 1-5s for 10K rows
  - GMM: Moderate-High O(n * k * iterations * d²), typical: 5-30s for 10K rows
  - Hierarchical: High O(n²), typical: slow for >5K rows

═══════════════════════════════════════════════════════════════
📊 STATISTICAL FEATURES (NEW - Cross-feature Patterns & Outliers)
═══════════════════════════════════════════════════════════════

Row Statistics (Cross-feature Aggregations):
  • row_statistics_enabled: bool - Enable row-wise statistics
  • row_statistics: List[str] - Stats: ["mean", "std", "min", "max", "median", "sum", "range", "skew", "kurtosis"]
  • row_statistics_columns: List[str] | None - Specific columns (None = all numeric)
  • row_statistics_include_null_count: bool - Add null count per row
  • row_statistics_prefix: str - Column prefix (default: "row")

Percentile Ranking:
  • percentile_ranking_enabled: bool - Within-column percentile ranks
  • percentile_ranking_columns: List[str] | None - Specific columns
  • percentile_ranking_method: str - "average", "min", "max", "dense"

Z-Score Standardization:
  • z_score_enabled: bool - Standardized scores (mean=0, std=1)
  • z_score_columns: List[str] | None - Specific columns
  • z_score_robust: bool - Use median/MAD instead of mean/std

Outlier Detection:
  • outlier_detection_enabled: bool - Flag outliers
  • outlier_detection_methods: List[str] - ["iqr", "zscore", "isolation_forest", "lof"]
  • outlier_detection_threshold: float - Threshold for flagging
    → IQR: Multiplier for IQR (typical: 1.5-3.0)
    → Z-score: Standard deviations (typical: 2.5-4.0)
  • outlier_detection_contamination: float - Expected outlier fraction (0.001-0.20, typical: 0.05)
  • outlier_detection_add_scores: bool - Add outlier scores (not just flags)
  • outlier_detection_columns: List[str] | None - Specific columns

Quantile Features:
  • quantile_features_enabled: bool - Quantile-based transformations
  • quantile_features_n_quantiles: int - Number of quantiles (2-1000, typical: 10-100)
  • quantile_features_output_distribution: str - "uniform" or "normal"

Target-Based Statistical Features:
  • target_based_features_enabled: bool - Statistical relationships with target
  • target_based_features_statistics: List[str] - ["correlation", "mutual_info", "anova_f"]
  • target_based_features_bins: int - Bins for continuous target (3-20)

Use Cases:
  - Row-wise patterns across sensors/measurements
  - Data quality features (null counts, outlier flags)
  - Normalized feature representations
  - Outlier-aware modeling (fraud detection, anomaly detection)
  - Cross-feature signal strength

Best For:
  - Wide datasets with many numeric features (>10 columns)
  - Sensor data, financial metrics, lab results
  - Outlier-sensitive tasks (fraud, quality control)
  - All model types benefit from row statistics

Cost:
  - Row statistics: Very fast O(n * m)
  - Percentile ranking: Fast O(n log n) per column
  - Z-score: Very fast O(n) per column
  - Outlier detection (IQR/Z-score): Fast O(n) per column
  - Outlier detection (IsolationForest/LOF): Moderate O(n log n) to O(n²)

═══════════════════════════════════════════════════════════════
🔄 AGGREGATION & GROUPBY FEATURES (NEW - Hierarchical & Time-Series)
═══════════════════════════════════════════════════════════════

GroupBy Statistics (SQL-like GROUP BY aggregations):
  • groupby_enabled: bool - Enable group-level statistics
  • groupby_group_cols: List[str] - Columns to group by (e.g., ["customer_id", "store_id"])
  • groupby_agg_cols: List[str] | None - Columns to aggregate (None = all numeric)
  • groupby_agg_functions: List[str] - Functions: ["mean", "sum", "std", "min", "max", "median", "count", "nunique", "var", "skew", "kurt"]
  • groupby_add_count: bool - Add group size as feature
  • groupby_fill_missing_groups: bool - Handle new groups in test set
  • groupby_missing_fill_value: float - Value for missing groups (default: 0.0)

Rolling Window Features (Moving statistics):
  • rolling_enabled: bool - Enable rolling window features
  • rolling_windows: List[int] - Window sizes (e.g., [3, 7, 14, 28])
  • rolling_functions: List[str] - Functions: ["mean", "sum", "std", "min", "max"]
  • rolling_columns: List[str] | None - Specific columns
  • rolling_min_periods: int - Min observations for window (1+)
  • rolling_center: bool - Center-aligned windows

Expanding Window Features (Cumulative statistics):
  • expanding_enabled: bool - Enable expanding window features
  • expanding_functions: List[str] - Functions: ["mean", "sum", "std", "min", "max", "count"]
  • expanding_columns: List[str] | None - Specific columns
  • expanding_min_periods: int - Min observations (1+)

Lag Features (Historical values):
  • lag_enabled: bool - Enable lag features
  • lag_periods: List[int] - Lag periods (e.g., [1, 7, 14, 28, 365])
  • lag_columns: List[str] | None - Specific columns
  • lag_fill_value: float | None - Fill value for initial NaNs

Rank Features (Within-group rankings):
  • rank_enabled: bool - Enable rank features
  • rank_group_cols: List[str] | None - Group columns for ranking
  • rank_columns: List[str] | None - Columns to rank
  • rank_method: str - "average", "min", "max", "dense", "ordinal"
  • rank_pct: bool - Convert to percentile ranks (0-1)

Use Cases:
  - Customer transaction aggregations (total spend, avg order, transaction count)
  - Time-series forecasting (rolling means, lags, trends)
  - Hierarchical data (store-level stats, product-category patterns)
  - Historical pattern extraction (moving averages, cumulative sums)
  - Within-group rankings (customer ranking by spend in region)

Best For:
  - Hierarchical/grouped data (customer transactions, store sales)
  - Time-series data (sequential observations)
  - Panel data (multiple entities over time)
  - Tree models benefit greatly from group statistics
  - Forecasting tasks require rolling/lag features

Cost:
  - GroupBy: Moderate O(n log n) for sorting + O(n) aggregation
  - Rolling: Moderate O(n * w) where w = window size
  - Expanding: Fast O(n)
  - Lag: Very fast O(n)
  - Rank: Moderate O(n log n) per group

═══════════════════════════════════════════════════════════════
🌍 DOMAIN-SPECIFIC FEATURES (NEW - Industry Knowledge Encoding)
═══════════════════════════════════════════════════════════════

Master Controls:
  • domain_features_enabled: bool - Enable domain-specific features
  • domain_type: str - "finance", "ecommerce", "healthcare", "geospatial", "nlp"

FINANCE & TRADING:
  • finance_technical_indicators: bool - RSI, MACD, Bollinger Bands, Moving Averages
  • finance_indicators: List[str] - ["rsi", "macd", "bollinger", "sma", "ema", "roc", "stochastic"]
  • finance_price_col: str - Price column name (default: "close")
  • finance_high_col: str - High price column (default: "high")
  • finance_low_col: str - Low price column (default: "low")
  • finance_rsi_period: int - RSI period (5-50, typical: 14)
  • finance_macd_fast: int - MACD fast period (typical: 12)
  • finance_macd_slow: int - MACD slow period (typical: 26)
  • finance_bb_period: int - Bollinger Bands period (typical: 20)
  • finance_sma_periods: List[int] - SMA periods (e.g., [20, 50, 200])
  
  • finance_risk_ratios: bool - Sharpe, Sortino, Max Drawdown
  • finance_volatility: bool - Historical volatility, Beta
  • finance_returns_col: str - Returns column for risk metrics

E-COMMERCE & RETAIL:
  • ecommerce_rfm: bool - RFM analysis (Recency, Frequency, Monetary)
  • ecommerce_rfm_customer_col: str - Customer ID column
  • ecommerce_rfm_date_col: str - Transaction date column
  • ecommerce_rfm_amount_col: str - Transaction amount column
  • ecommerce_rfm_bins: int - Bins for RFM scores (3-5, typical: 5)
  
  • ecommerce_clv: bool - Customer Lifetime Value features
  • ecommerce_basket_analysis: bool - Market basket features
  • ecommerce_seasonality: bool - Purchase seasonality patterns

HEALTHCARE & MEDICAL:
  • healthcare_vital_ratios: bool - Heart rate/BP ratios, BMI
  • healthcare_bmi_weight_col: str - Weight column (kg)
  • healthcare_bmi_height_col: str - Height column (m)
  • healthcare_clinical_scores: bool - Risk scores, severity indices
  • healthcare_lab_ranges: bool - Normal range deviations

GEOSPATIAL:
  • geospatial_distance_features: bool - Haversine distance calculations
  • geospatial_lat_col: str - Latitude column (default: "latitude")
  • geospatial_lon_col: str - Longitude column (default: "longitude")
  • geospatial_reference_points: List[Tuple[float, float]] - POIs for distance calc
  • geospatial_coordinate_transforms: bool - Grid, zones, quadrants
  • geospatial_proximity_features: bool - Nearest neighbor distances

NLP & TEXT (Domain-specific beyond basic text):
  • nlp_pos_features: bool - Part-of-speech tag frequencies
  • nlp_dependency_features: bool - Syntactic dependency patterns
  • nlp_semantic_similarity: bool - Semantic similarity to reference texts
  • nlp_keyword_extraction: bool - TF-IDF based keyword features

Use Cases:
  - Finance: Trading algorithms, risk assessment, portfolio analysis
  - E-commerce: Customer segmentation, churn prediction, recommendation
  - Healthcare: Diagnosis support, risk stratification, treatment planning
  - Geospatial: Location-based services, logistics, real estate
  - NLP: Advanced text classification, information extraction

Best For:
  - Domain-specific prediction tasks
  - Incorporating expert knowledge as features
  - When domain context is critical to prediction
  - Datasets with clear domain alignment

Cost:
  - Finance indicators: Moderate (rolling computations)
  - E-commerce RFM: Moderate (groupby aggregations)
  - Healthcare ratios: Very fast (simple math)
  - Geospatial distance: Moderate (distance computations)
  - NLP advanced: High (requires spaCy/transformers)

Recommendation Guidelines:
  - Use ONLY when dataset clearly belongs to domain
  - Verify required columns exist (price, date, lat/lon, etc.)
  - Start with lightweight features, expand if beneficial
  - Combine with general techniques for best results

═══════════════════════════════════════════════════════════════
"""

    def _construct_recommendation_prompt(self, data_summary: Dict[str, Any]) -> str:
        """Construct FeatureCraft-aware prompt for LLM."""
        
        # Get comprehensive capabilities context
        capabilities = self._get_featurecraft_capabilities_context()
        
        prompt = f"""You are an expert ML feature engineering advisor with deep knowledge of FeatureCraft's capabilities.
Analyze this dataset and recommend SPECIFIC FeatureCraft configuration parameters for optimal results.

{capabilities}

═══════════════════════════════════════════════════════════════
📊 DATASET PROFILE
═══════════════════════════════════════════════════════════════

Basic Info:
  • Rows: {data_summary['n_rows']:,}
  • Features: {data_summary['n_features']}
  • Task: {data_summary['task']}
  • Estimator Family: {data_summary['estimator_family']}
  • Time Budget: {data_summary['time_budget']}

Feature Types:
  • Numeric: {data_summary['feature_types']['numeric']}
  • Categorical: {data_summary['feature_types']['categorical']}
  • Text: {data_summary['feature_types']['text']}
  • Datetime: {data_summary['feature_types']['datetime']}

Data Quality Issues:
  • High missing columns: {data_summary['data_quality']['high_missing_cols']}
  • High cardinality columns: {data_summary['data_quality']['high_cardinality_cols']}
  • Outlier issues: {data_summary['complexity_indicators']['has_outliers']}
  • Skewed features: {data_summary['complexity_indicators']['has_skewed_features']}

"""
        
        if 'class_imbalance' in data_summary:
            prompt += f"""Class Imbalance:
  • Minority class ratio: {data_summary['class_imbalance']['minority_class_ratio']:.1%}
  • Is imbalanced: {data_summary['class_imbalance']['is_imbalanced']}

"""
        
        prompt += """═══════════════════════════════════════════════════════════════
🎯 YOUR TASK
═══════════════════════════════════════════════════════════════

Recommend SPECIFIC FeatureCraft configuration parameters in JSON format:

{
  "reasoning": "Comprehensive explanation of your specific choices based on FeatureCraft capabilities and dataset characteristics. Explain trade-offs, computational costs, and expected benefits for each major decision.",
  
  "estimated_feature_count": 50,
  "risk_level": "low",  // "low", "medium", "high" based on feature-to-sample ratio
  
  "config_overrides": {
    // ═══ INTERACTIONS ═══
    // Be SPECIFIC with operations and limits!
    "interactions_enabled": true,
    "interactions_use_arithmetic": true,
    "interactions_arithmetic_ops": ["multiply", "divide"],  // Specific ops!
    "interactions_max_arithmetic_pairs": 25,
    
    "interactions_use_polynomial": false,  // Expensive for trees
    "interactions_polynomial_degree": 2,
    "interactions_polynomial_max_features": 8,
    
    "interactions_use_ratios": true,
    "interactions_ratios_include_proportions": true,
    "interactions_ratios_include_log": false,
    "interactions_max_ratio_pairs": 20,
    
    "interactions_use_products": false,  // Very expensive
    "interactions_use_categorical_numeric": true,
    "interactions_cat_num_strategy": "group_stats",
    "interactions_max_cat_num_pairs": 15,
    
    // ═══ ENCODINGS ═══
    // Choose based on cardinality & estimator!
    "use_target_encoding": true,  // High cardinality
    "te_smoothing": 20.0,
    "te_noise": 0.01,
    "use_frequency_encoding": false,  // Redundant with target encoding
    "use_woe": true,  // Binary classification with imbalance
    "use_ordinal": false,
    
    "low_cardinality_max": 10,
    "mid_cardinality_max": 50,
    "rare_level_threshold": 0.01,
    
    // ═══ MATHEMATICAL TRANSFORMS (NEW!) ═══
    // Auto-select optimal transform per column based on distribution
    "transform_strategy": "auto",  // Or: "log", "log1p", "sqrt", "box_cox", "yeo_johnson", "reciprocal", "exponential", "none"
    "skew_threshold": 1.0,  // Lower for linear models (1.0), higher for trees (2.0)
    "log_shift": 0.00001,  // For log(x + shift) when x has zeros
    "sqrt_handle_negatives": "abs",  // Signed square root
    "reciprocal_epsilon": 1e-10,  // Prevent division by zero
    "boxcox_lambda": null,  // Auto-optimize lambda
    
    // ═══ BINNING / DISCRETIZATION (NEW!) ═══
    // Convert continuous → categorical for linear models
    "binning_enabled": true,  // CRITICAL for linear models to learn thresholds!
    "binning_strategy": "auto",  // Or: "equal_width", "equal_frequency", "kmeans", "decision_tree", "custom"
    "binning_n_bins": 5,  // 3-7 typical
    "binning_encode": "ordinal",  // Or "onehot"
    "binning_prefer_supervised": true,  // Use decision_tree when target correlation strong
    "binning_skewness_threshold": 1.0,  // For auto strategy selection
    // "binning_columns": ["age", "income"],  // Optional: specific columns only
    
    // ═══ SCALING ═══
    // CRITICAL: Match to estimator family!
    "scaler_tree": "none",  // Trees don't need scaling
    "scaler_linear": "standard",
    "scaler_svm": "standard",
    "scaler_robust_if_outliers": true,
    "outlier_share_threshold": 0.05,
    "winsorize": false,
    
    // ═══ TEXT (if applicable) ═══
    "text_extract_statistics": true,  // Lightweight
    "text_extract_sentiment": true,  // Moderate cost
    "text_sentiment_method": "textblob",
    "text_use_word_embeddings": false,  // Expensive
    "text_use_sentence_embeddings": false,  // Very expensive
    "text_extract_ner": false,  // Expensive, needs spaCy
    "text_use_topic_modeling": false,  // Expensive
    "tfidf_max_features": 5000,
    "ngram_range": [1, 2],
    "svd_components_for_trees": 100,
    
    // ═══ DATETIME (if applicable) ═══
    "dt_extract_basic": true,
    "dt_extract_cyclical": true,
    "dt_extract_boolean_flags": true,
    "dt_extract_season": true,
    "dt_extract_business": false,
    
    // ═══ SELECTION ═══
    // Prevent overfitting!
    "use_mi": true,
    "mi_top_k": 30,  // Specific number
    "corr_drop_threshold": 0.95,
    "vif_drop_threshold": 10.0,
    "use_woe_selection": false,
    
    // ═══ IMBALANCE ═══
    "use_smote": true,  // For imbalanced data
    "smote_threshold": 0.15,
    "smote_k_neighbors": 5,
    "use_undersample": false,
    
    // ═══ MISSING VALUES ═══
    "numeric_simple_impute_max": 0.05,
    "numeric_advanced_impute_max": 0.30,
    "add_missing_indicators": true,
    "categorical_missing_indicator_min": 0.05,
    
    // ═══ CV (for target encoding) ═══
    "cv_n_splits": 5,
    "cv_strategy": "stratified",
    
    // ═══ CLUSTERING (NEW!) ═══
    // Unsupervised pattern discovery & segmentation
    "clustering_enabled": false,  // Enable for customer segmentation, anomaly detection
    "clustering_methods": ["kmeans"],  // Or: ["dbscan", "gmm", "hierarchical"]
    "clustering_n_clusters": 5,  // 3-10 typical
    "clustering_extract_cluster_id": true,
    "clustering_extract_distance": true,
    "clustering_extract_probabilities": false,  // GMM only
    "clustering_extract_outlier_flag": false,  // DBSCAN only
    // "clustering_columns": ["feature1", "feature2"],  // Optional: specific columns
    
    // ═══ STATISTICAL FEATURES (NEW!) ═══
    // Row-wise patterns & outlier detection
    "row_statistics_enabled": false,  // Enable for wide datasets with many numeric features
    "row_statistics": ["mean", "std", "min", "max"],
    "row_statistics_include_null_count": true,
    
    "outlier_detection_enabled": false,  // Enable for fraud, anomaly detection
    "outlier_detection_methods": ["iqr"],  // Or: ["zscore", "isolation_forest"]
    "outlier_detection_threshold": 1.5,  // IQR multiplier
    
    "percentile_ranking_enabled": false,  // Normalized ranks within columns
    
    // ═══ AGGREGATION & GROUPBY (NEW!) ═══
    // Hierarchical data & time-series patterns
    "groupby_enabled": false,  // Enable for customer/store-level aggregations
    // "groupby_group_cols": ["customer_id"],  // Required if enabled
    // "groupby_agg_functions": ["mean", "sum", "std", "count"],
    
    "rolling_enabled": false,  // Enable for time-series data
    // "rolling_windows": [3, 7, 14],  // Window sizes
    // "rolling_functions": ["mean", "std"],
    
    "lag_enabled": false,  // Enable for forecasting tasks
    // "lag_periods": [1, 7, 28],  // Lag periods
    
    // ═══ DOMAIN-SPECIFIC (NEW!) ═══
    // Industry knowledge encoding
    "domain_features_enabled": false,  // Enable ONLY when dataset clearly matches domain
    // "domain_type": "finance",  // Or: "ecommerce", "healthcare", "geospatial"
    
    // FINANCE (if domain_type="finance"):
    // "finance_technical_indicators": true,
    // "finance_indicators": ["rsi", "macd", "bollinger"],
    // "finance_price_col": "close",
    
    // E-COMMERCE (if domain_type="ecommerce"):
    // "ecommerce_rfm": true,
    // "ecommerce_rfm_customer_col": "customer_id",
    // "ecommerce_rfm_date_col": "purchase_date",
    // "ecommerce_rfm_amount_col": "amount",
    
    // HEALTHCARE (if domain_type="healthcare"):
    // "healthcare_vital_ratios": true,
    // "healthcare_bmi_weight_col": "weight_kg",
    // "healthcare_bmi_height_col": "height_m",
    
    // GEOSPATIAL (if domain_type="geospatial"):
    // "geospatial_distance_features": true,
    // "geospatial_lat_col": "latitude",
    // "geospatial_lon_col": "longitude",
    
    // Add any other specific FeatureCraft parameters...
  }
}

═══════════════════════════════════════════════════════════════
📋 CRITICAL GUIDELINES
═══════════════════════════════════════════════════════════════

1. **Recommend ONLY FeatureCraft parameters that exist** (see capabilities above)
2. **Consider dataset size vs computational cost**:
   - Small (<1K rows): Conservative, no interactions
   - Medium (1K-100K): Selective interactions, moderate text features
   - Large (>100K): Can afford more, but still selective
3. **Match techniques to estimator family**:
   - Trees: No scaling, arithmetic interactions, avoid polynomial
   - Linear/SVM: Scaling essential, polynomial can help
   - KNN/NN: Scaling essential, distance-based considerations
4. **Text feature cost tiers**:
   - Lightweight: statistics, sentiment (always OK)
   - Moderate: TF-IDF with SVD (OK for most datasets)
   - Expensive: word embeddings, NER (>10K rows recommended)
   - Very expensive: sentence embeddings, topic modeling (>50K rows recommended)
5. **Prioritize techniques with highest ROI** for this specific dataset
6. **Include specific parameter values**, not just true/false
7. **Feature explosion prevention**:
   - Estimated features should be < 0.2 x n_rows (low risk)
   - Use mi_top_k for dimensionality control
   - Conservative interaction limits for small datasets
8. **Encoding strategy**:
   - Low cardinality (<10 unique): One-hot encoding (automatic)
   - Mid cardinality (10-50): Target encoding
   - High cardinality (>50): Target encoding + smoothing
   - Very high (>1000): Consider hashing or WoE
9. **Class imbalance** (<20% minority): Enable SMOTE or WoE
10. **Computational budget** (time_budget parameter):
    - "fast": Minimal interactions, basic text, no expensive features
    - "balanced": Selective interactions, moderate text features (default)
    - "thorough": More aggressive, enable advanced features if dataset size allows
11. **Clustering features** - Consider when:
    - Customer segmentation tasks (use K-Means or GMM)
    - Anomaly/fraud detection (use DBSCAN with outlier flags)
    - Medium-large datasets (>1K rows, >5 numeric features)
    - Patterns not obvious from raw features
12. **Statistical features** - Consider when:
    - Wide datasets (>10 numeric columns) → row statistics
    - Fraud/anomaly detection → outlier detection
    - Sensor/measurement data → row-wise aggregations
    - Data quality concerns → null counts, outlier flags
13. **Aggregation/GroupBy features** - Consider when:
    - Hierarchical data (customer transactions, store sales) → groupby stats
    - Time-series data → rolling windows, lag features
    - Customer behavior patterns → group-level aggregations
    - Requires proper grouping columns (customer_id, store_id, etc.)
14. **Domain-specific features** - Consider when:
    - Finance data (price, volume) → technical indicators (RSI, MACD)
    - E-commerce transactions → RFM analysis
    - Healthcare data (vitals, labs) → BMI, vital ratios
    - Geospatial data (lat/lon) → distance features
    - ONLY use when domain clearly matches and required columns exist

═══════════════════════════════════════════════════════════════

Respond with ONLY valid JSON, no markdown formatting or explanations outside the JSON structure."""
        
        return prompt
    
    def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API."""
        try:
            # Build kwargs for API call
            kwargs = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You are an expert ML feature engineering advisor with deep expertise in FeatureCraft. Always respond with valid JSON only, following the exact structure specified in the prompt."},
                    {"role": "user", "content": prompt}
                ]
            }

            # Only add temperature if it's not the default value (1.0) to avoid compatibility issues
            if self.temperature != 1.0:
                kwargs["temperature"] = self.temperature

            response = self.client.chat.completions.create(**kwargs)
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            raise
    
    def _call_anthropic(self, prompt: str) -> str:
        """Call Anthropic API."""
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=3000,  # Increased for detailed config_overrides
                temperature=self.temperature,
                system="You are an expert ML feature engineering advisor with deep expertise in FeatureCraft. Always respond with valid JSON only, following the exact structure specified in the prompt.",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.content[0].text.strip()
        except Exception as e:
            logger.error(f"Anthropic API call failed: {e}")
            raise
    
    def _call_local(self, prompt: str) -> str:
        """Call local LLM (Ollama, LM Studio, etc.)."""
        # Placeholder for local LLM integration
        # Users can implement this with their preferred local model
        raise NotImplementedError(
            "Local LLM support not yet implemented. "
            "Use provider='openai' or provider='anthropic', or implement custom local LLM calling."
        )
    
    def _parse_ai_response(self, response: str, X: pd.DataFrame, insights: DatasetInsights) -> FeatureStrategy:
        """Parse LLM JSON response into FeatureStrategy."""
        try:
            # Clean response (remove markdown if present)
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()
            
            # Parse JSON
            data = json.loads(response)
            
            # Create strategy from parsed data
            strategy = FeatureStrategy(
                use_interactions=data.get("use_interactions", False),
                interaction_types=data.get("interaction_types", []),
                max_interaction_features=data.get("max_interaction_features", 50),
                use_target_encoding=data.get("use_target_encoding", True),
                use_frequency_encoding=data.get("use_frequency_encoding", False),
                text_strategy=data.get("text_strategy", "basic"),
                apply_feature_selection=data.get("apply_feature_selection", False),
                target_n_features=data.get("target_n_features"),
                reasoning=data.get("reasoning", "AI-generated recommendation"),
                estimated_feature_count=data.get("estimated_feature_count", X.shape[1]),
                risk_level=data.get("risk_level", "low"),
                config_overrides=data.get("config_overrides", {}),
            )
            
            # Always enforce minimums for tfidf_max_features and svd_components_for_trees
            strategy.config_overrides["tfidf_max_features"] = max(100, strategy.config_overrides.get("tfidf_max_features", 1000))
            strategy.config_overrides["svd_components_for_trees"] = max(2, strategy.config_overrides.get("svd_components_for_trees", 200))

            return strategy
        
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse AI response as JSON: {e}")
            logger.debug(f"Response: {response}")
            raise ValueError(f"Invalid JSON response from AI: {e}")
    
    def apply_strategy(self, base_config: FeatureCraftConfig, strategy: FeatureStrategy) -> FeatureCraftConfig:
        """Apply recommended strategy to configuration.
        
        Args:
            base_config: Base configuration
            strategy: Recommended strategy
        
        Returns:
            Updated configuration with strategy applied
        """
        # Get current config as dict
        config_dict = base_config.model_dump()
        
        # Apply strategy overrides
        if strategy.config_overrides:
            config_dict.update(strategy.config_overrides)
        
        # CRITICAL FIX: Enforce consistency across related parameters
        # This prevents the bug where AI says "interactions_enabled: False" but
        # individual interaction methods remain enabled due to default values.
        
        # Fix 1: If interactions are disabled, force ALL interaction sub-parameters to False
        if "interactions_enabled" in config_dict and not config_dict["interactions_enabled"]:
            logger.debug("AI disabled interactions - forcing all interaction sub-parameters to False")
            config_dict["interactions_use_arithmetic"] = False
            config_dict["interactions_use_polynomial"] = False
            config_dict["interactions_use_ratios"] = False
            config_dict["interactions_use_products"] = False
            config_dict["interactions_use_categorical_numeric"] = False
            config_dict["interactions_use_binned"] = False
            # Clear limits too
            config_dict["interactions_max_arithmetic_pairs"] = 0
            config_dict["interactions_max_ratio_pairs"] = 0
            config_dict["interactions_max_products"] = 0
            config_dict["interactions_max_cat_num_pairs"] = 0
        
        # Fix 2: If interactions are enabled, ensure at least one method is enabled
        # (prevent AI from enabling interactions_enabled but disabling all methods)
        if "interactions_enabled" in config_dict and config_dict["interactions_enabled"]:
            has_any_method = (
                config_dict.get("interactions_use_arithmetic", False) or
                config_dict.get("interactions_use_polynomial", False) or
                config_dict.get("interactions_use_ratios", False) or
                config_dict.get("interactions_use_products", False) or
                config_dict.get("interactions_use_categorical_numeric", False) or
                config_dict.get("interactions_use_binned", False)
            )
            if not has_any_method:
                logger.warning(
                    "AI enabled interactions_enabled but no interaction methods are active. "
                    "Disabling interactions_enabled to prevent empty FeatureUnion."
                )
                config_dict["interactions_enabled"] = False
        
        # Fix 3: If feature selection is disabled, ensure all selection methods are disabled
        # Check multiple indicators that feature selection should be off
        apply_selection = strategy.apply_feature_selection
        if not apply_selection or strategy.target_n_features is None:
            logger.debug("AI disabled feature selection - forcing all selection parameters to False/None")
            config_dict["use_mi"] = False
            config_dict["mi_top_k"] = None
            config_dict["use_woe_selection"] = False
            # Note: Keep correlation/VIF thresholds as they are preprocessing, not selection
        
        # Fix 4: If feature selection IS enabled, ensure proper configuration
        if apply_selection and strategy.target_n_features:
            logger.debug(f"AI enabled feature selection - ensuring mi_top_k is set to {strategy.target_n_features}")
            if strategy.selection_method == "mutual_info":
                config_dict["use_mi"] = True
                config_dict["mi_top_k"] = strategy.target_n_features
            elif strategy.selection_method == "tree_importance":
                # We don't have tree_importance selection yet, fall back to MI
                logger.warning("tree_importance selection not implemented, using mutual_info instead")
                config_dict["use_mi"] = True
                config_dict["mi_top_k"] = strategy.target_n_features
        
        # Fix 5: Ensure text feature limits are reasonable (prevent AI from setting too low)
        if "tfidf_max_features" in config_dict:
            config_dict["tfidf_max_features"] = max(100, config_dict["tfidf_max_features"])
        if "svd_components_for_trees" in config_dict:
            config_dict["svd_components_for_trees"] = max(2, config_dict["svd_components_for_trees"])
        
        # Create new config
        new_config = FeatureCraftConfig(**config_dict)
        
        # Log final state for debugging
        logger.info(f"✓ Applied AI strategy to config:")
        logger.info(f"  • interactions_enabled: {new_config.interactions_enabled}")
        if new_config.interactions_enabled:
            logger.info(f"    - arithmetic: {new_config.interactions_use_arithmetic}")
            logger.info(f"    - polynomial: {new_config.interactions_use_polynomial}")
            logger.info(f"    - ratios: {new_config.interactions_use_ratios}")
            logger.info(f"    - cat_num: {new_config.interactions_use_categorical_numeric}")
        logger.info(f"  • feature_selection (use_mi): {new_config.use_mi}")
        logger.info(f"  • mi_top_k: {new_config.mi_top_k}")
        logger.info(f"  • target_encoding: {new_config.use_target_encoding}")
        
        return new_config
    
    def print_strategy(self, strategy: FeatureStrategy):
        """Print strategy in a beautiful format."""
        content = f"""[bold cyan]AI Feature Engineering Strategy[/bold cyan]

[yellow]📊 Estimated Output:[/yellow]
  • Features: {strategy.estimated_feature_count}
  • Training Time: {strategy.estimated_training_time}
  • Risk Level: [{self._risk_color(strategy.risk_level)}]{strategy.risk_level.upper()}[/{self._risk_color(strategy.risk_level)}]

[yellow]🔧 Core Techniques:[/yellow]
  • Interactions: {'✓ Enabled' if strategy.use_interactions else '✗ Disabled'}
"""
        if strategy.use_interactions:
            content += f"    Types: {', '.join(strategy.interaction_types)}\n"
            content += f"    Max Features: {strategy.max_interaction_features}\n"
        
        content += f"""  • Encoding: {'Target' if strategy.use_target_encoding else 'Frequency/Hashing'}
  • Feature Selection: {'✓ Enabled' if strategy.apply_feature_selection else '✗ Disabled'}
"""
        
        if strategy.apply_feature_selection:
            content += f"    Target Features: {strategy.target_n_features}\n"
        
        # New techniques
        advanced_techniques = []
        
        if strategy.use_clustering:
            methods = ', '.join(strategy.clustering_methods) if strategy.clustering_methods else 'kmeans'
            advanced_techniques.append(f"  • Clustering: ✓ Enabled ({methods}, n_clusters={strategy.clustering_n_clusters})")
        
        if strategy.use_row_statistics:
            stats = ', '.join(strategy.row_statistics_to_extract) if strategy.row_statistics_to_extract else 'mean, std, min, max'
            advanced_techniques.append(f"  • Row Statistics: ✓ Enabled ({stats})")
        
        if strategy.use_outlier_detection:
            advanced_techniques.append(f"  • Outlier Detection: ✓ Enabled")
        
        if strategy.use_percentile_ranking:
            advanced_techniques.append(f"  • Percentile Ranking: ✓ Enabled")
        
        if strategy.use_groupby_stats:
            cols = ', '.join(strategy.groupby_columns[:2]) if strategy.groupby_columns else 'auto'
            advanced_techniques.append(f"  • GroupBy Statistics: ✓ Enabled (columns: {cols})")
        
        if strategy.use_rolling_features:
            advanced_techniques.append(f"  • Rolling Window Features: ✓ Enabled")
        
        if strategy.use_lag_features:
            advanced_techniques.append(f"  • Lag Features: ✓ Enabled")
        
        if strategy.use_domain_features:
            domain_name = strategy.domain_type or 'auto'
            features = ', '.join(strategy.domain_features_to_extract[:3]) if strategy.domain_features_to_extract else 'auto'
            advanced_techniques.append(f"  • Domain Features: ✓ Enabled ({domain_name}: {features})")
        
        if advanced_techniques:
            content += f"\n[yellow]🚀 Advanced Techniques:[/yellow]\n" + "\n".join(advanced_techniques) + "\n"
        
        content += f"\n[yellow]💡 Reasoning:[/yellow]\n{strategy.reasoning}"
        
        console.print(Panel(content, border_style="cyan", expand=False))
    
    @staticmethod
    def _risk_color(risk: str) -> str:
        """Get color for risk level."""
        return {"low": "green", "medium": "yellow", "high": "red"}.get(risk, "white")

