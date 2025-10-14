# FeatureCraft Agent: Comprehensive Design Document

**Version:** 1.0.0  
**Date:** October 10, 2025  
**Authors:** Principal Data Scientist & AI Agent Architect Team  

---

## Executive Summary

The **FeatureCraft Agent** is an advanced AI-driven system that orchestrates end-to-end automated feature engineering for tabular ML tasks (classification, regression, time series). It leverages the existing FeatureCraft library as its core toolset, adding intelligent orchestration, iterative optimization, and rigorous evaluation to automatically discover optimal feature engineering pipelines.

### Key Value Propositions

- **Zero-Configuration Intelligence**: Analyzes datasets automatically and selects optimal feature engineering strategies without manual tuning
- **Iterative Optimization**: Employs multi-stage search (heuristic warm-start → structured search → Bayesian/evolutionary tuning) to progressively refine pipelines
- **Rigorous Evaluation**: Implements leak-safe CV strategies, ablation studies, and drift detection to ensure robust, production-ready pipelines
- **Full Transparency**: Generates human-readable reports with explanations for every decision, feature importance, and performance metrics
- **Minimal Code Surface**: Thin orchestration layer (< 2000 LOC) that composes existing FeatureCraft primitives rather than reimplementing capabilities
- **Budget-Aware**: Respects compute/time constraints with early stopping, incremental feature addition, and smart caching

### Core Components

1. **Inspector Module**: Dataset fingerprinting, quality checks, statistical profiling, leakage detection
2. **Strategist Module**: Heuristic-based strategy selection using dataset characteristics and task constraints
3. **Composer Module**: Converts strategy → FeatureCraft pipeline using existing transformers/encoders
4. **Evaluator Module**: Leak-safe CV scoring, baseline comparison, ablation analysis, SHAP/permutation importance
5. **Optimizer Module**: Multi-stage search (greedy/beam/Bayesian) to refine pipelines iteratively
6. **Reporter Module**: Markdown/JSON/HTML reports with visualizations, explanations, and reproducibility artifacts

### Integration with FeatureCraft

The agent builds on FeatureCraft's existing capabilities:
- **Reuses 95%+ existing code**: `AutoFeatureEngineer`, encoders, transformers, selectors, statistical features, clustering, text processing
- **Extends with orchestration**: Adds Inspector, Strategist, Evaluator, Optimizer as thin wrappers
- **Backward-compatible**: Existing FeatureCraft users can opt-in to agent mode without breaking changes
- **Production-ready**: Exports standard sklearn pipelines + metadata for seamless deployment

### Expected Impact

- **40-70% reduction in manual feature engineering effort** (measured in practitioner hours)
- **15-30% improvement over baseline pipelines** (CV score) on benchmark datasets
- **10x reduction in pipeline search time** vs. random/grid search over full config space
- **Zero leakage incidents** (validated by comprehensive leak checks and time-based CV)

---

## 1. Capability & API Mapping of FeatureCraft

### Capabilities Matrix

| **Feature Engineering Operation** | **FeatureCraft Class/Method** | **Source File** | **Agent Usage** |
|-----------------------------------|-------------------------------|-----------------|----------------|
| **Core Pipeline Orchestration** |
| Automatic feature engineering pipeline | `AutoFeatureEngineer` | `src/featurecraft/pipeline.py` | Main pipeline wrapper; Agent calls `fit/transform` |
| Configuration management | `FeatureCraftConfig` | `src/featurecraft/config.py` | Agent modifies config based on strategy |
| Dataset analysis & profiling | `analyze_dataset()` | `src/featurecraft/insights.py` | Inspector uses for fingerprinting |
| Task detection | `detect_task()` | `src/featurecraft/insights.py` | Inspector auto-detects classification/regression |
| Pipeline explanation | `PipelineExplainer` | `src/featurecraft/explainability.py` | Reporter integrates explanations |
| **Encoding Strategies** |
| One-hot encoding | `make_ohe()` | `src/featurecraft/encoders.py` | Low-cardinality categoricals |
| Target encoding (K-Fold) | `KFoldTargetEncoder` | `src/featurecraft/encoders.py` | Mid-cardinality categoricals, leak-safe |
| Target encoding (Out-of-Fold) | `OutOfFoldTargetEncoder` | `src/featurecraft/encoders.py` | Mid-cardinality, CV-aware |
| Target encoding (Leave-One-Out) | `LeaveOneOutTargetEncoder` | `src/featurecraft/encoders.py` | Alternative to K-Fold |
| Frequency encoding | `FrequencyEncoder` | `src/featurecraft/encoders.py` | High-cardinality categoricals |
| Count encoding | `CountEncoder` | `src/featurecraft/encoders.py` | High-cardinality categoricals |
| Weight of Evidence (WoE) | `WoEEncoder` | `src/featurecraft/encoders.py` | Binary classification |
| Hashing encoding | `HashingEncoder` | `src/featurecraft/encoders.py` | Ultra-high cardinality (>1000 levels) |
| Ordinal encoding | `OrdinalEncoder` | `src/featurecraft/encoders.py` | Ordered categories |
| Binary encoding | `BinaryEncoder` | `src/featurecraft/encoders.py` | Space-efficient encoding |
| CatBoost encoding | `CatBoostEncoder` | `src/featurecraft/encoders.py` | Ordered target statistics |
| Entity embeddings | `EntityEmbeddingsEncoder` | `src/featurecraft/encoders.py` | Neural network-based representations |
| Rare category grouping | `RareCategoryGrouper` | `src/featurecraft/encoders.py` | Combine infrequent levels |
| **Numeric Transformations** |
| Power transformation (Yeo-Johnson) | `SkewedPowerTransformer` | `src/featurecraft/transformers.py` | Skewed distributions |
| Mathematical transforms | `MathematicalTransformer` | `src/featurecraft/transformers.py` | Auto-select log/sqrt/box-cox/etc. |
| Numeric type conversion | `NumericConverter` | `src/featurecraft/transformers.py` | Mixed-type columns |
| Ensure numeric output | `EnsureNumericOutput` | `src/featurecraft/transformers.py` | Pipeline compatibility |
| Binning/discretization | `BinningTransformer`, `AutoBinningSelector` | `src/featurecraft/transformers.py` | Quantile/KMeans/decision-tree binning |
| **Scaling** |
| Standard scaling | `choose_scaler("standard")` | `src/featurecraft/scalers.py` | Linear models, SVM |
| Min-max scaling | `choose_scaler("minmax")` | `src/featurecraft/scalers.py` | Neural networks, k-NN |
| Robust scaling | `choose_scaler("robust")` | `src/featurecraft/scalers.py` | Outlier-heavy data |
| MaxAbs scaling | `choose_scaler("maxabs")` | `src/featurecraft/scalers.py` | Sparse data preservation |
| **Imputation** |
| Simple imputation | `choose_numeric_imputer("simple")` | `src/featurecraft/imputers.py` | Low missingness (<5%) |
| KNN imputation | `choose_numeric_imputer("knn")` | `src/featurecraft/imputers.py` | Moderate missingness (5-30%) |
| Iterative imputation | `choose_numeric_imputer("iterative")` | `src/featurecraft/imputers.py` | Complex missingness patterns |
| Categorical imputation | `categorical_imputer()` | `src/featurecraft/imputers.py` | Most frequent / constant strategy |
| **Feature Interactions** |
| Arithmetic interactions | `ArithmeticInteractions` | `src/featurecraft/interactions.py` | +, -, *, / operations |
| Polynomial features | `PolynomialInteractions` | `src/featurecraft/interactions.py` | Cross-products, powers |
| Ratio features | `RatioFeatures` | `src/featurecraft/interactions.py` | Division-based ratios |
| Product interactions | `ProductInteractions` | `src/featurecraft/interactions.py` | Multiplicative interactions |
| Binned interactions | `BinnedInteractions` | `src/featurecraft/interactions.py` | Categorical × numeric |
| Categorical-numeric interactions | `CategoricalNumericInteractions` | `src/featurecraft/interactions.py` | Grouped interactions |
| **Time Series Features** |
| Datetime extraction | `DateTimeFeatures` | `src/featurecraft/transformers.py` | Year, month, day, hour, dow, etc. |
| Lag features | `LagFeaturesTransformer` | `src/featurecraft/aggregations.py` | Temporal dependencies |
| Rolling window stats | `RollingWindowTransformer` | `src/featurecraft/aggregations.py` | Moving averages, std, min, max |
| Expanding window stats | `ExpandingWindowTransformer` | `src/featurecraft/aggregations.py` | Cumulative statistics |
| Group-by statistics | `GroupByStatsTransformer` | `src/featurecraft/aggregations.py` | Entity-level aggregations |
| Rank features | `RankFeaturesTransformer` | `src/featurecraft/aggregations.py` | Temporal ranking |
| **Statistical Features** |
| Row-wise statistics | `RowStatisticsTransformer` | `src/featurecraft/statistical.py` | Mean/std/min/max per row |
| Outlier detection | `OutlierDetector` | `src/featurecraft/statistical.py` | IQR/Z-score flags |
| Percentile ranking | `PercentileRankTransformer` | `src/featurecraft/statistical.py` | Within-column percentiles |
| Z-score transformation | `ZScoreTransformer` | `src/featurecraft/statistical.py` | Standardized scores |
| Quantile transformation | `QuantileTransformer` | `src/featurecraft/statistical.py` | Quantile-based features |
| Target-based features | `TargetBasedFeaturesTransformer` | `src/featurecraft/statistical.py` | Statistical relationships with target |
| Missing value patterns | `MissingValuePatternsTransformer` | `src/featurecraft/statistical.py` | Missingness flags/counts |
| **Clustering Features** |
| K-Means clustering | `ClusteringFeatureExtractor("kmeans")` | `src/featurecraft/clustering.py` | Cluster membership + distances |
| DBSCAN clustering | `ClusteringFeatureExtractor("dbscan")` | `src/featurecraft/clustering.py` | Density-based clustering |
| Gaussian Mixture | `ClusteringFeatureExtractor("gmm")` | `src/featurecraft/clustering.py` | Probabilistic soft assignments |
| Hierarchical clustering | `ClusteringFeatureExtractor("hierarchical")` | `src/featurecraft/clustering.py` | Tree-based clustering |
| Multi-method clustering | `MultiMethodClusteringExtractor` | `src/featurecraft/clustering.py` | Ensemble clustering |
| Adaptive clustering | `AdaptiveClusteringExtractor` | `src/featurecraft/clustering.py` | Auto-select method + params |
| **Text Processing** |
| Text statistics | `TextStatisticsExtractor` | `src/featurecraft/text.py` | Char/word counts, sentence stats |
| Sentiment analysis | `SentimentAnalyzer` | `src/featurecraft/text.py` | Polarity, subjectivity |
| Named entity recognition | `NERFeatureExtractor` | `src/featurecraft/text.py` | Extract entities (persons, orgs, locations) |
| Topic modeling | `TopicModelingFeatures` | `src/featurecraft/text.py` | LDA topic distributions |
| Readability scores | `ReadabilityScoreExtractor` | `src/featurecraft/text.py` | Flesch-Kincaid, SMOG, etc. |
| Text preprocessing | `TextPreprocessor` | `src/featurecraft/text.py` | Cleaning, normalization, tokenization |
| TF-IDF/hashing | `build_text_pipeline()` | `src/featurecraft/text.py` | Vectorization + SVD |
| **Feature Selection** |
| Mutual information | `MutualInfoSelector` | `src/featurecraft/selection.py` | Non-linear relationships |
| Chi-square selection | `Chi2Selector` | `src/featurecraft/selection.py` | Categorical targets |
| Lasso selection | `LassoSelector` | `src/featurecraft/selection.py` | L1-regularized selection |
| Recursive feature elimination | `RFESelector` | `src/featurecraft/selection.py` | RFE with estimator |
| Sequential feature selection | `SequentialFeatureSelector` | `src/featurecraft/selection.py` | Forward/backward selection |
| Tree-based importance | `TreeImportanceSelector` | `src/featurecraft/selection.py` | Feature importance from trees |
| Boruta algorithm | `BorutaSelector` | `src/featurecraft/selection.py` | Statistical significance testing |
| WoE/IV selection | `WOEIVSelector` | `src/featurecraft/selection.py` | Information Value thresholding |
| Correlation pruning | `prune_correlated()` | `src/featurecraft/selection.py` | Remove high-correlation features |
| VIF multicollinearity | `compute_vif_drop()` | `src/featurecraft/selection.py` | Variance Inflation Factor |
| **Dimensionality Reduction** |
| PCA/SVD/NMF/UMAP | `DimensionalityReducer` | `src/featurecraft/reducers.py` | Linear/non-linear reduction |
| Multi-method reduction | `MultiMethodDimensionalityReducer` | `src/featurecraft/reducers.py` | Ensemble reduction |
| Adaptive reduction | `AdaptiveDimensionalityReducer` | `src/featurecraft/reducers.py` | Auto-select method |
| **Domain-Specific Features** |
| Finance technical indicators | `FinanceTechnicalIndicators` | `src/featurecraft/domain.py` | RSI, MACD, Bollinger Bands |
| Finance risk ratios | `FinanceRiskRatios` | `src/featurecraft/domain.py` | Sharpe, Sortino, max drawdown |
| E-commerce RFM | `EcommerceRFM` | `src/featurecraft/domain.py` | Recency, Frequency, Monetary |
| Healthcare vitals | `HealthcareVitals` | `src/featurecraft/domain.py` | BMI, BP ratios, HR variability |
| Geospatial features | `GeospatialFeatures` | `src/featurecraft/domain.py` | Haversine distance, POI proximity |
| **AI-Powered Intelligence** |
| AI feature advisor | `AIFeatureAdvisor` | `src/featurecraft/ai/advisor.py` | LLM-driven strategy recommendations |
| Feature engineering planner | `FeatureEngineeringPlanner` | `src/featurecraft/ai/planner.py` | Orchestrated planning |
| Adaptive optimizer | `AdaptiveConfigOptimizer` | `src/featurecraft/ai/optimizer.py` | Learning from feedback |
| **Data Quality & Validation** |
| Schema validation | `SchemaValidator` | `src/featurecraft/validation/schema_validator.py` | Input validation |
| Drift detection | Module functions | `src/featurecraft/drift.py` | PSI, KS tests |
| Leakage guards | `LeakageGuard` utilities | `src/featurecraft/utils/leakage.py` | Prevent target leakage |
| **Imbalance Handling** |
| SMOTE oversampling | Config flag `use_smote` | `src/featurecraft/imbalance.py` | Synthetic minority oversampling |
| **Explainability** |
| SHAP integration | Utility functions | `src/featurecraft/shap_utils.py` | SHAP value computation |
| Pipeline explanations | `PipelineExplanation` dataclass | `src/featurecraft/explainability.py` | Human-readable decisions |
| **Reporting & Visualization** |
| HTML report builder | `ReportBuilder` | `src/featurecraft/report.py` | Interactive HTML reports |
| Plotting utilities | Various functions | `src/featurecraft/plots.py` | Distributions, correlations, etc. |

### Identified Gaps & Proposed Extensions

| **Gap** | **Proposed Solution** | **Backward-Compatible?** |
|---------|----------------------|-------------------------|
| **Automated CV strategy selection** | Add `CVStrategySelector` to choose stratified/group/time-based splits based on data characteristics | ✅ Yes (new utility) |
| **Pipeline-level ablation testing** | Add `AblationEvaluator` to measure per-operation impact | ✅ Yes (new evaluator) |
| **Multi-objective optimization** | Extend `AdaptiveConfigOptimizer` to balance score vs. complexity vs. speed | ✅ Yes (optional mode) |
| **Leakage risk scoring** | Add `LeakageRiskScorer` to quantify leakage risk per feature/operation | ✅ Yes (new validator) |
| **Feature cost profiling** | Add `FeatureCostProfiler` to track compute time per feature family | ✅ Yes (new profiler) |
| **Pipeline versioning** | Add `PipelineVersionTracker` for experiment lineage | ✅ Yes (new utility) |
| **Iterative feature addition** | Add `IncrementalPipelineBuilder` for greedy forward/backward search | ✅ Yes (new composer) |
| **Ensemble pipeline voting** | Add `PipelineEnsembler` to combine top-k pipelines | ✅ Yes (new meta-learner) |

All proposed extensions are **backward-compatible** and implemented as optional utilities that complement existing FeatureCraft functionality without breaking changes.

---

## 2. Dataset Fingerprint Plan

The **Inspector Module** computes a comprehensive dataset fingerprint to inform feature engineering strategy selection. This fingerprint is computed once at the start of the agent workflow and cached for subsequent operations.

### Fingerprint Schema

```python
@dataclass
class DatasetFingerprint:
    """Comprehensive dataset characteristics for strategy selection."""
    
    # === Basic Statistics ===
    n_rows: int
    n_cols: int
    n_features_original: int  # Excludes target
    target_name: str
    task_type: TaskType  # CLASSIFICATION, REGRESSION, TIME_SERIES
    
    # === Column Type Counts ===
    n_numeric: int
    n_categorical: int
    n_text: int
    n_datetime: int
    n_id_like: int  # High-cardinality unique (IDs, hashes)
    n_constant: int  # Constant or near-constant columns
    n_duplicate: int  # Duplicate columns
    
    # === Data Quality Indicators ===
    missing_summary: Dict[str, float]  # col → missing_rate
    high_missing_cols: List[str]  # missing_rate > 0.3
    constant_cols: List[str]  # nunique <= 1 or variance < 1e-8
    duplicate_col_groups: List[List[str]]  # Groups of identical columns
    
    # === Categorical Analysis ===
    cardinality_summary: Dict[str, int]  # col → nunique
    low_cardinality_cols: List[str]  # nunique <= 10
    mid_cardinality_cols: List[str]  # 10 < nunique <= 50
    high_cardinality_cols: List[str]  # 50 < nunique <= 1000
    ultra_high_cardinality_cols: List[str]  # nunique > 1000
    rare_category_ratios: Dict[str, float]  # col → fraction of categories with freq < 1%
    
    # === Numeric Distribution Analysis ===
    skewness_summary: Dict[str, float]  # col → skewness
    kurtosis_summary: Dict[str, float]  # col → kurtosis
    heavily_skewed_cols: List[str]  # |skewness| > 1.5
    outlier_share_summary: Dict[str, float]  # col → fraction beyond 1.5*IQR
    high_outlier_cols: List[str]  # outlier_share > 0.05
    near_zero_variance_cols: List[str]  # variance < 1e-4
    
    # === Target Analysis ===
    target_dtype: str
    target_nunique: int
    target_missing_rate: float
    class_balance: Optional[Dict[str, float]]  # For classification
    minority_class_ratio: Optional[float]  # For classification
    is_imbalanced: bool  # minority_class_ratio < 0.2
    target_skewness: Optional[float]  # For regression
    
    # === Correlation & Multicollinearity ===
    high_correlation_pairs: List[Tuple[str, str, float]]  # (col1, col2, corr) where |corr| > 0.95
    multicollinear_groups: List[List[str]]  # Groups with VIF > 10
    target_correlation_top: List[Tuple[str, float]]  # (col, corr) top-10 by |corr| with target
    
    # === Mutual Information (Task-Aware) ===
    mutual_info_scores: Dict[str, float]  # col → MI score with target
    low_mi_cols: List[str]  # MI score < 0.01 (weak predictive signal)
    
    # === Time Series Diagnostics (if applicable) ===
    time_column: Optional[str]
    is_time_series: bool
    time_granularity: Optional[str]  # 'second', 'minute', 'hour', 'day', 'week', 'month'
    time_gaps: Optional[Dict[str, Any]]  # Mean/median/max gap, missing timestamps
    has_regular_intervals: bool
    
    # === Entity/Group Diagnostics (for GroupKFold) ===
    entity_columns: List[str]  # Suspected entity/group columns (moderate cardinality, repeated IDs)
    entity_imbalance: Optional[Dict[str, float]]  # col → gini coefficient of group sizes
    
    # === Text Column Diagnostics ===
    text_length_stats: Dict[str, Dict[str, float]]  # col → {mean_len, std_len, max_len}
    text_vocab_size: Dict[str, int]  # col → unique token count
    
    # === Leakage Risk Signals ===
    leakage_risk_cols: List[str]  # Perfect correlation with target, future info
    id_like_with_high_mi: List[str]  # High-cardinality cols with suspiciously high MI
    
    # === Computational Complexity Estimates ===
    estimated_feature_count_baseline: int  # Expected features from default config
    estimated_memory_gb: float  # Rough memory footprint estimate
    
    # === Metadata ===
    fingerprint_timestamp: str
    featurecraft_version: str
```

### Fingerprint Computation Steps

1. **Basic profiling** (fast):
   - Row/col counts, dtypes
   - Column type detection (numeric/categorical/text/datetime/ID-like)
   - Missing value rates per column
   
2. **Quality checks** (fast):
   - Constant/near-constant columns (variance < 1e-8)
   - Duplicate columns (pairwise equality)
   - High missingness (> 30%)
   
3. **Statistical summaries** (moderate):
   - Skewness, kurtosis for numeric columns
   - Outlier share (IQR method)
   - Cardinality for categorical columns
   - Rare category ratios
   
4. **Target analysis** (moderate):
   - Task detection (classification vs. regression)
   - Class balance (classification)
   - Target skewness (regression)
   - Imbalance detection
   
5. **Correlation analysis** (moderate to slow):
   - Pairwise correlations (Pearson for numeric)
   - Spearman rank correlation (for ordinal/monotonic relationships)
   - High-correlation pairs (|corr| > 0.95)
   - Top-10 target correlations
   
6. **Mutual information** (slow):
   - MI scores for all features vs. target (task-aware: classification vs. regression)
   - Identify low-MI features (< 0.01)
   
7. **Multicollinearity** (slow):
   - VIF computation for numeric features (sample if > 50 features)
   - Identify multicollinear groups (VIF > 10)
   
8. **Time series diagnostics** (if datetime columns present):
   - Detect time column (monotonic, high cardinality)
   - Compute time granularity (median gap)
   - Check for gaps/irregular intervals
   
9. **Entity/group diagnostics**:
   - Detect entity columns (moderate cardinality, repeated values)
   - Compute group size distribution (for GroupKFold recommendation)
   
10. **Leakage risk detection**:
    - Perfect correlation with target (|corr| > 0.99)
    - High-cardinality cols with suspiciously high MI (potential ID leakage)
    - Datetime columns that might contain future information

### Task Detection & Metric Selection Logic

```python
def detect_task_and_metrics(
    y: pd.Series,
    time_column: Optional[str],
    config: FeatureCraftConfig
) -> Tuple[TaskType, List[str]]:
    """Detect task type and select appropriate metrics."""
    
    # Time series detection
    if time_column is not None:
        return TaskType.TIME_SERIES, ["RMSE", "MAE", "MAPE", "sMAPE"]
    
    # Classification vs. Regression
    if pd.api.types.is_numeric_dtype(y):
        nunique = y.nunique()
        if nunique <= 15:
            # Classification
            if nunique == 2:
                # Binary classification
                return TaskType.CLASSIFICATION, ["LogLoss", "AUC-ROC", "F1", "Precision", "Recall"]
            else:
                # Multi-class classification
                return TaskType.CLASSIFICATION, ["LogLoss", "Accuracy", "F1-Macro", "F1-Weighted"]
        else:
            # Regression
            return TaskType.REGRESSION, ["RMSE", "MAE", "R2", "MAPE"]
    else:
        # Non-numeric target → classification
        nunique = y.nunique()
        if nunique == 2:
            return TaskType.CLASSIFICATION, ["LogLoss", "AUC-ROC", "F1"]
        else:
            return TaskType.CLASSIFICATION, ["LogLoss", "Accuracy", "F1-Macro"]
```

### Safe Defaults for Missing Information

| **Missing Information** | **Safe Default** | **Rationale** |
|------------------------|------------------|---------------|
| Task type not auto-detectable | `CLASSIFICATION` | More common in tabular ML; safer to assume discrete target |
| Primary metric not specified | Binary: `LogLoss` / Multi-class: `LogLoss` / Regression: `RMSE` | Standard, differentiable, widely used |
| Time column not provided | Check for monotonic datetime with high cardinality | Heuristic auto-detection |
| Entity/group column not provided | Use `StratifiedKFold` or `KFold` | Standard CV without grouping |
| Compute budget not specified | `"balanced"` (15 min wall time) | Reasonable default for most tasks |
| Memory limit not specified | Estimate from dataset size × 10 | Conservative 10x overhead |

---

## 3. Feature Operations Taxonomy & Heuristics

The **Strategist Module** uses if-then rules tied to the dataset fingerprint to select feature engineering operations. Below is the taxonomy organized by feature family.

### 3.1 Numeric Features

| **Operation** | **FeatureCraft Method** | **Heuristic Trigger** | **Parameters** |
|--------------|------------------------|----------------------|---------------|
| **Imputation: Mean/Median** | `choose_numeric_imputer("simple")` | `missing_rate <= 0.05` | Strategy: `"median"` (robust) |
| **Imputation: KNN** | `choose_numeric_imputer("knn")` | `0.05 < missing_rate <= 0.20` | `n_neighbors=5` |
| **Imputation: Iterative** | `choose_numeric_imputer("iterative")` | `0.20 < missing_rate <= 0.30` | Max iter: `10` |
| **Drop column** | Remove from pipeline | `missing_rate > 0.30 AND low_mi` | N/A |
| **Missing indicator** | `add_missing_indicators=True` | `missing_rate > 0.05` | Binary flag column |
| **Standard scaling** | `choose_scaler("standard")` | `estimator_family="linear" OR "svm"` | Mean=0, std=1 |
| **Min-max scaling** | `choose_scaler("minmax")` | `estimator_family="knn" OR "nn"` | Range [0, 1] |
| **Robust scaling** | `choose_scaler("robust")` | `outlier_share > 0.05` | Median + IQR |
| **No scaling** | `choose_scaler("none")` | `estimator_family="tree"` | Trees don't need scaling |
| **Yeo-Johnson transform** | `SkewedPowerTransformer` | `|skewness| > 1.5 AND task=REGRESSION` | Auto-fit lambda |
| **Log transform** | `MathematicalTransformer("log")` | `skewness > 2 AND all_positive` | log(x + 1e-5) |
| **Winsorization** | `config.winsorize=True` | `outlier_share > 0.10` | Clip at [1%, 99%] percentiles |
| **Quantile binning** | `BinningTransformer("quantile")` | `estimator_family="linear" AND n_bins<=10` | Equal-frequency bins |
| **KMeans binning** | `BinningTransformer("kmeans")` | `Heavy multimodal distribution` | Data-driven boundaries |
| **Polynomial interactions** | `PolynomialInteractions(degree=2)` | `estimator_family="linear" AND n_features<=20` | Degree=2 only (avoid explosion) |
| **Arithmetic interactions** | `ArithmeticInteractions(['+', '-', '*', '/'])` | `estimator_family="tree" AND n_features<=30` | Top-k by MI pairs |
| **Ratio features** | `RatioFeatures` | `Domain=finance OR n_features<=15` | Pairwise divisions |

### 3.2 Categorical Features

| **Operation** | **FeatureCraft Method** | **Heuristic Trigger** | **Parameters** |
|--------------|------------------------|----------------------|---------------|
| **Most-frequent imputation** | `categorical_imputer("most_frequent")` | Always (safe default) | N/A |
| **Rare category grouping** | `RareCategoryGrouper` | `rare_category_ratio > 0.05` | Threshold: 1% |
| **One-hot encoding** | `make_ohe()` | `cardinality <= low_cardinality_max (10)` | Handle unknown: `"infrequent_if_exist"` |
| **Target encoding (K-Fold)** | `KFoldTargetEncoder` | `10 < cardinality <= 50 AND task=CLASSIFICATION` | K=5, smoothing=0.3, noise=0.01 |
| **Target encoding (OOF)** | `OutOfFoldTargetEncoder` | `10 < cardinality <= 50 AND task=REGRESSION` | Fold-aware encoding |
| **Frequency encoding** | `FrequencyEncoder` | `cardinality > 50 OR estimator_family="tree"` | Category → frequency |
| **Count encoding** | `CountEncoder` | `cardinality > 50 AND task=TIME_SERIES` | Category → count |
| **WoE encoding** | `WoEEncoder` | `task=CLASSIFICATION (binary) AND cardinality>10` | Weight of Evidence |
| **Hashing encoding** | `HashingEncoder` | `cardinality > 1000` | n_components=256 |
| **Ordinal encoding** | `OrdinalEncoder` | `ordinal_maps provided in config` | User-specified ordering |

**Leakage-Safe Encoding for CV:**
- Always use **out-of-fold** statistics for target encoding (K-Fold or Leave-One-Out)
- Forbid target encoders in time series without explicit train/test split
- Fit encoders only on training fold, transform on validation fold

### 3.3 Datetime & Time Series

| **Operation** | **FeatureCraft Method** | **Heuristic Trigger** | **Parameters** |
|--------------|------------------------|----------------------|---------------|
| **Extract calendar features** | `DateTimeFeatures(extract=["year", "month", ...])` | Always for datetime columns | Year, month, day, dow, hour, minute |
| **Cyclical encoding** | `DateTimeFeatures(cyclical=True)` | Always for datetime columns | Sin/cos for month, dow, hour |
| **Seasonality** | `DateTimeFeatures(extract=["season"])` | Always | Winter/Spring/Summer/Fall |
| **Business logic** | `DateTimeFeatures(extract=["is_weekend", ...])` | Always | is_weekend, is_month_start, is_quarter_end |
| **Lag features** | `LagFeaturesTransformer(lags=[1,2,3,7])` | `task=TIME_SERIES` | Lags: 1, 2, 3, 7, 14 (depends on granularity) |
| **Rolling mean** | `RollingWindowTransformer(window=7, stats=["mean"])` | `task=TIME_SERIES` | Window: 7, 14, 30 |
| **Rolling std** | `RollingWindowTransformer(window=7, stats=["std"])` | `task=TIME_SERIES` | Volatility signal |
| **Expanding mean** | `ExpandingWindowTransformer(stats=["mean"])` | `task=TIME_SERIES` | Cumulative average |
| **Recency/age features** | `(current_date - datetime_col).days` | Always | Days since event |
| **Event windows** | Manual feature: `is_within_X_days_of_event` | Domain-specific (holidays, campaigns) | User-provided event dates |
| **Groupwise lags** | `LagFeaturesTransformer(group_by=entity_col)` | `entity_columns detected` | Per-entity lags |
| **Groupwise rolling** | `RollingWindowTransformer(group_by=entity_col)` | `entity_columns detected` | Per-entity rolling stats |

**Leakage-Safe CV for Time Series:**
- **Enforce** `TimeSeriesSplit` with gap/embargo (no shuffle, no random splits)
- **Forbid** target encoders that peek into future (use only past statistics)
- **Forbid** rolling/lag features that look forward
- **Add embargo period** (e.g., 1 week gap between train/test) to prevent information leakage

### 3.4 Groupby & Relational Features

| **Operation** | **FeatureCraft Method** | **Heuristic Trigger** | **Parameters** |
|--------------|------------------------|----------------------|---------------|
| **Groupby aggregations** | `GroupByStatsTransformer(group_by=entity_col)` | `entity_columns detected AND n_rows>1000` | Stats: count, nunique, mean, median, std, sum |
| **Ratios within group** | Manual: `feature / group_mean(feature)` | `entity_columns detected` | Deviation from group average |
| **Rank within group** | `RankFeaturesTransformer(group_by=entity_col)` | `entity_columns detected` | Temporal ranking within entity |

**Leakage Guards:**
- Fit group statistics **only on training data**, transform on train+test
- For CV, compute group stats **per fold** (train-only statistics)
- Never use future information (e.g., future group statistics in time series)

### 3.5 Text Features

| **Operation** | **FeatureCraft Method** | **Heuristic Trigger** | **Parameters** |
|--------------|------------------------|----------------------|---------------|
| **Text statistics** | `TextStatisticsExtractor` | Always for text columns | char_count, word_count, avg_word_len, sentence_count |
| **Sentiment analysis** | `SentimentAnalyzer` | `n_text_cols <= 3 AND n_rows <= 100k` | Polarity, subjectivity (TextBlob/VADER) |
| **NER features** | `NERFeatureExtractor` | `n_text_cols <= 2 AND n_rows <= 50k` | Entity counts (person, org, location) |
| **Readability scores** | `ReadabilityScoreExtractor` | `n_text_cols <= 3` | Flesch-Kincaid, SMOG, etc. |
| **TF-IDF vectorization** | `build_text_pipeline(use_tfidf=True)` | `n_rows <= 100k` | Max features: 5000, ngrams: (1,2), SVD: 200 |
| **Hashing vectorization** | `build_text_pipeline(use_hashing=True)` | `n_rows > 100k OR vocab_size > 50k` | n_features: 4096, ngrams: (1,2) |
| **Simple embeddings** | Config option (future) | `budget="thorough" AND n_rows <= 50k` | Pretrained word2vec/GloVe, avg pooling |

**Budget-Aware Text Processing:**
- Skip NER/sentiment if `n_rows > 100k` (too slow)
- Use hashing instead of TF-IDF for large vocabularies
- Limit SVD components for tree models (200) vs. linear models (500)

### 3.6 Redundancy Control

| **Operation** | **FeatureCraft Method** | **Heuristic Trigger** | **Parameters** |
|--------------|------------------------|----------------------|---------------|
| **Correlation pruning** | `prune_correlated(threshold=0.95)` | `n_features > 50` | Drop one from pairs with |corr| > 0.95 |
| **VIF pruning** | `compute_vif_drop(threshold=10)` | `n_features > 20 AND estimator_family="linear"` | Drop features with VIF > 10 |
| **PCA** | `DimensionalityReducer("pca")` | `n_features > 100 AND budget="fast"` | Retain 95% variance |
| **UMAP** | `DimensionalityReducer("umap")` | `n_features > 100 AND budget="thorough"` | n_components: 20 |
| **Permutation importance pruning** | After first CV pass | `n_features > 50` | Drop bottom 20% by importance |

### 3.7 Heuristic Trigger Rules Summary

```python
# Pseudocode for heuristic triggers
if fingerprint.missing_rate > 0.40:
    # Compare: impute vs. drop vs. missing indicators
    strategy.operations.append("compare_imputation_strategies")

if any(col in fingerprint.high_cardinality_cols for col in categoricals):
    if task == "CLASSIFICATION":
        strategy.encoding = "target_kfold_cv"  # Leak-safe with CV
    else:
        strategy.encoding = "frequency"

if fingerprint.heavily_skewed_cols and task == "REGRESSION":
    strategy.transforms.append("yeo_johnson")

if fingerprint.is_time_series:
    strategy.cv_strategy = "TimeSeriesSplit"
    strategy.cv_embargo_days = 7
    strategy.forbid_operations = ["target_encoding_lookahead"]

if fingerprint.n_features <= 20 and estimator_family == "linear":
    strategy.interactions.append("polynomial_degree2")

if fingerprint.outlier_share > 0.05:
    strategy.scaling = "robust"
else:
    strategy.scaling = choose_scaler_by_estimator(estimator_family)
```

---

## 4. Evaluation Framework & Guardrails

The **Evaluator Module** ensures robust, leak-free evaluation of candidate pipelines.

### 4.1 Baselines

1. **Raw Baseline**: Minimal preprocessing (impute median/mode, one-hot encode, no scaling)
2. **Auto Baseline**: Default FeatureCraft pipeline with standard config
3. **Custom Baseline** (optional): User-provided pipeline

All candidate pipelines are scored relative to baselines.

### 4.2 Models for Scoring Pipelines

**Fast, Strong Learners:**
- **LightGBM**: Fast, handles missing values, tree-based
- **XGBoost**: Strong performance, GPU support
- **CatBoost**: Handles categoricals natively (for comparison)

**Interpretability Baseline (optional):**
- **Logistic Regression** (classification) or **Ridge** (regression) with standard preprocessing

**Configuration:**
- Small hyperparameter budget (10-20 trials with early stopping)
- Fixed random seeds for reproducibility
- Use default hyperparameters first, tune only if time permits

### 4.3 CV Strategy Selection

```python
def select_cv_strategy(
    fingerprint: DatasetFingerprint,
    entity_column: Optional[str],
    config: FeatureCraftConfig
) -> BaseCrossValidator:
    """Select appropriate CV strategy based on data characteristics."""
    
    if fingerprint.is_time_series:
        # Time series: strictly chronological splits
        return TimeSeriesSplit(
            n_splits=5,
            test_size=len(y) // 10,  # 10% test per fold
            gap=fingerprint.time_gaps.get("median_gap_days", 7)  # Embargo
        )
    
    elif entity_column is not None:
        # Entity/group structure: GroupKFold
        return GroupKFold(n_splits=5)
    
    elif fingerprint.task_type == TaskType.CLASSIFICATION:
        # Classification: stratified to preserve class balance
        return StratifiedKFold(n_splits=5, shuffle=True, random_state=config.random_state)
    
    else:
        # Regression: standard K-Fold
        return KFold(n_splits=5, shuffle=True, random_state=config.random_state)
```

### 4.4 Metrics

**Primary Metric (optimized):**
- Binary classification: `LogLoss` (differentiable, calibrated)
- Multi-class classification: `LogLoss`
- Regression: `RMSE` (common, penalizes large errors)
- Time series: `MAPE` or `sMAPE` (percentage-based, interpretable)

**Secondary Metrics (monitored):**
- Binary: `AUC-ROC`, `F1`, `Precision`, `Recall`
- Multi-class: `Accuracy`, `F1-Macro`, `F1-Weighted`
- Regression: `MAE`, `R²`, `MAPE`

**Calibration (for classification):**
- Optionally apply `CalibratedClassifierCV` after pipeline fitting
- Evaluate calibration curve (Brier score, reliability diagram)

### 4.5 Ablation Studies

**Per-Operation Ablation:**
- Fit pipeline with operation X **removed**
- Compare CV score vs. full pipeline
- Quantify contribution: `Δscore = score_full - score_without_X`

**Per-Feature-Family Ablation:**
- Remove entire feature families (e.g., all interaction features, all text features)
- Measure impact on performance

**Permutation Importance:**
- After fitting, permute each feature and measure score drop
- Rank features by importance

**SHAP (budget permitting):**
- Compute SHAP values for top-k features on validation set
- Visualize feature contributions

### 4.6 Early Stopping & Budget Control

```python
@dataclass
class ComputeBudget:
    """Resource limits for agent execution."""
    max_wall_time_minutes: int = 60  # Total time budget
    max_pipelines: int = 50  # Max number of pipelines to evaluate
    max_fit_time_seconds: int = 300  # Per-pipeline fit time limit
    max_memory_gb: float = 16.0  # Memory limit
    early_stop_patience: int = 10  # Stop if no improvement in N pipelines
```

**Budget Controls:**
1. **Time limits**: Use `joblib.Parallel(timeout=...)` for per-fit limits
2. **Early stopping**: Track best score; stop if no improvement in `patience` iterations
3. **Memory checks**: Monitor RSS; skip expensive operations if approaching limit
4. **Fail-fast**: Skip unstable configurations (e.g., iterative imputer diverging)

### 4.7 Leakage Checks

**Automated Leakage Detection:**
1. **Train/Test Distribution Drift**:
   - Compute PSI (Population Stability Index) for each feature
   - Compute KS (Kolmogorov-Smirnov) statistic
   - Flag features with PSI > 0.25 or KS > 0.10 (possible leakage)

2. **Time Leakage Scans** (for time series):
   - Check that all lags/rolling features use only past data
   - Verify no future information in train set

3. **Target Encoders with CV-Only Statistics**:
   - Ensure target encoding uses out-of-fold statistics
   - Never fit on full train set and transform on test set

4. **Feature Creation Windows**:
   - For rolling/expanding features, verify they don't peek forward

**Leakage Risk Score:**
```python
def compute_leakage_risk_score(pipeline, X_train, X_test, y_train) -> float:
    """Compute a leakage risk score in [0, 1]."""
    risk = 0.0
    
    # Check 1: Perfect feature-target correlation
    for col in X_train.columns:
        if abs(X_train[col].corr(y_train)) > 0.99:
            risk += 0.5
    
    # Check 2: Train-test drift
    psi_scores = compute_psi(X_train, X_test)
    if any(psi > 0.25 for psi in psi_scores.values()):
        risk += 0.3
    
    # Check 3: ID-like features with high MI
    # (implementation details omitted)
    
    return min(risk, 1.0)
```

---

## 5. Search & Optimization Strategy

The **Optimizer Module** employs a multi-stage search to progressively refine pipelines.

### Stage 1: Heuristic Warm Start

**Objective:** Build 3-6 reasonable candidate pipelines using fingerprint heuristics.

**Method:**
1. Use `Strategist` to generate initial strategy from fingerprint
2. Create pipeline configurations with variations:
   - **Conservative**: Minimal feature engineering, fast
   - **Balanced** (default): Moderate feature engineering
   - **Aggressive**: Full feature engineering (interactions, clustering, etc.)
   - **Estimator-specific**: Tailored for tree/linear/svm/knn/nn

**Output:** Top-3 pipelines by CV score → pass to Stage 2

### Stage 2: Structured Search

**Objective:** Greedily improve top-k pipelines by adding/removing operations.

**Method:**
- **Greedy Forward Selection**:
  1. Start with minimal pipeline (impute + encode + scale)
  2. Iteratively add one operation at a time (interactions, clustering, text features, etc.)
  3. Keep operation if CV score improves by > 0.5%
  4. Stop when no operation improves score or budget exhausted

- **Beam Search** (optional):
  1. Maintain beam of size k=3 (top-k pipelines)
  2. For each pipeline, expand with candidate operations
  3. Evaluate all expansions, keep top-k
  4. Repeat until convergence or budget exhausted

**Output:** Top-2 pipelines → pass to Stage 3

### Stage 3: Bayesian Optimization / Evolutionary (Optional)

**Objective:** Tune discrete choices (encoders, scalers) and key hyperparameters.

**Method:**
- Use **Bayesian Optimization** (via `Optuna` or `scikit-optimize`) over:
  - Encoder type (OHE vs. target vs. frequency)
  - Scaler type (standard vs. robust vs. minmax)
  - Interaction degree (1, 2, 3)
  - Feature selection top-k
  - Imputation strategy

- **Search space size**: Keep tractable (< 100 hyperparameter combinations)
- **Budget**: 20-50 trials max

**Alternative: Evolutionary Algorithm**:
- Represent pipeline as chromosome (list of operations + params)
- Mutation: Add/remove/replace operations
- Crossover: Combine two pipelines
- Selection: Tournament selection by CV score
- Terminate: After 50 generations or no improvement

**Output:** Top-1 pipeline → pass to Stage 4

### Stage 4: Pruning & Consolidation

**Objective:** Remove redundant features, re-fit with tighter CV.

**Method:**
1. **Correlation pruning**: Drop features with |corr| > 0.95
2. **Permutation importance pruning**: Drop bottom 20% by importance
3. **Re-fit top-2 pipelines** with 10-fold CV (higher confidence)
4. **Finalize best pipeline**

**Output:** Final pipeline + metadata

### Caching Strategy

**Cache Intermediate Artifacts:**
- **Fingerprint**: Compute once, reuse across all pipelines
- **Imputed data**: Reuse if imputation strategy unchanged
- **Encoded categoricals**: Reuse if encoder unchanged
- **Correlation matrix**: Compute once
- **Mutual information scores**: Compute once

**Persistence:**
- Save artifacts to disk (`artifacts/cache/`) with content hashing
- Reload on subsequent runs if dataset hash matches

---

## 6. Agent Architecture (Roles, Tools, Memory)

### 6.1 Modules & Responsibilities

```
FeatureCraftAgent
├── Inspector       # Dataset fingerprinting, quality checks, leakage detection
├── Strategist      # Strategy selection based on fingerprint + constraints
├── Composer        # Convert strategy → FeatureCraft pipeline configuration
├── Evaluator       # CV scoring, ablation, baselines, metrics tracking
├── Optimizer       # Multi-stage search (heuristic → greedy → Bayesian)
└── Reporter        # Generate reports, visualizations, artifacts
```

#### **Inspector Module**

**Responsibilities:**
- Compute dataset fingerprint (Section 2)
- Detect task type (classification/regression/time-series)
- Identify data quality issues (missing values, outliers, duplicates, constant columns)
- Compute leakage risk scores
- Cache fingerprint for reuse

**Key Methods:**
```python
class Inspector:
    def fingerprint(self, X: pd.DataFrame, y: pd.Series) -> DatasetFingerprint:
        """Compute comprehensive dataset fingerprint."""
        ...
    
    def detect_task(self, y: pd.Series) -> TaskType:
        """Auto-detect classification/regression/time-series."""
        ...
    
    def check_data_quality(self, X: pd.DataFrame) -> List[Issue]:
        """Identify data quality issues."""
        ...
    
    def estimate_leakage_risk(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Compute leakage risk score per feature."""
        ...
```

#### **Strategist Module**

**Responsibilities:**
- Map fingerprint → initial feature engineering strategy
- Apply heuristic rules (Section 3.7)
- Generate 3-6 candidate strategies (conservative/balanced/aggressive)
- Select CV strategy (stratified/group/time-based)

**Key Methods:**
```python
class Strategist:
    def recommend_strategy(
        self,
        fingerprint: DatasetFingerprint,
        config: FeatureCraftConfig,
        estimator_family: str
    ) -> FeatureStrategy:
        """Generate initial strategy from fingerprint."""
        ...
    
    def generate_variants(
        self,
        base_strategy: FeatureStrategy
    ) -> List[FeatureStrategy]:
        """Generate conservative/balanced/aggressive variants."""
        ...
    
    def select_cv_strategy(
        self,
        fingerprint: DatasetFingerprint
    ) -> BaseCrossValidator:
        """Choose appropriate CV strategy."""
        ...
```

#### **Composer Module**

**Responsibilities:**
- Convert `FeatureStrategy` → `FeatureCraftConfig` overrides
- Build sklearn `Pipeline` using FeatureCraft transformers
- Validate pipeline compatibility
- Export pipeline specification (JSON)

**Key Methods:**
```python
class Composer:
    def build_pipeline(
        self,
        strategy: FeatureStrategy,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Pipeline:
        """Convert strategy → FeatureCraft pipeline."""
        ...
    
    def strategy_to_config(
        self,
        strategy: FeatureStrategy,
        base_config: FeatureCraftConfig
    ) -> FeatureCraftConfig:
        """Convert strategy → config overrides."""
        ...
    
    def validate_pipeline(self, pipeline: Pipeline) -> bool:
        """Check pipeline compatibility."""
        ...
```

#### **Evaluator Module**

**Responsibilities:**
- Fit pipelines with CV scoring
- Track metrics (primary + secondaries)
- Compute baselines (raw, auto)
- Run ablation studies (per-operation, per-feature-family)
- Compute permutation/SHAP importance
- Detect leakage (train-test drift, time leakage)

**Key Methods:**
```python
class Evaluator:
    def evaluate_pipeline(
        self,
        pipeline: Pipeline,
        X: pd.DataFrame,
        y: pd.Series,
        cv: BaseCrossValidator,
        scoring: str
    ) -> EvaluationResult:
        """Fit pipeline with CV and return scores."""
        ...
    
    def compute_baseline(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        baseline_type: str  # "raw" or "auto"
    ) -> EvaluationResult:
        """Compute baseline scores."""
        ...
    
    def ablation_study(
        self,
        pipeline: Pipeline,
        X: pd.DataFrame,
        y: pd.Series,
        cv: BaseCrossValidator
    ) -> Dict[str, float]:
        """Measure per-operation impact."""
        ...
    
    def check_leakage(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series
    ) -> Dict[str, Any]:
        """Detect leakage signals."""
        ...
```

#### **Optimizer Module**

**Responsibilities:**
- Stage 1: Warm start with heuristic pipelines
- Stage 2: Greedy forward selection / beam search
- Stage 3: Bayesian optimization (optional)
- Stage 4: Pruning & consolidation
- Track best pipeline across iterations
- Implement early stopping

**Key Methods:**
```python
class Optimizer:
    def warm_start(
        self,
        strategist: Strategist,
        fingerprint: DatasetFingerprint
    ) -> List[Pipeline]:
        """Generate initial candidate pipelines."""
        ...
    
    def greedy_forward_selection(
        self,
        base_pipeline: Pipeline,
        X: pd.DataFrame,
        y: pd.Series,
        cv: BaseCrossValidator
    ) -> Pipeline:
        """Greedily add operations to improve score."""
        ...
    
    def bayesian_optimize(
        self,
        pipeline: Pipeline,
        X: pd.DataFrame,
        y: pd.Series,
        cv: BaseCrossValidator,
        n_trials: int
    ) -> Pipeline:
        """Tune hyperparameters with Bayesian optimization."""
        ...
    
    def prune_and_consolidate(
        self,
        pipeline: Pipeline,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Pipeline:
        """Remove redundant features, re-fit."""
        ...
```

#### **Reporter Module**

**Responsibilities:**
- Generate Markdown/HTML/JSON reports
- Create visualizations (feature importance, CV scores, ablations)
- Export artifacts (pipeline.joblib, metadata.json, explanation.md)
- Summarize pipeline decisions with explanations

**Key Methods:**
```python
class Reporter:
    def generate_report(
        self,
        result: AgentResult,
        output_dir: str
    ) -> str:
        """Generate comprehensive report."""
        ...
    
    def export_artifacts(
        self,
        pipeline: Pipeline,
        metadata: Dict[str, Any],
        output_dir: str
    ) -> None:
        """Export pipeline + metadata."""
        ...
    
    def explain_decisions(
        self,
        strategy: FeatureStrategy,
        fingerprint: DatasetFingerprint
    ) -> str:
        """Generate human-readable explanation."""
        ...
```

### 6.2 Tool Adapters (Function-Calling Style)

**JSON Schemas for Operations:**

```json
{
  "load_data": {
    "description": "Load dataset from file or DataFrame",
    "parameters": {
      "path": {"type": "string"},
      "target_column": {"type": "string"},
      "time_column": {"type": "string", "optional": true},
      "entity_column": {"type": "string", "optional": true}
    }
  },
  "fingerprint": {
    "description": "Compute dataset fingerprint",
    "parameters": {
      "X": {"type": "DataFrame"},
      "y": {"type": "Series"}
    },
    "returns": "DatasetFingerprint"
  },
  "build_pipeline": {
    "description": "Build FeatureCraft pipeline from strategy",
    "parameters": {
      "strategy": {"type": "FeatureStrategy"},
      "X": {"type": "DataFrame"},
      "y": {"type": "Series"}
    },
    "returns": "Pipeline"
  },
  "fit_evaluate_cv": {
    "description": "Fit pipeline with CV and return scores",
    "parameters": {
      "pipeline": {"type": "Pipeline"},
      "X": {"type": "DataFrame"},
      "y": {"type": "Series"},
      "cv": {"type": "BaseCrossValidator"},
      "scoring": {"type": "string"}
    },
    "returns": "EvaluationResult"
  },
  "rank_candidates": {
    "description": "Rank pipelines by score",
    "parameters": {
      "results": {"type": "List[EvaluationResult]"}
    },
    "returns": "List[EvaluationResult]"
  },
  "export_artifacts": {
    "description": "Export final pipeline and metadata",
    "parameters": {
      "pipeline": {"type": "Pipeline"},
      "metadata": {"type": "Dict"},
      "output_dir": {"type": "string"}
    }
  }
}
```

### 6.3 Memory & State

**Run Ledger (Minimal State):**
```python
@dataclass
class RunLedger:
    """Minimal state for reproducibility."""
    run_id: str
    dataset_hash: str  # Hash of X.columns + X.shape
    target_name: str
    task_type: TaskType
    estimator_family: str
    random_seed: int
    cv_splits_hash: str  # Hash of CV split indices
    config_snapshot: Dict[str, Any]  # Initial config
    timestamp: str
```

**Pipeline Registry:**
```python
@dataclass
class PipelineEntry:
    """Single pipeline evaluation record."""
    pipeline_id: str
    strategy: FeatureStrategy
    config: FeatureCraftConfig
    cv_score_mean: float
    cv_score_std: float
    metrics: Dict[str, float]  # All metrics
    fit_time_seconds: float
    transform_time_seconds: float
    n_features_out: int
    leakage_risk_score: float
    timestamp: str
```

**Artifact Store Paths:**
```
artifacts/
├── run_{run_id}/
│   ├── ledger.json
│   ├── fingerprint.json
│   ├── pipelines/
│   │   ├── pipeline_001.joblib
│   │   ├── pipeline_001_metadata.json
│   │   ├── pipeline_002.joblib
│   │   └── ...
│   ├── evaluations/
│   │   ├── eval_001.json
│   │   └── ...
│   ├── cache/
│   │   ├── imputed_data.parquet
│   │   ├── encoded_data.parquet
│   │   └── ...
│   ├── final_pipeline.joblib
│   ├── final_metadata.json
│   ├── report.md
│   ├── report.html
│   └── explanation.md
```

---

## 7. Integration Plan with FeatureCraft

### 7.1 Direct Integration: Reusing Existing Classes

**Example 1: Building a Pipeline with FeatureCraft Transformers**

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from featurecraft import (
    RareCategoryGrouper,
    KFoldTargetEncoder,
    FrequencyEncoder,
    SkewedPowerTransformer,
    choose_scaler,
    choose_numeric_imputer,
    ArithmeticInteractions,
    ClusteringFeatureExtractor
)

# Agent Composer builds this pipeline based on strategy
def build_pipeline_example(strategy: FeatureStrategy, X: pd.DataFrame, y: pd.Series):
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    
    # Numeric preprocessing
    numeric_pipeline = Pipeline([
        ("imputer", choose_numeric_imputer("knn")),
        ("skew_transform", SkewedPowerTransformer() if strategy.apply_skew_transform else "passthrough"),
        ("scaler", choose_scaler("robust"))
    ])
    
    # Categorical preprocessing
    low_card_cols = [col for col in categorical_cols if X[col].nunique() <= 10]
    mid_card_cols = [col for col in categorical_cols if 10 < X[col].nunique() <= 50]
    high_card_cols = [col for col in categorical_cols if X[col].nunique() > 50]
    
    categorical_pipeline = ColumnTransformer([
        ("low_card_ohe", make_ohe(), low_card_cols),
        ("mid_card_target", KFoldTargetEncoder(n_splits=5), mid_card_cols),
        ("high_card_freq", FrequencyEncoder(), high_card_cols)
    ])
    
    # Main preprocessing
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_cols),
        ("categorical", categorical_pipeline, categorical_cols)
    ])
    
    # Feature engineering
    feature_engineering = []
    if strategy.use_interactions:
        feature_engineering.append(("interactions", ArithmeticInteractions(operations=["multiply", "divide"])))
    if strategy.use_clustering:
        feature_engineering.append(("clustering", ClusteringFeatureExtractor(method="kmeans", n_clusters=5)))
    
    # Full pipeline
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        *feature_engineering
    ])
    
    return pipeline
```

**Example 2: Using AutoFeatureEngineer with Agent-Modified Config**

```python
from featurecraft import AutoFeatureEngineer, FeatureCraftConfig

# Agent Strategist generates config overrides
def apply_strategy_to_config(strategy: FeatureStrategy, base_config: FeatureCraftConfig) -> FeatureCraftConfig:
    overrides = {}
    
    # Encoding strategy
    if strategy.encoding_priority == "target":
        overrides["use_target_encoding"] = True
        overrides["use_frequency_encoding"] = False
    elif strategy.encoding_priority == "frequency":
        overrides["use_target_encoding"] = False
        overrides["use_frequency_encoding"] = True
    
    # Interactions
    if strategy.use_interactions:
        overrides["interactions_enabled"] = True
        if "arithmetic" in strategy.interaction_types:
            overrides["interactions_use_arithmetic"] = True
        if "polynomial" in strategy.interaction_types:
            overrides["interactions_use_polynomial"] = True
            overrides["interactions_polynomial_degree"] = 2
    
    # Clustering
    if strategy.use_clustering:
        overrides["use_clustering_features"] = True
        overrides["clustering_method"] = strategy.clustering_methods[0]
        overrides["clustering_n_clusters"] = strategy.clustering_n_clusters
    
    # Text
    if strategy.text_strategy == "advanced":
        overrides["text_extract_sentiment"] = True
        overrides["text_extract_ner"] = True
    elif strategy.text_strategy == "minimal":
        overrides["text_extract_sentiment"] = False
        overrides["text_extract_ner"] = False
    
    # Apply overrides
    config_dict = base_config.model_dump()
    config_dict.update(overrides)
    return FeatureCraftConfig(**config_dict)

# Agent Composer uses this
def build_afe_pipeline(strategy: FeatureStrategy, base_config: FeatureCraftConfig):
    optimized_config = apply_strategy_to_config(strategy, base_config)
    afe = AutoFeatureEngineer(config=optimized_config)
    return afe
```

### 7.2 Proposed Thin Wrappers

**PipelineBuilder (Composer Utility):**
```python
class PipelineBuilder:
    """Utility to convert FeatureStrategy → FeatureCraft pipeline."""
    
    def __init__(self, operation_registry: OperationRegistry):
        self.registry = operation_registry
    
    def build(self, strategy: FeatureStrategy, X: pd.DataFrame, y: pd.Series) -> Pipeline:
        """Build pipeline from strategy."""
        steps = []
        
        # Imputation
        steps.append(self.registry.get_imputer(strategy))
        
        # Encoding
        steps.append(self.registry.get_encoder(strategy))
        
        # Scaling
        steps.append(self.registry.get_scaler(strategy))
        
        # Feature engineering
        if strategy.use_interactions:
            steps.append(self.registry.get_interactions(strategy))
        if strategy.use_clustering:
            steps.append(self.registry.get_clustering(strategy))
        
        # Feature selection
        if strategy.apply_feature_selection:
            steps.append(self.registry.get_selector(strategy))
        
        return Pipeline(steps)
```

**OperationRegistry (Mapping Strategy → FeatureCraft Objects):**
```python
class OperationRegistry:
    """Registry mapping operation names → FeatureCraft classes."""
    
    def get_imputer(self, strategy: FeatureStrategy):
        if strategy.imputation_strategy == "simple":
            return choose_numeric_imputer("simple")
        elif strategy.imputation_strategy == "knn":
            return choose_numeric_imputer("knn")
        elif strategy.imputation_strategy == "iterative":
            return choose_numeric_imputer("iterative")
    
    def get_encoder(self, strategy: FeatureStrategy):
        if strategy.encoding_priority == "target":
            return KFoldTargetEncoder(n_splits=5)
        elif strategy.encoding_priority == "frequency":
            return FrequencyEncoder()
        elif strategy.encoding_priority == "hashing":
            return HashingEncoder(n_components=256)
    
    def get_scaler(self, strategy: FeatureStrategy):
        return choose_scaler(strategy.scaling_method)
    
    def get_interactions(self, strategy: FeatureStrategy):
        if "arithmetic" in strategy.interaction_types:
            return ArithmeticInteractions(operations=["multiply", "divide"])
        elif "polynomial" in strategy.interaction_types:
            return PolynomialInteractions(degree=2)
        elif "ratios" in strategy.interaction_types:
            return RatioFeatures()
    
    def get_clustering(self, strategy: FeatureStrategy):
        return ClusteringFeatureExtractor(
            method=strategy.clustering_methods[0],
            n_clusters=strategy.clustering_n_clusters
        )
    
    def get_selector(self, strategy: FeatureStrategy):
        if strategy.selection_method == "mutual_info":
            return MutualInfoSelector(k=strategy.target_n_features)
        elif strategy.selection_method == "tree_importance":
            return TreeImportanceSelector(k=strategy.target_n_features)
```

### 7.3 Scikit-learn Compatibility

All FeatureCraft transformers already implement `fit/transform` API, ensuring seamless integration with sklearn pipelines. The agent's `Composer` module simply orchestrates existing transformers into `Pipeline` and `ColumnTransformer` objects.

**Key Compatibility Points:**
- ✅ All transformers have `fit(X, y)` and `transform(X)` methods
- ✅ Transformers support `get_feature_names_out()` (sklearn 1.0+)
- ✅ Pipelines are serializable with `joblib.dump()`
- ✅ `AutoFeatureEngineer` is a sklearn estimator (has `get_params/set_params`)

---

**(Continues in next section...)**

# FeatureCraft Agent — Design & Implementation (Agentified, Refined)

> A production-ready plan to run FeatureCraft as a **feature-engineering agent** that studies a dataset, selects and builds features from your library, iterates to improve CV score, and exports explainable artifacts. This document refines your original pseudocode, fills gaps, and plugs it into an agent framework.

---

## 0) Executive Summary

**What it does:**

* Inspects a dataset → fingerprints schema, quality, leakage risks.
* Strategizes → proposes feature-engineering plans tailored to fingerprint + estimator family.
* Composes pipelines from your **feature ops** library (sklearn-compatible).
* Evaluates via CV; keeps improvements over baselines.
* Optimizes iteratively (greedy addition → optional Bayesian HPO → pruning).
* Reports with ablation, permutation importance, optional SHAP; exports pipelines and HTML/MD/JSON reports.

**Primary agent framework:** **LangGraph** (LangChain’s graph-native agent runtime). It gives strong control-flow guarantees, retries, and deterministic orchestration that fits your 6-stage pipeline.

**Alternative:** CrewAI (multi-agent roles) for teams that prefer a human-like role split. See Appendix A.

---

## 1) Why LangGraph for this use case

* **Deterministic control-flow**: Your process is a fixed 6-stage DAG with gated branches (e.g., only tune if budget allows, only proceed if baseline beaten). Graph nodes/edges map cleanly.
* **Tool-centric**: Real work is done via Python tools (inspection, compose, evaluate). Nodes simply call tools and update state.
* **Resumability & State**: Central typed state ("RunState") holds fingerprint, strategies, candidates, artifacts. LangGraph persists and resumes.
* **Budget & Policies**: Easy to encode thresholds (≥1%/≥0.5%), time budgets, and early stops as guard edges.
* **Minimal LLM dependency**: LLM optional (for reporting/narration). Training stays in sklearn/Optuna.

---

## 2) System Architecture

### 2.1 High-level modules (unchanged in spirit)

1. **Inspector** → dataset fingerprint & QA
2. **Strategist** → generate strategies from fingerprint
3. **Composer** → build sklearn pipelines from ops registry
4. **Evaluator** → CV, baselines, metrics
5. **Optimizer** → greedy feature addition → optional Bayesian HPO → pruning
6. **Reporter** → artifacts & explainability outputs

### 2.2 Data flow (text diagram)

```
Raw Data (X, y)
  └─▶ Inspector ──fingerprint/QA──▶ Strategist ──plans──▶ Composer ──candidates──▶ Evaluator
                                                                                         │
                                                                                         ├─ if no candidate beats baseline: return baseline_auto
                                                                                         ▼
                                                                                      Optimizer ──▶ Finalist
                                                                                         │
                                                                                         ▼
                                                                                      Reporter ──▶ Artifacts & Reports
```

### 2.3 Run State (shared across nodes)

```python
@dataclass
class RunState:
    run_id: str
    config: AgentConfig
    budget: ComputeBudget

    # Data & meta
    X: pd.DataFrame
    y: pd.Series
    target_name: str

    # Stage outputs
    fingerprint: Optional[DatasetFingerprint] = None
    cv_strategy: Optional[BaseCrossValidator] = None
    baselines: Optional[Baselines] = None  # raw, auto
    initial_strategies: List[FeatureStrategy] = field(default_factory=list)

    candidates: List[Candidate] = field(default_factory=list)  # (strategy, pipeline, result)
    finalists: List[Candidate] = field(default_factory=list)
    best: Optional[Candidate] = None

    # Explainability
    ablation: Optional[AblationResults] = None
    perm_importance: Optional[Dict[str, float]] = None
    shap_values: Optional[Any] = None
    leakage_report: Optional[Dict[str, Any]] = None

    # Artifacts
    artifact_store: ArtifactStore = None
    ledger: RunLedger = None
```

---

## 3) Technology Stack

* **Core**: `pandas`, `numpy`, `scikit-learn`, `scipy`
* **Encoders**: `category_encoders` (target/frequency/one-hot)
* **Imbalance**: `imbalanced-learn` (optional; SMOTE/undersampling)
* **Optimization**: `optuna` for Bayesian HPO
* **Explainability**: `shap` (budget-gated), `sklearn.inspection.permutation_importance`
* **NLP (optional)**: `vaderSentiment`/`textblob` for sentiment; `spaCy` for NER; `textstat` for readability
* **Logging/CLI**: `rich`, `typer` or `click`
* **Artifacts**: `joblib`/`cloudpickle`; optional `mlflow`
* **Agent runtime**: **LangGraph** (plus `langchain-core`)

---

## 4) Contracts & Registries

### 4.1 Transformer contract (sklearn-compatible)

```python
class BaseOp(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        raise NotImplementedError
    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        # strongly recommended for pruning/reporting
        return np.array([f"{self.__class__.__name__}_{i}" for i in range(self._n_out)])
```

### 4.2 Feature Ops Registry

```python
FEATURE_OPS = {
    "imputer_num": NumImputer(),
    "imputer_cat": CatImputer(),
    "onehot": OneHotEncoder(handle_unknown="ignore", sparse_output=False),
    "freq_encode": FrequencyEncoder(),  # custom
    "target_encode": TargetEncoder(),   # from category_encoders
    "arith_interactions": ArithmeticInteractions(["*", "/"]),
    "poly2": PolynomialFeatures(degree=2, include_bias=False),
    "kmeans5": KMeansFeaturizer(n_clusters=5),
    "row_stats": RowStats(["mean","std","min","max"]),
    # ... add your own here
}
```

### 4.3 Estimator Families

* **tree**: LightGBM/XGBoost/RandomForest/ExtraTrees (no scaling needed; robust to monotonic transforms)
* **linear**: Logistic/Linear/ElasticNet (needs scaling; benefits from polynomial features and clean encodings)
* **other**: SVM (scaled), CatBoost (built-in categorical handling), etc.

### 4.4 DatasetFingerprint additions

Include fields used downstream (fixes gaps from original):

```python
@dataclass
class DatasetFingerprint:
    n_rows: int
    n_cols: int
    task_type: Literal["classification","regression"]
    categorical_cols: List[str]
    numeric_cols: List[str]
    text_cols: List[str]
    datetime_cols: List[str]
    n_features_original: int

    missing_summary: Dict[str, float]
    high_missing_cols: List[str]

    cardinality_summary: Dict[str, int]
    low_cardinality_cols: List[str]
    mid_cardinality_cols: List[str]
    high_cardinality_cols: List[str]

    skewness_summary: Dict[str, float]
    outlier_share_summary: Dict[str, float]
    heavily_skewed_cols: List[str]

    class_balance: Optional[Dict[Any, float]]
    minority_class_ratio: Optional[float]
    is_imbalanced: bool

    high_correlation_pairs: List[Tuple[str,str,float]]
    multicollinear_groups: List[List[str]]

    mutual_info_scores: Dict[str, float]
    low_mi_cols: List[str]

    is_time_series: bool
    # optional: time_column, granularity, gaps
    leakage_risk_cols: List[str]
```

---

## 5) Policies: Budget, Thresholds, and Guardrails

* **Baseline gate**: keep candidates only if `cv_mean ≥ baseline_auto_mean * 1.01` (≥1%).
* **Greedy add gate**: keep an added op only if `new_cv ≥ current_cv * 1.005` (≥0.5%).
* **HPO gate**: `budget.has_budget_for_bayesian()`; trials from config.
* **SHAP gate**: `budget.has_budget_for_shap()`; sample limit.
* **Max iterations**: `budget.max_iterations` in greedy forward.
* **Leakage**: run final leakage checks on train/test transformed splits.
* **Reproducibility**: `random_seed` across all RNGs.

---

## 6) Evaluation & Metrics

* **Primary metric**: from config (e.g., `logloss`, `roc_auc`, `rmse`).
* **CV choice**: `TimeSeriesSplit` if time-dependent; `GroupKFold` if entity/group; `StratifiedKFold` for classification; otherwise `KFold`.
* **Confidence**: finalists re-evaluated with more splits (e.g., 10-fold).
* **Ablation**: drop-feature or drop-block analysis with CV.
* **Permutation importance**: on the finalist pipeline.
* **Optional SHAP**: budget-gated for model understanding.

---

## 7) Reproducibility, Tracking, and Artifacts

* **Artifact store**: `/artifacts/run_{run_id}`
* Save: `fingerprint.json`, `baselines.json`, candidate summaries, final pipeline (`best_pipeline.joblib`), `report.{md,html,json}`.
* **MLflow** (optional): log params/metrics/artifacts; tag run_id.
* **Ledger**: append step-level decisions, thresholds, and deltas for auditability.

---

## 8) Refined End-to-End Pseudocode (corrected)

```python
def run_featurecraft_agent(X, y, target_name, config, budget):
    run_id = generate_run_id()
    ledger = create_run_ledger(run_id, X, y, target_name, config)
    artifact_store = ArtifactStore(f"artifacts/run_{run_id}")

    print("🔍 Stage 1/6: Inspecting Dataset...")
    inspector = Inspector(config)
    fingerprint = inspector.fingerprint(X, y)
    artifact_store.save("fingerprint.json", as_dict(fingerprint))

    issues = inspector.check_data_quality(X, y)
    if any(issue.severity == "ERROR" for issue in issues):
        print(f"❌ Critical data quality issues found: {issues}")
        return failure_result(issues)

    leakage_risks = inspector.estimate_leakage_risk(X, y)
    high_risk_cols = [c for c, risk in leakage_risks.items() if risk > 0.5]
    if high_risk_cols:
        print(f"⚠️ High leakage risk: {high_risk_cols}")

    print("🧠 Stage 2/6: Generating Strategies...")
    strategist = Strategist(config)
    initial_strategies = strategist.generate_initial_strategies(
        fingerprint=fingerprint,
        estimator_family=config.estimator_family,
        budget=budget,
    )
    cv_strategy = strategist.select_cv_strategy(fingerprint, config)

    print("📊 Stage 3/6: Evaluating Baselines...")
    evaluator = Evaluator(config, cv_strategy)
    baseline_raw = evaluator.compute_baseline(X, y, baseline_type="raw")
    baseline_auto = evaluator.compute_baseline(X, y, baseline_type="auto")
    artifact_store.save("baselines.json", {
        "raw": baseline_raw.to_dict(),
        "auto": baseline_auto.to_dict()
    })

    print("🚀 Stage 4/6: Building Heuristic Pipelines...")
    composer = Composer(config)
    optimizer = Optimizer(config, evaluator, artifact_store)  # evaluator injected

    candidate_pipelines = []
    for strategy in initial_strategies:
        pipeline = composer.build_pipeline(strategy, X, y)
        result = evaluator.evaluate_pipeline(pipeline, X, y)
        if result.cv_score_mean > baseline_auto.cv_score_mean * 1.01:
            candidate_pipelines.append((strategy, pipeline, result))
            print(f"  ✓ {strategy.name}: {result.cv_score_mean:.4f}")
        else:
            print(f"  ✗ {strategy.name}: {result.cv_score_mean:.4f} (≤ baseline)")

    if not candidate_pipelines:
        print("⚠️ No pipelines beat baseline. Using auto baseline.")
        return baseline_auto

    candidate_pipelines.sort(key=lambda x: x[2].cv_score_mean, reverse=True)
    best_k = candidate_pipelines[:3]

    print("🔬 Stage 5/6: Iterative Optimization...")
    print("  → Greedy Forward Selection...")
    refined = []
    for strategy, pipeline, _ in best_k:
        refined_pipeline = optimizer.greedy_forward_selection(
            base_pipeline=pipeline, X=X, y=y, budget=budget.stage_budget(stage=5)
        )
        refined_result = evaluator.evaluate_pipeline(refined_pipeline, X, y)
        refined.append((strategy, refined_pipeline, refined_result))
        print(f"    ✓ Refined {strategy.name}: {refined_result.cv_score_mean:.4f}")

    refined.sort(key=lambda x: x[2].cv_score_mean, reverse=True)
    top_2 = refined[:2]

    if budget.has_budget_for_bayesian():
        print("  → Bayesian Hyperparameter Tuning...")
        tuned = []
        for strategy, pipeline, _ in top_2:
            tuned_pipeline = optimizer.bayesian_optimize(
                pipeline=pipeline, X=X, y=y, n_trials=budget.n_bayesian_trials
            )
            tuned_result = evaluator.evaluate_pipeline(tuned_pipeline, X, y)
            tuned.append((strategy, tuned_pipeline, tuned_result))
            print(f"    ✓ Tuned {strategy.name}: {tuned_result.cv_score_mean:.4f}")
        top_2 = tuned

    print("  → Pruning & Consolidation...")
    final_candidates = []
    for strategy, pipeline, _ in top_2:
        pruned_pipeline = optimizer.prune_and_consolidate(pipeline, X, y)
        final_result = evaluator.evaluate_pipeline(pruned_pipeline, X, y, n_splits=10)
        final_candidates.append((strategy, pruned_pipeline, final_result))
        print(f"    ✓ Pruned {strategy.name}: {final_result.cv_score_mean:.4f}")

    final_candidates.sort(key=lambda x: x[2].cv_score_mean, reverse=True)
    best_strategy, best_pipeline, best_result = final_candidates[0]

    print(f"\n✅ Best Pipeline: {best_strategy.name}")
    print(f"   Score: {best_result.cv_score_mean:.4f} ± {best_result.cv_score_std:.4f}")
    print(f"   Improvement: {(best_result.cv_score_mean / baseline_auto.cv_score_mean - 1) * 100:.1f}%")

    print("📝 Stage 6/6: Reporting...")
    ablation_results = evaluator.ablation_study(best_pipeline, X, y)
    importance_scores = evaluator.permutation_importance(best_pipeline, X, y)
    shap_values = None
    if budget.has_budget_for_shap():
        shap_values = evaluator.compute_shap(best_pipeline, X, y, n_samples=1000)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=config.random_seed)
    X_tr_t = best_pipeline.fit_transform(X_tr, y_tr)
    X_te_t = best_pipeline.transform(X_te)
    leakage_report = evaluator.check_leakage(X_tr_t, X_te_t, y_tr)

    reporter = Reporter(config, artifact_store)
    agent_result = AgentResult(
        run_id=run_id,
        fingerprint=fingerprint,
        best_strategy=best_strategy,
        best_pipeline=best_pipeline,
        best_result=best_result,
        baseline_raw=baseline_raw,
        baseline_auto=baseline_auto,
        ablation_results=ablation_results,
        importance_scores=importance_scores,
        shap_values=shap_values,
        leakage_report=leakage_report,
        all_candidates=final_candidates,
        ledger=ledger,
    )

    reporter.export_artifacts(agent_result)
    reporter.generate_report(agent_result, format="markdown")
    reporter.generate_report(agent_result, format="html")
    reporter.generate_report(agent_result, format="json")

    print(f"\n✅ Completed. Artifacts: {artifact_store.root_dir}")
    return agent_result
```

---

## 9) Module Details (refined)

### 9.1 Inspector

* **Type detection**: `numeric/categorical/text/datetime` by dtype + heuristics.
* **Missingness**: per-column rate; flag >30%.
* **Cardinality**: low ≤10, mid 11–50, high >50.
* **Skewness & outliers**: `scipy.stats.skew`; outlier share via IQR or z-score.
* **Class imbalance**: minority ratio <0.2.
* **Correlation & multicollinearity**: Pearson/Spearman for numeric; VIF groups (threshold 10.0). Guard non-numeric.
* **Mutual information**: `mutual_info_classif` or `mutual_info_regression` (separate discrete/continuous handling).
* **Time series**: detect datetime column; granularity, gaps; set `is_time_series`.
* **Leakage signals**: near-perfect correlation with target for numeric; for categoricals, check very high predictive MI + suspicious naming (e.g., *_target, *_label*). Provide a risk score.

### 9.2 Strategist

* Produces **conservative/balanced/aggressive** plans + **estimator-specific** variants.
* Sets: encoding priority, scaling method, interaction choices, clustering usage, text features, feature selection method/target N.
* Selects **CV strategy** based on time-series/grouping/task type.

### 9.3 Composer

* Stitches **ColumnTransformers** for type-specific preprocessing.
* Pulls ops by name from `FEATURE_OPS`.
* Ensures every custom op is robust to NaNs and implements `get_feature_names_out` when possible.

### 9.4 Evaluator

* Common `fit/eval` routine with chosen CV and metric.
* Supports baselines: `raw` (minimal) and `auto` (default pipeline without heuristics).
* Optional `n_splits` override for finalist evaluation.

### 9.5 Optimizer

* **Greedy forward selection**: try adding ops from a candidate list; accept if ≥0.5% gain.
* **Bayesian HPO**: Optuna search spaces per estimator & key transforms.
* **Pruning**: drop features with |corr|>0.95 (post-transform) and very low permutation importance; re-fit pipeline with a `ColumnSelector`.

### 9.6 Reporter

* Generates **ablation**, **permutation importance**, optional **SHAP** plots.
* Writes **Markdown/HTML/JSON**. Artifacts include pipeline, metrics, fingerprint, candidate summary.

---

## 10) LangGraph Integration (code sketch)

```python
from langgraph.graph import StateGraph, END

# 1) Define state type (RunState from §2.3)

# 2) Node functions (each returns partial state updates)

def node_inspect(state: RunState) -> dict:
    inspector = Inspector(state.config)
    fp = inspector.fingerprint(state.X, state.y)
    issues = inspector.check_data_quality(state.X, state.y)
    state.artifact_store.save("fingerprint.json", as_dict(fp))
    return {"fingerprint": fp}

def node_strategize(state: RunState) -> dict:
    strat = Strategist(state.config)
    strategies = strat.generate_initial_strategies(state.fingerprint, state.config.estimator_family, state.budget)
    cv = strat.select_cv_strategy(state.fingerprint, state.config)
    return {"initial_strategies": strategies, "cv_strategy": cv}

def node_baselines(state: RunState) -> dict:
    ev = Evaluator(state.config, state.cv_strategy)
    raw = ev.compute_baseline(state.X, state.y, "raw")
    auto = ev.compute_baseline(state.X, state.y, "auto")
    state.artifact_store.save("baselines.json", {"raw": raw.to_dict(), "auto": auto.to_dict()})
    return {"baselines": Baselines(raw=raw, auto=auto)}

def node_candidates(state: RunState) -> dict:
    comp = Composer(state.config)
    ev = Evaluator(state.config, state.cv_strategy)
    cands = []
    for s in state.initial_strategies:
        p = comp.build_pipeline(s, state.X, state.y)
        r = ev.evaluate_pipeline(p, state.X, state.y)
        if r.cv_score_mean > state.baselines.auto.cv_score_mean * 1.01:
            cands.append(Candidate(strategy=s, pipeline=p, result=r))
    return {"candidates": cands}

def node_optimize(state: RunState) -> dict:
    if not state.candidates:
        return {"best": None}  # handled by policy edge
    ev = Evaluator(state.config, state.cv_strategy)
    opt = Optimizer(state.config, ev, state.artifact_store)
    cands = sorted(state.candidates, key=lambda c: c.result.cv_score_mean, reverse=True)[:3]
    refined = []
    for c in cands:
        p2 = opt.greedy_forward_selection(c.pipeline, state.X, state.y, state.budget.stage_budget(5))
        r2 = ev.evaluate_pipeline(p2, state.X, state.y)
        refined.append(Candidate(c.strategy, p2, r2))
    refined.sort(key=lambda c: c.result.cv_score_mean, reverse=True)
    top2 = refined[:2]
    if state.budget.has_budget_for_bayesian():
        tuned = []
        for c in top2:
            p3 = opt.bayesian_optimize(c.pipeline, state.X, state.y, state.budget.n_bayesian_trials)
            r3 = ev.evaluate_pipeline(p3, state.X, state.y)
            tuned.append(Candidate(c.strategy, p3, r3))
        top2 = tuned
    finals = []
    for c in top2:
        p4 = opt.prune_and_consolidate(c.pipeline, state.X, state.y)
        r4 = ev.evaluate_pipeline(p4, state.X, state.y, n_splits=10)
        finals.append(Candidate(c.strategy, p4, r4))
    finals.sort(key=lambda c: c.result.cv_score_mean, reverse=True)
    best = finals[0]
    return {"finalists": finals, "best": best}

def node_report(state: RunState) -> dict:
    ev = Evaluator(state.config, state.cv_strategy)
    ablation = ev.ablation_study(state.best.pipeline, state.X, state.y)
    perm = ev.permutation_importance(state.best.pipeline, state.X, state.y)
    shap_vals = None
    if state.budget.has_budget_for_shap():
        shap_vals = ev.compute_shap(state.best.pipeline, state.X, state.y, n_samples=1000)
    # leakage check
    Xtr, Xte, ytr, yte = train_test_split(state.X, state.y, test_size=0.2, random_state=state.config.random_seed)
    Xtr_t = state.best.pipeline.fit_transform(Xtr, ytr)
    Xte_t = state.best.pipeline.transform(Xte)
    leak = ev.check_leakage(Xtr_t, Xte_t, ytr)
    rep = Reporter(state.config, state.artifact_store)
    rep.export_artifacts_from_state(state, ablation, perm, shap_vals, leak)
    rep.generate_report_from_state(state)
    return {"ablation": ablation, "perm_importance": perm, "shap_values": shap_vals, "leakage_report": leak}

# 3) Build the graph and policy edges
sg = StateGraph(RunState)
sg.add_node("inspect", node_inspect)
sg.add_node("strategize", node_strategize)
sg.add_node("baselines", node_baselines)
sg.add_node("candidates", node_candidates)
sg.add_node("optimize", node_optimize)
sg.add_node("report", node_report)

sg.set_entry_point("inspect")
sg.add_edge("inspect", "strategize")
sg.add_edge("strategize", "baselines")
sg.add_edge("baselines", "candidates")

# branch: if no candidates beat baseline, skip to report with baseline_auto

def has_candidates(state: RunState):
    return bool(state.candidates)

sg.add_conditional_edges("candidates", has_candidates, {True: "optimize", False: "report"})
sg.add_edge("optimize", "report")
sg.add_edge("report", END)

graph = sg.compile()
```

**Note:** you can keep the LLM out of the loop entirely, or use a small LLM node to write the human-facing narrative in the report.

---

## 11) Config & CLI

### 11.1 Pydantic config

```python
class AgentConfig(BaseModel):
    estimator_family: Literal["tree","linear","svm","catboost"] = "tree"
    primary_metric: str = "logloss"
    time_budget_minutes: int = 60
    max_pipelines: int = 50
    enable_bayesian_optimization: bool = True
    n_bayesian_trials: int = 30
    enable_shap: bool = False
    random_seed: int = 42
    output_dir: str = "artifacts"
    entity_column: Optional[str] = None
```

### 11.2 CLI (Typer)

```python
@app.command()
def run(
    data_path: Path,
    target: str,
    config_path: Optional[Path] = None,
):
    df = pd.read_csv(data_path)
    X, y = df.drop(columns=[target]), df[target]
    config = AgentConfig(**yaml.safe_load(open(config_path))) if config_path else AgentConfig()
    budget = ComputeBudget.from_config(config)
    result = run_featurecraft_agent(X, y, target, config, budget)
    print(result.summary())
```

---

## 12) Example Usage (refined)

```python
from featurecraft.agent import FeatureCraftAgent, AgentConfig

df = pd.read_csv("dataset.csv")
X, y = df.drop(columns=["target"]), df["target"]

config = AgentConfig(
    estimator_family="tree",
    primary_metric="roc_auc",
    time_budget_minutes=45,
    n_bayesian_trials=40,
    enable_shap=False,
    output_dir="artifacts/my_experiment",
)

agent = FeatureCraftAgent(config=config)
result = agent.run(X=X, y=y, target_name="target")
print(result.summary())
print(f"Best score: {result.best_score:.4f}")
print(f"Improvement: {result.improvement_pct:.1f}%")

pipeline = result.load_pipeline()
X_new_t = pipeline.transform(X_new)
```

---

## 13) Testing & Quality Checklist

* Unit-test each transformer (NaN-safe, deterministic, correct shapes, names).
* Snapshot-test Composer outputs per strategy.
* Integration-test greedy selection gates (0.5% threshold) with synthetic datasets.
* Reproducibility test with fixed seeds.
* Performance smoke tests on medium datasets (≤100k rows) within time budget.

---

## 14) Roadmap

* Add **entity leakage** checks (train/test by entity)
* Cache intermediate transformed matrices to speed greedy trials
* Add **CatBoost** strategy variant
* Add **automated drift** report (train–test distribution shift)
* Distributed HPO via Optuna’s RDB storage

---

## Appendix A) CrewAI Alternative (multi-agent roles)

**Roles:**

* Inspector (Data QA)
* Strategist (FE planner)
* Composer (Pipeline builder)
* Evaluator (CV)
* Optimizer (Search/prune)
* Reporter (Narratives)
* Supervisor (budget/timekeeper)

**When to choose CrewAI:** you want human-like role chat, async-like collaboration, or external human-in-the-loop steps. Control flow is looser than LangGraph but suitable for exploratory workflows.

**Integration tips:**

* Share a single ArtifactStore and RunState-like memory.
* Gate expensive tools by Supervisor’s policy messages.
* Keep training in tools; use LLMs for planning/explanations only.
