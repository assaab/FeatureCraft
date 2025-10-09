# FeatureCraft Feature Engineering Techniques

This document provides a comprehensive overview of all feature engineering techniques implemented in FeatureCraft, organized by technique category, method, use case, and library.

## 📊 Encoding Techniques

| Technique | Method | Use Case | Library |
|-----------|--------|----------|---------|
| **Rare Category Grouping** | `RareCategoryGrouper` | Handle infrequent categories | Custom |
| **Hashing Encoding** | `HashingEncoder` | High-cardinality categorical features | Custom |
| **KFold Target Encoding** | `KFoldTargetEncoder` | Binary classification with target encoding | Custom |
| **Leave-One-Out Target Encoding** | `LeaveOneOutTargetEncoder` | Target encoding with leakage prevention | Custom |
| **Weight of Evidence Encoding** | `WoEEncoder` | Binary classification feature encoding | Custom |
| **Ordinal Encoding** | `OrdinalEncoder` | Categorical to ordinal conversion | Custom |
| **Out-of-Fold Target Encoding** | `OutOfFoldTargetEncoder` | Cross-validation aware target encoding | Custom |
| **Frequency Encoding** | `FrequencyEncoder` | Categorical feature frequency encoding | Custom |
| **Count Encoding** | `CountEncoder` | Categorical feature count encoding | Custom |

## 📝 Text Processing Techniques

| Technique | Method | Use Case | Library |
|-----------|--------|----------|---------|
| **Text Statistics** | `TextStatisticsExtractor` | Character/word counts, sentence analysis | Custom |
| **Sentiment Analysis** | `SentimentAnalyzer` | Polarity and subjectivity scores | TextBlob/VADER |
| **Named Entity Recognition** | `NERFeatureExtractor` | Extract entities (persons, organizations, locations) | spaCy |
| **Topic Modeling** | `TopicModelingFeatures` | Latent Dirichlet Allocation topic distributions | scikit-learn |
| **Readability Scores** | `ReadabilityScoreExtractor` | Text complexity (Flesch-Kincaid, SMOG) | textstat |
| **Text Preprocessing** | `TextPreprocessor` | Cleaning, normalization, tokenization | Custom |
| **Text Vectorization** | TF-IDF, CountVectorizer | Feature extraction for ML | scikit-learn |

**Note**: Text processing features are now available in the main FeatureCraft API and can be used directly: `from featurecraft import SentimentAnalyzer, NERFeatureExtractor, TopicModelingFeatures`

## 🔢 Mathematical Transformations

| Technique | Method | Use Case | Library |
|-----------|--------|----------|---------|
| **Numeric Conversion** | `NumericConverter` | Mixed type column conversion | Custom |
| **Power Transformation** | `SkewedPowerTransformer` | Handle skewed distributions with Yeo-Johnson | Custom |
| **Log Transformation** | `LogTransformer` | Normalize right-skewed data | Custom |
| **Log1p Transformation** | `Log1pTransformer` | Stabilize variance for small values (log(x+1)) | Custom |
| **Square Root Transformation** | `SqrtTransformer` | Moderate skewness reduction | Custom |
| **Reciprocal Transformation** | `ReciprocalTransformer` | Handle inverse relationships (1/x) | Custom |
| **Box-Cox Transformation** | `BoxCoxTransformer` | Normalize distributions for positive data | Custom |
| **Exponential Transformation** | `ExponentialTransformer` | Create polynomial features | Custom |
| **Yeo-Johnson Transformation** | `YeoJohnsonWrapper` | Normalize non-positive data | Custom |
| **Intelligent Auto-Selection** | `MathematicalTransformer` | Automatically select optimal transform per column | Custom |

## 🧮 Statistical Features

| Technique | Method | Use Case | Library |
|-----------|--------|----------|---------|
| **Row-wise Statistics** | `RowStatisticsTransformer` | Cross-feature aggregations (mean, std, min, max per row) | Custom |
| **Percentile Ranking** | `PercentileRankTransformer` | Within-column percentile ranking | Custom |
| **Z-Score Standardization** | `ZScoreTransformer` | Standardized scores across features | Custom |
| **Outlier Detection** | `OutlierDetector` | IQR and Z-score based outlier flagging | Custom |
| **Quantile-based Features** | `QuantileTransformer` | Quantile-based transformations | Custom |
| **Target-based Features** | `TargetBasedFeaturesTransformer` | Statistical relationships with target variable | Custom |
| **Missing Value Patterns** | `MissingValuePatternsTransformer` | Analyze missing value patterns across features | Custom |

## 🏭 Clustering-based Features

| Technique | Method | Use Case | Library |
|-----------|--------|----------|---------|
| **K-Means Clustering** | `ClusteringFeatureExtractor` | Cluster membership and distances | scikit-learn |
| **DBSCAN Clustering** | `ClusteringFeatureExtractor` | Density-based clustering with outlier detection | scikit-learn |
| **Gaussian Mixture** | `ClusteringFeatureExtractor` | Probabilistic clustering with soft assignments | scikit-learn |
| **Hierarchical Clustering** | `ClusteringFeatureExtractor` | Tree-based clustering for nested structures | scikit-learn |
| **Multi-Method Clustering** | `MultiMethodClusteringExtractor` | Ensemble clustering from multiple algorithms | Custom |
| **Adaptive Clustering** | `AdaptiveClusteringExtractor` | Auto-select optimal clustering method and parameters | Custom |

## 🌍 Domain-Specific Features

### Finance & Trading
| Technique | Method | Use Case | Library |
|-----------|--------|----------|---------|
| **Technical Indicators** | `FinanceTechnicalIndicators` | RSI, MACD, Bollinger Bands, Moving Averages | Custom |
| **Risk Ratios** | `FinanceRiskRatios` | Sharpe ratio, Sortino ratio, Maximum drawdown | Custom |
| **Volatility Measures** | `FinanceVolatility` | Historical volatility, Beta calculations | Custom |

### E-commerce & Retail
| Technique | Method | Use Case | Library |
|-----------|--------|----------|---------|
| **RFM Analysis** | `EcommerceRFM` | Recency, Frequency, Monetary customer analysis | Custom |
| **Customer Segmentation** | `EcommerceRFM` | RFM-based customer segments (Champion, Loyal, etc.) | Custom |
| **Purchase Patterns** | Transaction analysis | Purchase timing and value patterns | Custom |

### Healthcare & Medical
| Technique | Method | Use Case | Library |
|-----------|--------|----------|---------|
| **Vital Sign Ratios** | `HealthcareVitals` | BMI, blood pressure ratios, heart rate variability | Custom |
| **Clinical Scores** | `ClinicalScores` | APACHE, SOFA, Charlson comorbidity scores | Custom |
| **Medical Measurements** | `MedicalMeasurements` | Laboratory value interpretations | Custom |

### Natural Language Processing
| Technique | Method | Use Case | Library |
|-----------|--------|----------|---------|
| **Text Statistics** | `NLPTextStats` | Character/word counts, readability scores | Custom |
| **Part-of-Speech Features** | `NLPPoSFeatures` | Grammatical feature extraction | spaCy/NLTK |
| **Sentiment Analysis** | `NLPSentiment` | Polarity and emotion detection | TextBlob/NLTK |

### Geospatial Features
| Technique | Method | Use Case | Library |
|-----------|--------|----------|---------|
| **Distance Calculations** | `GeospatialFeatures` | Haversine distance between coordinates | Custom |
| **Proximity Features** | `GeospatialFeatures` | Distance to points of interest (POIs) | Custom |
| **Coordinate Transformations** | `GeospatialFeatures` | Lat/lon to radians, cartesian coordinates | Custom |
| **Spatial Binning** | `GeospatialFeatures` | Geohash-like spatial discretization | Custom |

## ⚡ Feature Interactions

| Technique | Method | Use Case | Library |
|-----------|--------|----------|---------|
| **Arithmetic Interactions** | `ArithmeticInteractions` | Linear combinations, ratios, products | Custom |
| **Polynomial Features** | `PolynomialFeatures` | Cross-products and powers | scikit-learn |
| **Ratio Features** | Division operations | Proportional relationships | Custom |
| **Product Interactions** | Multiplication operations | Interaction effects | Custom |
| **Binned Interactions** | Categorical × Numeric | Discretized interactions | Custom |

## ⏰ Time Series Features

| Technique | Method | Use Case | Library |
|-----------|--------|----------|---------|
| **Lag Features** | `LagFeaturesTransformer` | Temporal dependencies | Custom |
| **Rolling Statistics** | `RollingWindowTransformer` | Moving averages and statistics | Custom |
| **Expanding Statistics** | `ExpandingWindowTransformer` | Cumulative statistics over time | Custom |
| **Datetime Feature Extraction** | `DateTimeFeatures` | Year, month, day, hour, minute, second, etc. | Custom |
| **Cyclical Encoding** | `DateTimeFeatures` | Sin/cos transforms for periodic patterns | Custom |
| **Seasonality Features** | `DateTimeFeatures` | Season extraction (winter/spring/summer/fall) | Custom |
| **Business Logic Features** | `DateTimeFeatures` | Business hours, weekends, month/quarter boundaries | Custom |
| **Relative Time Features** | `DateTimeFeatures` | Days since reference date | Custom |
| **Rank-based Features** | `RankFeaturesTransformer` | Temporal ranking and ordering | Custom |

## 🎯 Feature Selection Techniques

| Technique | Method | Use Case | Library |
|-----------|--------|----------|---------|
| **Mutual Information Selection** | `MutualInfoSelector` | Non-linear feature relationships | scikit-learn |
| **Chi-Square Selection** | `Chi2Selector` | Statistical significance for categorical targets | scikit-learn |
| **Lasso Selection** | `LassoSelector` | L1-regularized feature selection | scikit-learn |
| **Recursive Feature Elimination** | `RFESelector` | Recursive feature elimination with estimator | scikit-learn |
| **Sequential Selection** | `SequentialFeatureSelector` | Forward/backward feature selection | mlxtend |
| **Tree-based Selection** | `TreeImportanceSelector` | Feature importance from tree models | Custom |
| **Boruta Algorithm** | `BorutaSelector` | Statistical significance testing | boruta |
| **WOE/IV Selection** | `WOEIVSelector` | Weight of Evidence/Information Value | Custom |
| **Correlation-based Pruning** | `prune_correlated` | Remove highly correlated features | Custom |
| **Variance Inflation Factor** | `compute_vif_drop` | Detect multicollinearity | Custom |
| **Dimensionality Reduction** | `DimensionalityReducer` | Reduce feature space | Custom |
| **Multi-Method Reduction** | `MultiMethodDimensionalityReducer` | Ensemble dimensionality reduction | Custom |
| **Adaptive Reduction** | `AdaptiveDimensionalityReducer` | Auto-select optimal reduction method | Custom |

## 🔧 Imputation Techniques

| Technique | Method | Use Case | Library |
|-----------|--------|----------|---------|
| **Simple Imputation** | `SimpleImputer` | Basic missing value filling | scikit-learn |
| **K-Nearest Neighbors** | `KNNImputer` | Distance-based imputation | scikit-learn |
| **Iterative Imputation** | `IterativeImputer` | Multivariate imputation | scikit-learn |
| **Median/Mean Imputation** | Strategy-based | Robust central tendency | scikit-learn |
| **Missing Indicators** | Add binary flags | Missingness patterns | Custom |

## ⚖️ Scaling Techniques

| Technique | Method | Use Case | Library |
|-----------|--------|----------|---------|
| **Standard Scaling** | `StandardScaler` | Zero mean, unit variance | scikit-learn |
| **Min-Max Scaling** | `MinMaxScaler` | Scale to [0,1] range | scikit-learn |
| **Robust Scaling** | `RobustScaler` | Outlier-resistant scaling | scikit-learn |
| **Max-Abs Scaling** | `MaxAbsScaler` | Preserve sparsity | scikit-learn |

## 🤖 AI-Powered Techniques

| Technique | Method | Use Case | Library |
|-----------|--------|----------|---------|
| **AI Feature Advisor** | `AIFeatureAdvisor` | LLM-driven strategy recommendations | Custom (OpenAI/Anthropic) |
| **Feature Engineering Planner** | `FeatureEngineeringPlanner` | Orchestrated pipeline planning | Custom |
| **Adaptive Configuration** | `AdaptiveConfigOptimizer` | Learning from performance feedback | Custom |
| **Pipeline Explanation** | `PipelineExplainer` | Explain feature engineering decisions | Custom |
| **Transformation Explanation** | `TransformationExplanation` | Explain individual transformations | Custom |
| **Decision Categories** | `DecisionCategory` | Categorize feature engineering decisions | Custom |

## 🏗️ Advanced Techniques

### Binning/Discretization
- **Multiple Strategies**: `BinningTransformer`
  - Equal width binning (uniform intervals)
  - Equal frequency binning (quantile-based)
  - K-means clustering binning (data-driven boundaries)
  - Decision tree binning (supervised, target-aware)
  - Custom binning (user-defined edges)

### Outlier Handling
- **Winsorization**: `WinsorizerTransformer`
- **Percentile-based clipping**: `WinsorizerTransformer`
- **Z-score based detection**: Built-in outlier detection

### Categorical Cleaning
- **Normalization**: `CategoricalCleaner`
- **Missing value handling**: `CategoricalMissingIndicator`
- **String standardization**: Built-in cleaning functions

### Pipeline Integration Features
- **Auto Feature Engineer**: `AutoFeatureEngineer` - End-to-end automated pipeline
- **Report Builder**: `ReportBuilder` - Generate comprehensive feature reports
- **Configuration Management**: `FeatureCraftConfig` - Pipeline configuration
- **Pipeline Export**: Export trained pipelines for production use

## 📈 Pipeline Integration

All techniques are designed to work seamlessly with:
- **Scikit-learn pipelines**
- **Feature unions for parallel processing**
- **Automatic hyperparameter tuning**
- **Cross-validation compatibility**
- **Production deployment** (pipeline export functionality)
- **Explainability integration** (built-in explanations)
- **Configuration management** (centralized settings)
- **Performance monitoring** (drift detection and reporting)

## 🔍 Technique Selection Logic

FeatureCraft automatically selects appropriate techniques based on:

1. **Data Characteristics**:
   - Column types (numeric, categorical, text, datetime, geospatial)
   - Missing value patterns and data quality
   - Cardinality levels and feature distributions
   - Domain context (finance, healthcare, e-commerce, etc.)
   - Temporal patterns and seasonality
   - Text complexity and structure

2. **Estimator Family**:
   - Tree-based models (no scaling, label encoding, clustering features)
   - Linear models (standard scaling, one-hot encoding, statistical features)
   - SVM models (standard scaling, one-hot encoding, polynomial features)
   - KNN models (min-max scaling, label encoding, distance features)
   - Neural networks (min-max scaling, label encoding, embedding features)
   - Ensemble models (all techniques, meta-features)

3. **Task Type**:
   - Binary classification (target encoding, statistical relationships)
   - Multi-class classification (one-hot encoding, clustering features)
   - Regression (scaling, outlier handling, domain-specific features)
   - Time series forecasting (lag features, rolling statistics, technical indicators)
   - Anomaly detection (clustering, outlier detection, statistical patterns)
   - Customer segmentation (clustering, RFM analysis, behavioral features)

4. **Domain-Specific Logic**:
   - Finance: Technical indicators, risk ratios, volatility measures
   - Healthcare: Vital signs, clinical scores, medical measurements
   - E-commerce: RFM analysis, customer lifetime value, purchase patterns
   - NLP: Text statistics, sentiment analysis, readability scores
   - Geospatial: Distance calculations, proximity features, coordinate systems

## 🚀 Performance Optimizations

- **Memory efficient**: Sparse matrix support where applicable
- **Parallel processing**: Multi-core support for expensive operations (clustering, statistical computations)
- **Incremental learning**: Online learning for large datasets
- **Caching**: Avoid recomputation of expensive features (cluster centroids, statistical summaries)
- **Adaptive computation**: Skip unnecessary computations based on data characteristics
- **Batch processing**: Optimized batch sizes for large datasets
- **Memory mapping**: Handle large datasets that don't fit in memory
- **GPU acceleration**: CUDA support for compatible operations (distance calculations, clustering)

---

*This documentation covers all feature engineering techniques implemented in FeatureCraft v1.0. Techniques are continuously improved and new methods may be added in future versions.*
