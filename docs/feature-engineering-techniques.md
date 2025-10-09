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
| **Text Statistics** | `TextStatisticsExtractor` | Basic text metrics (char count, word count, etc.) | Custom |
| **Bag-of-Words** | `CountVectorizer` | Document classification | scikit-learn |
| **TF-IDF** | `TfidfVectorizer` | Keyword extraction | scikit-learn |
| **N-grams** | `CountVectorizer` (1,2) or (1,3) | Capture phrases | scikit-learn |
| **Hashing Vectorization** | `HashingVectorizer` | High-dimensional text data | scikit-learn |
| **Named Entity Recognition** | `NERFeatureExtractor` | Extract entities (persons, orgs, locations) | spaCy |
| **Readability Scores** | `ReadabilityScoreExtractor` | Text complexity (Flesch-Kincaid, SMOG) | textstat |
| **Sentiment Analysis** | Polarity scoring | Opinion mining | TextBlob |
| **Topic Modeling** | `LatentDirichletAllocation` | Document clustering | scikit-learn |
| **Text Dimensionality Reduction** | `TruncatedSVD` | Reduce text feature dimensions | scikit-learn |

## 🔢 Mathematical Transformations

| Technique | Method | Use Case | Library |
|-----------|--------|----------|---------|
| **Numeric Conversion** | `NumericConverter` | Mixed type column conversion | Custom |
| **Power Transformation** | `SkewedPowerTransformer` | Handle skewed distributions | Custom |
| **Log Transformation** | `LogTransformer` | Normalize right-skewed data | Custom |
| **Log1p Transformation** | `Log1pTransformer` | Stabilize variance for small values | Custom |
| **Square Root Transformation** | `SqrtTransformer` | Moderate skewness reduction | Custom |
| **Reciprocal Transformation** | `ReciprocalTransformer` | Handle inverse relationships | Custom |
| **Box-Cox Transformation** | `BoxCoxTransformer` | Normalize distributions | Custom |
| **Exponential Transformation** | `ExponentialTransformer` | Create polynomial features | Custom |
| **Yeo-Johnson Transformation** | `YeoJohnsonWrapper` | Normalize non-positive data | Custom |
| **Mathematical Combinations** | `MathematicalTransformer` | Intelligent auto-selection | Custom |

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
| **Lag Features** | `make_lags` | Temporal dependencies | Custom |
| **Rolling Statistics** | `make_rolling` | Moving averages and statistics | Custom |
| **Datetime Feature Extraction** | `DateTimeFeatures` | Comprehensive date/time components | Custom |
| **Cyclical Encoding** | Sin/Cos transforms | Periodic patterns | Custom |
| **Seasonality Features** | Season extraction | Seasonal patterns | Custom |
| **Business Logic Features** | Business hour/day detection | Business context | Custom |
| **Relative Time Features** | Days since reference | Temporal relationships | Custom |

## 🎯 Feature Selection Techniques

| Technique | Method | Use Case | Library |
|-----------|--------|----------|---------|
| **Mutual Information Selection** | `MutualInfoSelector` | Non-linear feature relationships | scikit-learn |
| **Correlation-based Pruning** | `prune_correlated` | Remove highly correlated features | Custom |
| **Variance Inflation Factor** | `compute_vif_drop` | Detect multicollinearity | Custom |
| **Dimensionality Reduction** | `DimensionalityReducer` | Reduce feature space | Custom |

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

## 🏗️ Advanced Techniques

### Binning/Discretization
- **Multiple Strategies**: `BinningTransformer`
  - Equal width binning
  - Equal frequency binning
  - K-means clustering binning
  - Decision tree binning
  - Quantile-based binning

### Outlier Handling
- **Winsorization**: `WinsorizerTransformer`
- **Percentile-based clipping**: `WinsorizerTransformer`
- **Z-score based detection**: Built-in outlier detection

### Categorical Cleaning
- **Normalization**: `CategoricalCleaner`
- **Missing value handling**: `CategoricalMissingIndicator`
- **String standardization**: Built-in cleaning functions

## 📈 Pipeline Integration

All techniques are designed to work seamlessly with:
- **Scikit-learn pipelines**
- **Feature unions for parallel processing**
- **Automatic hyperparameter tuning**
- **Cross-validation compatibility**

## 🔍 Technique Selection Logic

FeatureCraft automatically selects appropriate techniques based on:

1. **Data Characteristics**:
   - Column types (numeric, categorical, text, datetime)
   - Missing value patterns
   - Cardinality levels
   - Distribution shapes

2. **Estimator Family**:
   - Tree-based models (no scaling, label encoding)
   - Linear models (standard scaling, one-hot encoding)
   - SVM models (standard scaling, one-hot encoding)
   - KNN models (min-max scaling, label encoding)
   - Neural networks (min-max scaling, label encoding)

3. **Task Type**:
   - Binary classification
   - Multi-class classification
   - Regression
   - Time series forecasting

## 🚀 Performance Optimizations

- **Memory efficient**: Sparse matrix support where applicable
- **Parallel processing**: Multi-core support for expensive operations
- **Incremental learning**: Online learning for large datasets
- **Caching**: Avoid recomputation of expensive features

---

*This documentation covers all feature engineering techniques implemented in FeatureCraft v1.0. Techniques are continuously improved and new methods may be added in future versions.*
