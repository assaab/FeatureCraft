# FeatureCraft Configuration Guide

FeatureCraft provides a comprehensive, layered configuration system that allows you to customize every aspect of the feature engineering pipeline.

## Configuration Sources & Precedence

Configuration can be provided through multiple sources with the following precedence (highest to lowest):

1. **CLI `--set` overrides** (highest priority)
2. **Explicit config file** (YAML/TOML/JSON via `--config`)
3. **Environment variables** (`FEATURECRAFT__*`)
4. **Python API kwargs**
5. **Library defaults** (lowest priority)

## Configuration Methods

### 1. Python API

```python
from featurecraft import FeatureCraftConfig, AutoFeatureEngineer

# Direct instantiation
cfg = FeatureCraftConfig(
    low_cardinality_max=15,
    use_smote=True,
    smote_threshold=0.10,
    random_state=42,
)

afe = AutoFeatureEngineer(config=cfg)
afe.fit(X_train, y_train, estimator_family="tree")
```

### 2. Config File (YAML/JSON/TOML)

```yaml
# featurecraft-config.yaml
random_state: 42
artifacts_dir: "my_artifacts"

# Encoding
low_cardinality_max: 12
mid_cardinality_max: 60
rare_level_threshold: 0.02
use_target_encoding: true

# Imbalance
use_smote: true
smote_threshold: 0.10
smote_k_neighbors: 5

# Text
text_use_hashing: false
tfidf_max_features: 10000

# Reducers
reducer_kind: "pca"
reducer_components: 50
```

Load in Python:
```python
from featurecraft.settings import load_config

cfg = load_config(config_file="featurecraft-config.yaml")
afe = AutoFeatureEngineer(config=cfg)
```

Or via CLI:
```bash
featurecraft fit --config featurecraft-config.yaml \
    --input train.csv --target label --out artifacts/
```

### 3. Environment Variables

```bash
export FEATURECRAFT__RANDOM_STATE=42
export FEATURECRAFT__LOW_CARDINALITY_MAX=15
export FEATURECRAFT__USE_SMOTE=true
export FEATURECRAFT__SMOTE_THRESHOLD=0.10
```

Nested keys use double underscores:
```bash
export FEATURECRAFT__REDUCER__KIND=pca
export FEATURECRAFT__REDUCER__COMPONENTS=50
```

### 4. CLI Flags

```bash
featurecraft fit \
    --input train.csv \
    --target label \
    --set low_cardinality_max=12 \
    --set use_smote=true \
    --set smote_threshold=0.10 \
    --set reducer_kind=pca \
    --random-state 42 \
    --out artifacts/
```

### 5. Interactive Wizard

```bash
featurecraft wizard --output my-config.yaml
```

The wizard will prompt you for key configuration options and generate a YAML config file.

## Parameter Categories

### General Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `random_state` | int | 42 | Random seed for reproducibility |
| `verbosity` | int | 1 | Logging verbosity (0-3) |
| `artifacts_dir` | str | "artifacts" | Output directory for artifacts |
| `dry_run` | bool | false | Dry run mode (no file writes) |
| `fail_fast` | bool | false | Stop on first error |

### Missing Values

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `numeric_simple_impute_max` | float | 0.05 | Max missingness for simple imputation |
| `numeric_advanced_impute_max` | float | 0.30 | Max missingness for iterative imputation |
| `categorical_impute_strategy` | str | "most_frequent" | Categorical imputation strategy |
| `add_missing_indicators` | bool | true | Add binary missing indicators |

### Encoding

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `low_cardinality_max` | int | 10 | Max unique values for one-hot encoding |
| `mid_cardinality_max` | int | 50 | Max unique values for target encoding |
| `rare_level_threshold` | float | 0.01 | Frequency threshold for grouping rare categories |
| `ohe_handle_unknown` | str | "infrequent_if_exist" | How OHE handles unknown categories |
| `hashing_n_features_tabular` | int | 256 | Hash features for high-cardinality categoricals |
| `use_target_encoding` | bool | true | Enable target encoding |
| `use_leave_one_out_te` | bool | false | Use Leave-One-Out TE instead of K-Fold |
| `target_encoding_noise` | float | 0.01 | Gaussian noise for target encoding |
| `target_encoding_smoothing` | float | 0.3 | Smoothing factor for target encoding |
| `use_ordinal` | bool | false | Use ordinal encoding |
| `ordinal_maps` | dict | {} | Manual ordinal category ordering |
| `use_woe` | bool | false | Use Weight of Evidence encoding |

### Scaling & Transforms

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `skew_threshold` | float | 1.0 | Absolute skewness threshold for power transforms |
| `outlier_share_threshold` | float | 0.05 | Outlier fraction threshold for robust scaling |
| `scaler_linear` | str | "standard" | Scaler for linear models |
| `scaler_svm` | str | "standard" | Scaler for SVM |
| `scaler_knn` | str | "minmax" | Scaler for k-NN |
| `scaler_nn` | str | "minmax" | Scaler for neural networks |
| `scaler_tree` | str | "none" | Scaler for tree models |
| `scaler_robust_if_outliers` | bool | true | Auto-use RobustScaler if outliers detected |
| `winsorize` | bool | false | Apply winsorization to clip outliers |
| `clip_percentiles` | tuple | (0.01, 0.99) | Percentiles for clipping |

### Feature Selection

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `corr_drop_threshold` | float | 0.95 | Correlation threshold for dropping features |
| `vif_drop_threshold` | float | 10.0 | VIF threshold for multicollinearity |
| `use_mi` | bool | false | Use mutual information for selection |
| `mi_top_k` | int | None | Keep top K features by MI |

### Text Processing

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tfidf_max_features` | int | 20000 | Max TF-IDF features |
| `ngram_range` | tuple | (1, 2) | N-gram range for text |
| `text_use_hashing` | bool | false | Use HashingVectorizer instead of TF-IDF |
| `text_hashing_features` | int | 16384 | Hash features for text |
| `text_char_ngrams` | bool | false | Use character n-grams |
| `svd_components_for_trees` | int | 200 | SVD components for text with trees |

### Time Series & Datetime

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ts_default_lags` | list[int] | [1, 7, 28] | Default lag periods |
| `ts_default_windows` | list[int] | [3, 7, 28] | Default rolling window sizes |
| `use_fourier` | bool | false | Add Fourier features for cyclical patterns |
| `fourier_orders` | list[int] | [3, 7] | Fourier series orders |
| `holiday_country` | str | None | ISO country code for holidays (e.g., 'US') |
| `time_column` | str | None | Name of time/date column |
| `time_order` | str | None | Column to sort by for time-ordered ops |

### Dimensionality Reduction

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `reducer_kind` | str | None | Reducer type: 'pca', 'svd', 'umap', or None |
| `reducer_components` | int | None | Number of components |
| `reducer_variance` | float | None | Explained variance ratio for PCA |

### Imbalance Handling

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_smote` | bool | false | Enable SMOTE oversampling |
| `smote_threshold` | float | 0.10 | Minority ratio threshold to trigger SMOTE |
| `smote_k_neighbors` | int | 5 | Number of neighbors for SMOTE |
| `smote_strategy` | str | "auto" | SMOTE sampling strategy |
| `use_undersample` | bool | false | Enable random undersampling |
| `class_weight_threshold` | float | 0.20 | Minority ratio for class_weight advisory |

### Drift Detection

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_drift_detection` | bool | false | Enable drift detection |
| `drift_psi_threshold` | float | 0.25 | PSI threshold for categorical drift |
| `drift_ks_threshold` | float | 0.10 | KS threshold for numeric drift |
| `reference_path` | str | None | Path to reference dataset |

### Explainability

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_shap` | bool | false | Enable SHAP explainability |
| `shap_max_samples` | int | 100 | Max samples for SHAP computation |

### Sampling & Cross-Validation

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sample_n` | int | None | Fixed number of samples |
| `sample_frac` | float | None | Fraction of samples |
| `stratify_by` | str | None | Column for stratified sampling |
| `cv_n_splits` | int | 5 | Number of CV folds |
| `use_group_kfold` | bool | false | Use GroupKFold CV |
| `groups_column` | str | None | Column for group-based CV |

### Reporting

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `template_dir` | str | None | Custom templates directory |
| `embed_figures` | bool | true | Embed figures in HTML reports |
| `open_report` | bool | false | Auto-open report in browser |
| `report_filename` | str | "report.html" | Report filename |
| `max_corr_features` | int | 60 | Max features in correlation heatmap |

## Advanced Usage

### Runtime Config Override

```python
afe = AutoFeatureEngineer()

# Override at fit time
runtime_cfg = FeatureCraftConfig(use_smote=True, smote_threshold=0.15)
afe.fit(X, y, config=runtime_cfg)
```

### Context Manager for Temporary Overrides

```python
afe = AutoFeatureEngineer()

# Temporary override
with afe.with_overrides(use_smote=True, smote_threshold=0.08):
    afe.fit(X_train, y_train)
# Original config restored after context
```

### Sklearn-Style Parameter Access

```python
afe = AutoFeatureEngineer()

# Get parameters
params = afe.get_params(deep=True)
print(params['use_smote'])

# Set parameters
afe.set_params(use_smote=True, low_cardinality_max=15)
afe.fit(X, y)
```

### Print Effective Configuration

```bash
# Show merged config as YAML
featurecraft print-config --format yaml

# With overrides
featurecraft print-config \
    --config my-config.yaml \
    --set use_smote=true \
    --format yaml

# Export JSON Schema
featurecraft print-config --schema
```

## Interactive Mode

Enable interactive prompts during CLI operations:

```bash
# Interactive fit with consent-driven prompts
featurecraft fit --interactive \
    --input train.csv --target label --out artifacts/

# The CLI will ask:
# - Enable SMOTE for detected imbalance?
# - Choose imputation strategy for high missingness?
# - Set hashing dimensions for high-cardinality columns?
```

Interactive mode only prompts when relevant issues are detected.

## Configuration File Examples

### Minimal Config

```yaml
random_state: 42
artifacts_dir: "my_output"
```

### Production Config for Tree Models

```yaml
# General
random_state: 42
verbosity: 2
artifacts_dir: "prod_artifacts"

# Encoding - aggressive for high-cardinality data
low_cardinality_max: 15
mid_cardinality_max: 100
rare_level_threshold: 0.005
use_target_encoding: true

# No scaling for tree models
scaler_tree: "none"

# Handle imbalance
use_smote: false  # Better to use class_weight in trees
class_weight_threshold: 0.15

# Reduce high-dimensional text
text_use_hashing: false
tfidf_max_features: 5000
svd_components_for_trees: 100

# Reporting
open_report: false
embed_figures: true
```

### Production Config for Linear Models

```yaml
# General
random_state: 42
verbosity: 1
artifacts_dir: "linear_artifacts"

# Encoding - conservative for interpretability
low_cardinality_max: 10
mid_cardinality_max: 30
use_target_encoding: false

# Scaling mandatory for linear
scaler_linear: "standard"
scaler_robust_if_outliers: true
winsorize: true
clip_percentiles: [0.01, 0.99]

# Handle imbalance with SMOTE
use_smote: true
smote_threshold: 0.15
smote_k_neighbors: 5

# Feature selection
corr_drop_threshold: 0.90
vif_drop_threshold: 5.0
use_mi: true
mi_top_k: 50

# Dimensionality reduction
reducer_kind: "pca"
reducer_variance: 0.95
```

## Validation & Schema

All configuration parameters are validated using Pydantic:

```python
from featurecraft import FeatureCraftConfig

# Invalid configuration raises validation error
try:
    cfg = FeatureCraftConfig(smote_threshold=1.5)  # Out of range
except Exception as e:
    print(f"Validation error: {e}")

# Export JSON Schema
schema = FeatureCraftConfig.model_json_schema()
print(schema)
```

## Environment Variable Reference

All parameters can be set via environment variables with the `FEATURECRAFT__` prefix:

```bash
# General
export FEATURECRAFT__RANDOM_STATE=42
export FEATURECRAFT__VERBOSITY=2

# Encoding
export FEATURECRAFT__LOW_CARDINALITY_MAX=15
export FEATURECRAFT__USE_TARGET_ENCODING=true

# SMOTE
export FEATURECRAFT__USE_SMOTE=true
export FEATURECRAFT__SMOTE_THRESHOLD=0.10

# Text
export FEATURECRAFT__TEXT_USE_HASHING=true
export FEATURECRAFT__TEXT_HASHING_FEATURES=8192

# Reducers
export FEATURECRAFT__REDUCER_KIND=pca
export FEATURECRAFT__REDUCER_COMPONENTS=50
```

## Troubleshooting

### Configuration Not Applied

Check precedence order - CLI overrides take highest priority:
```bash
# This --set will override any config file setting
featurecraft fit --config my.yaml --set use_smote=false ...
```

### Invalid Parameter Values

Use `print-config --schema` to see valid ranges:
```bash
featurecraft print-config --schema | jq '.properties.smote_threshold'
```

### Config File Not Found

Ensure the path is correct:
```bash
# Relative to current directory
featurecraft fit --config ./configs/prod.yaml ...

# Absolute path
featurecraft fit --config /path/to/config.yaml ...
```

### Environment Variables Not Loading

Check the prefix and nesting:
```bash
# Correct
export FEATURECRAFT__USE_SMOTE=true

# Incorrect (missing double underscore after prefix)
export FEATURECRAFT_USE_SMOTE=true
```

## Best Practices

1. **Use config files for production** - More maintainable than CLI flags
2. **Version control configs** - Track configuration changes alongside code
3. **Use environment variables for secrets** - Don't commit credentials
4. **Test configurations** - Use `--dry-run` to validate before running
5. **Document overrides** - Comment why specific values are chosen
6. **Start with defaults** - Only override what you need
7. **Use wizard for discovery** - Generate initial config with `featurecraft wizard`

## See Also

- [Getting Started Guide](getting-started.md)
- [Optimization Guide](optimization-guide.md)
- [Benchmarks](benchmarks.md)
- [API Reference](../README.md#python-api)
- [CLI Reference](../README.md#cli-interface)
