# FeatureCraft Advanced Features

This guide covers advanced features in FeatureCraft including drift detection, leakage prevention, schema validation, and explainability.

## Table of Contents

- [Drift Detection](#drift-detection)
- [Leakage Guard](#leakage-guard)
- [Schema Validation](#schema-validation)
- [Frequency & Count Encoding](#frequency--count-encoding)
- [SHAP Explainability](#shap-explainability)

---

## Drift Detection

Monitor distribution drift between training and production data to detect data quality issues and model degradation.

### What is Drift?

**Data drift** occurs when the statistical properties of production data differ significantly from training data. This can lead to:
- Decreased model performance
- Invalid predictions
- Silent failures in production

### How It Works

FeatureCraft detects drift using:
- **PSI (Population Stability Index)** for categorical features
- **KS (Kolmogorov-Smirnov)** statistic for numeric features

### Basic Usage

```python
from featurecraft import AutoFeatureEngineer, FeatureCraftConfig

# Enable drift detection
config = FeatureCraftConfig(
    enable_drift_detection=True,
    drift_psi_threshold=0.25,      # PSI > 0.25 = significant drift
    drift_ks_threshold=0.10,       # KS > 0.10 = significant drift
)

afe = AutoFeatureEngineer(config=config)

# Analyze with drift detection
summary = afe.analyze(
    current_df,
    target="target",
    reference_path="training_data.csv"  # Path to reference dataset
)

# Check drift results
if summary.drift_report:
    for col, (score, severity) in summary.drift_report['results'].items():
        if severity in ["WARN", "CRITICAL"]:
            print(f"⚠️ {col}: {severity} (score={score:.4f})")
```

### CLI Usage

```bash
# Analyze with drift detection
featurecraft analyze \
    --input production_data.csv \
    --target target \
    --reference training_data.csv \
    --set enable_drift_detection=true \
    --out artifacts/
```

### Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enable_drift_detection` | `false` | Enable drift detection |
| `drift_psi_threshold` | `0.25` | PSI threshold for categorical drift (>0.25 = significant) |
| `drift_ks_threshold` | `0.10` | KS threshold for numeric drift (>0.1 = significant) |
| `reference_path` | `None` | Path to reference dataset (CSV/parquet) |

### Interpreting Results

**Severity Levels:**
- `OK`: No significant drift detected
- `WARN`: Moderate drift (monitor closely)
- `CRITICAL`: Significant drift (investigate immediately)

**PSI Ranges (Categorical):**
- `< 0.10`: No significant drift
- `0.10 - 0.25`: Moderate drift
- `> 0.25`: Significant drift

**KS Ranges (Numeric):**
- `< 0.05`: No significant drift
- `0.05 - 0.10`: Moderate drift
- `> 0.10`: Significant drift

### Best Practices

1. **Regular Monitoring**: Check drift weekly or monthly
2. **Set Alerts**: Trigger retraining when drift exceeds thresholds
3. **Log Drift Reports**: Track drift over time
4. **Investigate Root Causes**: High drift may indicate data quality issues
5. **Retrain Models**: Significant drift typically requires retraining

---

## Leakage Guard

Prevent target leakage in production pipelines by ensuring transformers never receive target variables during inference.

### What is Target Leakage?

**Target leakage** occurs when the model has access to information at prediction time that it wouldn't have in production. This leads to:
- Unrealistically high training performance
- Poor production performance
- Invalid model evaluation

### How It Works

FeatureCraft's `LeakageGuardMixin` raises an error if any transformer receives `y` during `transform()` (as opposed to `fit()`).

### Built-in Protection

All FeatureCraft encoders that use target information (e.g., `OutOfFoldTargetEncoder`) inherit `LeakageGuardMixin`:

```python
from featurecraft import AutoFeatureEngineer

afe = AutoFeatureEngineer()
afe.fit(X_train, y_train, estimator_family="tree")

# ✅ CORRECT: Only X passed to transform
X_test_transformed = afe.transform(X_test)

# ❌ ERROR: Raises ValueError if y is accidentally passed
try:
    X_test_transformed = afe.transform(X_test, y_test)
except ValueError as e:
    print(f"Leakage prevented: {e}")
```

### Custom Transformer with Leakage Guard

If you're building custom transformers:

```python
from sklearn.base import BaseEstimator, TransformerMixin
from featurecraft.utils.leakage import LeakageGuardMixin

class MyCustomEncoder(BaseEstimator, TransformerMixin, LeakageGuardMixin):
    def __init__(self, raise_on_target_in_transform=True):
        self.raise_on_target_in_transform = raise_on_target_in_transform
    
    def fit(self, X, y):
        # Use y for training
        self.mapping_ = self._compute_mapping(X, y)
        return self
    
    def transform(self, X, y=None):
        # Guard against accidental y usage
        self.ensure_no_target_in_transform(y)
        
        # Transform logic (without y)
        return self._apply_mapping(X)
```

### Disabling the Guard

For special cases (e.g., research pipelines):

```python
config = FeatureCraftConfig(
    raise_on_target_in_transform=False  # Disable leakage guard
)
afe = AutoFeatureEngineer(config=config)
```

⚠️ **Warning**: Only disable in controlled environments where leakage is intentional.

---

## Schema Validation

Validate that production data matches training data schema to catch data quality issues early.

### What is Schema Validation?

**Schema validation** ensures that:
- All expected columns are present
- Data types match
- No unexpected columns appear
- Value ranges are reasonable

### Usage

```python
from featurecraft.validation.schema_validator import SchemaValidator

# Create validator during training
validator = SchemaValidator()
validator.fit(X_train)

# Validate production data
try:
    validator.validate(X_prod)
    print("✅ Schema validation passed")
except ValueError as e:
    print(f"❌ Schema validation failed: {e}")
```

### AutoFeatureEngineer Integration

FeatureCraft automatically validates schemas during `transform()`:

```python
afe = AutoFeatureEngineer()
afe.fit(X_train, y_train)

# Automatically validates X_test schema
X_test_transformed = afe.transform(X_test)
```

### Validation Checks

| Check | Description |
|-------|-------------|
| **Column Presence** | All training columns exist in production data |
| **Data Types** | Column types match (int, float, object) |
| **Unexpected Columns** | Warns about columns not in training data |
| **Null Patterns** | Detects unexpected missing value patterns |

### Best Practices

1. **Validate Early**: Check schema before expensive transformations
2. **Log Failures**: Track schema violations over time
3. **Alert on Changes**: Schema changes may indicate upstream issues
4. **Version Schemas**: Save schemas alongside models

---

## Frequency & Count Encoding

Advanced categorical encoding techniques for high-cardinality features.

### What are Frequency and Count Encoders?

- **Frequency Encoding**: Maps categories to their relative frequency (0-1)
- **Count Encoding**: Maps categories to their absolute count

These encoders are particularly effective for:
- High-cardinality categoricals (>50 unique values)
- Tree-based models (XGBoost, LightGBM)
- Fraud detection and anomaly detection

### Usage

```python
from featurecraft import AutoFeatureEngineer, FeatureCraftConfig

config = FeatureCraftConfig(
    use_frequency_encoding=True,  # Enable frequency encoding
    use_count_encoding=True,      # Enable count encoding
    mid_cardinality_max=100,      # Apply to features with 10-100 unique values
)

afe = AutoFeatureEngineer(config=config)
afe.fit(X_train, y_train, estimator_family="tree")
```

### When to Use

**Frequency Encoding** works well when:
- Frequency itself is predictive (common categories → different behavior)
- Working with tree-based models
- Need to preserve ordering by popularity

**Count Encoding** works well when:
- Absolute counts matter (e.g., transaction counts)
- Sample size information is predictive
- Combined with other encoding strategies

### Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_frequency_encoding` | `false` | Enable frequency encoding |
| `use_count_encoding` | `false` | Enable count encoding |
| `low_cardinality_max` | `10` | Use one-hot encoding if ≤ this |
| `mid_cardinality_max` | `50` | Use target/freq encoding if ≤ this |
| `hashing_n_features_tabular` | `256` | Hash features if > mid_cardinality_max |

### Example: High-Cardinality Dataset

```python
# Santander or IEEE Fraud dataset
config = FeatureCraftConfig(
    low_cardinality_max=15,
    mid_cardinality_max=100,
    use_frequency_encoding=True,
    use_count_encoding=True,
    use_target_encoding=True,
)

afe = AutoFeatureEngineer(config=config)
afe.fit(X_train, y_train)
```

---

## SHAP Explainability

Generate SHAP (SHapley Additive exPlanations) values for model interpretability.

### What is SHAP?

**SHAP** explains individual predictions by computing the contribution of each feature to the prediction.

### Enable SHAP

```python
from featurecraft import AutoFeatureEngineer, FeatureCraftConfig

config = FeatureCraftConfig(
    enable_shap=True,
    shap_max_samples=100,  # Number of samples for SHAP computation
)

afe = AutoFeatureEngineer(config=config)
```

### Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enable_shap` | `false` | Enable SHAP explainability |
| `shap_max_samples` | `100` | Max samples for SHAP (higher = slower but more accurate) |

### Usage with Models

```python
from sklearn.ensemble import RandomForestClassifier
import shap

# Fit pipeline
afe = AutoFeatureEngineer(config=FeatureCraftConfig(enable_shap=True))
X_train_transformed = afe.fit_transform(X_train, y_train)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train_transformed, y_train)

# Generate SHAP explanations
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train_transformed[:100])

# Plot
shap.summary_plot(shap_values, X_train_transformed[:100])
```

### Best Practices

1. **Limit Samples**: Use `shap_max_samples` to control computation time
2. **Cache Results**: SHAP computation can be expensive
3. **Interpret Carefully**: SHAP shows feature importance, not causality
4. **Combine with Domain Knowledge**: Use SHAP to validate assumptions

---

## Complete Example: Production Pipeline

```python
from featurecraft import AutoFeatureEngineer, FeatureCraftConfig
import pandas as pd

# Production-ready configuration
config = FeatureCraftConfig(
    # Encoding
    use_frequency_encoding=True,
    use_count_encoding=True,
    use_target_encoding=True,
    low_cardinality_max=15,
    mid_cardinality_max=100,
    
    # Drift detection
    enable_drift_detection=True,
    drift_psi_threshold=0.25,
    drift_ks_threshold=0.10,
    
    # Leakage prevention
    raise_on_target_in_transform=True,
    
    # Explainability
    enable_shap=False,  # Enable in analysis, disable in production
    
    # General
    random_state=42,
    verbosity=2,
)

# Training phase
afe = AutoFeatureEngineer(config=config)
X_train_transformed = afe.fit_transform(X_train, y_train, estimator_family="tree")

# Export pipeline
afe.export("artifacts/production_pipeline")

# Production phase (later)
afe_prod = AutoFeatureEngineer.load("artifacts/production_pipeline/pipeline.joblib")

# Validate and transform
X_prod_transformed = afe_prod.transform(X_prod)  # Automatic schema validation

# Check drift (monthly)
summary = afe_prod.analyze(
    X_prod,
    target="target",
    reference_path="artifacts/training_data.csv"
)

if summary.drift_report:
    critical_drifts = [
        col for col, (score, severity) in summary.drift_report['results'].items()
        if severity == "CRITICAL"
    ]
    if critical_drifts:
        print(f"⚠️ ALERT: Critical drift detected in {len(critical_drifts)} features")
        print("Consider retraining the model")
```

---

## See Also

- [Configuration Guide](configuration.md) - All configuration parameters
- [Optimization Guide](optimization-guide.md) - Performance tuning
- [Getting Started](getting-started.md) - Basic usage
- [Benchmarks](benchmarks.md) - Real-world examples

