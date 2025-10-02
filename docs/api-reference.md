# FeatureCraft API Reference

Complete reference for FeatureCraft's Python API.

## Table of Contents

- [Core Classes](#core-classes)
  - [AutoFeatureEngineer](#autofeatureengineer)
  - [FeatureCraftConfig](#featurecraftconfig)
- [Data Classes](#data-classes)
  - [DatasetInsights](#datasetinsights)
  - [Issue](#issue)
- [Utility Functions](#utility-functions)
  - [load_config](#load_config)
  - [save_config](#save_config)
- [Reporting](#reporting)
  - [ReportBuilder](#reportbuilder)

---

## Core Classes

### AutoFeatureEngineer

The main class for automatic feature engineering.

#### Constructor

```python
AutoFeatureEngineer(config: Optional[FeatureCraftConfig] = None)
```

**Parameters:**
- `config` (FeatureCraftConfig, optional): Configuration object. If None, uses defaults.

**Example:**
```python
from featurecraft import AutoFeatureEngineer, FeatureCraftConfig

# With default config
afe = AutoFeatureEngineer()

# With custom config
config = FeatureCraftConfig(random_state=42, use_smote=True)
afe = AutoFeatureEngineer(config=config)
```

#### Methods

##### `analyze()`

Analyze dataset and generate insights report.

```python
analyze(
    df: pd.DataFrame,
    target: str,
    *,
    reference_path: Optional[str] = None,
    output_dir: Optional[str] = None
) -> DatasetInsights
```

**Parameters:**
- `df` (pd.DataFrame): Input dataset
- `target` (str): Target column name
- `reference_path` (str, optional): Path to reference dataset for drift detection
- `output_dir` (str, optional): Directory to save HTML report

**Returns:**
- `DatasetInsights`: Dataset analysis summary

**Raises:**
- `ValueError`: If target column not found or dataset is empty
- `TypeError`: If df is not a DataFrame

**Example:**
```python
import pandas as pd
from featurecraft import AutoFeatureEngineer

df = pd.read_csv("data.csv")
afe = AutoFeatureEngineer()

summary = afe.analyze(df, target="target", output_dir="artifacts")
print(f"Task: {summary.task}")
print(f"Issues: {len(summary.issues)}")
```

##### `fit()`

Fit feature engineering pipeline.

```python
fit(
    X: pd.DataFrame,
    y: pd.Series,
    estimator_family: str = "tree",
    *,
    groups: Optional[pd.Series] = None,
    config: Optional[FeatureCraftConfig] = None
) -> AutoFeatureEngineer
```

**Parameters:**
- `X` (pd.DataFrame): Feature DataFrame
- `y` (pd.Series): Target Series
- `estimator_family` (str): One of `"tree"`, `"linear"`, `"svm"`, `"knn"`, `"nn"`
- `groups` (pd.Series, optional): Group labels for GroupKFold CV
- `config` (FeatureCraftConfig, optional): Override config for this fit

**Returns:**
- `AutoFeatureEngineer`: Self for method chaining

**Raises:**
- `ValueError`: If X or y are empty
- `TypeError`: If X is not DataFrame or y is not Series

**Example:**
```python
afe = AutoFeatureEngineer()
afe.fit(X_train, y_train, estimator_family="tree")
```

##### `transform()`

Transform features using fitted pipeline.

```python
transform(X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame
```

**Parameters:**
- `X` (pd.DataFrame): Feature DataFrame to transform
- `y` (pd.Series, optional): Must be None (raises error if not)

**Returns:**
- `pd.DataFrame`: Transformed features

**Raises:**
- `ValueError`: If y is provided (leakage guard)
- `RuntimeError`: If called before fit()

**Example:**
```python
X_test_transformed = afe.transform(X_test)
```

##### `fit_transform()`

Fit pipeline and transform training data.

```python
fit_transform(
    X: pd.DataFrame,
    y: pd.Series,
    estimator_family: str = "tree",
    *,
    groups: Optional[pd.Series] = None,
    config: Optional[FeatureCraftConfig] = None
) -> pd.DataFrame
```

**Parameters:**
- Same as `fit()`

**Returns:**
- `pd.DataFrame`: Transformed training features

**Example:**
```python
X_train_transformed = afe.fit_transform(X_train, y_train, estimator_family="tree")
```

##### `export()`

Export fitted pipeline and metadata.

```python
export(output_dir: str) -> Dict[str, Any]
```

**Parameters:**
- `output_dir` (str): Directory to save artifacts

**Returns:**
- `dict`: Summary with exported file paths

**Raises:**
- `RuntimeError`: If called before fit()

**Exports:**
- `{output_dir}/pipeline.joblib`: Fitted sklearn Pipeline
- `{output_dir}/metadata.json`: Configuration and summary
- `{output_dir}/feature_names.txt`: Output feature names

**Example:**
```python
afe.fit(X_train, y_train)
afe.export("artifacts/my_pipeline")
```

##### `load()` (classmethod)

Load exported pipeline.

```python
@classmethod
load(cls, path: str) -> AutoFeatureEngineer
```

**Parameters:**
- `path` (str): Path to pipeline.joblib or directory containing it

**Returns:**
- `AutoFeatureEngineer`: Loaded instance

**Example:**
```python
afe = AutoFeatureEngineer.load("artifacts/my_pipeline/pipeline.joblib")
```

##### `get_params()`

Get configuration parameters (sklearn-compatible).

```python
get_params(deep: bool = True) -> Dict[str, Any]
```

**Parameters:**
- `deep` (bool): If True, return nested parameters

**Returns:**
- `dict`: Parameter dictionary

**Example:**
```python
params = afe.get_params()
print(params['use_smote'])
```

##### `set_params()`

Set configuration parameters (sklearn-compatible).

```python
set_params(**params) -> AutoFeatureEngineer
```

**Parameters:**
- `**params`: Parameters to set

**Returns:**
- `AutoFeatureEngineer`: Self for method chaining

**Example:**
```python
afe.set_params(use_smote=True, random_state=42)
```

##### `with_overrides()` (context manager)

Temporarily override configuration.

```python
with_overrides(**overrides) -> ContextManager[AutoFeatureEngineer]
```

**Parameters:**
- `**overrides`: Parameters to temporarily override

**Example:**
```python
afe = AutoFeatureEngineer()

with afe.with_overrides(use_smote=True, random_state=999):
    afe.fit(X_train, y_train)
# Original config restored
```

#### Properties

##### `pipeline_`

```python
@property
pipeline_ -> sklearn.pipeline.Pipeline
```

The fitted sklearn Pipeline (available after `fit()`).

**Raises:**
- `AttributeError`: If accessed before fit()

**Example:**
```python
afe.fit(X_train, y_train)
print(afe.pipeline_.named_steps)
```

##### `feature_names_out_`

```python
@property
feature_names_out_ -> List[str]
```

Output feature names after transformation.

**Raises:**
- `AttributeError`: If accessed before fit()

**Example:**
```python
afe.fit(X_train, y_train)
print(f"Output features: {afe.feature_names_out_}")
```

---

### FeatureCraftConfig

Pydantic model for configuration management.

#### Constructor

```python
FeatureCraftConfig(**kwargs)
```

**Parameters:**
- `**kwargs`: Configuration parameters (see [Configuration Guide](configuration.md))

**Example:**
```python
from featurecraft import FeatureCraftConfig

config = FeatureCraftConfig(
    random_state=42,
    low_cardinality_max=15,
    use_smote=True,
    smote_threshold=0.10
)
```

#### Key Parameters

See [Configuration Guide](configuration.md) for complete parameter list.

**General:**
- `random_state` (int): Random seed (default: 42)
- `verbosity` (int): Logging level 0-3 (default: 1)
- `artifacts_dir` (str): Output directory (default: "artifacts")

**Encoding:**
- `low_cardinality_max` (int): Max unique for one-hot (default: 10)
- `mid_cardinality_max` (int): Max unique for target encoding (default: 50)
- `use_target_encoding` (bool): Enable target encoding (default: True)

**Imbalance:**
- `use_smote` (bool): Enable SMOTE (default: False)
- `smote_threshold` (float): Minority ratio threshold (default: 0.10)

**Drift:**
- `enable_drift_detection` (bool): Enable drift detection (default: False)
- `drift_psi_threshold` (float): PSI threshold (default: 0.25)
- `drift_ks_threshold` (float): KS threshold (default: 0.10)

#### Methods

##### `model_dump()`

Export configuration as dictionary.

```python
model_dump() -> Dict[str, Any]
```

**Returns:**
- `dict`: Configuration dictionary

**Example:**
```python
config = FeatureCraftConfig(random_state=42)
print(config.model_dump())
```

##### `model_dump_json()`

Export configuration as JSON string.

```python
model_dump_json(indent: int = 2) -> str
```

**Returns:**
- `str`: JSON-formatted configuration

**Example:**
```python
config = FeatureCraftConfig(random_state=42)
with open("config.json", "w") as f:
    f.write(config.model_dump_json())
```

---

## Data Classes

### DatasetInsights

Contains dataset analysis results.

#### Attributes

- `task` (str): Detected task type (`"classification"`, `"regression"`)
- `n_samples` (int): Number of rows
- `n_features` (int): Number of features
- `n_numeric` (int): Number of numeric features
- `n_categorical` (int): Number of categorical features
- `n_datetime` (int): Number of datetime features
- `n_text` (int): Number of text features
- `target_type` (str): Target data type
- `class_balance` (Optional[Dict[str, float]]): Class distribution (classification only)
- `missing_summary` (Dict[str, float]): Missing value percentages by column
- `issues` (List[Issue]): Detected data quality issues
- `drift_report` (Optional[dict]): Drift detection results (if enabled)

**Example:**
```python
summary = afe.analyze(df, target="target")
print(f"Task: {summary.task}")
print(f"Samples: {summary.n_samples}")
print(f"Features: {summary.n_features}")
for issue in summary.issues:
    print(f"  - {issue.severity}: {issue.message}")
```

---

### Issue

Represents a data quality issue.

#### Attributes

- `column` (Optional[str]): Column name (if column-specific)
- `severity` (str): Severity level (`"ERROR"`, `"WARNING"`, `"INFO"`)
- `message` (str): Human-readable description
- `recommendation` (Optional[str]): Suggested fix

**Example:**
```python
for issue in summary.issues:
    if issue.severity == "ERROR":
        print(f"❌ {issue.column}: {issue.message}")
        if issue.recommendation:
            print(f"   Fix: {issue.recommendation}")
```

---

## Utility Functions

### load_config()

Load configuration from file.

```python
load_config(
    config_file: str,
    overrides: Optional[Dict[str, Any]] = None
) -> FeatureCraftConfig
```

**Parameters:**
- `config_file` (str): Path to YAML/JSON/TOML config file
- `overrides` (dict, optional): Override specific parameters

**Returns:**
- `FeatureCraftConfig`: Loaded configuration

**Example:**
```python
from featurecraft import load_config

config = load_config("config.yaml", overrides={"random_state": 999})
```

---

### save_config()

Save configuration to file.

```python
save_config(
    config: FeatureCraftConfig,
    output_path: str,
    format: str = "yaml"
) -> None
```

**Parameters:**
- `config` (FeatureCraftConfig): Configuration to save
- `output_path` (str): Output file path
- `format` (str): Format (`"yaml"`, `"json"`, or `"toml"`)

**Example:**
```python
from featurecraft import FeatureCraftConfig, save_config

config = FeatureCraftConfig(random_state=42, use_smote=True)
save_config(config, "my_config.yaml", format="yaml")
```

---

## Reporting

### ReportBuilder

Generate HTML reports with visualizations.

#### Constructor

```python
ReportBuilder(
    template_dir: Optional[str] = None,
    embed_figures: bool = True
)
```

**Parameters:**
- `template_dir` (str, optional): Custom template directory
- `embed_figures` (bool): Embed images in HTML (default: True)

#### Methods

##### `build()`

Generate HTML report.

```python
build(
    insights: DatasetInsights,
    output_path: str,
    open_browser: bool = False
) -> str
```

**Parameters:**
- `insights` (DatasetInsights): Analysis results
- `output_path` (str): Output HTML file path
- `open_browser` (bool): Auto-open in browser

**Returns:**
- `str`: Path to generated report

**Example:**
```python
from featurecraft import ReportBuilder

summary = afe.analyze(df, target="target")
builder = ReportBuilder()
builder.build(summary, "report.html", open_browser=True)
```

---

## Complete Example

```python
import pandas as pd
from featurecraft import (
    AutoFeatureEngineer,
    FeatureCraftConfig,
    load_config,
    save_config
)

# Load data
df = pd.read_csv("data.csv")
X = df.drop(columns=["target"])
y = df["target"]

# Create configuration
config = FeatureCraftConfig(
    random_state=42,
    low_cardinality_max=15,
    use_smote=True,
    smote_threshold=0.10,
    enable_drift_detection=True
)

# Save config for reproducibility
save_config(config, "experiment_config.yaml")

# Initialize and analyze
afe = AutoFeatureEngineer(config=config)
summary = afe.analyze(df, target="target", output_dir="artifacts")

# Fit and transform
X_train_transformed = afe.fit_transform(X, y, estimator_family="tree")

# Get output features
print(f"Output features: {afe.feature_names_out_}")

# Export pipeline
afe.export("artifacts/pipeline")

# Load pipeline later
afe_loaded = AutoFeatureEngineer.load("artifacts/pipeline/pipeline.joblib")
X_test_transformed = afe_loaded.transform(X_test)
```

---

## See Also

- [Configuration Guide](configuration.md) - Detailed parameter reference
- [Getting Started](getting-started.md) - Quick start guide
- [Advanced Features](advanced-features.md) - Drift, leakage, SHAP
- [Optimization Guide](optimization-guide.md) - Performance tuning

