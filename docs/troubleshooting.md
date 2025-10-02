# FeatureCraft Troubleshooting Guide

Common issues and solutions for FeatureCraft users.

## Table of Contents

- [Installation Issues](#installation-issues)
- [Data Loading & Validation](#data-loading--validation)
- [Configuration Problems](#configuration-problems)
- [Pipeline Errors](#pipeline-errors)
- [Memory & Performance](#memory--performance)
- [Encoding Issues](#encoding-issues)
- [Drift Detection](#drift-detection)
- [Export & Import](#export--import)
- [Integration Issues](#integration-issues)

---

## Installation Issues

### Issue: `pip install featurecraft` fails

**Symptoms:**
```
ERROR: Could not find a version that satisfies the requirement featurecraft
```

**Solutions:**

1. **Update pip:**
```bash
python -m pip install --upgrade pip
```

2. **Check Python version:**
```bash
python --version  # Must be 3.9 or higher
```

3. **Try with specific index:**
```bash
pip install featurecraft --index-url https://pypi.org/simple/
```

4. **Install from source:**
```bash
git clone https://github.com/featurecraft/featurecraft.git
cd featurecraft
pip install -e .
```

---

### Issue: Import error after installation

**Symptoms:**
```python
ImportError: cannot import name 'AutoFeatureEngineer' from 'featurecraft'
```

**Solutions:**

1. **Check installation:**
```bash
pip show featurecraft
```

2. **Verify correct Python environment:**
```bash
which python
pip list | grep featurecraft
```

3. **Reinstall:**
```bash
pip uninstall featurecraft
pip install featurecraft
```

---

### Issue: Missing optional dependencies

**Symptoms:**
```
ModuleNotFoundError: No module named 'imbalanced_learn'
```

**Solution:**

Install extras:
```bash
# For SMOTE and imbalance handling
pip install "featurecraft[extras]"

# For all optional features
pip install "featurecraft[all]"
```

---

## Data Loading & Validation

### Issue: "X must be a pandas DataFrame"

**Symptoms:**
```python
TypeError: X must be a pandas DataFrame, got <class 'numpy.ndarray'>
```

**Solution:**

Convert to DataFrame:
```python
import pandas as pd
import numpy as np

# If you have numpy array
X_array = np.array([[1, 2, 3], [4, 5, 6]])
X_df = pd.DataFrame(X_array, columns=['feat1', 'feat2', 'feat3'])

# Now this works
afe.fit(X_df, y)
```

---

### Issue: "Target column not found"

**Symptoms:**
```python
ValueError: Column 'target' not found in DataFrame
```

**Solutions:**

1. **Check column name (case-sensitive):**
```python
print(df.columns.tolist())
# If you see 'Target' instead of 'target', use:
summary = afe.analyze(df, target='Target')
```

2. **Check for whitespace:**
```python
df.columns = df.columns.str.strip()  # Remove leading/trailing spaces
```

3. **Verify column exists:**
```python
assert 'target' in df.columns, f"Available columns: {df.columns.tolist()}"
```

---

### Issue: Empty DataFrame error

**Symptoms:**
```python
ValueError: X cannot be empty
```

**Solution:**

Check for empty data:
```python
print(f"Shape: {X.shape}")
print(f"Is empty: {X.empty}")

# Remove empty rows
X = X.dropna(how='all')
y = y[X.index]
```

---

### Issue: Index mismatch between X and y

**Symptoms:**
```python
ValueError: X and y must have the same index
```

**Solution:**

Reset and align indices:
```python
# Option 1: Reset indices
X = X.reset_index(drop=True)
y = y.reset_index(drop=True)

# Option 2: Align on index
X, y = X.align(y, join='inner', axis=0)

# Option 3: Use loc with common index
common_idx = X.index.intersection(y.index)
X = X.loc[common_idx]
y = y.loc[common_idx]
```

---

## Configuration Problems

### Issue: Configuration file not found

**Symptoms:**
```bash
FileNotFoundError: Config file not found: config.yaml
```

**Solutions:**

1. **Use absolute path:**
```bash
featurecraft fit --config /full/path/to/config.yaml --input data.csv --target target
```

2. **Verify file exists:**
```bash
ls -l config.yaml
pwd  # Check current directory
```

3. **Check file extension:**
```bash
# Ensure .yaml, not .yml (or vice versa)
mv config.yml config.yaml
```

---

### Issue: Invalid configuration parameter

**Symptoms:**
```python
ValidationError: Invalid value for 'smote_threshold': must be between 0.0 and 1.0
```

**Solution:**

Check parameter constraints:
```python
from featurecraft import FeatureCraftConfig

# This will show validation error with details
try:
    config = FeatureCraftConfig(smote_threshold=1.5)
except Exception as e:
    print(f"Error: {e}")

# Correct: value must be in [0.0, 1.0]
config = FeatureCraftConfig(smote_threshold=0.15)
```

---

### Issue: Environment variables not loading

**Symptoms:**
```python
# Set FEATURECRAFT__RANDOM_STATE=42 but config still uses default
```

**Solutions:**

1. **Check prefix (double underscore):**
```bash
# Correct
export FEATURECRAFT__RANDOM_STATE=42

# Incorrect (single underscore)
export FEATURECRAFT_RANDOM_STATE=42
```

2. **Verify environment variable is set:**
```bash
echo $FEATURECRAFT__RANDOM_STATE
```

3. **Check boolean values:**
```bash
# Correct
export FEATURECRAFT__USE_SMOTE=true

# Incorrect (Python syntax)
export FEATURECRAFT__USE_SMOTE=True
```

---

## Pipeline Errors

### Issue: "Pipeline not fitted"

**Symptoms:**
```python
RuntimeError: Call fit() before transform()
```

**Solution:**

Ensure fit is called first:
```python
afe = AutoFeatureEngineer()
# ❌ This will fail
# X_transformed = afe.transform(X_test)

# ✅ Correct order
afe.fit(X_train, y_train)
X_transformed = afe.transform(X_test)
```

---

### Issue: Target leakage error

**Symptoms:**
```python
ValueError: OutOfFoldTargetEncoder.transform() received target variable (y=Series). 
This is a potential label leakage risk.
```

**Solution:**

Don't pass y to transform:
```python
# ❌ Incorrect (causes leakage)
X_test_transformed = afe.transform(X_test, y_test)

# ✅ Correct (no leakage)
X_test_transformed = afe.transform(X_test)
```

If intentional (e.g., research), disable guard:
```python
config = FeatureCraftConfig(raise_on_target_in_transform=False)
afe = AutoFeatureEngineer(config=config)
```

---

### Issue: Feature names mismatch

**Symptoms:**
```python
ValueError: Feature names seen during fit differ from transform
```

**Solutions:**

1. **Check column names:**
```python
print("Training columns:", X_train.columns.tolist())
print("Test columns:", X_test.columns.tolist())

# Ensure same columns in same order
X_test = X_test[X_train.columns]
```

2. **Handle missing columns:**
```python
# Add missing columns with defaults
for col in X_train.columns:
    if col not in X_test.columns:
        X_test[col] = 0  # or np.nan
```

3. **Remove extra columns:**
```python
# Keep only training columns
X_test = X_test[X_train.columns]
```

---

### Issue: SMOTE error with too few samples

**Symptoms:**
```python
ValueError: Expected n_neighbors <= n_samples, but n_neighbors = 5, n_samples = 3
```

**Solutions:**

1. **Reduce k_neighbors:**
```python
config = FeatureCraftConfig(
    use_smote=True,
    smote_k_neighbors=2  # Reduce from default 5
)
```

2. **Disable SMOTE for small datasets:**
```python
if len(y) < 10:
    config = FeatureCraftConfig(use_smote=False)
else:
    config = FeatureCraftConfig(use_smote=True)
```

---

## Memory & Performance

### Issue: Out of memory error

**Symptoms:**
```python
MemoryError: Unable to allocate array
```

**Solutions:**

1. **Sample data:**
```python
config = FeatureCraftConfig(
    sample_frac=0.5,  # Use 50% of data
    random_state=42
)
```

2. **Reduce text features:**
```python
config = FeatureCraftConfig(
    tfidf_max_features=5000,  # Reduce from default 20000
    svd_components_for_trees=50
)
```

3. **Disable expensive features:**
```python
config = FeatureCraftConfig(
    use_smote=False,  # SMOTE can be memory-intensive
    enable_shap=False,
    reducer_kind=None
)
```

4. **Process in chunks (for inference):**
```python
chunk_size = 10000
for i in range(0, len(X_test), chunk_size):
    X_chunk = X_test.iloc[i:i+chunk_size]
    X_chunk_transformed = afe.transform(X_chunk)
    # Process chunk...
```

---

### Issue: Slow performance

**Symptoms:**
Pipeline takes hours to fit.

**Solutions:**

1. **Reduce CV folds:**
```python
config = FeatureCraftConfig(
    cv_n_splits=3  # Reduce from default 5
)
```

2. **Disable expensive operations:**
```python
config = FeatureCraftConfig(
    use_target_encoding=False,  # Target encoding is slow
    use_mi=False,  # Mutual information is slow
    enable_shap=False
)
```

3. **Sample for development:**
```python
# For testing, use small sample
config = FeatureCraftConfig(sample_n=10000)
```

4. **Profile to find bottlenecks:**
```python
import time

start = time.time()
afe.fit(X_train, y_train)
print(f"Fit time: {time.time() - start:.2f}s")
```

---

## Encoding Issues

### Issue: High cardinality warning

**Symptoms:**
```
WARNING: Column 'user_id' has 10000 unique values (high cardinality)
```

**Solutions:**

1. **Use hashing:**
```python
config = FeatureCraftConfig(
    mid_cardinality_max=100,
    hashing_n_features_tabular=512
)
```

2. **Use frequency encoding:**
```python
config = FeatureCraftConfig(
    use_frequency_encoding=True,
    use_count_encoding=True
)
```

3. **Drop high-cardinality columns:**
```python
X = X.drop(columns=['user_id', 'session_id'])
```

---

### Issue: Unknown categories in test set

**Symptoms:**
```python
Warning: Found unknown categories in column 'city': ['NewCity']
```

**Solution:**

This is handled automatically by FeatureCraft:
```python
# Default behavior handles unknown categories
config = FeatureCraftConfig(
    ohe_handle_unknown="infrequent_if_exist"  # Handles unseen categories
)
```

No action needed unless you want to customize:
```python
# Use prior for unseen categories in target encoding
config = FeatureCraftConfig(
    use_target_encoding=True,
    te_smoothing=50.0  # Higher = more regularization
)
```

---

## Drift Detection

### Issue: Drift detection not running

**Symptoms:**
```python
summary.drift_report is None
```

**Solutions:**

1. **Enable drift detection:**
```python
config = FeatureCraftConfig(enable_drift_detection=True)
summary = afe.analyze(df, target='target', reference_path='train.csv')
```

2. **Verify reference file exists:**
```python
import os
assert os.path.exists('train.csv'), "Reference file not found"
```

3. **Check reference_path parameter:**
```python
# Must pass reference_path to analyze()
summary = afe.analyze(
    current_df,
    target='target',
    reference_path='reference.csv'  # Must be provided
)
```

---

### Issue: High drift detected

**Symptoms:**
```
WARNING: 15 features show CRITICAL drift
```

**Solutions:**

1. **Investigate root cause:**
```python
if summary.drift_report:
    for col, (score, severity) in summary.drift_report['results'].items():
        if severity == 'CRITICAL':
            print(f"{col}: score={score:.4f}")
            # Check distributions
            print(f"  Training: {X_train[col].describe()}")
            print(f"  Production: {X_prod[col].describe()}")
```

2. **Adjust thresholds:**
```python
# Less sensitive
config = FeatureCraftConfig(
    drift_psi_threshold=0.35,  # Increase from 0.25
    drift_ks_threshold=0.15    # Increase from 0.10
)
```

3. **Retrain model:**
If drift is real (not threshold issue), retrain:
```python
# Combine old and new data
X_combined = pd.concat([X_train, X_prod], ignore_index=True)
y_combined = pd.concat([y_train, y_prod], ignore_index=True)

# Retrain
afe_new = AutoFeatureEngineer()
afe_new.fit(X_combined, y_combined)
```

---

## Export & Import

### Issue: Pipeline export fails

**Symptoms:**
```python
RuntimeError: Cannot export pipeline before fitting
```

**Solution:**

Ensure fit is called first:
```python
afe = AutoFeatureEngineer()
afe.fit(X_train, y_train)  # Must fit first
afe.export("artifacts/")   # Now this works
```

---

### Issue: Loading pipeline fails

**Symptoms:**
```python
FileNotFoundError: Pipeline file not found
```

**Solutions:**

1. **Check file path:**
```python
import os
pipeline_path = "artifacts/pipeline.joblib"
assert os.path.exists(pipeline_path), f"File not found: {pipeline_path}"

afe = AutoFeatureEngineer.load(pipeline_path)
```

2. **Verify joblib compatibility:**
```python
import joblib
print(f"Joblib version: {joblib.__version__}")

# If version mismatch, try:
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
afe = AutoFeatureEngineer.load(pipeline_path)
```

---

### Issue: Version mismatch warning

**Symptoms:**
```
UserWarning: Trying to unpickle estimator trained with scikit-learn 1.3 but running 1.4
```

**Solution:**

This is usually safe but test thoroughly:
```python
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

afe = AutoFeatureEngineer.load("pipeline.joblib")

# Verify it works
X_test_transformed = afe.transform(X_test)
assert X_test_transformed.shape[0] == X_test.shape[0], "Transform failed"
```

If issues persist, retrain with current versions.

---

## Integration Issues

### Issue: Sklearn GridSearchCV compatibility

**Symptoms:**
```python
AttributeError: AutoFeatureEngineer has no attribute 'get_params'
```

**Solution:**

FeatureCraft implements sklearn interface:
```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

# Create pipeline
pipeline = Pipeline([
    ('features', afe),
    ('model', RandomForestClassifier())
])

# Grid search works
param_grid = {
    'features__use_smote': [True, False],
    'model__n_estimators': [100, 200]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=3)
grid_search.fit(X_train, y_train)
```

---

### Issue: Pandas version compatibility

**Symptoms:**
```python
AttributeError: 'DataFrame' object has no attribute 'iteritems'
```

**Solution:**

Update pandas:
```bash
pip install --upgrade "pandas>=1.5,<3.0"
```

Or downgrade if needed:
```bash
pip install "pandas>=1.5,<2.0"
```

---

## Getting Help

If you can't find a solution here:

1. **Check documentation:**
   - [Getting Started](getting-started.md)
   - [Configuration Guide](configuration.md)
   - [API Reference](api-reference.md)

2. **Search existing issues:**
   - [GitHub Issues](https://github.com/featurecraft/featurecraft/issues)

3. **Create a minimal reproducible example:**
```python
import pandas as pd
from featurecraft import AutoFeatureEngineer

# Minimal example that reproduces the issue
X = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
y = pd.Series([0, 1, 0])

afe = AutoFeatureEngineer()
# ... your code that fails ...
```

4. **Open an issue:**
   - Include error message
   - Include Python version: `python --version`
   - Include FeatureCraft version: `pip show featurecraft`
   - Include minimal reproducible example

---

## See Also

- [Configuration Guide](configuration.md) - All parameters
- [API Reference](api-reference.md) - Complete API
- [CLI Reference](cli-reference.md) - Command-line usage
- [Advanced Features](advanced-features.md) - Drift, leakage, SHAP

