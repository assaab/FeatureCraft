# Getting Started with FeatureCraft

**FeatureCraft** is an automated feature engineering library that handles data preprocessing, feature transformation, and encoding—all in one scikit-learn-compatible pipeline. Perfect for both beginners and experienced data scientists.

---

## 📦 Installation

Get started in seconds:

```bash
pip install featurecraft
```

For additional features (advanced encoding, drift detection):
```bash
pip install "featurecraft[extras]"
```

---

## 🚀 Quick Start (3 Steps)

### Step 1: Load Your Data

```python
import pandas as pd
from featurecraft.pipeline import AutoFeatureEngineer

df = pd.read_csv("your_data.csv")
```

### Step 2: Create and Fit Pipeline

```python
# Initialize the auto feature engineer
afe = AutoFeatureEngineer()

# Separate features and target
X = df.drop(columns=["target"])
y = df["target"]

# Fit and transform in one step
X_transformed = afe.fit_transform(X, y, estimator_family="tree")
print(f"✓ Original shape: {X.shape} → Transformed: {X_transformed.shape}")
```

### Step 3: Export Your Pipeline

```python
# Save for production use
afe.export("artifacts/")
# Creates: pipeline.joblib, metadata.json, feature_names.txt
```

**That's it!** Your data is now ML-ready with proper encoding, scaling, and transformations.

---

## 🎯 Choose Your Model Type

Different ML models need different preprocessing. FeatureCraft handles this automatically with **estimator families**:

| Family | Scaling | Best For |
|--------|---------|----------|
| `tree` | None | XGBoost, LightGBM, Random Forest, CatBoost |
| `linear` | StandardScaler | Linear/Logistic Regression, Ridge, Lasso |
| `svm` | StandardScaler | Support Vector Machines |
| `knn` | MinMaxScaler | K-Nearest Neighbors |
| `nn` | MinMaxScaler | Neural Networks |

**Example:**
```python
# For XGBoost/Random Forest
X_tree = afe.fit_transform(X, y, estimator_family="tree")

# For Logistic Regression
X_linear = afe.fit_transform(X, y, estimator_family="linear")
```

---

## 📊 Analyze Your Data First (Optional but Recommended)

Before engineering features, understand your dataset:

```python
# Run comprehensive analysis
summary = afe.analyze(df, target="target")

# See what it found
print(f"📋 Task Type: {summary.task}")  # classification or regression
print(f"⚠️  Data Issues: {len(summary.issues)}")
print(f"📈 Features: {summary.n_features}")

# Export interactive HTML report
afe.export_report("artifacts/report.html")
```

**The report includes:**
- Feature distributions and correlations
- Missing value patterns
- Outlier detection
- Feature importance estimates
- Actionable recommendations

---

## 🎨 CLI Usage (No Code Required)

Prefer command line? FeatureCraft has you covered:

```bash
# Step 1: Analyze your dataset
featurecraft analyze --input data.csv --target target --out artifacts/

# Step 2: Create and fit pipeline
featurecraft fit-transform --input data.csv --target target --out artifacts/ --estimator-family tree

# Step 3: View results
open artifacts/report.html  # macOS/Linux
start artifacts/report.html  # Windows
```

---

## ⚙️ Customize Configuration (Advanced)

Need more control? Adjust FeatureCraft's behavior:

```python
from featurecraft.config import FeatureCraftConfig

config = FeatureCraftConfig(
    low_cardinality_max=15,        # Max unique values for categorical
    outlier_share_threshold=0.1,   # Outlier detection sensitivity
    random_state=42                # Reproducibility
)

afe = AutoFeatureEngineer(config=config)
```

---

## 📁 What Gets Created?

### After `afe.export("artifacts/")`

- **`pipeline.joblib`** - Your fitted scikit-learn pipeline (load with `joblib.load()`)
- **`metadata.json`** - Configuration, feature counts, and settings
- **`feature_names.txt`** - Names of all engineered features

### After `afe.export_report("report.html")`

- **`report.html`** - Beautiful interactive report with visualizations

---

## 🎓 Complete Example

Here's everything together:

```python
import pandas as pd
from featurecraft.pipeline import AutoFeatureEngineer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# 1. Load data
df = pd.read_csv("data.csv")

# 2. Analyze (optional)
afe = AutoFeatureEngineer()
summary = afe.analyze(df, target="target")
print(f"Detected {summary.task} task")

# 3. Transform
X = df.drop(columns=["target"])
y = df["target"]
X_transformed = afe.fit_transform(X, y, estimator_family="tree")

# 4. Train model
model = RandomForestClassifier(random_state=42)
scores = cross_val_score(model, X_transformed, y, cv=5)
print(f"CV Score: {scores.mean():.3f} (+/- {scores.std():.3f})")

# 5. Save everything
afe.export("artifacts/")
```

---

## 🎉 Next Steps

Ready to dive deeper? Check out:

- **[Advanced Features](advanced-features.md)** - Time series, text encoding, drift detection
- **[API Reference](api-reference.md)** - Full documentation of all methods
- **[Configuration](configuration.md)** - Detailed configuration options
- **[CLI Reference](cli-reference.md)** - Complete command-line guide

---

**Questions or issues?** Open an issue on GitHub or check the [Troubleshooting Guide](troubleshooting.md).