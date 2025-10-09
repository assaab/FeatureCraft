# FeatureCraft

<div align="center">
  <p><strong>Automatic feature engineering with beautiful explanations - zero configuration, maximum transparency!</strong></p>
  <p>Intelligent preprocessing pipelines + automatic explanations for every decision + sklearn integration</p>

  [![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
  [![PyPI version](https://img.shields.io/pypi/v/featurecraft.svg)](https://pypi.org/project/featurecraft/)

</div>

## Quick Start

```bash
pip install featurecraft
```

```python
from featurecraft.pipeline import AutoFeatureEngineer
import pandas as pd

# Load data
df = pd.read_csv("data.csv")
X, y = df.drop("target", axis=1), df["target"]

# Fit and transform with automatic explanations!
afe = AutoFeatureEngineer()
X_transformed = afe.fit_transform(X, y, estimator_family="tree")

# Beautiful formatted explanations automatically print showing:
# - Why each transformation was chosen
# - What columns were affected
# - Configuration parameters used
# - Performance tips and recommendations
```

That's it! Zero configuration, maximum transparency.

## Table of Contents

- [Quick Start](#quick-start)
- [About The Project](#about-the-project)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
  - [CLI Usage](#cli-usage)
  - [Python API](#python-api)
  - [Estimator Families](#estimator-families)
  - [Configuration](#configuration)
  - [Output Artifacts](#output-artifacts)
- [Examples](#examples)
- [Documentation](#documentation)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)

## About The Project

FeatureCraft is a comprehensive feature engineering library that automates the process of transforming raw tabular data into machine learning-ready features. It provides intelligent preprocessing, feature selection, encoding, and scaling tailored for different estimator families.

### Key Features

✨ **Automatic Feature Engineering**: Intelligent preprocessing pipeline that handles missing values, outliers, categorical encoding, and feature scaling

🔮 **Automatic Explainability**: Beautiful, automatic explanations of every transformation decision printed to console - understand WHY each preprocessing choice was made without any extra code!

🔍 **Dataset Analysis**: Comprehensive insights into your data including distributions, correlations, and data quality issues

📊 **Multiple Estimator Support**: Optimized preprocessing for tree-based models, linear models, SVMs, k-NN, and neural networks

🛠️ **Sklearn Integration**: Seamless integration with scikit-learn pipelines and ecosystem

📈 **HTML Reports**: Interactive visualizations and insights reports

⚡ **CLI & Python API**: Choose between command-line interface or programmatic usage

🕐 **Time Series Support**: Optional time-series aware preprocessing

🤖 **AI-Powered Intelligence**: Advanced LLM-driven feature engineering with adaptive optimization

### AI-Powered Intelligence

FeatureCraft includes sophisticated AI components that leverage Large Language Models and machine learning to make intelligent feature engineering decisions:

#### 🤖 AIFeatureAdvisor
Uses Large Language Models (OpenAI, Anthropic, or local models) to analyze your dataset characteristics and recommend optimal feature engineering strategies. It considers:
- Column types and distributions
- Missing value patterns
- Outlier characteristics
- Feature interactions and relationships
- Domain-specific preprocessing requirements

#### 📋 FeatureEngineeringPlanner
Orchestrates the entire feature engineering workflow:
- Analyzes dataset characteristics
- Gets AI recommendations for optimal strategies
- Applies smart optimizations based on data patterns
- Configures preprocessing pipelines automatically

#### 🔄 AdaptiveConfigOptimizer
Learns from model performance feedback to continuously improve feature engineering strategies:
- Tracks performance metrics across different datasets
- Identifies successful patterns and configurations
- Adapts recommendations based on historical results
- Prevents overfitting through intelligent feature selection

### Why FeatureCraft?

* **Automated Workflow**: No need to manually handle different data types and preprocessing steps
* **Automatic Explanations**: See beautiful formatted explanations of every decision - enabled by default with zero extra code!
* **Best Practices**: Implements proven feature engineering techniques
* **Performance Optimized**: Different preprocessing strategies for different model types
* **Production Ready**: Exports sklearn-compatible pipelines for deployment
* **Comprehensive Analysis**: Deep insights into your dataset characteristics
* **Explainable AI**: Understand why transformations are applied and how they affect your data
* **Transparency**: Rich explanations help build trust and debug preprocessing decisions

## Getting Started

### Prerequisites

- Python 3.9 or higher
- pandas >= 1.5
- scikit-learn >= 1.3
- numpy >= 1.23

### Installation

```bash
pip install featurecraft
```

All features including AI-powered planning, enhanced encoders, SHAP explainability, and schema validation are included by default.

## Usage

### CLI Usage

```bash
# Analyze dataset and generate comprehensive report
featurecraft analyze --input data.csv --target target_column --out artifacts/

# Fit preprocessing pipeline and transform data
featurecraft fit-transform --input data.csv --target target_column --out artifacts/ --estimator-family tree

# Open the generated HTML report
open artifacts/report.html
```

### Python API

#### Basic Usage with Automatic Explanations

```python
import pandas as pd
from featurecraft.pipeline import AutoFeatureEngineer
from featurecraft.config import FeatureCraftConfig

# Load your data
df = pd.read_csv("your_data.csv")
X, y = df.drop(columns=["target_column"]), df["target_column"]

# Initialize with automatic explanations (enabled by default)
config = FeatureCraftConfig(
    explain_transformations=True,  # Enable explanations (default)
    explain_auto_print=True        # Auto-print after fit (default)
)
afe = AutoFeatureEngineer(config=config)

# Fit and transform - explanations print automatically!
Xt = afe.fit_transform(X, y, estimator_family="tree")
# Beautiful formatted explanations appear here showing:
#   - Column classifications
#   - Imputation strategies
#   - Encoding decisions
#   - Scaling choices
#   - Feature transformations
#   - And WHY each decision was made!

print(f"Transformed {X.shape[1]} features into {Xt.shape[1]} features")

# Export pipeline for production use
afe.export("artifacts")
```

#### Advanced: Analyze + Manual Explanation Control

```python
# Analyze dataset first (optional but recommended)
summary = afe.analyze(df, target="target_column")
print(f"Detected task: {summary.task}")
print(f"Found {len(summary.issues)} data quality issues")

# Fit with manual explanation control
config = FeatureCraftConfig(explain_auto_print=False)  # Disable auto-print
afe = AutoFeatureEngineer(config=config)
afe.fit(X, y, estimator_family="linear")

# Print or save explanations manually
afe.print_explanation()  # Rich console output
afe.save_explanation("artifacts/explanation.md", format="markdown")
afe.save_explanation("artifacts/explanation.json", format="json")
```

### Estimator Families

Choose the preprocessing strategy based on your model type:

| Family | Models | Scaling | Encoding | Best For |
|--------|--------|---------|----------|----------|
| `tree` | XGBoost, LightGBM, Random Forest | None | Label Encoding | Tree-based models |
| `linear` | Linear/Logistic Regression | StandardScaler | One-hot + Target | Linear models |
| `svm` | SVM, SVC | StandardScaler | One-hot | Support Vector Machines |
| `knn` | k-Nearest Neighbors | MinMaxScaler | Label Encoding | Distance-based models |
| `nn` | Neural Networks | MinMaxScaler | Label Encoding | Deep learning |

### Configuration

Customize the preprocessing behavior:

```python
from featurecraft.config import FeatureCraftConfig

# Custom configuration
config = FeatureCraftConfig(
    low_cardinality_max=15,        # Max unique values for low-cardinality features
    outlier_share_threshold=0.1,   # Threshold for outlier detection
    random_state=42,               # For reproducible results
    
    # Explainability (enabled by default!)
    explain_transformations=True,  # Enable detailed explanations (default: True)
    explain_auto_print=True,       # Auto-print after fit() (default: True)
    explain_save_path=None,        # Optional: auto-save to file path
)

afe = AutoFeatureEngineer(config=config)

# Note: You can also use AutoFeatureEngineer() without config 
# and get automatic explanations by default!
```

### Output Artifacts

The `export()` method creates:

- `pipeline.joblib`: Fitted sklearn Pipeline ready for production
- `metadata.json`: Configuration and processing summary
- `feature_names.txt`: List of all output feature names
- `explanation.md`: Human-readable explanation of transformation decisions (when explanations enabled)
- `explanation.json`: Machine-readable explanation data (when explanations enabled)

The `analyze()` method generates:

- `report.html`: Interactive HTML report with plots, insights, and recommendations

## Examples

Check out the [examples](./examples/) directory for comprehensive usage examples:

- **[01_quickstart.py](./examples/01_quickstart.py)**: Basic usage with automatic explanations on Iris and Wine datasets
- **[02_kaggle_benchmark.py](./examples/02_kaggle_benchmark.py)**: Kaggle dataset benchmarking
- **[03_complex_kaggle_benchmark.py](./examples/03_complex_kaggle_benchmark.py)**: Advanced benchmarking with complex datasets
- **[06_explainability_demo.py](./examples/06_explainability_demo.py)**: Deep dive into explanation features and export formats

Run the quickstart example to see automatic explanations in action:

```bash
python examples/01_quickstart.py
```

This will demonstrate automatic feature engineering with beautiful formatted explanations for the Iris and Wine datasets!

## Documentation

### Getting Started
- **[Getting Started](./docs/getting-started.md)**: Quick start guide for new users
- **[API Reference](./docs/api-reference.md)**: Complete Python API documentation
- **[CLI Reference](./docs/cli-reference.md)**: Command-line interface guide

### Configuration & Optimization
- **[Configuration Guide](./docs/configuration.md)**: Comprehensive parameter reference
- **[Optimization Guide](./docs/optimization-guide.md)**: Performance tuning and best practices

### Advanced Topics
- **[Advanced Features](./docs/advanced-features.md)**: Drift detection, leakage prevention, SHAP
- **[Benchmarks](./docs/benchmarks.md)**: Real-world dataset performance results
- **[Troubleshooting](./docs/troubleshooting.md)**: Common issues and solutions

## Roadmap

See the [open issues](https://github.com/featurecraft/featurecraft/issues) for a list of proposed features and known issues.

- [ ] Enhanced time series preprocessing
- [ ] Feature selection algorithms
- [ ] Integration with popular ML frameworks
- [ ] GPU acceleration support
- [ ] Advanced outlier detection methods
- [ ] Automated feature interaction detection

## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".

Don't forget to give the project a star! Thanks again!


## License

Distributed under the MIT License. See `LICENSE` for more information.

---

<div align="center">
  <p><strong>Built with ❤️ for the machine learning community</strong></p>
</div>
