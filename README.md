# FeatureCraft

<div align="center">
  <p><strong>Automatic feature engineering, insights, and sklearn pipelines for tabular ML with optional time-series support.</strong></p>

  [![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
  [![PyPI version](https://badge.fury.io/py/featurecraft.svg)](https://badge.fury.io/py/featurecraft)

</div>

## Table of Contents

- [About The Project](#about-the-project)
  - [Built With](#built-with)
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
- [Contact](#contact)
- [Acknowledgments](#acknowledgments)

## About The Project

FeatureCraft is a comprehensive feature engineering library that automates the process of transforming raw tabular data into machine learning-ready features. It provides intelligent preprocessing, feature selection, encoding, and scaling tailored for different estimator families.

### Key Features

✨ **Automatic Feature Engineering**: Intelligent preprocessing pipeline that handles missing values, outliers, categorical encoding, and feature scaling

🔍 **Dataset Analysis**: Comprehensive insights into your data including distributions, correlations, and data quality issues

🔮 **Explainability & Transparency**: Detailed explanations of why and how transformations are applied, with rich console output and export capabilities

📊 **Multiple Estimator Support**: Optimized preprocessing for tree-based models, linear models, SVMs, k-NN, and neural networks

🛠️ **Sklearn Integration**: Seamless integration with scikit-learn pipelines and ecosystem

📈 **HTML Reports**: Interactive visualizations and insights reports

⚡ **CLI & Python API**: Choose between command-line interface or programmatic usage

🕐 **Time Series Support**: Optional time-series aware preprocessing

### Why FeatureCraft?

* **Automated Workflow**: No need to manually handle different data types and preprocessing steps
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
# Basic installation
pip install featurecraft

# With optional extras for enhanced functionality
pip install "featurecraft[extras]"

# Full installation with all dependencies
pip install "featurecraft[all]"
```

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

```python
import pandas as pd
from featurecraft.pipeline import AutoFeatureEngineer

# Load your data
df = pd.read_csv("your_data.csv")

# Initialize the feature engineer
afe = AutoFeatureEngineer()

# Analyze dataset (optional but recommended)
summary = afe.analyze(df, target="target_column")
print(f"Detected task: {summary.task}")
print(f"Found {len(summary.issues)} data quality issues")

# Prepare features and target
X, y = df.drop(columns=["target_column"]), df["target_column"]

# Fit and transform with estimator-specific preprocessing
Xt = afe.fit_transform(X, y, estimator_family="tree")
print(f"Transformed {X.shape[1]} features into {Xt.shape[1]} features")

# Export pipeline for production use
afe.export("artifacts")

# Access detailed explanations (optional)
explanation = afe.get_explanation()
afe.print_explanation()  # Rich console output
afe.save_explanation("artifacts/explanation.md", format="markdown")
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
    explain_transformations=True,  # Enable detailed explanations (default: True)
    explain_auto_print=True        # Auto-print explanations after fitting (default: True)
)

afe = AutoFeatureEngineer(config=config)
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

- **[01_quickstart.py](./examples/01_quickstart.py)**: Basic usage with multiple datasets
- **[02_kaggle_benchmark.py](./examples/02_kaggle_benchmark.py)**: Kaggle dataset benchmarking
- **[03_complex_kaggle_benchmark.py](./examples/03_complex_kaggle_benchmark.py)**: Advanced benchmarking with complex datasets
- **[06_explainability_demo.py](./examples/06_explainability_demo.py)**: Understanding transformation decisions and explanations

Run the quickstart example:

```bash
cd examples
python 01_quickstart.py --cases iris,wine,breast_cancer --artifacts ../artifacts
```

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
