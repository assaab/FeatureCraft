# Kaggle Benchmark Suite

This document describes the Kaggle benchmark suite (`02_kaggle_benchmark.py`) that tests the FeatureCraft library against top Kaggle competition solutions.

## Overview

The benchmark compares three approaches:

1. **Featurecraft (Automated)** - Fully automated feature engineering using the FeatureCraft library
2. **Kaggle Top (Manual)** - Hand-crafted features based on top Kaggle solutions
3. **Baseline** - Minimal preprocessing (imputation + scaling/encoding)

## Supported Datasets

### 1. Titanic - Binary Classification
- **Competition**: [Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic)
- **Task**: Predict passenger survival
- **Download**: `kaggle competitions download -c titanic`
- **Extract to**: `./data/titanic/train.csv`
- **Top Kaggle Features**:
  - Title extraction from Name
  - Family size (SibSp + Parch + 1)
  - IsAlone indicator
  - Age binning
  - Fare per person
  - Cabin deck extraction

### 2. House Prices - Regression
- **Competition**: [House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
- **Task**: Predict house sale prices
- **Download**: `kaggle competitions download -c house-prices-advanced-regression-techniques`
- **Extract to**: `./data/house-prices/train.csv`
- **Top Kaggle Features**:
  - Total square footage (basement + living area)
  - Age features (current year - year built)
  - Quality × Area interactions
  - Total bathrooms
  - Total porch area
  - Binary indicators (has pool, garage, basement, fireplace)

### 3. Bike Sharing Demand - Regression
- **Competition**: [Bike Sharing Demand](https://www.kaggle.com/c/bike-sharing-demand)
- **Task**: Predict bike rental demand
- **Download**: `kaggle competitions download -c bike-sharing-demand`
- **Extract to**: `./data/bike-sharing/train.csv`
- **Top Kaggle Features**:
  - Datetime extraction (hour, day, month, year, weekday)
  - Peak hour indicators
  - Weekend/rush hour flags
  - Weather × temperature interactions
  - Comfort index (temp / humidity)
  - Windchill factor

### 4. Telco Customer Churn - Binary Classification
- **Dataset**: [Telco Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn)
- **Task**: Predict customer churn
- **Download**: `kaggle datasets download -d blastchar/telco-customer-churn`
- **Extract to**: `./data/telco-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv`
- **Features**: Multiple services, contract type, payment method, charges

### 5. Credit Card Fraud Detection - Imbalanced Classification
- **Dataset**: [Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Task**: Detect fraudulent transactions
- **Download**: `kaggle datasets download -d mlg-ulb/creditcardfraud`
- **Extract to**: `./data/creditcard/creditcard.csv`
- **Note**: Highly imbalanced dataset (0.17% fraud rate)

## Installation & Setup

### 1. Install Dependencies

```bash
# Install featurecraft
pip install featurecraft

# Install Kaggle API
pip install kaggle

# Install other requirements
pip install pandas numpy scikit-learn matplotlib seaborn rich
```

### 2. Setup Kaggle API

```bash
# Place your kaggle.json in the correct location
# Linux/Mac: ~/.kaggle/kaggle.json
# Windows: C:\Users\<username>\.kaggle\kaggle.json

# Set permissions (Linux/Mac)
chmod 600 ~/.kaggle/kaggle.json
```

### 3. Download Datasets

```bash
# Create data directory
mkdir -p data

# Download Titanic
mkdir -p data/titanic
cd data/titanic
kaggle competitions download -c titanic
unzip titanic.zip
cd ../..

# Download House Prices
mkdir -p data/house-prices
cd data/house-prices
kaggle competitions download -c house-prices-advanced-regression-techniques
unzip house-prices-advanced-regression-techniques.zip
cd ../..

# Download Bike Sharing
mkdir -p data/bike-sharing
cd data/bike-sharing
kaggle competitions download -c bike-sharing-demand
unzip bike-sharing-demand.zip
cd ../..

# Download Telco Churn
mkdir -p data/telco-churn
cd data/telco-churn
kaggle datasets download -d blastchar/telco-customer-churn
unzip telco-customer-churn.zip
cd ../..

# Download Credit Card Fraud
mkdir -p data/creditcard
cd data/creditcard
kaggle datasets download -d mlg-ulb/creditcardfraud
unzip creditcardfraud.zip
cd ../..
```

## Usage

### Run All Datasets

```bash
python examples/02_kaggle_benchmark.py --datasets all
```

### Run Specific Datasets

```bash
# Single dataset
python examples/02_kaggle_benchmark.py --datasets titanic

# Multiple datasets
python examples/02_kaggle_benchmark.py --datasets titanic,house_prices,bike_sharing
```

### Custom Artifacts Directory

```bash
python examples/02_kaggle_benchmark.py --artifacts ./my_results --datasets all
```

## Output

The benchmark generates:

### 1. Console Output
- Real-time progress updates
- Feature engineering statistics
- Cross-validation scores
- Test metrics
- Comparative analysis

### 2. JSON Results
- Location: `artifacts/kaggle_benchmark/kaggle_benchmark_results.json`
- Contains detailed results for all approaches and datasets

### 3. Comparison Plots
- Location: `artifacts/kaggle_benchmark/comparison_plots.png`
- Shows:
  - Feature count by approach
  - Cross-validation scores
  - Total time comparison
  - Success rate

### 4. Pipeline Artifacts
- Location: `artifacts/kaggle_benchmark/{dataset}/featurecraft_pipeline/`
- Exported FeatureCraft pipelines for each dataset

## Understanding Results

### Metrics by Task

**Classification:**
- `cv_score`: Cross-validated ROC AUC (binary) or Accuracy (multiclass)
- `accuracy`: Test accuracy
- `f1_macro`: Macro-averaged F1 score
- `roc_auc`: Test ROC AUC
- `log_loss`: Logarithmic loss

**Regression:**
- `cv_score`: Cross-validated negative MSE
- `rmse`: Root Mean Squared Error
- `mae`: Mean Absolute Error
- `r2`: R² coefficient of determination
- `mape`: Mean Absolute Percentage Error

### Interpreting Comparisons

**Positive indicators for FeatureCraft:**
- Higher CV score than baseline/kaggle_top
- Competitive test metrics
- Reasonable feature expansion (not too many features)
- Similar or better time efficiency

**What to look for:**
- Is automated FE competitive with manual expert features?
- Does it generalize well (CV vs test performance)?
- Is the feature engineering efficient (time)?

## Expected Performance

Based on typical results:

| Dataset | Approach | Expected CV Score | Notes |
|---------|----------|-------------------|-------|
| Titanic | Baseline | ~0.82 | ROC AUC |
| Titanic | Kaggle Top | ~0.85 | ROC AUC |
| Titanic | Featurecraft | ~0.84 | ROC AUC |
| House Prices | Baseline | ~30K | RMSE |
| House Prices | Kaggle Top | ~25K | RMSE |
| House Prices | Featurecraft | ~27K | RMSE |
| Bike Sharing | Baseline | ~0.30 | R² |
| Bike Sharing | Kaggle Top | ~0.45 | R² |
| Bike Sharing | Featurecraft | ~0.40 | R² |

*Note: Actual results depend on the specific implementations and may vary.*

## Fallback Mode

If Kaggle datasets are not available, the script automatically generates synthetic datasets that mimic the structure of the real datasets. This allows testing without downloads but won't provide realistic comparisons.

To get meaningful results, download the actual Kaggle datasets.

## Troubleshooting

### Issue: "Could not find featurecraft library"
**Solution**: Install featurecraft: `pip install featurecraft`

### Issue: "Kaggle API credentials not found"
**Solution**: Setup kaggle.json as described in Setup section

### Issue: "Dataset file not found"
**Solution**: Download datasets using commands above, or script will use synthetic fallback

### Issue: High memory usage with Credit Card Fraud dataset
**Solution**: Script automatically samples to 50K rows, but you can reduce further if needed

### Issue: Slow performance
**Solution**: 
- Run fewer datasets at once
- Reduce cross-validation folds in the code
- Use synthetic datasets (faster but less realistic)

## Customization

### Add Your Own Dataset

1. Create a load function:
```python
def load_my_dataset() -> Tuple[pd.DataFrame, pd.Series, str]:
    # Load data
    df = pd.read_csv("path/to/data.csv")
    y = df['target']
    X = df.drop(columns=['target'])
    return X, y, "classification"  # or "regression"
```

2. Create Kaggle top features function:
```python
def create_kaggle_top_features_my_dataset(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    # Add your expert features
    X['new_feature'] = X['feat1'] * X['feat2']
    return X
```

3. Add to main():
```python
available_datasets = {
    'my_dataset': load_my_dataset,
    # ... existing datasets
}
```

### Modify Models

Edit the `evaluate_model()` function to use different models:

```python
# Use XGBoost instead
import xgboost as xgb
model = xgb.XGBClassifier(n_estimators=100, random_state=42)
```

## Best Practices

1. **Start Small**: Test with one dataset first (e.g., Titanic)
2. **Compare Fairly**: Use same train/test splits and random seeds
3. **Check Overfitting**: Compare CV scores vs test scores
4. **Time Constraints**: Set reasonable timeouts for large datasets
5. **Document Features**: Keep track of which manual features work best

## Citation

If you use this benchmark in research or publications, please cite:

```
FeatureCraft Kaggle Benchmark Suite
https://github.com/yourusername/Feature-Engineering-Python
```

## License

Same as the main FeatureCraft project license.

