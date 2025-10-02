"""
Kaggle Benchmark Suite for FeatureCraft
========================================

Advanced benchmarking script comparing automated feature engineering against 
expert-level manual feature engineering from top Kaggle competition solutions.

Datasets Supported:
-------------------
1. Titanic - Binary classification
2. House Prices - Regression  
3. Bike Sharing Demand - Regression
4. Telco Customer Churn - Binary classification
5. Credit Card Fraud Detection - Imbalanced binary classification

Comparison Approaches:
---------------------
- Featurecraft: Automated feature engineering pipeline
- Kaggle Top: Manual feature engineering from top solutions
- Baseline: Minimal preprocessing only

Note: Datasets must be downloaded separately and placed in ./data/
"""

# ============================================================================
# IMPORTS
# ============================================================================

import matplotlib
matplotlib.use('Agg')

import sys
import json
import time
import logging
import warnings
import argparse
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score, mean_squared_error,
    mean_absolute_error, r2_score, log_loss, mean_absolute_percentage_error
)
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

try:
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.panel import Panel
except ImportError:
    print("Installing rich for better console output...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "rich"])
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.panel import Panel


# ============================================================================
# CONFIGURATION
# ============================================================================

warnings.filterwarnings('ignore')
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
console = Console()


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class KaggleBenchmarkResult:
    """Result container for Kaggle dataset benchmark."""
    dataset: str
    approach: str  # 'featurecraft', 'kaggle_top', 'baseline'
    task: str
    n_rows: int
    n_features_in: int
    n_features_out: int
    fit_time: float
    transform_time: float
    train_time: float
    cv_score: float
    cv_std: float
    test_score: Dict[str, float]
    feature_engineering_details: str
    status: str = "success"
    error: Optional[str] = None


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def find_library() -> Tuple[Any, Any]:
    """Discover the feature engineering library."""
    try:
        from featurecraft.pipeline import AutoFeatureEngineer as MainClass
        import featurecraft as module
        logger.info(f"[OK] Found library: featurecraft.pipeline.AutoFeatureEngineer")
        return module, MainClass
    except ImportError:
        raise RuntimeError(
            "Could not find featurecraft library. Please install it first: pip install featurecraft"
        )


# ============================================================================
# DATASET LOADERS
# ============================================================================

def load_titanic() -> Tuple[pd.DataFrame, pd.Series, str]:
    """
    Load Titanic dataset from Kaggle or seaborn (binary classification).
    
    Expected path: ./data/titanic/train.csv
    Fallback: seaborn's built-in titanic dataset
    """
    train_path = Path("./data/titanic/train.csv")
    
    if train_path.exists():
        df = pd.read_csv(train_path)
        logger.info(f"[OK] Loaded Kaggle Titanic dataset: {df.shape}")
    else:
        # Fallback to seaborn's built-in dataset
        df = sns.load_dataset('titanic')
        logger.info(f"[OK] Loaded seaborn Titanic dataset: {df.shape}")
    
    # Prepare features and target
    target_col = 'survived' if 'survived' in df.columns else 'Survived'
    y = df[target_col].copy()
    
    # Drop target and ID columns
    drop_cols = [target_col]
    if 'PassengerId' in df.columns:
        drop_cols.append('PassengerId')
    X = df.drop(columns=drop_cols)
    
    # Remove non-informative columns
    cols_to_drop = [col for col in ['deck', 'embark_town', 'alive'] if col in X.columns]
    X = X.drop(columns=cols_to_drop, errors='ignore')
    
    return X, y, "classification"


def load_house_prices() -> Tuple[pd.DataFrame, pd.Series, str]:
    """
    Load House Prices dataset (regression).
    
    Expected path: ./data/house-prices/train.csv
    Download from: https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data
    """
    train_path = Path("./data/house-prices/train.csv")
    
    if not train_path.exists():
        raise FileNotFoundError(
            f"House Prices dataset not found at {train_path}. "
            "Please download from: https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data"
        )
    
    df = pd.read_csv(train_path)
    logger.info(f"[OK] Loaded House Prices dataset: {df.shape}")
    
    # Prepare features and target
    y = df['SalePrice'].copy()
    X = df.drop(columns=['SalePrice', 'Id'])
    
    return X, y, "regression"


def load_bike_sharing() -> Tuple[pd.DataFrame, pd.Series, str]:
    """
    Load Bike Sharing Demand dataset (regression).
    
    Expected path: ./data/bike-sharing/train.csv
    Download from: https://www.kaggle.com/c/bike-sharing-demand/data
    """
    train_path = Path("./data/bike-sharing/train.csv")
    
    if not train_path.exists():
        raise FileNotFoundError(
            f"Bike Sharing dataset not found at {train_path}. "
            "Please download from: https://www.kaggle.com/c/bike-sharing-demand/data"
        )
    
    df = pd.read_csv(train_path)
    logger.info(f"[OK] Loaded Bike Sharing dataset: {df.shape}")
    
    # Prepare features and target
    target_col = 'count' if 'count' in df.columns else 'cnt'
    y = df[target_col].copy()
    
    # Drop target and leakage columns
    drop_cols = [target_col]
    for col in ['casual', 'registered']:
        if col in df.columns:
            drop_cols.append(col)
    X = df.drop(columns=drop_cols)
    
    return X, y, "regression"


def load_telco_churn() -> Tuple[pd.DataFrame, pd.Series, str]:
    """
    Load Telco Customer Churn dataset (binary classification).
    
    Expected path: ./data/telco-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv
    Download from: https://www.kaggle.com/blastchar/telco-customer-churn
    """
    data_path = Path("./data/telco-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    
    if not data_path.exists():
        raise FileNotFoundError(
            f"Telco Churn dataset not found at {data_path}. "
            "Please download from: https://www.kaggle.com/blastchar/telco-customer-churn"
        )
    
    df = pd.read_csv(data_path)
    logger.info(f"[OK] Loaded Telco Churn dataset: {df.shape}")
    
    # Prepare target
    target_col = 'Churn'
    y = df[target_col].map({'Yes': 1, 'No': 0}) if df[target_col].dtype == 'object' else df[target_col]
    
    # Drop target and ID columns
    drop_cols = [target_col]
    if 'customerID' in df.columns:
        drop_cols.append('customerID')
    X = df.drop(columns=drop_cols)
    
    # Handle TotalCharges if it's stored as string
    if 'TotalCharges' in X.columns and X['TotalCharges'].dtype == 'object':
        X['TotalCharges'] = pd.to_numeric(X['TotalCharges'], errors='coerce')
    
    return X, y, "classification"


def load_credit_card_fraud() -> Tuple[pd.DataFrame, pd.Series, str]:
    """
    Load Credit Card Fraud Detection dataset (imbalanced classification).
    
    Expected path: ./data/creditcard/creditcard.csv
    Download from: https://www.kaggle.com/mlg-ulb/creditcardfraud
    
    Note: Large dataset is sampled to 50,000 rows for faster benchmarking.
    """
    data_path = Path("./data/creditcard/creditcard.csv")
    
    if not data_path.exists():
        raise FileNotFoundError(
            f"Credit Card Fraud dataset not found at {data_path}. "
            "Please download from: https://www.kaggle.com/mlg-ulb/creditcardfraud"
        )
    
    df = pd.read_csv(data_path)
    
    # Sample to reduce size for faster benchmarking
    if len(df) > 50000:
        df = df.sample(n=50000, random_state=42)
        logger.info(f"[OK] Loaded Credit Card Fraud dataset (sampled): {df.shape}")
    else:
        logger.info(f"[OK] Loaded Credit Card Fraud dataset: {df.shape}")
    
    # Prepare features and target
    y = df['Class'].copy()
    X = df.drop(columns=['Class'])
    
    return X, y, "classification"


# ============================================================================
# MANUAL FEATURE ENGINEERING (Kaggle Top Solutions)
# ============================================================================

def create_kaggle_top_features_titanic(X: pd.DataFrame) -> pd.DataFrame:
    """
    Manual feature engineering based on top Kaggle Titanic solutions.
    
    Top solutions typically include:
    - Title extraction from Name
    - Family size features
    - Fare per person
    - Age binning
    - Cabin deck extraction
    """
    X = X.copy()
    
    # Extract title from name if exists
    if 'Name' in X.columns or 'name' in X.columns:
        name_col = 'Name' if 'Name' in X.columns else 'name'
        X['Title'] = X[name_col].str.extract(r' ([A-Za-z]+)\.', expand=False)
        # Group rare titles
        rare_titles = ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']
        X['Title'] = X['Title'].replace(rare_titles, 'Rare')
        X['Title'] = X['Title'].replace(['Mlle', 'Ms'], 'Miss')
        X['Title'] = X['Title'].replace('Mme', 'Mrs')
        X = X.drop(columns=[name_col])
    
    # Family size features
    if 'SibSp' in X.columns and 'Parch' in X.columns:
        X['FamilySize'] = X['SibSp'] + X['Parch'] + 1
        X['IsAlone'] = (X['FamilySize'] == 1).astype(int)
    elif 'sibsp' in X.columns and 'parch' in X.columns:
        X['FamilySize'] = X['sibsp'] + X['parch'] + 1
        X['IsAlone'] = (X['FamilySize'] == 1).astype(int)
    
    # Age binning
    age_col = 'Age' if 'Age' in X.columns else 'age'
    if age_col in X.columns:
        X['Age_bin'] = pd.cut(X[age_col], bins=[0, 12, 18, 35, 60, 100], labels=['Child', 'Teen', 'Adult', 'Middle', 'Senior'])
    
    # Fare per person
    fare_col = 'Fare' if 'Fare' in X.columns else 'fare'
    if fare_col in X.columns and 'FamilySize' in X.columns:
        X['Fare_Per_Person'] = X[fare_col] / X['FamilySize']
    
    # Cabin deck
    cabin_col = 'Cabin' if 'Cabin' in X.columns else 'cabin'
    if cabin_col in X.columns:
        X['Cabin_Deck'] = X[cabin_col].str[0]
        X = X.drop(columns=[cabin_col])
    
    # Drop ticket if exists
    X = X.drop(columns=[col for col in ['Ticket', 'ticket'] if col in X.columns], errors='ignore')
    
    return X


def create_kaggle_top_features_house_prices(X: pd.DataFrame) -> pd.DataFrame:
    """
    Manual feature engineering based on top Kaggle House Prices solutions.
    
    Top solutions typically include:
    - Total square footage
    - Age features
    - Quality × condition interactions
    - Polynomial features for important variables
    """
    X = X.copy()
    
    # Total square footage (if columns exist)
    if 'TotalBsmtSF' in X.columns and 'GrLivArea' in X.columns:
        X['TotalSF'] = X['TotalBsmtSF'] + X['GrLivArea']
    
    # Age features
    if 'YearBuilt' in X.columns:
        X['Age'] = 2024 - X['YearBuilt']
        X = X.drop(columns=['YearBuilt'])
    
    if 'YearRemodAdd' in X.columns:
        X['RemodAge'] = 2024 - X['YearRemodAdd']
        X = X.drop(columns=['YearRemodAdd'])
    
    # Quality interactions
    if 'OverallQual' in X.columns and 'GrLivArea' in X.columns:
        X['QualArea'] = X['OverallQual'] * X['GrLivArea']
    
    # Bathrooms total
    if 'FullBath' in X.columns and 'HalfBath' in X.columns:
        X['TotalBath'] = X['FullBath'] + 0.5 * X['HalfBath']
    
    if 'BsmtFullBath' in X.columns and 'BsmtHalfBath' in X.columns:
        X['TotalBath'] = X.get('TotalBath', 0) + X['BsmtFullBath'] + 0.5 * X['BsmtHalfBath']
    
    # Porch total
    porch_cols = ['OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch']
    existing_porch = [col for col in porch_cols if col in X.columns]
    if existing_porch:
        X['TotalPorchSF'] = X[existing_porch].sum(axis=1)
    
    # Has pool, garage, basement, fireplace
    if 'PoolArea' in X.columns:
        X['HasPool'] = (X['PoolArea'] > 0).astype(int)
    
    if 'GarageArea' in X.columns:
        X['HasGarage'] = (X['GarageArea'] > 0).astype(int)
    
    if 'TotalBsmtSF' in X.columns:
        X['HasBsmt'] = (X['TotalBsmtSF'] > 0).astype(int)
    
    if 'Fireplaces' in X.columns:
        X['HasFireplace'] = (X['Fireplaces'] > 0).astype(int)
    
    return X


def create_kaggle_top_features_bike_sharing(X: pd.DataFrame) -> pd.DataFrame:
    """
    Manual feature engineering based on top Kaggle Bike Sharing solutions.
    
    Top solutions typically include:
    - Datetime feature extraction (hour, day, month, year, weekday)
    - Peak hour indicators
    - Weather interactions with temperature
    """
    X = X.copy()
    
    # Datetime features
    if 'datetime' in X.columns:
        X['datetime'] = pd.to_datetime(X['datetime'])
        X['hour'] = X['datetime'].dt.hour
        X['day'] = X['datetime'].dt.day
        X['month'] = X['datetime'].dt.month
        X['year'] = X['datetime'].dt.year
        X['dayofweek'] = X['datetime'].dt.dayofweek
        X['is_weekend'] = (X['dayofweek'] >= 5).astype(int)
        
        # Peak hours
        X['is_rush_hour'] = X['hour'].isin([7, 8, 9, 17, 18, 19]).astype(int)
        X['is_night'] = X['hour'].isin([0, 1, 2, 3, 4, 5, 6]).astype(int)
        
        X = X.drop(columns=['datetime'])
    
    # Weather interactions
    if 'weather' in X.columns and 'temp' in X.columns:
        X['weather_temp'] = X['weather'] * X['temp']
    
    # Comfort index
    if 'temp' in X.columns and 'humidity' in X.columns:
        X['comfort_index'] = X['temp'] / (X['humidity'] + 1)
    
    # Windchill
    if 'temp' in X.columns and 'windspeed' in X.columns:
        X['windchill'] = X['temp'] - X['windspeed'] * 0.5
    
    return X


# ============================================================================
# BASELINE PREPROCESSING
# ============================================================================

def create_baseline_pipeline(X: pd.DataFrame, y: pd.Series, task: str) -> Pipeline:
    """
    Create a simple baseline pipeline with minimal preprocessing.
    """
    # Identify numeric and categorical columns
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Numeric pipeline
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical pipeline
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor


# ============================================================================
# BENCHMARK APPROACHES
# ============================================================================

def run_featurecraft_approach(
    EngineClass: Any,
    X: pd.DataFrame,
    y: pd.Series,
    task: str,
    artifacts_dir: Path
) -> Dict[str, Any]:
    """Run featurecraft automated feature engineering."""
    result = {
        'n_features_in': X.shape[1],
        'n_features_out': X.shape[1],
        'fit_time': 0.0,
        'transform_time': 0.0,
        'status': 'success',
        'error': None,
        'details': 'Automated feature engineering with featurecraft'
    }
    
    try:
        # Initialize
        afe = EngineClass()
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42,
            stratify=y if task == "classification" else None
        )
        
        # Fit
        start = time.time()
        afe.fit(X_train, y_train)
        result['fit_time'] = time.time() - start
        
        # Transform
        start = time.time()
        Xt_train = afe.transform(X_train)
        Xt_test = afe.transform(X_test)
        result['transform_time'] = time.time() - start
        
        result['n_features_out'] = Xt_train.shape[1]
        result['X_train'] = Xt_train
        result['X_test'] = Xt_test
        result['y_train'] = y_train
        result['y_test'] = y_test
        
        # Export pipeline
        try:
            export_dir = artifacts_dir / "featurecraft_pipeline"
            export_dir.mkdir(exist_ok=True, parents=True)
            if hasattr(afe, 'export'):
                afe.export(str(export_dir))
        except Exception as e:
            logger.debug(f"Export failed: {e}")
        
    except Exception as e:
        result['status'] = 'failed'
        result['error'] = str(e)
        logger.error(f"Featurecraft failed: {e}")
        traceback.print_exc()
        
        # Fallback to original data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42,
            stratify=y if task == "classification" else None
        )
        result['X_train'] = X_train
        result['X_test'] = X_test
        result['y_train'] = y_train
        result['y_test'] = y_test
    
    return result


def run_kaggle_top_approach(
    X: pd.DataFrame,
    y: pd.Series,
    task: str,
    dataset_name: str
) -> Dict[str, Any]:
    """Run manual feature engineering based on top Kaggle solutions."""
    result = {
        'n_features_in': X.shape[1],
        'n_features_out': X.shape[1],
        'fit_time': 0.0,
        'transform_time': 0.0,
        'status': 'success',
        'error': None,
        'details': 'Manual feature engineering from top Kaggle solutions'
    }
    
    try:
        # Apply dataset-specific feature engineering
        start = time.time()
        if dataset_name == 'titanic':
            X_engineered = create_kaggle_top_features_titanic(X)
        elif dataset_name == 'house_prices':
            X_engineered = create_kaggle_top_features_house_prices(X)
        elif dataset_name == 'bike_sharing':
            X_engineered = create_kaggle_top_features_bike_sharing(X)
        else:
            # Generic processing
            X_engineered = X.copy()
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X_engineered, y, test_size=0.2, random_state=42,
            stratify=y if task == "classification" else None
        )
        
        # Apply baseline preprocessing
        preprocessor = create_baseline_pipeline(X_train, y_train, task)
        
        # Fit
        preprocessor.fit(X_train, y_train)
        result['fit_time'] = time.time() - start
        
        # Transform
        start = time.time()
        Xt_train = preprocessor.transform(X_train)
        Xt_test = preprocessor.transform(X_test)
        result['transform_time'] = time.time() - start
        
        result['n_features_out'] = Xt_train.shape[1]
        result['X_train'] = Xt_train
        result['X_test'] = Xt_test
        result['y_train'] = y_train
        result['y_test'] = y_test
        
    except Exception as e:
        result['status'] = 'failed'
        result['error'] = str(e)
        logger.error(f"Kaggle top approach failed: {e}")
        traceback.print_exc()
        
        # Fallback
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42,
            stratify=y if task == "classification" else None
        )
        preprocessor = create_baseline_pipeline(X_train, y_train, task)
        preprocessor.fit(X_train, y_train)
        Xt_train = preprocessor.transform(X_train)
        Xt_test = preprocessor.transform(X_test)
        
        result['X_train'] = Xt_train
        result['X_test'] = Xt_test
        result['y_train'] = y_train
        result['y_test'] = y_test
    
    return result


def run_baseline_approach(
    X: pd.DataFrame,
    y: pd.Series,
    task: str
) -> Dict[str, Any]:
    """Run baseline with minimal preprocessing."""
    result = {
        'n_features_in': X.shape[1],
        'n_features_out': X.shape[1],
        'fit_time': 0.0,
        'transform_time': 0.0,
        'status': 'success',
        'error': None,
        'details': 'Minimal preprocessing (imputation + scaling/encoding)'
    }
    
    try:
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42,
            stratify=y if task == "classification" else None
        )
        
        # Create and fit preprocessor
        start = time.time()
        preprocessor = create_baseline_pipeline(X_train, y_train, task)
        preprocessor.fit(X_train, y_train)
        result['fit_time'] = time.time() - start
        
        # Transform
        start = time.time()
        Xt_train = preprocessor.transform(X_train)
        Xt_test = preprocessor.transform(X_test)
        result['transform_time'] = time.time() - start
        
        result['n_features_out'] = Xt_train.shape[1]
        result['X_train'] = Xt_train
        result['X_test'] = Xt_test
        result['y_train'] = y_train
        result['y_test'] = y_test
        
    except Exception as e:
        result['status'] = 'failed'
        result['error'] = str(e)
        logger.error(f"Baseline approach failed: {e}")
    
    return result


# ============================================================================
# MODEL EVALUATION
# ============================================================================

def evaluate_model(
    X_train: np.ndarray,
    y_train: pd.Series,
    X_test: np.ndarray,
    y_test: pd.Series,
    task: str
) -> Tuple[float, float, Dict[str, float], float]:
    """
    Train and evaluate model with cross-validation.
    
    Returns: (cv_score, cv_std, test_metrics, train_time)
    """
    # Select model based on task
    if task == "classification":
        # Check class balance
        class_counts = pd.Series(y_train).value_counts()
        is_imbalanced = (class_counts.min() / class_counts.max()) < 0.1
        
        if is_imbalanced:
            model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        else:
            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    else:
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    
    # Cross-validation
    start = time.time()
    if task == "classification":
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc' if len(np.unique(y_train)) == 2 else 'accuracy', n_jobs=-1)
    else:
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
        cv_scores = -cv_scores  # Make positive
    
    cv_score = cv_scores.mean()
    cv_std = cv_scores.std()
    
    # Train on full training set
    model.fit(X_train, y_train)
    train_time = time.time() - start
    
    # Test metrics
    y_pred = model.predict(X_test)
    test_metrics = {}
    
    if task == "classification":
        test_metrics['accuracy'] = accuracy_score(y_test, y_pred)
        test_metrics['f1_macro'] = f1_score(y_test, y_pred, average='macro')
        
        if len(np.unique(y_train)) == 2:
            y_proba = model.predict_proba(X_test)[:, 1]
            test_metrics['roc_auc'] = roc_auc_score(y_test, y_proba)
            test_metrics['log_loss'] = log_loss(y_test, y_proba)
    else:
        test_metrics['rmse'] = np.sqrt(mean_squared_error(y_test, y_pred))
        test_metrics['mae'] = mean_absolute_error(y_test, y_pred)
        test_metrics['r2'] = r2_score(y_test, y_pred)
        
        # MAPE (avoid division by zero)
        mask = y_test != 0
        if mask.sum() > 0:
            test_metrics['mape'] = mean_absolute_percentage_error(y_test[mask], y_pred[mask])
    
    return cv_score, cv_std, test_metrics, train_time


# ============================================================================
# VISUALIZATION
# ============================================================================

def create_comparison_plots(results: List[KaggleBenchmarkResult], artifacts_dir: Path):
    """Create comparison plots for different approaches."""
    try:
        # Convert to DataFrame for easier plotting
        df = pd.DataFrame([asdict(r) for r in results])
        
        # Plot 1: Feature count comparison
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Feature engineering effectiveness
        ax = axes[0, 0]
        pivot = df.pivot_table(values='n_features_out', index='dataset', columns='approach', aggfunc='mean')
        pivot.plot(kind='bar', ax=ax, rot=45)
        ax.set_title('Feature Count by Approach', fontsize=14, fontweight='bold')
        ax.set_ylabel('Number of Features')
        ax.legend(title='Approach')
        ax.grid(axis='y', alpha=0.3)
        
        # Performance comparison (CV score)
        ax = axes[0, 1]
        pivot = df.pivot_table(values='cv_score', index='dataset', columns='approach', aggfunc='mean')
        pivot.plot(kind='bar', ax=ax, rot=45)
        ax.set_title('Cross-Validation Score by Approach', fontsize=14, fontweight='bold')
        ax.set_ylabel('CV Score')
        ax.legend(title='Approach')
        ax.grid(axis='y', alpha=0.3)
        
        # Time comparison
        ax = axes[1, 0]
        df['total_time'] = df['fit_time'] + df['transform_time'] + df['train_time']
        pivot = df.pivot_table(values='total_time', index='dataset', columns='approach', aggfunc='mean')
        pivot.plot(kind='bar', ax=ax, rot=45)
        ax.set_title('Total Time by Approach', fontsize=14, fontweight='bold')
        ax.set_ylabel('Time (seconds)')
        ax.legend(title='Approach')
        ax.grid(axis='y', alpha=0.3)
        
        # Success rate
        ax = axes[1, 1]
        success_rate = df.groupby('approach')['status'].apply(lambda x: (x == 'success').sum() / len(x) * 100)
        success_rate.plot(kind='bar', ax=ax, color=['green', 'blue', 'orange'])
        ax.set_title('Success Rate by Approach', fontsize=14, fontweight='bold')
        ax.set_ylabel('Success Rate (%)')
        ax.set_ylim(0, 105)
        ax.grid(axis='y', alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=0)
        
        plt.tight_layout()
        plot_path = artifacts_dir / "comparison_plots.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"[OK] Comparison plots saved to {plot_path}")
        
    except Exception as e:
        logger.warning(f"Failed to create comparison plots: {e}")


# ============================================================================
# MAIN ORCHESTRATION
# ============================================================================

def main():
    """
    Main orchestration function for running Kaggle benchmarks.
    
    Compares three approaches across multiple datasets:
    - Featurecraft (automated)
    - Kaggle Top (manual expert solutions)
    - Baseline (minimal preprocessing)
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="FeatureCraft Kaggle Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--artifacts", 
        type=str, 
        default="./artifacts/kaggle_benchmark",
        help="Directory to save benchmark artifacts"
    )
    parser.add_argument(
        "--datasets", 
        type=str, 
        default="all",
        help="Comma-separated dataset names or 'all' (titanic,house_prices,bike_sharing,telco_churn,credit_card_fraud)"
    )
    args = parser.parse_args()
    
    # Setup artifacts directory
    artifacts_base = Path(args.artifacts)
    artifacts_base.mkdir(parents=True, exist_ok=True)
    
    # Display header
    console.print("\n" + "="*80)
    console.print(Panel.fit(
        "[bold blue]FeatureCraft Kaggle Benchmark Suite[/bold blue]\n"
        "[yellow]Comparing Automated vs Expert Feature Engineering[/yellow]",
        border_style="blue"
    ))
    console.print("="*80 + "\n")
    
    # Discover FeatureCraft library
    try:
        module, EngineClass = find_library()
        console.print(f"[green][OK][/green] Library found: {EngineClass.__name__}\n")
    except RuntimeError as e:
        console.print(f"[red][FAIL][/red] {e}")
        sys.exit(1)
    
    # Define available datasets and their loaders
    available_datasets = {
        'titanic': load_titanic,
        'house_prices': load_house_prices,
        'bike_sharing': load_bike_sharing,
        'telco_churn': load_telco_churn,
        'credit_card_fraud': load_credit_card_fraud
    }
    
    # Select datasets to benchmark
    if args.datasets == "all":
        selected_datasets = list(available_datasets.keys())
    else:
        selected_datasets = [d.strip() for d in args.datasets.split(",")]
    
    # Display configuration
    console.print(f"[cyan]Datasets:[/cyan] {', '.join(selected_datasets)}")
    console.print(f"[cyan]Approaches:[/cyan] Featurecraft (automated), Kaggle Top (manual), Baseline")
    console.print()
    
    # Run benchmarks across all selected datasets
    all_results: List[KaggleBenchmarkResult] = []
    
    # Remove Progress spinner and use plain prints for progress
    for dataset_name in selected_datasets:
        # Validate dataset name
        if dataset_name not in available_datasets:
            console.print(f"[red][FAIL][/red] Unknown dataset: {dataset_name}")
            continue
        # Display dataset header
        console.print(f"\n{'='*70}")
        console.print(f"Dataset: {dataset_name.upper()}")
        console.print(f"{'='*70}\n")
        try:
            print(f"[INFO] Loading {dataset_name}...")
            X, y, task = available_datasets[dataset_name]()
            # Display dataset info
            console.print(f"[green][OK][/green] Loaded: {X.shape[0]} rows, "
                f"{X.shape[1]} features, task={task}")
            console.print(
                f"  [OK] Target distribution: {dict(pd.Series(y).value_counts().head())}\n"
            )
            # Create dataset artifacts directory
            artifacts_dir = artifacts_base / dataset_name
            artifacts_dir.mkdir(exist_ok=True, parents=True)
            # Test all three approaches
            approaches = [
                ('baseline', lambda: run_baseline_approach(X, y, task)),
                ('kaggle_top', lambda: run_kaggle_top_approach(X, y, task, dataset_name)),
                ('featurecraft', lambda: run_featurecraft_approach(EngineClass, X, y, task, artifacts_dir))
            ]
            for approach_name, approach_func in approaches:
                console.print(f"-> Testing: {approach_name}")
                try:
                    print(f"[INFO] Feature engineering ({approach_name})...")
                    fe_result = approach_func()
                    # Check for failures
                    if fe_result['status'] == 'failed':
                        console.print(f"  [red][FAIL][/red] {fe_result['error']}\n")
                        continue
                    # Display feature engineering results
                    console.print(
                        f"  [green][OK][/green] Features: "
                        f"{fe_result['n_features_in']}  {fe_result['n_features_out']}"
                    )
                    console.print(
                        f"  [green][OK][/green] Time: "
                        f"fit={fe_result['fit_time']:.2f}s, "
                        f"transform={fe_result['transform_time']:.2f}s"
                    )
                    # Train and evaluate model
                    print(f"[INFO] Training and evaluating...")
                    cv_score, cv_std, test_metrics, train_time = evaluate_model(
                        fe_result['X_train'],
                        fe_result['y_train'],
                        fe_result['X_test'],
                        fe_result['y_test'],
                        task
                    )
                    # Display evaluation results
                    console.print(f"  [green][OK][/green] CV Score: {cv_score:.4f} (±{cv_std:.4f})")
                    test_metrics_str = ', '.join([f'{k}={v:.4f}' for k, v in test_metrics.items()])
                    console.print(f"  [green][OK][/green] Test metrics: {test_metrics_str}")
                    console.print(f"  [green][OK][/green] Train time: {train_time:.2f}s\n")
                    # Store result
                    result = KaggleBenchmarkResult(
                        dataset=dataset_name,
                        approach=approach_name,
                        task=task,
                        n_rows=len(X),
                        n_features_in=fe_result['n_features_in'],
                        n_features_out=fe_result['n_features_out'],
                        fit_time=fe_result['fit_time'],
                        transform_time=fe_result['transform_time'],
                        train_time=train_time,
                        cv_score=cv_score,
                        cv_std=cv_std,
                        test_score=test_metrics,
                        feature_engineering_details=fe_result['details'],
                        status='success'
                    )
                    all_results.append(result)
                except Exception as e:
                    console.print(f"  [red][FAIL][/red] {e}\n")
                    logger.error(traceback.format_exc())
        except Exception as e:
            console.print(f"[red][FAIL][/red] Dataset loading failed: {e}")
            logger.error(traceback.format_exc())
    
    # Display summary header
    console.print(f"\n[bold blue]{'='*80}[/bold blue]")
    console.print(f"[bold blue]BENCHMARK RESULTS SUMMARY[/bold blue]")
    console.print(f"[bold blue]{'='*80}[/bold blue]\n")
    
    if not all_results:
        console.print("[red]No results to display. All benchmarks failed.[/red]")
        return
    
    # Create results table
    table = Table(show_header=True, header_style="bold magenta", title="Kaggle Benchmark Results")
    table.add_column("Dataset", style="cyan")
    table.add_column("Approach", style="yellow")
    table.add_column("Task", style="white")
    table.add_column("Feat In->Out", justify="center")
    table.add_column("CV Score", justify="right")
    table.add_column("Test Metrics", style="green")
    table.add_column("Time(s)", justify="right")
    
    for res in all_results:
        test_metrics_str = ", ".join([f"{k}={v:.3f}" for k, v in res.test_score.items()])
        total_time = res.fit_time + res.transform_time + res.train_time
        
        table.add_row(
            res.dataset,
            res.approach,
            res.task,
            f"{res.n_features_in}->{res.n_features_out}",
            f"{res.cv_score:.4f}±{res.cv_std:.3f}",
            test_metrics_str,
            f"{total_time:.1f}"
        )
    
    console.print(table)
    
    # Perform comparative analysis
    console.print(f"\n[bold green]COMPARATIVE ANALYSIS[/bold green]\n")
    
    df_results = pd.DataFrame([asdict(r) for r in all_results])
    
    for dataset in df_results['dataset'].unique():
        dataset_results = df_results[df_results['dataset'] == dataset]
        console.print(f"[cyan]{dataset.upper()}:[/cyan]")
        
        # Identify best approach
        best_idx = dataset_results['cv_score'].idxmax()
        best = dataset_results.loc[best_idx]
        console.print(
            f"  Best: [bold green]{best['approach']}[/bold green] "
            f"(CV={best['cv_score']:.4f})"
        )
        
        # Compare FeatureCraft vs other approaches
        if 'featurecraft' in dataset_results['approach'].values:
            fc_result = dataset_results[dataset_results['approach'] == 'featurecraft'].iloc[0]
            baseline_result = dataset_results[dataset_results['approach'] == 'baseline'].iloc[0]
            
            # vs Baseline
            improvement = (fc_result['cv_score'] - baseline_result['cv_score']) / baseline_result['cv_score'] * 100
            color = 'green' if improvement > 0 else 'red'
            console.print(f"  Featurecraft vs Baseline: [{color}]{improvement:+.1f}%[/{color}]")
            
            # vs Kaggle Top (if available)
            if 'kaggle_top' in dataset_results['approach'].values:
                kaggle_result = dataset_results[dataset_results['approach'] == 'kaggle_top'].iloc[0]
                improvement_vs_kaggle = (fc_result['cv_score'] - kaggle_result['cv_score']) / kaggle_result['cv_score'] * 100
                color = 'green' if improvement_vs_kaggle > 0 else 'red'
                console.print(f"  Featurecraft vs Kaggle Top: [{color}]{improvement_vs_kaggle:+.1f}%[/{color}]")
        
        console.print()
    
    # Save results to JSON
    results_json = artifacts_base / "kaggle_benchmark_results.json"
    results_json.write_text(json.dumps([asdict(r) for r in all_results], indent=2))
    console.print(f"[green][OK][/green] Results saved to {results_json}")
    
    # Create comparison plots
    create_comparison_plots(all_results, artifacts_base)
    console.print(f"[green][OK][/green] All artifacts saved to {artifacts_base}")
    
    # Done!
    console.print("\n[bold green][OK] Benchmark complete![/bold green]\n")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()

