"""
@03_complex_kaggle_benchmark.py

Advanced benchmarking script for featurecraft library on COMPLEX Kaggle datasets.
Tests the library's capabilities on real-world, messy, production-like data.

Complex Datasets tested:
1. IEEE-CIS Fraud Detection - 434 features, high cardinality, heavy imbalance (~3.5%)
2. Home Credit Default Risk - Multi-table relational data (7 tables)
3. Santander Customer Transaction - 200 anonymized numeric features, class imbalance

For each dataset, we compare:
- Featurecraft automated pipeline
- Advanced manual feature engineering
- Baseline preprocessing

Stress tests:
- High cardinality categorical encoding
- Severe class imbalance handling
- Heavy missing data (50%+)
- Multi-table aggregation
- High-dimensional numeric data
"""

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
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score, precision_score, recall_score,
    log_loss, average_precision_score, confusion_matrix
)
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import joblib

# GPU-enabled XGBoost
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("[!] XGBoost not available. Installing for GPU acceleration...")
    import subprocess
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "xgboost"])
        import xgboost as xgb
        XGB_AVAILABLE = True
    except:
        print("[!] Failed to install XGBoost. Will fall back to sklearn.")

try:
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
    from rich.panel import Panel
    from rich import box
except ImportError:
    print("Installing rich for better console output...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "rich"])
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
    from rich.panel import Panel
    from rich import box

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

console = Console()


@dataclass
class ComplexBenchmarkResult:
    """Result container for complex Kaggle dataset benchmark."""
    dataset: str
    approach: str  # 'featurecraft', 'advanced_manual', 'baseline'
    task: str
    n_rows: int
    n_features_in: int
    n_features_out: int
    fit_time: float
    transform_time: float
    train_time: float
    cv_roc_auc: float
    cv_std: float
    test_metrics: Dict[str, float]
    memory_usage_mb: float
    feature_engineering_details: str
    challenges_addressed: List[str]
    status: str = "success"
    error: Optional[str] = None


def find_library() -> Tuple[Any, Any]:
    """Discover the feature engineering library."""
    try:
        from featurecraft.pipeline import AutoFeatureEngineer as MainClass
        import featurecraft as module
        logger.info(f"[+] Found library: featurecraft.pipeline.AutoFeatureEngineer")
        return module, MainClass
    except ImportError:
        raise RuntimeError(
            "Could not find featurecraft library. Please install it first: pip install featurecraft"
        )


def load_ieee_fraud_detection() -> Tuple[pd.DataFrame, pd.Series, str, List[str]]:
    """
    Load IEEE-CIS Fraud Detection dataset (complex classification).
    
    Download from: https://www.kaggle.com/c/ieee-fraud-detection/data
    
    Challenges:
    - 434 features across transaction + identity tables
    - High cardinality categoricals (card types, email domains)
    - Heavy class imbalance (~3.5% fraud)
    - 50%+ missing data in many columns
    - Mixed feature types
    """
    challenges = [
        "High cardinality categoricals (>1000 unique values)",
        "Heavy class imbalance (96.5% vs 3.5%)",
        "Extreme missing data (50%+ in many columns)",
        "434 mixed-type features"
    ]
    
    try:
        transaction_path = Path("./data/ieee-fraud-detection/train_transaction.csv")
        identity_path = Path("./data/ieee-fraud-detection/train_identity.csv")
        
        if transaction_path.exists():
            console.print(f"[cyan]Loading IEEE Fraud Detection from disk...[/cyan]")
            df_trans = pd.read_csv(transaction_path)
            
            # Sample to reduce memory if too large
            if len(df_trans) > 300000:
                df_trans = df_trans.sample(n=300000, random_state=42)
                console.print(f"[yellow]  Sampled to 300k rows for memory efficiency[/yellow]")
            
            if identity_path.exists():
                df_identity = pd.read_csv(identity_path)
                df = df_trans.merge(df_identity, on='TransactionID', how='left')
                console.print(f"[green][+][/green] Merged transaction + identity tables")
            else:
                df = df_trans
                console.print(f"[yellow]⚠[/yellow] Identity table not found, using transaction only")
            
            logger.info(f"[+] Loaded IEEE Fraud dataset: {df.shape}")
        else:
            console.print(f"[yellow]⚠[/yellow] IEEE dataset not found, generating synthetic fraud data...")
            df = generate_synthetic_ieee_fraud()
            logger.info(f"[+] Generated synthetic fraud dataset: {df.shape}")
        
        # Prepare features and target
        target_col = 'isFraud'
        y = df[target_col].copy()
        X = df.drop(columns=['TransactionID', target_col] if 'TransactionID' in df.columns else [target_col])
        
        return X, y, "classification", challenges
    
    except Exception as e:
        logger.error(f"Failed to load IEEE Fraud dataset: {e}")
        raise


def generate_synthetic_ieee_fraud(n_samples: int = 100000) -> pd.DataFrame:
    """Generate synthetic fraud detection data with similar complexity."""
    np.random.seed(42)
    
    # Transaction features (V1-V339 are PCA components in real data)
    data = {}
    
    # Numeric features (simulate PCA components)
    for i in range(1, 100):
        data[f'V{i}'] = np.random.randn(n_samples)
    
    # Transaction amount
    data['TransactionAmt'] = np.random.lognormal(mean=4, sigma=1.5, size=n_samples)
    
    # High cardinality categoricals
    data['card1'] = np.random.randint(1000, 20000, n_samples)  # High cardinality
    data['card2'] = np.random.choice([100, 200, 300, 400, 500, np.nan], n_samples, p=[0.3, 0.25, 0.2, 0.15, 0.05, 0.05])
    data['card3'] = np.random.choice([100, 150, 185, np.nan], n_samples, p=[0.6, 0.2, 0.1, 0.1])
    data['card4'] = np.random.choice(['visa', 'mastercard', 'discover', 'american express'], n_samples)
    data['card5'] = np.random.choice([100, 102, 111, 114, 117, 123, 142, 150, 166, 204, 226], n_samples)
    data['card6'] = np.random.choice(['debit', 'credit', np.nan], n_samples, p=[0.5, 0.45, 0.05])
    
    # Email domains (high cardinality with missing)
    email_domains = [f'email_domain_{i}' for i in range(500)] + [np.nan] * 100
    data['P_emaildomain'] = np.random.choice(email_domains, n_samples)
    data['R_emaildomain'] = np.random.choice(email_domains, n_samples)
    
    # Address info with heavy missing
    data['addr1'] = np.random.choice(range(1, 500), n_samples)
    data['addr2'] = np.random.choice(range(1, 100), n_samples)
    
    # Device info with missing (50%+)
    devices = [f'device_{i}' for i in range(50)] + [np.nan]
    data['DeviceType'] = np.random.choice(['desktop', 'mobile', np.nan], n_samples, p=[0.4, 0.4, 0.2])
    data['DeviceInfo'] = np.random.choice(devices, n_samples)
    
    # Add more features with heavy missing data
    for i in range(1, 20):
        missing_rate = np.random.uniform(0.3, 0.7)
        values = np.random.randn(n_samples)
        mask = np.random.random(n_samples) < missing_rate
        values[mask] = np.nan
        data[f'M{i}'] = values
    
    df = pd.DataFrame(data)
    df['TransactionID'] = range(1, n_samples + 1)
    
    # Create target with imbalance (3.5% fraud)
    fraud_prob = 0.035
    df['isFraud'] = np.random.choice([0, 1], n_samples, p=[1-fraud_prob, fraud_prob])
    
    # Add some correlation between features and fraud
    df.loc[df['TransactionAmt'] > df['TransactionAmt'].quantile(0.95), 'isFraud'] = \
        np.random.choice([0, 1], sum(df['TransactionAmt'] > df['TransactionAmt'].quantile(0.95)), p=[0.8, 0.2])
    
    return df


def load_home_credit_default() -> Tuple[pd.DataFrame, pd.Series, str, List[str]]:
    """
    Load Home Credit Default Risk dataset (multi-table relational).
    
    Download from: https://www.kaggle.com/c/home-credit-default-risk/data
    
    Challenges:
    - 7 different data tables (application, bureau, previous_application, etc.)
    - Temporal features requiring aggregation
    - 122 features in main table alone
    - Heavy missing data
    """
    challenges = [
        "Multi-table relational data (7 tables)",
        "Feature aggregation across tables",
        "Temporal patterns",
        "122+ raw features with heavy missingness"
    ]
    
    try:
        main_path = Path("./data/home-credit-default-risk/application_train.csv")
        bureau_path = Path("./data/home-credit-default-risk/bureau.csv")
        prev_app_path = Path("./data/home-credit-default-risk/previous_application.csv")
        
        if main_path.exists():
            console.print(f"[cyan]Loading Home Credit dataset from disk...[/cyan]")
            df = pd.read_csv(main_path)
            
            # Sample if too large
            if len(df) > 100000:
                df = df.sample(n=100000, random_state=42)
                console.print(f"[yellow]  Sampled to 100k rows for memory efficiency[/yellow]")
            
            # Merge bureau data if available
            if bureau_path.exists():
                bureau = pd.read_csv(bureau_path)
                # Aggregate bureau data per SK_ID_CURR
                bureau_agg = bureau.groupby('SK_ID_CURR').agg({
                    'DAYS_CREDIT': ['min', 'max', 'mean'],
                    'CREDIT_DAY_OVERDUE': ['max', 'mean'],
                    'AMT_CREDIT_SUM': ['sum', 'mean', 'max'],
                    'AMT_CREDIT_SUM_DEBT': ['sum', 'mean'],
                }).reset_index()
                bureau_agg.columns = ['SK_ID_CURR'] + [f'BUREAU_{c[0]}_{c[1]}' for c in bureau_agg.columns[1:]]
                df = df.merge(bureau_agg, on='SK_ID_CURR', how='left')
                console.print(f"[green][+][/green] Merged bureau aggregations")
            
            # Merge previous application data if available
            if prev_app_path.exists():
                prev_app = pd.read_csv(prev_app_path)
                prev_agg = prev_app.groupby('SK_ID_CURR').agg({
                    'AMT_ANNUITY': ['min', 'max', 'mean'],
                    'AMT_APPLICATION': ['min', 'max', 'mean', 'sum'],
                    'AMT_CREDIT': ['min', 'max', 'mean'],
                    'DAYS_DECISION': ['min', 'max', 'mean'],
                }).reset_index()
                prev_agg.columns = ['SK_ID_CURR'] + [f'PREV_{c[0]}_{c[1]}' for c in prev_agg.columns[1:]]
                df = df.merge(prev_agg, on='SK_ID_CURR', how='left')
                console.print(f"[green][+][/green] Merged previous application aggregations")
            
            logger.info(f"[+] Loaded Home Credit dataset: {df.shape}")
        else:
            console.print(f"[yellow]⚠[/yellow] Home Credit dataset not found, generating synthetic data...")
            df = generate_synthetic_home_credit()
            logger.info(f"[+] Generated synthetic home credit dataset: {df.shape}")
        
        # Prepare features and target
        target_col = 'TARGET'
        y = df[target_col].copy()
        X = df.drop(columns=['SK_ID_CURR', target_col] if 'SK_ID_CURR' in df.columns else [target_col])
        
        return X, y, "classification", challenges
    
    except Exception as e:
        logger.error(f"Failed to load Home Credit dataset: {e}")
        raise


def generate_synthetic_home_credit(n_samples: int = 50000) -> pd.DataFrame:
    """Generate synthetic home credit data with similar complexity."""
    np.random.seed(42)
    
    data = {
        'SK_ID_CURR': range(1, n_samples + 1),
        
        # Application info
        'NAME_CONTRACT_TYPE': np.random.choice(['Cash loans', 'Revolving loans'], n_samples, p=[0.9, 0.1]),
        'CODE_GENDER': np.random.choice(['M', 'F', 'XNA'], n_samples, p=[0.35, 0.65, 0.001]),
        'FLAG_OWN_CAR': np.random.choice(['Y', 'N'], n_samples),
        'FLAG_OWN_REALTY': np.random.choice(['Y', 'N'], n_samples, p=[0.7, 0.3]),
        'CNT_CHILDREN': np.random.poisson(0.4, n_samples),
        'AMT_INCOME_TOTAL': np.random.lognormal(mean=11.5, sigma=0.6, size=n_samples),
        'AMT_CREDIT': np.random.lognormal(mean=12.5, sigma=0.7, size=n_samples),
        'AMT_ANNUITY': np.random.lognormal(mean=9.5, sigma=0.6, size=n_samples),
        'AMT_GOODS_PRICE': np.random.lognormal(mean=12.3, sigma=0.7, size=n_samples),
        
        # Temporal features (negative days)
        'DAYS_BIRTH': -np.random.randint(7000, 25000, n_samples),
        'DAYS_EMPLOYED': -np.random.randint(0, 15000, n_samples),
        'DAYS_REGISTRATION': -np.random.randint(0, 20000, n_samples),
        'DAYS_ID_PUBLISH': -np.random.randint(0, 7000, n_samples),
        
        # Employment info
        'NAME_INCOME_TYPE': np.random.choice([
            'Working', 'Commercial associate', 'Pensioner', 'State servant', 'Unemployed'
        ], n_samples, p=[0.5, 0.25, 0.15, 0.08, 0.02]),
        
        'NAME_EDUCATION_TYPE': np.random.choice([
            'Secondary / secondary special', 'Higher education', 
            'Incomplete higher', 'Lower secondary', 'Academic degree'
        ], n_samples, p=[0.7, 0.2, 0.05, 0.04, 0.01]),
        
        'NAME_FAMILY_STATUS': np.random.choice([
            'Married', 'Single / not married', 'Civil marriage', 'Separated', 'Widow'
        ], n_samples, p=[0.6, 0.2, 0.1, 0.05, 0.05]),
        
        'NAME_HOUSING_TYPE': np.random.choice([
            'House / apartment', 'With parents', 'Municipal apartment', 'Rented apartment', 'Office apartment'
        ], n_samples, p=[0.7, 0.15, 0.08, 0.05, 0.02]),
        
        # Document flags
        'FLAG_MOBIL': np.random.choice([0, 1], n_samples, p=[0.001, 0.999]),
        'FLAG_EMP_PHONE': np.random.choice([0, 1], n_samples, p=[0.2, 0.8]),
        'FLAG_WORK_PHONE': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'FLAG_CONT_MOBILE': np.random.choice([0, 1], n_samples, p=[0.002, 0.998]),
        'FLAG_PHONE': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'FLAG_EMAIL': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
    }
    
    # Add document flags (many will be mostly 0)
    for i in range(2, 21):
        data[f'FLAG_DOCUMENT_{i}'] = np.random.choice([0, 1], n_samples, p=[0.95, 0.05])
    
    # External sources (with heavy missing)
    for i in [1, 2, 3]:
        values = np.random.uniform(0, 1, n_samples)
        mask = np.random.random(n_samples) < 0.4  # 40% missing
        values[mask] = np.nan
        data[f'EXT_SOURCE_{i}'] = values
    
    # Region and occupation info
    data['REGION_POPULATION_RELATIVE'] = np.random.uniform(0.0005, 0.07, n_samples)
    data['REGION_RATING_CLIENT'] = np.random.choice([1, 2, 3], n_samples, p=[0.2, 0.6, 0.2])
    data['REGION_RATING_CLIENT_W_CITY'] = np.random.choice([1, 2, 3], n_samples, p=[0.2, 0.6, 0.2])
    
    # Add aggregated bureau features (simulating joined data)
    data['BUREAU_DAYS_CREDIT_min'] = -np.random.randint(0, 3000, n_samples)
    data['BUREAU_DAYS_CREDIT_max'] = -np.random.randint(0, 1000, n_samples)
    data['BUREAU_AMT_CREDIT_SUM_sum'] = np.random.lognormal(mean=11, sigma=1.5, size=n_samples)
    
    # Add missing values to simulate real messiness
    for col in ['AMT_ANNUITY', 'DAYS_EMPLOYED', 'BUREAU_AMT_CREDIT_SUM_sum']:
        if col in data:
            mask = np.random.random(n_samples) < 0.15
            temp = np.array(data[col], dtype=float)
            temp[mask] = np.nan
            data[col] = temp
    
    df = pd.DataFrame(data)
    
    # Create target with imbalance (8% default)
    default_prob = 0.08
    df['TARGET'] = np.random.choice([0, 1], n_samples, p=[1-default_prob, default_prob])
    
    # Add correlations
    df.loc[df['EXT_SOURCE_1'].notna() & (df['EXT_SOURCE_1'] < 0.3), 'TARGET'] = \
        np.random.choice([0, 1], sum(df['EXT_SOURCE_1'].notna() & (df['EXT_SOURCE_1'] < 0.3)), p=[0.7, 0.3])
    
    return df


def load_santander_transaction() -> Tuple[pd.DataFrame, pd.Series, str, List[str]]:
    """
    Load Santander Customer Transaction Prediction dataset.
    
    Download from: https://www.kaggle.com/c/santander-customer-transaction-prediction/data
    
    Challenges:
    - 200 anonymized numeric features
    - Class imbalance (10% positive)
    - Pure numeric feature engineering challenge
    - Distribution shifts and adversarial validation needed
    """
    challenges = [
        "200 anonymized numeric features",
        "Class imbalance (90% vs 10%)",
        "No feature semantics (pure numeric optimization)",
        "Distribution shifts between train/test"
    ]
    
    try:
        train_path = Path("./data/santander-transaction/train.csv")
        
        if train_path.exists():
            console.print(f"[cyan]Loading Santander dataset from disk...[/cyan]")
            df = pd.read_csv(train_path)
            
            # Sample if too large
            if len(df) > 100000:
                df = df.sample(n=100000, random_state=42)
                console.print(f"[yellow]  Sampled to 100k rows for memory efficiency[/yellow]")
            
            logger.info(f"[+] Loaded Santander dataset: {df.shape}")
        else:
            console.print(f"[yellow]⚠[/yellow] Santander dataset not found, generating synthetic data...")
            df = generate_synthetic_santander()
            logger.info(f"[+] Generated synthetic Santander dataset: {df.shape}")
        
        # Prepare features and target
        target_col = 'target'
        y = df[target_col].copy()
        X = df.drop(columns=['ID_code', target_col] if 'ID_code' in df.columns else [target_col])
        
        return X, y, "classification", challenges
    
    except Exception as e:
        logger.error(f"Failed to load Santander dataset: {e}")
        raise


def generate_synthetic_santander(n_samples: int = 30000, n_features: int = 200) -> pd.DataFrame:
    """Generate synthetic Santander-like data with 200 numeric features."""
    np.random.seed(42)
    
    # Generate 200 numeric features with various distributions
    data = {}
    
    for i in range(n_features):
        if i % 5 == 0:
            # Normal distribution
            data[f'var_{i}'] = np.random.randn(n_samples)
        elif i % 5 == 1:
            # Skewed distribution
            data[f'var_{i}'] = np.random.lognormal(0, 1, n_samples)
        elif i % 5 == 2:
            # Uniform distribution
            data[f'var_{i}'] = np.random.uniform(-10, 10, n_samples)
        elif i % 5 == 3:
            # Heavy-tailed distribution
            data[f'var_{i}'] = np.random.standard_t(df=3, size=n_samples)
        else:
            # Mixture distribution
            mask = np.random.random(n_samples) < 0.5
            data[f'var_{i}'] = np.where(mask, np.random.randn(n_samples), np.random.randn(n_samples) + 3)
    
    df = pd.DataFrame(data)
    df['ID_code'] = [f'ID_{i:06d}' for i in range(n_samples)]
    
    # Create target with 10% positive class
    df['target'] = np.random.choice([0, 1], n_samples, p=[0.9, 0.1])
    
    # Add some signal - make certain feature combinations predictive
    signal_features = [f'var_{i}' for i in range(0, 20)]
    for feat in signal_features:
        # High values in certain features correlate with positive class
        high_vals = df[feat] > df[feat].quantile(0.9)
        df.loc[high_vals, 'target'] = np.random.choice([0, 1], sum(high_vals), p=[0.6, 0.4])
    
    return df


def create_advanced_manual_features_ieee(X: pd.DataFrame) -> pd.DataFrame:
    """
    Advanced manual feature engineering for IEEE Fraud Detection.
    Based on top Kaggle solutions.
    """
    X = X.copy()
    
    # Transaction amount features
    if 'TransactionAmt' in X.columns:
        X['TransactionAmt_log'] = np.log1p(X['TransactionAmt'])
        X['TransactionAmt_decimal'] = (X['TransactionAmt'] - X['TransactionAmt'].astype(int))
        X['TransactionAmt_rounded'] = X['TransactionAmt'].round(0)
    
    # Card aggregations (frequency encoding for high cardinality)
    card_cols = [col for col in X.columns if col.startswith('card')]
    for col in card_cols:
        if X[col].dtype in ['object', 'category']:
            freq_encoding = X[col].value_counts(normalize=True)
            X[f'{col}_freq'] = X[col].map(freq_encoding)
    
    # Email domain features
    email_cols = [col for col in X.columns if 'email' in col.lower()]
    for col in email_cols:
        if col in X.columns:
            X[f'{col}_isna'] = X[col].isna().astype(int)
    
    # V features interactions (if they exist)
    v_cols = [col for col in X.columns if col.startswith('V')]
    if len(v_cols) >= 10:
        # Create some V feature statistics
        v_data = X[v_cols[:20]]
        X['V_mean'] = v_data.mean(axis=1)
        X['V_std'] = v_data.std(axis=1)
        X['V_max'] = v_data.max(axis=1)
        X['V_min'] = v_data.min(axis=1)
    
    # Missing count
    X['missing_count'] = X.isnull().sum(axis=1)
    
    return X


def create_advanced_manual_features_home_credit(X: pd.DataFrame) -> pd.DataFrame:
    """
    Advanced manual feature engineering for Home Credit.
    Based on top Kaggle solutions.
    """
    X = X.copy()
    
    # Age features
    if 'DAYS_BIRTH' in X.columns:
        X['AGE_YEARS'] = -X['DAYS_BIRTH'] / 365
        X['AGE_BINNED'] = pd.cut(X['AGE_YEARS'], bins=[0, 25, 35, 45, 55, 100], labels=False)
    
    if 'DAYS_EMPLOYED' in X.columns:
        X['EMPLOYED_YEARS'] = -X['DAYS_EMPLOYED'] / 365
        X['EMPLOYED_YEARS'] = X['EMPLOYED_YEARS'].clip(lower=0)  # Handle anomalies
    
    # Financial ratios
    if 'AMT_INCOME_TOTAL' in X.columns and 'AMT_CREDIT' in X.columns:
        X['CREDIT_INCOME_RATIO'] = X['AMT_CREDIT'] / (X['AMT_INCOME_TOTAL'] + 1)
    
    if 'AMT_ANNUITY' in X.columns and 'AMT_INCOME_TOTAL' in X.columns:
        X['ANNUITY_INCOME_RATIO'] = X['AMT_ANNUITY'] / (X['AMT_INCOME_TOTAL'] + 1)
    
    if 'AMT_CREDIT' in X.columns and 'AMT_ANNUITY' in X.columns:
        X['CREDIT_TERM'] = X['AMT_CREDIT'] / (X['AMT_ANNUITY'] + 1)
    
    # External sources combinations
    ext_sources = [col for col in X.columns if col.startswith('EXT_SOURCE')]
    if len(ext_sources) >= 2:
        ext_data = X[ext_sources]
        X['EXT_SOURCE_MEAN'] = ext_data.mean(axis=1)
        X['EXT_SOURCE_STD'] = ext_data.std(axis=1)
        X['EXT_SOURCE_MIN'] = ext_data.min(axis=1)
        X['EXT_SOURCE_MAX'] = ext_data.max(axis=1)
    
    # Document flags sum
    doc_cols = [col for col in X.columns if col.startswith('FLAG_DOCUMENT')]
    if doc_cols:
        X['DOCUMENT_FLAGS_SUM'] = X[doc_cols].sum(axis=1)
    
    # Contact flags sum
    contact_cols = [col for col in X.columns if 'FLAG_' in col and any(x in col for x in ['PHONE', 'EMAIL', 'MOBIL'])]
    if contact_cols:
        X['CONTACT_FLAGS_SUM'] = X[contact_cols].sum(axis=1)
    
    return X


def create_advanced_manual_features_santander(X: pd.DataFrame) -> pd.DataFrame:
    """
    Advanced manual feature engineering for Santander.
    Based on top Kaggle solutions - pure numeric transformations.
    """
    X = X.copy()
    
    var_cols = [col for col in X.columns if col.startswith('var_')]
    
    if len(var_cols) >= 50:
        # Statistical features across all variables
        var_data = X[var_cols[:50]]  # Use subset to avoid memory issues
        
        X['vars_mean'] = var_data.mean(axis=1)
        X['vars_std'] = var_data.std(axis=1)
        X['vars_min'] = var_data.min(axis=1)
        X['vars_max'] = var_data.max(axis=1)
        X['vars_median'] = var_data.median(axis=1)
        X['vars_skew'] = var_data.skew(axis=1)
        
        # Count features
        X['vars_positive_count'] = (var_data > 0).sum(axis=1)
        X['vars_negative_count'] = (var_data < 0).sum(axis=1)
        X['vars_zero_count'] = (var_data == 0).sum(axis=1)
    
    # Frequency encoding - count unique rounded values
    for col in var_cols[:30]:  # Limit to avoid explosion
        X[f'{col}_rounded'] = X[col].round(2)
    
    return X


def create_baseline_pipeline(X: pd.DataFrame, y: pd.Series, task: str) -> Pipeline:
    """Create a baseline pipeline with minimal preprocessing."""
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Limit categorical features if too many unique values
    categorical_features = [
        col for col in categorical_features 
        if X[col].nunique() < 100  # Skip very high cardinality
    ]
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())  # Robust to outliers
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False, max_categories=50))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'  # Drop very high cardinality categoricals
    )
    
    return preprocessor


def run_featurecraft_approach(
    EngineClass: Any,
    X: pd.DataFrame,
    y: pd.Series,
    task: str,
    artifacts_dir: Path,
    use_ai: bool = True
) -> Dict[str, Any]:
    """Run featurecraft automated feature engineering with AI optimization.
    
    Args:
        EngineClass: AutoFeatureEngineer class
        X: Features
        y: Target
        task: Task type
        artifacts_dir: Artifacts directory
        use_ai: Enable AI-powered optimization (recommended)
    """
    result = {
        'n_features_in': X.shape[1],
        'n_features_out': X.shape[1],
        'fit_time': 0.0,
        'transform_time': 0.0,
        'memory_usage_mb': 0.0,
        'status': 'success',
        'error': None,
        'details': 'AI-powered automated feature engineering' if use_ai else 'Automated feature engineering'
    }
    
    try:
        # Initialize with AI-powered optimization
        from featurecraft.config import FeatureCraftConfig
        import os
        
        config = FeatureCraftConfig(
            validate_schema=True,  # Keep validation enabled for data quality
            schema_coerce=True,    # Auto-fix minor type and range issues
            explain_transformations=False,  # Disable explanations to avoid Unicode issues
            explain_auto_print=False,
        )
        
        # Create with AI optimization enabled
        api_key = os.getenv("OPENAI_API_KEY")
        if use_ai and not api_key:
            console.print("  [yellow]⚠ OPENAI_API_KEY not set - AI mode will fail. Set use_ai=False for heuristics[/yellow]")
        
        afe = EngineClass(
            config=config,
            use_ai_advisor=use_ai,
            ai_api_key=api_key,
            ai_model="gpt-4o-mini",
            time_budget="balanced",
        )
        
        # Split with stratification for imbalanced data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        console.print(f"[cyan]  Training on {len(X_train)} samples, testing on {len(X_test)}[/cyan]")
        
        # Fit
        start = time.time()
        afe.fit(X_train, y_train)
        result['fit_time'] = time.time() - start
        
        # Transform
        start = time.time()
        Xt_train = afe.transform(X_train)
        Xt_test = afe.transform(X_test)
        result['transform_time'] = time.time() - start
        
        # Memory usage
        if isinstance(Xt_train, pd.DataFrame):
            result['memory_usage_mb'] = Xt_train.memory_usage(deep=True).sum() / 1024 / 1024
        else:
            result['memory_usage_mb'] = Xt_train.nbytes / 1024 / 1024
        
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
                console.print(f"[green]  [+] Pipeline exported to {export_dir}[/green]")
        except Exception as e:
            logger.debug(f"Export failed: {e}")
        
    except Exception as e:
        result['status'] = 'failed'
        result['error'] = str(e)
        logger.error(f"Featurecraft failed: {e}")
        traceback.print_exc()
        
        # Fallback
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        result['X_train'] = X_train
        result['X_test'] = X_test
        result['y_train'] = y_train
        result['y_test'] = y_test
    
    return result


def run_advanced_manual_approach(
    X: pd.DataFrame,
    y: pd.Series,
    task: str,
    dataset_name: str
) -> Dict[str, Any]:
    """Run advanced manual feature engineering."""
    result = {
        'n_features_in': X.shape[1],
        'n_features_out': X.shape[1],
        'fit_time': 0.0,
        'transform_time': 0.0,
        'memory_usage_mb': 0.0,
        'status': 'success',
        'error': None,
        'details': 'Advanced manual feature engineering (Kaggle expert-level)'
    }
    
    try:
        start = time.time()
        
        # Apply dataset-specific feature engineering
        if 'ieee' in dataset_name or 'fraud' in dataset_name:
            X_engineered = create_advanced_manual_features_ieee(X)
        elif 'home' in dataset_name or 'credit' in dataset_name:
            X_engineered = create_advanced_manual_features_home_credit(X)
        elif 'santander' in dataset_name:
            X_engineered = create_advanced_manual_features_santander(X)
        else:
            X_engineered = X.copy()
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X_engineered, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Apply baseline preprocessing
        preprocessor = create_baseline_pipeline(X_train, y_train, task)
        preprocessor.fit(X_train, y_train)
        result['fit_time'] = time.time() - start
        
        # Transform
        start = time.time()
        Xt_train = preprocessor.transform(X_train)
        Xt_test = preprocessor.transform(X_test)
        result['transform_time'] = time.time() - start
        
        # Memory usage
        result['memory_usage_mb'] = Xt_train.nbytes / 1024 / 1024
        
        result['n_features_out'] = Xt_train.shape[1]
        result['X_train'] = Xt_train
        result['X_test'] = Xt_test
        result['y_train'] = y_train
        result['y_test'] = y_test
        
    except Exception as e:
        result['status'] = 'failed'
        result['error'] = str(e)
        logger.error(f"Advanced manual approach failed: {e}")
        traceback.print_exc()
        
        # Fallback
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
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
        'memory_usage_mb': 0.0,
        'status': 'success',
        'error': None,
        'details': 'Minimal preprocessing baseline'
    }
    
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        start = time.time()
        preprocessor = create_baseline_pipeline(X_train, y_train, task)
        preprocessor.fit(X_train, y_train)
        result['fit_time'] = time.time() - start
        
        start = time.time()
        Xt_train = preprocessor.transform(X_train)
        Xt_test = preprocessor.transform(X_test)
        result['transform_time'] = time.time() - start
        
        result['memory_usage_mb'] = Xt_train.nbytes / 1024 / 1024
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


def evaluate_model_imbalanced(
    X_train: np.ndarray,
    y_train: pd.Series,
    X_test: np.ndarray,
    y_test: pd.Series,
    task: str
) -> Tuple[float, float, Dict[str, float], float]:
    """
    Train and evaluate model with focus on imbalanced classification metrics.
    Uses GPU-accelerated XGBoost if available.
    
    Returns: (cv_roc_auc, cv_std, test_metrics, train_time)
    """
    # Use GPU-accelerated XGBoost if available (XGBoost 2.0+ API)
    if XGB_AVAILABLE:
        # Try to use GPU, fall back to CPU if GPU not available
        try:
            # Check XGBoost version and use appropriate GPU API
            if hasattr(xgb, '__version__') and xgb.__version__ >= '2.0.0':
                # XGBoost 2.0+ uses 'device' parameter instead of 'gpu_id' and 'tree_method'
                try:
                    # Try to use CUDA GPU 1 (NVIDIA)
                    model = xgb.XGBClassifier(
                        n_estimators=200,
                        learning_rate=0.05,
                        max_depth=6,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        tree_method='hist',  # Use hist for tree construction
                        device='cuda:1',  # Use CUDA GPU 1 (NVIDIA)
                        random_state=42,
                        eval_metric='auc',
                        use_label_encoder=False
                    )
                    console.print("  [bold green][+] Using XGBoost with CUDA GPU 1[/bold green]")
                except Exception as e:
                    console.print(f"  [yellow]⚠ CUDA GPU 1 not available, trying GPU 0: {e}[/yellow]")
                    try:
                        model = xgb.XGBClassifier(
                            n_estimators=200,
                            learning_rate=0.05,
                            max_depth=6,
                            subsample=0.8,
                            colsample_bytree=0.8,
                            tree_method='hist',
                            device='cuda:0',  # Fall back to CUDA GPU 0
                            random_state=42,
                            eval_metric='auc',
                            use_label_encoder=False
                        )
                        console.print("  [bold green][+] Using XGBoost with CUDA GPU 0[/bold green]")
                    except Exception as e2:
                        console.print(f"  [yellow]⚠ CUDA not available, using CPU: {e2}[/yellow]")
                        model = xgb.XGBClassifier(
                            n_estimators=200,
                            learning_rate=0.05,
                            max_depth=6,
                            subsample=0.8,
                            colsample_bytree=0.8,
                            tree_method='hist',
                            device='cpu',  # CPU fallback
                            random_state=42,
                            eval_metric='auc',
                            use_label_encoder=False
                        )
            else:
                # Older XGBoost version - use legacy GPU API
                try:
                    model = xgb.XGBClassifier(
                        n_estimators=200,
                        learning_rate=0.05,
                        max_depth=6,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        tree_method='gpu_hist',
                        gpu_id=1,  # Try GPU 1 first
                        random_state=42,
                        eval_metric='auc',
                        use_label_encoder=False
                    )
                    console.print("  [bold green][+] Using XGBoost GPU 1 (legacy API)[/bold green]")
                except Exception as e:
                    console.print(f"  [yellow]⚠ GPU 1 not available, trying GPU 0: {e}[/yellow]")
                    try:
                        model = xgb.XGBClassifier(
                            n_estimators=200,
                            learning_rate=0.05,
                            max_depth=6,
                            subsample=0.8,
                            colsample_bytree=0.8,
                            tree_method='gpu_hist',
                            gpu_id=0,  # Fall back to GPU 0
                            random_state=42,
                            eval_metric='auc',
                            use_label_encoder=False
                        )
                        console.print("  [bold green][+] Using XGBoost GPU 0 (legacy API)[/bold green]")
                    except Exception as e2:
                        console.print(f"  [yellow]⚠ GPU not available, using CPU: {e2}[/yellow]")
                        model = xgb.XGBClassifier(
                            n_estimators=200,
                            learning_rate=0.05,
                            max_depth=6,
                            subsample=0.8,
                            colsample_bytree=0.8,
                            tree_method='hist',
                            random_state=42,
                            eval_metric='auc',
                            use_label_encoder=False
                        )
        except Exception as e:
            console.print(f"  [yellow]⚠ XGBoost setup failed, using CPU: {e}[/yellow]")
            model = xgb.XGBClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                tree_method='hist',
                random_state=42,
                eval_metric='auc',
                use_label_encoder=False
            )
    else:
        # Fallback to sklearn
        console.print("  [yellow]⚠ XGBoost not available, using sklearn GradientBoostingClassifier (CPU)[/yellow]")
        model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            random_state=42
        )
    
    # Stratified K-Fold for imbalanced data
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Manual cross-validation with ROC-AUC (to avoid sklearn/xgboost compatibility issues)
    start = time.time()
    cv_scores = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
        # Split data
        X_fold_train = X_train[train_idx] if isinstance(X_train, np.ndarray) else X_train.iloc[train_idx]
        X_fold_val = X_train[val_idx] if isinstance(X_train, np.ndarray) else X_train.iloc[val_idx]
        y_fold_train = y_train.iloc[train_idx] if hasattr(y_train, 'iloc') else y_train[train_idx]
        y_fold_val = y_train.iloc[val_idx] if hasattr(y_train, 'iloc') else y_train[val_idx]
        
        # Clone model for each fold
        if XGB_AVAILABLE:
            fold_model = model.__class__(**model.get_params())
        else:
            from sklearn.base import clone
            fold_model = clone(model)
        
        # Train and predict
        fold_model.fit(X_fold_train, y_fold_train, verbose=False)
        y_fold_proba = fold_model.predict_proba(X_fold_val)[:, 1]
        
        # Calculate ROC-AUC
        fold_auc = roc_auc_score(y_fold_val, y_fold_proba)
        cv_scores.append(fold_auc)
    
    cv_roc_auc = np.mean(cv_scores)
    cv_std = np.std(cv_scores)
    
    # Train on full training set
    model.fit(X_train, y_train, verbose=False)
    train_time = time.time() - start
    
    # Predictions
    y_pred = model.predict(X_test)
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = y_pred  # Fallback
    
    # Comprehensive test metrics for imbalanced classification
    test_metrics = {
        'roc_auc': roc_auc_score(y_test, y_proba),
        'avg_precision': average_precision_score(y_test, y_proba),
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
        'log_loss': log_loss(y_test, y_proba),
    }
    
    # Confusion matrix elements
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    test_metrics['true_negative'] = int(tn)
    test_metrics['false_positive'] = int(fp)
    test_metrics['false_negative'] = int(fn)
    test_metrics['true_positive'] = int(tp)
    
    return cv_roc_auc, cv_std, test_metrics, train_time


def create_comparison_plots(results: List[ComplexBenchmarkResult], artifacts_dir: Path):
    """Create comprehensive comparison plots."""
    try:
        df = pd.DataFrame([asdict(r) for r in results])
        
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. ROC-AUC comparison
        ax1 = fig.add_subplot(gs[0, :2])
        pivot = df.pivot_table(values='cv_roc_auc', index='dataset', columns='approach', aggfunc='mean')
        pivot.plot(kind='bar', ax=ax1, rot=45, width=0.8)
        ax1.set_title('ROC-AUC Score by Approach (Higher is Better)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('ROC-AUC Score')
        ax1.set_ylim(0.5, 1.0)
        ax1.legend(title='Approach', loc='lower right')
        ax1.grid(axis='y', alpha=0.3)
        ax1.axhline(y=0.5, color='r', linestyle='--', alpha=0.3, label='Random')
        
        # 2. Feature engineering effectiveness
        ax2 = fig.add_subplot(gs[0, 2])
        pivot = df.pivot_table(values='n_features_out', index='dataset', columns='approach', aggfunc='mean')
        pivot.plot(kind='bar', ax=ax2, rot=45)
        ax2.set_title('Features Created', fontsize=12, fontweight='bold')
        ax2.set_ylabel('# Features')
        ax2.legend(title='Approach', fontsize=8)
        ax2.grid(axis='y', alpha=0.3)
        
        # 3. Processing time comparison
        ax3 = fig.add_subplot(gs[1, 0])
        df['total_time'] = df['fit_time'] + df['transform_time'] + df['train_time']
        pivot = df.pivot_table(values='total_time', index='dataset', columns='approach', aggfunc='mean')
        pivot.plot(kind='bar', ax=ax3, rot=45)
        ax3.set_title('Total Time (s)', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Time (seconds)')
        ax3.legend(title='Approach', fontsize=8)
        ax3.grid(axis='y', alpha=0.3)
        ax3.set_yscale('log')
        
        # 4. Memory usage
        ax4 = fig.add_subplot(gs[1, 1])
        pivot = df.pivot_table(values='memory_usage_mb', index='dataset', columns='approach', aggfunc='mean')
        pivot.plot(kind='bar', ax=ax4, rot=45)
        ax4.set_title('Memory Usage (MB)', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Memory (MB)')
        ax4.legend(title='Approach', fontsize=8)
        ax4.grid(axis='y', alpha=0.3)
        
        # 5. Success rate
        ax5 = fig.add_subplot(gs[1, 2])
        success_rate = df.groupby('approach')['status'].apply(lambda x: (x == 'success').sum() / len(x) * 100)
        colors = ['#2ecc71', '#3498db', '#e74c3c']
        success_rate.plot(kind='bar', ax=ax5, color=colors[:len(success_rate)])
        ax5.set_title('Success Rate', fontsize=12, fontweight='bold')
        ax5.set_ylabel('Success Rate (%)')
        ax5.set_ylim(0, 105)
        ax5.grid(axis='y', alpha=0.3)
        plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45)
        
        # 6. Detailed metrics heatmap (ROC-AUC)
        ax6 = fig.add_subplot(gs[2, :])
        heatmap_data = df.pivot_table(values='cv_roc_auc', index='approach', columns='dataset', aggfunc='mean')
        sns.heatmap(heatmap_data, annot=True, fmt='.4f', cmap='RdYlGn', center=0.75, 
                   vmin=0.5, vmax=1.0, ax=ax6, cbar_kws={'label': 'ROC-AUC'})
        ax6.set_title('ROC-AUC Heatmap by Dataset and Approach', fontsize=14, fontweight='bold')
        ax6.set_xlabel('Dataset')
        ax6.set_ylabel('Approach')
        
        plt.suptitle('FeatureCraft Complex Kaggle Benchmark - Comprehensive Analysis', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        plot_path = artifacts_dir / "complex_comparison_plots.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        console.print(f"[green][+][/green] Comparison plots saved to {plot_path}")
        
    except Exception as e:
        logger.warning(f"Failed to create comparison plots: {e}")
        traceback.print_exc()


def main():
    """Main orchestration function."""
    parser = argparse.ArgumentParser(description="FeatureCraft Complex Kaggle Benchmark Suite")
    parser.add_argument("--artifacts", type=str, default="./artifacts/complex_kaggle_benchmark", 
                       help="Artifacts directory")
    parser.add_argument("--datasets", type=str, default="santander", 
                       help="Comma-separated: all, ieee_fraud, home_credit, santander")
    args = parser.parse_args()
    
    # Setup
    artifacts_base = Path(args.artifacts)
    artifacts_base.mkdir(parents=True, exist_ok=True)
    
    # Header
    console.print("\n" + "="*90)
    console.print(Panel.fit(
        "[bold red]FeatureCraft COMPLEX Kaggle Benchmark Suite[/bold red]\n"
        "[bold yellow]Testing on Production-Scale, Messy, Real-World Datasets[/bold yellow]\n"
        "[cyan]High Cardinality - Heavy Imbalance - Missing Data - Multi-Table[/cyan]",
        border_style="red",
        box=box.DOUBLE
    ))
    console.print("="*90 + "\n")
    
    # Find library
    try:
        module, EngineClass = find_library()
        console.print(f"[green]+[/green] Library found: {EngineClass.__name__}\n")
    except RuntimeError as e:
        console.print(f"[red][-][/red] {e}")
        sys.exit(1)
    
    # Define datasets
    available_datasets = {
        'ieee_fraud': load_ieee_fraud_detection,
        'home_credit': load_home_credit_default,
        'santander': load_santander_transaction,
    }
    
    if args.datasets == "all":
        selected_datasets = list(available_datasets.keys())
    else:
        selected_datasets = [d.strip() for d in args.datasets.split(",")]
    
    console.print(f"[cyan]Datasets to benchmark:[/cyan] {', '.join(selected_datasets)}")
    console.print(f"[cyan]Approach:[/cyan] Featurecraft (automated) ONLY - Model Training ENABLED\n")
    
    # Run benchmarks
    all_results: List[ComplexBenchmarkResult] = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        console=console
    ) as progress:
        
        for dataset_name in selected_datasets:
            if dataset_name not in available_datasets:
                console.print(f"[red][-][/red] Unknown dataset: {dataset_name}")
                continue
            
            console.print(f"\n[bold yellow]{'='*80}[/bold yellow]")
            console.print(f"[bold yellow]DATASET: {dataset_name.upper().replace('_', ' ')}[/bold yellow]")
            console.print(f"[bold yellow]{'='*80}[/bold yellow]\n")
            
            try:
                # Load dataset
                task_load = progress.add_task(f"[cyan]Loading {dataset_name}...", total=None)
                X, y, task, challenges = available_datasets[dataset_name]()
                progress.remove_task(task_load)
                
                # Print dataset info
                console.print(f"[green][+][/green] Loaded: {X.shape[0]:,} rows × {X.shape[1]:,} features")
                console.print(f"[cyan]  Task:[/cyan] {task}")
                
                # Class distribution
                class_dist = y.value_counts(normalize=True).sort_index()
                console.print(f"[cyan]  Class distribution:[/cyan] {dict(class_dist.apply(lambda x: f'{x*100:.2f}%'))}")
                
                # Missing data
                missing_pct = (X.isnull().sum().sum() / (X.shape[0] * X.shape[1])) * 100
                console.print(f"[cyan]  Missing data:[/cyan] {missing_pct:.1f}%")
                
                # Challenges
                console.print(f"[bold red]  Challenges:[/bold red]")
                for challenge in challenges:
                    console.print(f"    - {challenge}")
                console.print()
                
                artifacts_dir = artifacts_base / dataset_name
                artifacts_dir.mkdir(exist_ok=True, parents=True)
                
                # Test ONLY FeatureCraft approach (model training enabled)
                approaches = [
                    ('featurecraft', lambda: run_featurecraft_approach(EngineClass, X, y, task, artifacts_dir))
                ]
                
                for approach_name, approach_func in approaches:
                    console.print(f"[bold cyan]-> Approach: {approach_name.upper().replace('_', ' ')}[/bold cyan]")
                    
                    try:
                        # Feature engineering
                        task_fe = progress.add_task(f"  [cyan]Feature engineering...", total=None)
                        fe_result = approach_func()
                        progress.remove_task(task_fe)
                        
                        if fe_result['status'] == 'failed':
                            console.print(f"  [red][-] Failed: {fe_result['error']}[/red]\n")
                            continue
                        
                        console.print(f"  [green][+][/green] Features: {fe_result['n_features_in']:,} -> {fe_result['n_features_out']:,}")
                        console.print(f"  [green][+][/green] Fit: {fe_result['fit_time']:.2f}s | Transform: {fe_result['transform_time']:.2f}s")
                        console.print(f"  [green][+][/green] Memory: {fe_result['memory_usage_mb']:.1f} MB")
                        
                        # Train and evaluate
                        task_eval = progress.add_task(f"  [cyan]Training & evaluating (5-fold CV)...", total=None)
                        cv_roc_auc, cv_std, test_metrics, train_time = evaluate_model_imbalanced(
                            fe_result['X_train'],
                            fe_result['y_train'],
                            fe_result['X_test'],
                            fe_result['y_test'],
                            task
                        )
                        progress.remove_task(task_eval)
                        
                        console.print(f"  [bold green][+] CV ROC-AUC: {cv_roc_auc:.4f} (±{cv_std:.4f})[/bold green]")
                        console.print(f"  [green][+][/green] Test ROC-AUC: {test_metrics['roc_auc']:.4f} | AP: {test_metrics['avg_precision']:.4f}")
                        console.print(f"  [green][+][/green] Precision: {test_metrics['precision']:.4f} | Recall: {test_metrics['recall']:.4f} | F1: {test_metrics['f1_score']:.4f}")
                        console.print(f"  [green][+][/green] Train time: {train_time:.2f}s\n")
                        
                        # Store result
                        result = ComplexBenchmarkResult(
                            dataset=dataset_name,
                            approach=approach_name,
                            task=task,
                            n_rows=len(X),
                            n_features_in=fe_result['n_features_in'],
                            n_features_out=fe_result['n_features_out'],
                            fit_time=fe_result['fit_time'],
                            transform_time=fe_result['transform_time'],
                            train_time=train_time,
                            cv_roc_auc=cv_roc_auc,
                            cv_std=cv_std,
                            test_metrics=test_metrics,
                            memory_usage_mb=fe_result['memory_usage_mb'],
                            feature_engineering_details=fe_result['details'],
                            challenges_addressed=challenges,
                            status='success'
                        )
                        all_results.append(result)
                        
                    except Exception as e:
                        console.print(f"  [red][-] Failed: {e}[/red]\n")
                        logger.error(traceback.format_exc())
                
            except Exception as e:
                console.print(f"[red][-] Dataset loading failed: {e}[/red]")
                logger.error(traceback.format_exc())
    
    # Summary
    console.print(f"\n[bold blue]{'='*90}[/bold blue]")
    console.print(f"[bold blue]RESULTS SUMMARY - COMPLEX KAGGLE BENCHMARK[/bold blue]")
    console.print(f"[bold blue]{'='*90}[/bold blue]\n")
    
    if not all_results:
        console.print("[red]No results to display[/red]")
        return
    
    # Create results table (FeatureCraft only)
    table = Table(
        show_header=True,
        header_style="bold magenta",
        title="FeatureCraft Results Summary",
        box=box.DOUBLE_EDGE
    )
    table.add_column("Dataset", style="cyan", width=15)
    table.add_column("Features", justify="center", width=12)
    table.add_column("CV ROC-AUC", justify="right", width=15)
    table.add_column("Test ROC-AUC", justify="right", style="green", width=12)
    table.add_column("Precision", justify="right", width=10)
    table.add_column("Recall", justify="right", width=10)
    table.add_column("F1-Score", justify="right", width=10)
    table.add_column("Train Time(s)", justify="right", width=12)

    for res in all_results:
        total_time = res.fit_time + res.transform_time + res.train_time

        table.add_row(
            res.dataset.replace('_', ' ').title(),
            f"{res.n_features_in}->{res.n_features_out}",
            f"{res.cv_roc_auc:.4f}±{res.cv_std:.3f}",
            f"{res.test_metrics['roc_auc']:.4f}",
            f"{res.test_metrics['precision']:.3f}",
            f"{res.test_metrics['recall']:.3f}",
            f"{res.test_metrics['f1_score']:.3f}",
            f"{total_time:.1f}"
        )
    
    console.print(table)
    
    # FeatureCraft Performance Summary (Single Approach Test)
    console.print(f"\n[bold green]{'='*90}[/bold green]")
    console.print(f"[bold green]FEATURECRAFT PERFORMANCE SUMMARY[/bold green]")
    console.print(f"[bold green]{'='*90}[/bold green]\n")

    df_results = pd.DataFrame([asdict(r) for r in all_results])

    for dataset in df_results['dataset'].unique():
        dataset_results = df_results[df_results['dataset'] == dataset]
        console.print(f"[bold cyan]{dataset.upper().replace('_', ' ')}:[/bold cyan]")

        # Show FeatureCraft results
        if 'featurecraft' in dataset_results['approach'].values:
            fc_result = dataset_results[dataset_results['approach'] == 'featurecraft'].iloc[0]

            console.print(f"  [+] Featurecraft: {fc_result['n_features_in']:,} -> {fc_result['n_features_out']:,} features")
            console.print(f"  [TIME]  Fit time: {fc_result['fit_time']:.2f}s | Transform: {fc_result['transform_time']:.2f}s")
            console.print(f"  [MEM] Memory: {fc_result['memory_usage_mb']:.1f} MB")
            console.print(f"  [SCORE] CV ROC-AUC: {fc_result['cv_roc_auc']:.4f} (±{fc_result['cv_std']:.4f})")
            console.print(f"  [SCORE] Test ROC-AUC: {fc_result['test_metrics']['roc_auc']:.4f} | AP: {fc_result['test_metrics']['avg_precision']:.4f}")
            console.print(f"  [SCORE] Precision: {fc_result['test_metrics']['precision']:.4f} | Recall: {fc_result['test_metrics']['recall']:.4f} | F1: {fc_result['test_metrics']['f1_score']:.4f}")

        console.print()
    
    # Save results
    results_json = artifacts_base / "complex_benchmark_results.json"
    results_json.write_text(json.dumps([asdict(r) for r in all_results], indent=2, default=str))
    console.print(f"\n[green][+][/green] Results saved to {results_json}")
    
    # Skip comparison plots (single approach test)
    # create_comparison_plots(all_results, artifacts_base)
    
    console.print(f"[green][+][/green] All artifacts saved to {artifacts_base}")
    console.print("\n[bold green]🎉 FeatureCraft performance test complete! 🎉[/bold green]\n")


if __name__ == "__main__":
    main()

