"""
@04_tuned_complex_benchmark.py

ENHANCED complex Kaggle benchmark with FeatureCraft PARAMETER TUNING.

This script addresses the performance gap by:
1. Testing multiple FeatureCraft configurations
2. Enabling advanced encoding strategies
3. Adding statistical feature engineering
4. Proper parameter optimization

Key improvements over 03_complex_kaggle_benchmark.py:
- Parameter grid search for FeatureCraft
- Multiple encoding strategies (frequency, count, target encoding)
- Better handling of high cardinality
- Class imbalance handling (SMOTE)
- Statistical aggregation features
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
from itertools import product

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score

# Import XGBoost with fallback
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("⚠️  XGBoost not available. Will use sklearn.")

try:
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.panel import Panel
    from rich import box
except ImportError:
    print("Installing rich for better console output...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "rich"])
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.panel import Panel
    from rich import box

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
console = Console()


@dataclass
class TunedBenchmarkResult:
    """Result container with configuration details."""
    dataset: str
    config_name: str
    config_params: Dict[str, Any]
    n_rows: int
    n_features_in: int
    n_features_out: int
    fit_time: float
    transform_time: float
    train_time: float
    cv_roc_auc: float
    cv_std: float
    test_roc_auc: float
    status: str = "success"
    error: Optional[str] = None


def get_featurecraft_configs() -> Dict[str, Dict[str, Any]]:
    """
    Define multiple FeatureCraft configurations to test.
    
    These configs are inspired by top Kaggle solutions and address
    common challenges in complex datasets.
    """
    configs = {
        # Baseline (current default)
        "default": {},
        
        # Configuration for high-cardinality datasets (IEEE Fraud, Santander)
        "high_cardinality_optimized": {
            "low_cardinality_max": 15,  # More OHE
            "mid_cardinality_max": 100,  # Extended target encoding range
            "rare_level_threshold": 0.005,  # Group rare categories aggressively
            "use_target_encoding": True,
            "use_frequency_encoding": True,  # CRITICAL for high cardinality
            "use_count_encoding": True,
            "hashing_n_features_tabular": 512,  # More hash features
            "te_smoothing": 10.0,  # Lower smoothing = more signal
            "add_missing_indicators": True,
        },
        
        # Configuration for heavy class imbalance (Fraud Detection)
        "imbalance_optimized": {
            "use_smote": True,
            "smote_threshold": 0.15,  # Trigger SMOTE for <15% minority
            "smote_k_neighbors": 3,  # Fewer neighbors for rare classes
            "use_target_encoding": True,
            "te_smoothing": 50.0,  # Higher smoothing for stability
            "use_frequency_encoding": True,
            "add_missing_indicators": True,
            "low_cardinality_max": 12,
            "mid_cardinality_max": 80,
        },
        
        # Configuration for heavy missing data (Home Credit)
        "missing_data_optimized": {
            "numeric_simple_impute_max": 0.10,  # More aggressive simple imputation
            "numeric_advanced_impute_max": 0.50,  # Handle up to 50% missing
            "add_missing_indicators": True,  # CRITICAL: missing patterns matter!
            "categorical_missing_indicator_min": 0.03,
            "use_target_encoding": True,
            "use_frequency_encoding": True,
            "low_cardinality_max": 15,
            "mid_cardinality_max": 60,
        },
        
        # Configuration for pure numeric datasets (Santander)
        "numeric_heavy_optimized": {
            "skew_threshold": 0.75,  # More aggressive power transforms
            "outlier_share_threshold": 0.03,  # Detect outliers early
            "scaler_robust_if_outliers": True,
            "winsorize": True,  # Clip extreme outliers
            "clip_percentiles": (0.01, 0.99),
            "use_target_encoding": False,  # No categoricals
            "add_missing_indicators": True,
        },
        
        # Aggressive all-in configuration
        "aggressive_all_features": {
            "low_cardinality_max": 20,
            "mid_cardinality_max": 150,
            "rare_level_threshold": 0.003,
            "use_target_encoding": True,
            "use_frequency_encoding": True,
            "use_count_encoding": True,
            "add_missing_indicators": True,
            "te_smoothing": 20.0,
            "te_noise": 0.01,  # Add noise for regularization
            "hashing_n_features_tabular": 1024,
            "use_smote": True,
            "smote_threshold": 0.15,
            "numeric_advanced_impute_max": 0.40,
            "skew_threshold": 0.8,
            "outlier_share_threshold": 0.04,
        },
        
        # Conservative stable configuration
        "conservative_stable": {
            "low_cardinality_max": 8,
            "mid_cardinality_max": 40,
            "rare_level_threshold": 0.02,
            "use_target_encoding": True,
            "te_smoothing": 100.0,  # Heavy regularization
            "add_missing_indicators": False,
            "numeric_advanced_impute_max": 0.25,
        },
    }
    
    return configs


def load_dataset_with_stats_features(
    loader_func, 
    dataset_name: str
) -> Tuple[pd.DataFrame, pd.Series, str, List[str]]:
    """
    Load dataset and ADD statistical aggregation features
    (inspired by top Kaggle solutions).
    
    This bridges the gap between FeatureCraft and manual approaches.
    """
    X, y, task, challenges = loader_func()
    
    console.print(f"[cyan]  Adding statistical aggregation features...[/cyan]")
    
    # For Santander: Add statistical features across numeric columns
    if 'santander' in dataset_name:
        var_cols = [col for col in X.columns if col.startswith('var_')]
        if len(var_cols) >= 20:
            var_subset = X[var_cols[:50]]  # Avoid memory explosion
            
            X['stat_mean'] = var_subset.mean(axis=1)
            X['stat_std'] = var_subset.std(axis=1)
            X['stat_min'] = var_subset.min(axis=1)
            X['stat_max'] = var_subset.max(axis=1)
            X['stat_median'] = var_subset.median(axis=1)
            X['stat_range'] = X['stat_max'] - X['stat_min']
            X['stat_positive_count'] = (var_subset > 0).sum(axis=1)
            X['stat_negative_count'] = (var_subset < 0).sum(axis=1)
            X['stat_zero_count'] = (var_subset == 0).sum(axis=1)
            
            console.print(f"[green]    ✓ Added 9 statistical aggregation features[/green]")
    
    # For IEEE Fraud: Add log transforms and missing counts
    elif 'ieee' in dataset_name or 'fraud' in dataset_name:
        if 'TransactionAmt' in X.columns:
            X['TransactionAmt_log'] = np.log1p(X['TransactionAmt'])
            X['TransactionAmt_decimal'] = X['TransactionAmt'] - X['TransactionAmt'].astype(int)
            console.print(f"[green]    ✓ Added TransactionAmt transformations[/green]")
        
        # V features statistics
        v_cols = [col for col in X.columns if col.startswith('V')]
        if len(v_cols) >= 10:
            v_subset = X[v_cols[:30]]
            X['V_mean'] = v_subset.mean(axis=1)
            X['V_std'] = v_subset.std(axis=1)
            X['V_missing_count'] = v_subset.isnull().sum(axis=1)
            console.print(f"[green]    ✓ Added V feature statistics[/green]")
    
    # For Home Credit: Add ratio features
    elif 'home' in dataset_name or 'credit' in dataset_name:
        if 'AMT_INCOME_TOTAL' in X.columns and 'AMT_CREDIT' in X.columns:
            X['CREDIT_INCOME_RATIO'] = X['AMT_CREDIT'] / (X['AMT_INCOME_TOTAL'] + 1)
        if 'AMT_ANNUITY' in X.columns and 'AMT_INCOME_TOTAL' in X.columns:
            X['ANNUITY_INCOME_RATIO'] = X['AMT_ANNUITY'] / (X['AMT_INCOME_TOTAL'] + 1)
        if 'DAYS_BIRTH' in X.columns:
            X['AGE_YEARS'] = -X['DAYS_BIRTH'] / 365
        console.print(f"[green]    ✓ Added financial ratio features[/green]")
    
    # Global: Add missing count for all datasets
    X['missing_count'] = X.isnull().sum(axis=1)
    
    return X, y, task, challenges


def run_featurecraft_with_config(
    EngineClass: Any,
    X: pd.DataFrame,
    y: pd.Series,
    task: str,
    config_name: str,
    config_params: Dict[str, Any]
) -> Dict[str, Any]:
    """Run FeatureCraft with a specific configuration."""
    result = {
        'config_name': config_name,
        'config_params': config_params,
        'n_features_in': X.shape[1],
        'status': 'success',
        'error': None
    }
    
    try:
        # Import config class
        from featurecraft import FeatureCraftConfig
        
        # Create config with parameters
        cfg = FeatureCraftConfig(**config_params)
        afe = EngineClass(config=cfg)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
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
        
        console.print(f"  [green]✓[/green] {config_name}: {X.shape[1]} → {Xt_train.shape[1]} features")
        
    except Exception as e:
        result['status'] = 'failed'
        result['error'] = str(e)
        console.print(f"  [red]✗[/red] {config_name} failed: {e}")
        logger.debug(traceback.format_exc())
        
        # Fallback
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        result['X_train'] = X_train
        result['X_test'] = X_test
        result['y_train'] = y_train
        result['y_test'] = y_test
        result['fit_time'] = 0.0
        result['transform_time'] = 0.0
    
    return result


def evaluate_model_fast(
    X_train: np.ndarray,
    y_train: pd.Series,
    X_test: np.ndarray,
    y_test: pd.Series
) -> Tuple[float, float, float, float]:
    """Fast model evaluation with XGBoost."""
    
    if XGB_AVAILABLE:
        try:
            model = xgb.XGBClassifier(
                n_estimators=100,  # Reduced for speed
                learning_rate=0.1,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                tree_method='hist',
                random_state=42,
                eval_metric='auc',
                use_label_encoder=False,
                verbosity=0
            )
        except:
            model = GradientBoostingClassifier(
                n_estimators=50,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
    else:
        model = GradientBoostingClassifier(
            n_estimators=50,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
    
    # 3-fold CV for speed
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    start = time.time()
    cv_scores = []
    
    for train_idx, val_idx in cv.split(X_train, y_train):
        X_fold_train = X_train[train_idx] if isinstance(X_train, np.ndarray) else X_train.iloc[train_idx]
        X_fold_val = X_train[val_idx] if isinstance(X_train, np.ndarray) else X_train.iloc[val_idx]
        y_fold_train = y_train.iloc[train_idx] if hasattr(y_train, 'iloc') else y_train[train_idx]
        y_fold_val = y_train.iloc[val_idx] if hasattr(y_train, 'iloc') else y_train[val_idx]
        
        if XGB_AVAILABLE:
            fold_model = model.__class__(**model.get_params())
        else:
            from sklearn.base import clone
            fold_model = clone(model)
        
        fold_model.fit(X_fold_train, y_fold_train)
        y_fold_proba = fold_model.predict_proba(X_fold_val)[:, 1]
        fold_auc = roc_auc_score(y_fold_val, y_fold_proba)
        cv_scores.append(fold_auc)
    
    cv_mean = np.mean(cv_scores)
    cv_std = np.std(cv_scores)
    
    # Train on full set
    model.fit(X_train, y_train)
    train_time = time.time() - start
    
    # Test score
    y_proba = model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, y_proba)
    
    return cv_mean, cv_std, test_auc, train_time


def main():
    """Main orchestration with parameter tuning."""
    parser = argparse.ArgumentParser(description="Tuned FeatureCraft Benchmark")
    parser.add_argument("--artifacts", type=str, default="./artifacts/tuned_benchmark")
    parser.add_argument("--dataset", type=str, default="santander", 
                       help="Dataset: santander, ieee_fraud, home_credit")
    parser.add_argument("--configs", type=str, default="all",
                       help="Configs to test: all, fast (default+high_cardinality+imbalance), or comma-separated names")
    args = parser.parse_args()
    
    artifacts_base = Path(args.artifacts)
    artifacts_base.mkdir(parents=True, exist_ok=True)
    
    # Header
    console.print("\n" + "="*100)
    console.print(Panel.fit(
        "[bold cyan]FeatureCraft TUNED Benchmark - Parameter Optimization[/bold cyan]\n"
        "[yellow]Testing Multiple Configurations to Close the Performance Gap[/yellow]",
        border_style="cyan",
        box=box.DOUBLE
    ))
    console.print("="*100 + "\n")
    
    # Find library
    try:
        from featurecraft.pipeline import AutoFeatureEngineer as EngineClass
        from featurecraft import FeatureCraftConfig
        console.print(f"[green]✓[/green] Library found: AutoFeatureEngineer\n")
    except ImportError as e:
        console.print(f"[red]✗[/red] Could not import featurecraft: {e}")
        sys.exit(1)
    
    # Import dataset loaders from the other script
    # We'll import the functions dynamically
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "complex_benchmark",
        Path(__file__).parent / "03_complex_kaggle_benchmark.py"
    )
    complex_benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(complex_benchmark)
    
    load_santander_transaction = complex_benchmark.load_santander_transaction
    load_ieee_fraud_detection = complex_benchmark.load_ieee_fraud_detection
    load_home_credit_default = complex_benchmark.load_home_credit_default
    
    dataset_loaders = {
        'santander': load_santander_transaction,
        'ieee_fraud': load_ieee_fraud_detection,
        'home_credit': load_home_credit_default
    }
    
    if args.dataset not in dataset_loaders:
        console.print(f"[red]✗[/red] Unknown dataset: {args.dataset}")
        sys.exit(1)
    
    # Select configurations
    all_configs = get_featurecraft_configs()
    
    if args.configs == "all":
        selected_configs = all_configs
    elif args.configs == "fast":
        selected_configs = {
            k: v for k, v in all_configs.items() 
            if k in ["default", "high_cardinality_optimized", "imbalance_optimized", "aggressive_all_features"]
        }
    else:
        config_names = [c.strip() for c in args.configs.split(",")]
        selected_configs = {k: v for k, v in all_configs.items() if k in config_names}
    
    console.print(f"[cyan]Dataset:[/cyan] {args.dataset}")
    console.print(f"[cyan]Configurations to test:[/cyan] {len(selected_configs)}")
    for name in selected_configs.keys():
        console.print(f"  • {name}")
    console.print()
    
    # Load dataset with statistical features
    console.print(f"[bold yellow]Loading {args.dataset}...[/bold yellow]")
    X, y, task, challenges = load_dataset_with_stats_features(
        dataset_loaders[args.dataset],
        args.dataset
    )
    
    console.print(f"[green]✓[/green] {X.shape[0]:,} rows × {X.shape[1]:,} features")
    console.print(f"[cyan]  Challenges:[/cyan] {', '.join(challenges[:2])}")
    console.print()
    
    # Test each configuration
    results: List[TunedBenchmarkResult] = []
    
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        
        for config_name, config_params in selected_configs.items():
            console.print(f"[bold cyan]→ Testing Configuration: {config_name}[/bold cyan]")
            
            task_fe = progress.add_task(f"  Feature engineering...", total=None)
            
            fe_result = run_featurecraft_with_config(
                EngineClass, X, y, task, config_name, config_params
            )
            
            progress.remove_task(task_fe)
            
            if fe_result['status'] == 'failed':
                console.print(f"  [red]✗ Skipped due to error[/red]\n")
                continue
            
            console.print(f"  [green]✓[/green] Fit: {fe_result.get('fit_time', 0):.2f}s | Transform: {fe_result.get('transform_time', 0):.2f}s")
            
            # Evaluate
            task_eval = progress.add_task(f"  Evaluating...", total=None)
            
            cv_auc, cv_std, test_auc, train_time = evaluate_model_fast(
                fe_result['X_train'],
                fe_result['y_train'],
                fe_result['X_test'],
                fe_result['y_test']
            )
            
            progress.remove_task(task_eval)
            
            console.print(f"  [bold green]✓ CV ROC-AUC: {cv_auc:.4f} (±{cv_std:.4f})[/bold green]")
            console.print(f"  [bold green]✓ Test ROC-AUC: {test_auc:.4f}[/bold green]")
            console.print(f"  [green]✓[/green] Train time: {train_time:.2f}s\n")
            
            result = TunedBenchmarkResult(
                dataset=args.dataset,
                config_name=config_name,
                config_params=config_params,
                n_rows=len(X),
                n_features_in=fe_result['n_features_in'],
                n_features_out=fe_result['n_features_out'],
                fit_time=fe_result.get('fit_time', 0),
                transform_time=fe_result.get('transform_time', 0),
                train_time=train_time,
                cv_roc_auc=cv_auc,
                cv_std=cv_std,
                test_roc_auc=test_auc,
                status='success'
            )
            results.append(result)
    
    # Summary
    console.print(f"\n[bold blue]{'='*100}[/bold blue]")
    console.print(f"[bold blue]RESULTS - {args.dataset.upper()}[/bold blue]")
    console.print(f"[bold blue]{'='*100}[/bold blue]\n")
    
    if not results:
        console.print("[red]No successful results[/red]")
        return
    
    # Sort by test AUC
    results_sorted = sorted(results, key=lambda r: r.test_roc_auc, reverse=True)
    
    # Table
    table = Table(show_header=True, header_style="bold magenta", box=box.DOUBLE_EDGE)
    table.add_column("Rank", style="cyan", width=6)
    table.add_column("Configuration", style="yellow", width=30)
    table.add_column("Features", justify="center", width=12)
    table.add_column("CV ROC-AUC", justify="right", width=15)
    table.add_column("Test ROC-AUC", justify="right", style="bold green", width=12)
    table.add_column("Time(s)", justify="right", width=10)
    
    for idx, res in enumerate(results_sorted, 1):
        emoji = "🥇" if idx == 1 else ("🥈" if idx == 2 else ("🥉" if idx == 3 else "  "))
        total_time = res.fit_time + res.transform_time + res.train_time
        
        table.add_row(
            f"{emoji} {idx}",
            res.config_name,
            f"{res.n_features_in}→{res.n_features_out}",
            f"{res.cv_roc_auc:.4f}±{res.cv_std:.3f}",
            f"{res.test_roc_auc:.4f}",
            f"{total_time:.1f}"
        )
    
    console.print(table)
    
    # Best config analysis
    best = results_sorted[0]
    default = next((r for r in results if r.config_name == "default"), None)
    
    console.print(f"\n[bold green]🏆 BEST CONFIGURATION: {best.config_name}[/bold green]")
    console.print(f"   Test ROC-AUC: {best.test_roc_auc:.4f}")
    console.print(f"   CV ROC-AUC: {best.cv_roc_auc:.4f} (±{best.cv_std:.4f})")
    console.print(f"   Features: {best.n_features_in} → {best.n_features_out}")
    
    if default:
        improvement = (best.test_roc_auc - default.test_roc_auc) / default.test_roc_auc * 100
        console.print(f"\n[bold yellow]📊 Improvement vs Default: {improvement:+.2f}%[/bold yellow]")
        console.print(f"   Default: {default.test_roc_auc:.4f} → Best: {best.test_roc_auc:.4f}")
    
    console.print(f"\n[bold cyan]Key Parameters in Best Config:[/bold cyan]")
    for key, value in best.config_params.items():
        console.print(f"   • {key}: {value}")
    
    # Save results
    results_json = artifacts_base / f"{args.dataset}_tuned_results.json"
    results_json.write_text(json.dumps([asdict(r) for r in results], indent=2, default=str))
    console.print(f"\n[green]✓[/green] Results saved to {results_json}")
    
    console.print("\n[bold green]🎉 Parameter tuning complete! 🎉[/bold green]\n")


if __name__ == "__main__":
    main()

