"""Interactive wizard and consent-driven prompts for FeatureCraft CLI."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt

from .config import FeatureCraftConfig
from .logging import get_logger
from .settings import save_config

logger = get_logger(__name__)
console = Console()


def run_wizard(output_path: Optional[str] = None) -> FeatureCraftConfig:
    """Run interactive configuration wizard.
    
    Args:
        output_path: Path to save generated config file (YAML)
        
    Returns:
        Generated FeatureCraftConfig instance
    """
    console.print(Panel.fit(
        "[bold cyan]FeatureCraft Configuration Wizard[/bold cyan]\n\n"
        "Answer a few questions to generate your custom configuration.\n"
        "Press Enter to accept defaults (shown in brackets).",
        border_style="cyan"
    ))
    
    config_dict: Dict[str, Any] = {}
    
    # === General Settings ===
    console.print("\n[bold yellow]General Settings[/bold yellow]")
    
    random_state = Prompt.ask(
        "Random seed for reproducibility",
        default="42"
    )
    config_dict["random_state"] = int(random_state)
    
    artifacts_dir = Prompt.ask(
        "Artifacts output directory",
        default="artifacts"
    )
    config_dict["artifacts_dir"] = artifacts_dir
    
    # === Encoding Settings ===
    console.print("\n[bold yellow]Encoding Settings[/bold yellow]")
    
    low_card = Prompt.ask(
        "Max unique values for one-hot encoding (low cardinality)",
        default="10"
    )
    config_dict["low_cardinality_max"] = int(low_card)
    
    mid_card = Prompt.ask(
        "Max unique values for target encoding (mid cardinality)",
        default="50"
    )
    config_dict["mid_cardinality_max"] = int(mid_card)
    
    rare_threshold = Prompt.ask(
        "Rare category threshold (categories below this % are grouped)",
        default="0.01"
    )
    config_dict["rare_level_threshold"] = float(rare_threshold)
    
    # === SMOTE & Imbalance ===
    console.print("\n[bold yellow]Class Imbalance Handling[/bold yellow]")
    
    use_smote = Confirm.ask(
        "Enable SMOTE oversampling for imbalanced classification?",
        default=False
    )
    config_dict["use_smote"] = use_smote
    
    if use_smote:
        smote_threshold = Prompt.ask(
            "Minority class ratio threshold to trigger SMOTE (e.g., 0.10 = 10%)",
            default="0.10"
        )
        config_dict["smote_threshold"] = float(smote_threshold)
        
        smote_k = Prompt.ask(
            "Number of nearest neighbors for SMOTE",
            default="5"
        )
        config_dict["smote_k_neighbors"] = int(smote_k)
    
    # === Text Processing ===
    console.print("\n[bold yellow]Text Processing[/bold yellow]")
    
    text_hashing = Confirm.ask(
        "Use HashingVectorizer for text (faster, less memory)?",
        default=False
    )
    config_dict["text_use_hashing"] = text_hashing
    
    if text_hashing:
        text_hash_features = Prompt.ask(
            "Number of hashing features for text",
            default="16384"
        )
        config_dict["text_hashing_features"] = int(text_hash_features)
    else:
        tfidf_max = Prompt.ask(
            "Max TF-IDF features",
            default="20000"
        )
        config_dict["tfidf_max_features"] = int(tfidf_max)
    
    char_ngrams = Confirm.ask(
        "Use character n-grams for text?",
        default=False
    )
    config_dict["text_char_ngrams"] = char_ngrams
    
    # === Dimensionality Reduction ===
    console.print("\n[bold yellow]Dimensionality Reduction[/bold yellow]")
    
    use_reducer = Confirm.ask(
        "Enable dimensionality reduction (PCA/SVD/UMAP)?",
        default=False
    )
    
    if use_reducer:
        reducer_kind = Prompt.ask(
            "Reducer type",
            choices=["pca", "svd", "umap"],
            default="pca"
        )
        config_dict["reducer_kind"] = reducer_kind
        
        reducer_components = Prompt.ask(
            "Number of components",
            default="50"
        )
        config_dict["reducer_components"] = int(reducer_components)
    
    # === Time Series ===
    console.print("\n[bold yellow]Time Series Features[/bold yellow]")
    
    use_fourier = Confirm.ask(
        "Add Fourier features for cyclical time patterns?",
        default=False
    )
    config_dict["use_fourier"] = use_fourier
    
    if use_fourier:
        fourier_orders = Prompt.ask(
            "Fourier orders (comma-separated, e.g., 3,7)",
            default="3,7"
        )
        config_dict["fourier_orders"] = [int(x.strip()) for x in fourier_orders.split(",")]
    
    holiday_country = Prompt.ask(
        "Country code for holiday features (e.g., US, GB) or leave empty",
        default=""
    )
    if holiday_country:
        config_dict["holiday_country"] = holiday_country
    
    # === Drift Detection ===
    console.print("\n[bold yellow]Drift Detection[/bold yellow]")
    
    enable_drift = Confirm.ask(
        "Enable drift detection?",
        default=False
    )
    config_dict["enable_drift_detection"] = enable_drift
    
    if enable_drift:
        drift_psi = Prompt.ask(
            "PSI threshold for categorical drift (>0.25 = significant)",
            default="0.25"
        )
        config_dict["drift_psi_threshold"] = float(drift_psi)
        
        drift_ks = Prompt.ask(
            "KS threshold for numeric drift (>0.1 = significant)",
            default="0.10"
        )
        config_dict["drift_ks_threshold"] = float(drift_ks)
    
    # === SHAP Explainability ===
    console.print("\n[bold yellow]Explainability[/bold yellow]")
    
    enable_shap = Confirm.ask(
        "Enable SHAP explainability (requires shap package)?",
        default=False
    )
    config_dict["enable_shap"] = enable_shap
    
    if enable_shap:
        shap_samples = Prompt.ask(
            "Max samples for SHAP computation",
            default="100"
        )
        config_dict["shap_max_samples"] = int(shap_samples)
    
    # === Create Config ===
    try:
        cfg = FeatureCraftConfig(**config_dict)
    except Exception as e:
        console.print(f"[red]Configuration validation failed: {e}[/red]")
        raise typer.Exit(code=1)
    
    # === Save Config ===
    if output_path:
        try:
            save_config(cfg, output_path, format="yaml")
            console.print(f"\n[green]✓[/green] Configuration saved to: {output_path}")
        except Exception as e:
            console.print(f"[red]Failed to save config: {e}[/red]")
            raise typer.Exit(code=1)
    
    # === Show Equivalent CLI Flags ===
    console.print("\n[bold cyan]Equivalent CLI Flags:[/bold cyan]")
    cli_flags = _config_to_cli_flags(config_dict)
    console.print(f"  {cli_flags}")
    
    console.print("\n[green]✓ Configuration complete![/green]")
    return cfg


def _config_to_cli_flags(config_dict: Dict[str, Any]) -> str:
    """Convert config dict to CLI --set flags."""
    flags = []
    for key, value in config_dict.items():
        if isinstance(value, bool):
            value_str = str(value).lower()
        elif isinstance(value, list):
            value_str = ",".join(str(v) for v in value)
        else:
            value_str = str(value)
        flags.append(f"--set {key}={value_str}")
    return " ".join(flags)


def ask_smote_consent(minority_ratio: float, threshold: float = 0.10) -> bool:
    """Ask user if they want to enable SMOTE for imbalanced data.
    
    Args:
        minority_ratio: Detected minority class ratio
        threshold: Default SMOTE threshold
        
    Returns:
        True if user wants to enable SMOTE
    """
    console.print(
        f"\n[yellow]⚠[/yellow] Class imbalance detected: "
        f"minority class is {minority_ratio:.2%} of data"
    )
    return Confirm.ask(
        f"Enable SMOTE oversampling (threshold={threshold:.2%})?",
        default=True
    )


def ask_imputation_strategy(missing_rate: float) -> str:
    """Ask user to choose imputation strategy for high missingness.
    
    Args:
        missing_rate: Fraction of missing values
        
    Returns:
        Imputation strategy: 'simple' or 'iterative'
    """
    console.print(
        f"\n[yellow]⚠[/yellow] High missingness detected: "
        f"{missing_rate:.2%} of values are missing"
    )
    choice = Prompt.ask(
        "Choose imputation strategy",
        choices=["simple", "iterative"],
        default="iterative"
    )
    return choice


def ask_hashing_dimensions(n_high_cardinality_cols: int) -> int:
    """Ask user for hashing feature dimension for high-card columns.
    
    Args:
        n_high_cardinality_cols: Number of high-cardinality columns detected
        
    Returns:
        Number of hashing features
    """
    console.print(
        f"\n[yellow]⚠[/yellow] {n_high_cardinality_cols} high-cardinality "
        "columns detected (>50 unique values)"
    )
    dimensions = Prompt.ask(
        "Number of hashing features for encoding",
        default="256"
    )
    return int(dimensions)


def ask_reducer_choice(n_features: int) -> Optional[str]:
    """Ask user if they want dimensionality reduction.
    
    Args:
        n_features: Number of features before reduction
        
    Returns:
        Reducer kind ('pca', 'svd', 'umap') or None
    """
    console.print(
        f"\n[yellow]ℹ[/yellow] {n_features} features generated"
    )
    
    if not Confirm.ask("Enable dimensionality reduction?", default=False):
        return None
    
    choice = Prompt.ask(
        "Choose reducer",
        choices=["pca", "svd", "umap"],
        default="pca"
    )
    return choice


def interactive_fit_questions(
    X, y, cfg: FeatureCraftConfig
) -> Dict[str, Any]:
    """Ask interactive questions during fit if --interactive is enabled.
    
    Args:
        X: Feature DataFrame
        y: Target Series
        cfg: Current configuration
        
    Returns:
        Dict of configuration overrides based on user responses
    """
    overrides = {}
    
    # Check for class imbalance
    if hasattr(y, 'value_counts'):
        vc = y.value_counts()
        if len(vc) > 0:
            minority_ratio = float(vc.min() / vc.sum())
            if minority_ratio < cfg.smote_threshold and not cfg.use_smote:
                if ask_smote_consent(minority_ratio, cfg.smote_threshold):
                    overrides["use_smote"] = True
                    
                    # Ask for k_neighbors
                    k = Prompt.ask(
                        "SMOTE k_neighbors",
                        default=str(cfg.smote_k_neighbors)
                    )
                    overrides["smote_k_neighbors"] = int(k)
    
    # Check for high missingness
    missing_rate = X.isna().sum().sum() / (X.shape[0] * X.shape[1])
    if missing_rate > cfg.numeric_advanced_impute_max:
        strategy = ask_imputation_strategy(missing_rate)
        if strategy == "iterative":
            overrides["numeric_advanced_impute_max"] = min(0.5, missing_rate + 0.1)
    
    # Check for high-cardinality columns
    high_card_cols = []
    for col in X.select_dtypes(include=['object', 'category']).columns:
        if X[col].nunique() > cfg.mid_cardinality_max:
            high_card_cols.append(col)
    
    if len(high_card_cols) > 0:
        n_features = ask_hashing_dimensions(len(high_card_cols))
        overrides["hashing_n_features_tabular"] = n_features
    
    return overrides

