"""Scaling utilities for FeatureCraft."""

from __future__ import annotations

from typing import Optional

from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler

from .config import FeatureCraftConfig


def choose_scaler(
    estimator_family: str, 
    heavy_outliers: bool,
    cfg: Optional[FeatureCraftConfig] = None,
) -> StandardScaler | MinMaxScaler | RobustScaler | MaxAbsScaler | None:
    """Choose appropriate scaler based on estimator family and outlier profile.
    
    Args:
        estimator_family: Estimator family (tree, linear, svm, knn, nn)
        heavy_outliers: Whether heavy outliers are present
        cfg: Configuration with scaler preferences
        
    Returns:
        Scaler instance or None
    """
    if cfg is None:
        cfg = FeatureCraftConfig()
    
    fam = estimator_family.lower()
    
    # If configured to use RobustScaler for outliers and outliers detected
    if heavy_outliers and cfg.scaler_robust_if_outliers:
        return RobustScaler()
    
    # Get scaler choice from config based on estimator family
    scaler_choice = None
    if fam in {"tree", "gbm", "boost", "xgboost", "lightgbm"}:
        scaler_choice = cfg.scaler_tree
    elif fam in {"linear", "logistic", "ridge", "lasso"}:
        scaler_choice = cfg.scaler_linear
    elif fam in {"svm", "svr", "svc"}:
        scaler_choice = cfg.scaler_svm
    elif fam in {"knn", "k-nn", "nearest"}:
        scaler_choice = cfg.scaler_knn
    elif fam in {"nn", "neural", "mlp"}:
        scaler_choice = cfg.scaler_nn
    else:
        scaler_choice = cfg.scaler_linear  # Default fallback
    
    # Create scaler based on choice
    scaler_choice = scaler_choice.lower()
    if scaler_choice == "none":
        return None
    elif scaler_choice == "standard":
        return StandardScaler()
    elif scaler_choice == "minmax":
        return MinMaxScaler()
    elif scaler_choice == "robust":
        return RobustScaler()
    elif scaler_choice == "maxabs":
        return MaxAbsScaler()
    else:
        return StandardScaler()  # Safe default
