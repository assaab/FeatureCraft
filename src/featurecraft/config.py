"""Configuration settings for FeatureCraft."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator


class FeatureCraftConfig(BaseModel):
    """Configuration for FeatureCraft feature engineering pipeline.
    
    All parameters can be set via:
    - Python API: FeatureCraftConfig(param=value)
    - Environment variables: FEATURECRAFT__PARAM=value
    - Config file: YAML/JSON/TOML
    - CLI: --set param=value
    """
    
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    # ========== General ==========
    random_state: int = Field(default=42, ge=0, description="Random seed for reproducibility")
    verbosity: int = Field(default=1, ge=0, le=3, description="Logging verbosity (0=quiet, 3=debug)")
    artifacts_dir: str = Field(default="artifacts", description="Directory for artifacts and outputs")
    dry_run: bool = Field(default=False, description="Dry run mode (no file writes)")
    fail_fast: bool = Field(default=False, description="Stop on first error instead of continuing")

    # ========== Missing Values ==========
    numeric_simple_impute_max: float = Field(
        default=0.05, ge=0.0, le=1.0, 
        description="Threshold for simple imputation (<=5% missing)"
    )
    numeric_advanced_impute_max: float = Field(
        default=0.30, ge=0.0, le=1.0,
        description="Max missingness for advanced imputation (<=30%)"
    )
    categorical_impute_strategy: str = Field(
        default="most_frequent", 
        description="Strategy for categorical imputation: most_frequent, constant"
    )
    categorical_missing_indicator_min: float = Field(
        default=0.05, ge=0.0, le=1.0,
        description="Min missingness to add missing indicator"
    )
    add_missing_indicators: bool = Field(
        default=True, 
        description="Add binary missing indicators for high-missingness features"
    )

    # ========== Encoding ==========
    low_cardinality_max: int = Field(
        default=10, ge=1, le=1000,
        description="Max unique values for one-hot encoding"
    )
    mid_cardinality_max: int = Field(
        default=50, ge=1, le=10000,
        description="Max unique values for target encoding"
    )
    rare_level_threshold: float = Field(
        default=0.01, ge=0.0, le=1.0,
        description="Frequency threshold for grouping rare categories (<1% -> 'Other')"
    )
    missing_sentinel: str = Field(
        default="__MISSING__",
        description="Sentinel string to replace NaN/None in categorical features before encoding"
    )
    ohe_handle_unknown: str = Field(
        default="infrequent_if_exist",
        description="How OHE handles unknown categories"
    )
    hashing_n_features_tabular: int = Field(
        default=256, ge=8, le=8192,
        description="Number of hash features for high-cardinality categoricals"
    )
    use_target_encoding: bool = Field(
        default=True,
        description="Enable out-of-fold target encoding for mid-cardinality features"
    )
    use_leave_one_out_te: bool = Field(
        default=False,
        description="Use Leave-One-Out Target Encoding instead of out-of-fold K-Fold TE"
    )
    use_frequency_encoding: bool = Field(
        default=False,
        description="Enable frequency encoding (category → frequency count)"
    )
    use_count_encoding: bool = Field(
        default=False,
        description="Enable count encoding (category → occurrence count)"
    )
    target_encoding_noise: float = Field(
        default=0.01, ge=0.0, le=1.0,
        description="Gaussian noise std for target encoding regularization"
    )
    target_encoding_smoothing: float = Field(
        default=0.3, ge=0.0, le=10.0,
        description="Smoothing factor for target encoding (deprecated: use te_smoothing)"
    )
    use_ordinal: bool = Field(
        default=False,
        description="Use ordinal encoding for specified columns"
    )
    ordinal_maps: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Manual ordinal category ordering per column"
    )
    use_woe: bool = Field(
        default=False,
        description="Use Weight of Evidence encoding for binary classification"
    )

    # ========== Scaling & Transforms ==========
    skew_threshold: float = Field(
        default=1.0, ge=0.0,
        description="Absolute skewness threshold for power transforms"
    )
    outlier_share_threshold: float = Field(
        default=0.05, ge=0.0, le=1.0,
        description="Fraction of outliers (>1.5*IQR) to trigger robust scaling"
    )
    scaler_linear: str = Field(
        default="standard",
        description="Scaler for linear models: standard, minmax, robust, maxabs, none"
    )
    scaler_svm: str = Field(
        default="standard",
        description="Scaler for SVM: standard, minmax, robust, maxabs, none"
    )
    scaler_knn: str = Field(
        default="minmax",
        description="Scaler for k-NN: standard, minmax, robust, maxabs, none"
    )
    scaler_nn: str = Field(
        default="minmax",
        description="Scaler for neural networks: standard, minmax, robust, maxabs, none"
    )
    scaler_tree: str = Field(
        default="none",
        description="Scaler for tree models: none, standard, minmax, robust, maxabs"
    )
    scaler_robust_if_outliers: bool = Field(
        default=True,
        description="Automatically use RobustScaler if heavy outliers detected"
    )
    winsorize: bool = Field(
        default=False,
        description="Apply winsorization to clip extreme outliers"
    )
    clip_percentiles: Tuple[float, float] = Field(
        default=(0.01, 0.99),
        description="Percentiles for clipping if winsorize=True"
    )

    # ========== Selection ==========
    corr_drop_threshold: float = Field(
        default=0.95, ge=0.0, le=1.0,
        description="Correlation threshold for dropping redundant features"
    )
    vif_drop_threshold: float = Field(
        default=10.0, ge=1.0,
        description="VIF threshold for multicollinearity pruning"
    )
    use_mi: bool = Field(
        default=False,
        description="Use mutual information for feature selection"
    )
    mi_top_k: Optional[int] = Field(
        default=None, ge=1,
        description="Keep top K features by mutual information"
    )
    use_woe_selection: bool = Field(
        default=False,
        description="Use WoE/IV-based feature selection for binary classification"
    )
    woe_iv_threshold: float = Field(
        default=0.02, ge=0.0,
        description="Minimum Information Value threshold for WoE-based feature selection"
    )

    # ========== Text ==========
    tfidf_max_features: int = Field(
        default=20000, ge=100, le=1000000,
        description="Max features for TF-IDF vectorizer"
    )
    ngram_range: Tuple[int, int] = Field(
        default=(1, 2),
        description="N-gram range for text vectorization"
    )
    text_use_hashing: bool = Field(
        default=False,
        description="Use HashingVectorizer instead of TF-IDF for text"
    )
    text_hashing_features: int = Field(
        default=16384, ge=1024, le=131072,
        description="Number of features for text hashing"
    )
    text_char_ngrams: bool = Field(
        default=False,
        description="Use character n-grams for text"
    )
    hashing_n_features_text: int = Field(
        default=4096, ge=64, le=32768,
        description="(Deprecated: use text_hashing_features) Hash features for text"
    )
    svd_components_for_trees: int = Field(
        default=200, ge=2, le=1000,
        description="SVD components for text when using tree models"
    )

    # ========== Datetime & Time Series ==========
    ts_default_lags: List[int] = Field(
        default_factory=lambda: [1, 7, 28],
        description="Default lag periods for time series"
    )
    ts_default_windows: List[int] = Field(
        default_factory=lambda: [3, 7, 28],
        description="Default rolling window sizes for time series"
    )
    use_fourier: bool = Field(
        default=False,
        description="Add Fourier features for cyclical time patterns"
    )
    fourier_orders: List[int] = Field(
        default_factory=lambda: [3, 7],
        description="Fourier series orders (e.g., daily, weekly cycles)"
    )
    holiday_country: Optional[str] = Field(
        default=None,
        description="ISO country code for holiday features (e.g., 'US', 'GB')"
    )
    time_column: Optional[str] = Field(
        default=None,
        description="Name of time/date column for time series features"
    )
    time_order: Optional[str] = Field(
        default=None,
        description="Column to sort by for time-ordered operations"
    )

    # ========== Reducers ==========
    reducer_kind: Optional[str] = Field(
        default=None,
        description="Dimensionality reduction: none, pca, svd, umap, or None"
    )
    reducer_components: Optional[int] = Field(
        default=None, ge=2, le=1000,
        description="Number of components for reducer"
    )
    reducer_variance: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="Explained variance ratio for PCA (alternative to n_components)"
    )

    # ========== Imbalance ==========
    use_smote: bool = Field(
        default=False,
        description="Enable SMOTE oversampling for imbalanced classification"
    )
    smote_threshold: float = Field(
        default=0.10, ge=0.0, le=1.0,
        description="Minority class ratio threshold to trigger SMOTE (<10%)"
    )
    smote_k_neighbors: int = Field(
        default=5, ge=1, le=20,
        description="Number of nearest neighbors for SMOTE"
    )
    smote_strategy: str = Field(
        default="auto",
        description="SMOTE sampling strategy: auto, minority, all"
    )
    use_undersample: bool = Field(
        default=False,
        description="Enable random undersampling of majority class"
    )
    class_weight_threshold: float = Field(
        default=0.20, ge=0.0, le=1.0,
        description="Minority ratio threshold for class_weight advisory"
    )

    # ========== Drift ==========
    enable_drift_detection: bool = Field(
        default=False,
        description="Enable data drift detection and reporting"
    )
    enable_drift_report: bool = Field(
        default=False,
        description="Generate drift report in analyze() if reference_path provided"
    )
    drift_psi_threshold: float = Field(
        default=0.25, ge=0.0, le=1.0,
        description="PSI threshold for categorical drift (>0.25 = significant)"
    )
    drift_ks_threshold: float = Field(
        default=0.10, ge=0.0, le=1.0,
        description="KS statistic threshold for numeric drift (>0.1 = significant)"
    )
    reference_path: Optional[str] = Field(
        default=None,
        description="Path to reference dataset (CSV/parquet) for drift comparison in analyze()"
    )

    # ========== Explainability ==========
    enable_shap: bool = Field(
        default=False,
        description="Enable SHAP explainability features"
    )
    shap_max_samples: int = Field(
        default=100, ge=10, le=10000,
        description="Max samples for SHAP computation"
    )

    # ========== Sampling & CV ==========
    sample_n: Optional[int] = Field(
        default=None, ge=100,
        description="Fixed number of samples to use (for large datasets)"
    )
    sample_frac: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="Fraction of samples to use"
    )
    stratify_by: Optional[str] = Field(
        default=None,
        description="Column name for stratified sampling/splitting"
    )
    cv_n_splits: int = Field(
        default=5, ge=2, le=20,
        description="Number of cross-validation folds for target encoding and CV operations"
    )
    cv_strategy: str = Field(
        default="kfold",
        description="CV strategy for target encoding: kfold, stratified, group, time"
    )
    cv_shuffle: bool = Field(
        default=True,
        description="Whether to shuffle data in KFold/StratifiedKFold"
    )
    cv_random_state: Optional[int] = Field(
        default=None,
        description="Random state for CV splits (uses random_state if None)"
    )
    use_group_kfold: bool = Field(
        default=False,
        description="Use GroupKFold for CV (requires groups_column)"
    )
    groups_column: Optional[str] = Field(
        default=None,
        description="Column name for group-based CV splitting (for GroupKFold or group-aware encoding)"
    )
    
    # ========== Target Encoding ==========
    te_smoothing: float = Field(
        default=20.0, ge=0.0,
        description="Smoothing parameter for target encoding (higher = more regularization)"
    )
    te_noise: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Gaussian noise standard deviation for target encoding regularization"
    )
    te_prior: str = Field(
        default="global_mean",
        description="Prior strategy for target encoding: global_mean, median"
    )

    # ========== Reporting ==========
    template_dir: Optional[str] = Field(
        default=None,
        description="Custom templates directory for HTML reports"
    )
    embed_figures: bool = Field(
        default=True,
        description="Embed figures as base64 in HTML reports"
    )
    open_report: bool = Field(
        default=False,
        description="Automatically open report in browser after generation"
    )
    report_filename: str = Field(
        default="report.html",
        description="Filename for generated HTML report"
    )
    max_corr_features: int = Field(
        default=60, ge=2, le=500,
        description="Max features to include in correlation heatmap"
    )

    # ========== Schema Validation ==========
    validate_schema: bool = Field(
        default=True,
        description="Enable schema validation before fit/transform to detect data drift and type errors"
    )
    schema_path: Optional[str] = Field(
        default=None,
        description="Path to save/load learned DataFrame schema (auto-generated if None)"
    )
    schema_coerce: bool = Field(
        default=True,
        description="Attempt to coerce types during schema validation (False = strict)"
    )
    
    # ========== Leakage Prevention ==========
    raise_on_target_in_transform: bool = Field(
        default=True,
        description="Raise error if target (y) is passed to transform() to prevent leakage"
    )
    
    # ========== Performance & Caching ==========
    n_jobs: int = Field(
        default=-1,
        description="Number of parallel jobs for sklearn components (-1 = all cores, 1 = sequential)"
    )
    cache_dir: Optional[str] = Field(
        default=None,
        description="Directory for caching expensive transformations (None = no caching)"
    )
    
    # ========== Runtime (legacy/internal) ==========
    max_samples: Optional[int] = Field(
        default=None, ge=100,
        description="(Deprecated: use sample_n) Maximum samples for analysis"
    )

    @field_validator("mid_cardinality_max")
    @classmethod
    def check_mid_greater_than_low(cls, v, info):
        """Ensure mid_cardinality_max > low_cardinality_max."""
        if "low_cardinality_max" in info.data and v <= info.data["low_cardinality_max"]:
            raise ValueError("mid_cardinality_max must be greater than low_cardinality_max")
        return v
    
    @field_validator("clip_percentiles")
    @classmethod
    def check_clip_percentiles(cls, v):
        """Ensure clip percentiles are valid."""
        if len(v) != 2:
            raise ValueError("clip_percentiles must be a tuple of 2 values")
        if v[0] >= v[1]:
            raise ValueError("clip_percentiles[0] must be < clip_percentiles[1]")
        if not (0.0 <= v[0] < v[1] <= 1.0):
            raise ValueError("clip_percentiles must be in range [0, 1]")
        return v
    
    @classmethod
    def from_env(cls, prefix: str = "FEATURECRAFT") -> "FeatureCraftConfig":
        """Load configuration from environment variables.
        
        Example: FEATURECRAFT__LOW_CARDINALITY_MAX=15
        
        Args:
            prefix: Environment variable prefix
            
        Returns:
            FeatureCraftConfig instance
        """
        from .settings import load_from_env
        env_config = load_from_env(prefix)
        return cls(**env_config)

