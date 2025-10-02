"""Main AutoFeatureEngineer class for FeatureCraft."""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from rich.console import Console
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from .config import FeatureCraftConfig
from .encoders import (
    HashingEncoder,
    KFoldTargetEncoder,
    OutOfFoldTargetEncoder,
    RareCategoryGrouper,
    FrequencyEncoder,
    CountEncoder,
    make_ohe,
)
from .imputers import categorical_imputer, choose_numeric_imputer
from .insights import analyze_dataset, detect_task
from .logging import get_logger
from .validation.schema_validator import SchemaValidator
from .plots import (
    plot_boxplots,
    plot_correlation_heatmap,
    plot_countplots,
    plot_distributions,
    plot_missingness,
)
from .scalers import choose_scaler
from .text import build_text_pipeline
from .transformers import DateTimeFeatures, EnsureNumericOutput, NumericConverter, SkewedPowerTransformer
from .types import DatasetInsights, PipelineSummary, TaskType
from .validators import validate_input_frame

logger = get_logger(__name__)
console = Console()


class TextColumnSelector(FunctionTransformer):
    """Select a single text column and return as string series.
    
    This transformer is picklable because it doesn't use lambda functions.
    """

    def __init__(self, col: str):
        self.col = col
        # Don't pass func to parent - we'll override transform instead
        super().__init__(func=None)
    
    def transform(self, X):
        """Transform by selecting and converting the text column."""
        if isinstance(X, pd.DataFrame):
            return pd.Series(X[self.col]).astype(str).fillna("")
        else:
            # Handle array input
            return pd.Series(X[:, 0]).astype(str).fillna("")
    
    def fit(self, X, y=None):
        """Fit method (no-op for this transformer)."""
        return self


class AutoFeatureEngineer:
    """Main class for automatic feature engineering."""

    def __init__(self, config: FeatureCraftConfig | None = None) -> None:
        """Initialize with optional config."""
        self.cfg = config or FeatureCraftConfig()
        self.insights_: DatasetInsights | None = None
        self.pipeline_: Pipeline | None = None
        self.summary_: PipelineSummary | None = None
        self.feature_names_: list[str] | None = None
        self.estimator_family_: str = "tree"
        self.task_: TaskType | None = None

    # ---------- Configuration API ----------
    def set_params(self, **overrides) -> "AutoFeatureEngineer":
        """Set configuration parameters sklearn-style.
        
        Args:
            **overrides: Configuration parameters to update
            
        Returns:
            Self for method chaining
            
        Example:
            >>> afe = AutoFeatureEngineer()
            >>> afe.set_params(use_smote=True, low_cardinality_max=12)
            >>> afe.fit(X, y)
        """
        current_dict = self.cfg.model_dump()
        current_dict.update(overrides)
        try:
            self.cfg = FeatureCraftConfig(**current_dict)
            logger.debug(f"Updated {len(overrides)} configuration parameters")
        except Exception as e:
            logger.error(f"Failed to update configuration: {e}")
            raise ValueError(f"Invalid configuration parameters: {e}") from e
        return self

    def get_params(self, deep: bool = True) -> dict:
        """Get configuration parameters sklearn-style.
        
        Args:
            deep: If True, return all config parameters. If False, return wrapper.
            
        Returns:
            Configuration dictionary
            
        Example:
            >>> afe = AutoFeatureEngineer()
            >>> params = afe.get_params()
            >>> params['use_smote']
            False
        """
        if deep:
            return self.cfg.model_dump()
        return {"config": self.cfg}

    @contextmanager
    def with_overrides(self, **kwargs):
        """Context manager for temporary configuration overrides.
        
        Args:
            **kwargs: Temporary configuration overrides
            
        Yields:
            Self with overridden configuration
            
        Example:
            >>> afe = AutoFeatureEngineer()
            >>> with afe.with_overrides(use_smote=True):
            ...     afe.fit(X_train, y_train)
            >>> # Original config restored after context
        """
        original_cfg = deepcopy(self.cfg)
        try:
            self.set_params(**kwargs)
            yield self
        finally:
            self.cfg = original_cfg
            logger.debug("Restored original configuration after context")

    # ---------- Public API ----------
    def analyze(self, df: pd.DataFrame, target: str) -> DatasetInsights:
        """Analyze dataset and return insights.
        
        If config.enable_drift_report is True and config.reference_path is provided,
        computes drift metrics between reference and current datasets.
        
        Args:
            df: Dataset to analyze
            target: Target column name
            
        Returns:
            DatasetInsights with optional drift report
        """
        validate_input_frame(df, target)
        X = df.drop(columns=[target])
        y = df[target]

        insights = analyze_dataset(X, y, target_name=target, cfg=self.cfg)

        # Figures
        figures: dict[str, str] = {}
        _, b64 = plot_missingness(df)
        figures["missingness"] = b64
        for name, (_, s) in plot_distributions(df).items():
            figures[f"dist_{name}"] = s
        for name, (_, s) in plot_boxplots(df).items():
            figures[f"box_{name}"] = s
        for name, (_, s) in plot_countplots(df).items():
            figures[f"count_{name}"] = s
        if insights.correlations is not None and not insights.correlations.empty:
            _, b64 = plot_correlation_heatmap(insights.correlations)
            figures["corr_heatmap"] = b64
        insights.figures = figures

        # Optional: Drift detection
        if self.cfg.enable_drift_report and self.cfg.reference_path:
            try:
                reference_df = self._load_reference_data(self.cfg.reference_path)
                drift_report = self._compute_drift_report(reference_df, df)
                # Attach drift report to insights (extend DatasetInsights if needed)
                if hasattr(insights, '__dict__'):
                    insights.__dict__['drift_report'] = drift_report
                logger.info(f"Drift report generated: {drift_report.get('summary', {})}")
            except Exception as e:
                logger.warning(f"Drift detection failed: {e}")

        self.insights_ = insights
        self.task_ = insights.task
        return insights
    
    def _load_reference_data(self, path: str) -> pd.DataFrame:
        """Load reference dataset from path (CSV or parquet).
        
        Args:
            path: Path to reference dataset
            
        Returns:
            Reference DataFrame
        """
        import os
        if not os.path.exists(path):
            raise FileNotFoundError(f"Reference data not found: {path}")
        
        if path.endswith('.parquet'):
            return pd.read_parquet(path)
        else:
            return pd.read_csv(path)
    
    def _compute_drift_report(self, reference_df: pd.DataFrame, current_df: pd.DataFrame) -> dict:
        """Compute drift report between reference and current datasets.
        
        Args:
            reference_df: Reference (training) dataset
            current_df: Current (new) dataset
            
        Returns:
            Dict with drift results and summary
        """
        from .drift import DriftDetector, summarize_drift_report
        
        detector = DriftDetector(self.cfg)
        drift_results = detector.detect(reference_df, current_df)
        summary = summarize_drift_report(drift_results)
        
        return {
            "results": drift_results,
            "summary": summary,
        }

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        estimator_family: str = "tree",
        *,
        groups: Optional[pd.Series] = None,
        config: Optional[FeatureCraftConfig] = None,
    ) -> AutoFeatureEngineer:
        """Fit feature engineering pipeline.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            estimator_family: Estimator family (tree, linear, svm, knn, nn)
            groups: Optional group labels for GroupKFold CV
            config: Optional config override for this fit operation
            
        Returns:
            Self for method chaining
            
        Raises:
            ValueError: If X or y are empty or invalid
            TypeError: If X is not a DataFrame or y is not a Series
        """
        # Input validation - critical for production library
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"X must be a pandas DataFrame, got {type(X).__name__}")
        if not isinstance(y, pd.Series):
            raise TypeError(f"y must be a pandas Series, got {type(y).__name__}")
        
        if X.empty:
            raise ValueError("Cannot fit on empty DataFrame X. X must contain at least one row.")
        if len(y) == 0:
            raise ValueError("Cannot fit on empty Series y. y must contain at least one element.")
        if len(X) != len(y):
            raise ValueError(f"X and y must have the same length. Got X: {len(X)}, y: {len(y)}")
        
        # Validate sufficient data for pipeline operations
        if len(X) < 2:
            raise ValueError(
                f"Insufficient data: X has only {len(X)} row(s). "
                "At least 2 rows are required for feature engineering."
            )
        
        logger.debug(f"Validated input: X shape={X.shape}, y shape={y.shape}")
        
        # Apply config override if provided
        if config is not None:
            self.cfg = config
            logger.debug("Using runtime config override for fit")

        self.estimator_family_ = estimator_family
        self.pipeline_ = self._build_pipeline(X, y, estimator_family)
        self.pipeline_.fit(X, y)
        self.feature_names_ = self._get_feature_names(X)
        self.summary_ = PipelineSummary(
            feature_names=self.feature_names_ or [],
            n_features_out=len(self.feature_names_ or []),
            steps=[name for name, _ in self.pipeline_.steps],
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted pipeline."""
        if self.pipeline_ is None:
            raise RuntimeError("Pipeline is not fitted.")
        Xt = self.pipeline_.transform(X)
        Xt_arr = Xt.toarray() if hasattr(Xt, "toarray") else np.asarray(Xt)
        cols = self.feature_names_ or [f"f_{i}" for i in range(Xt_arr.shape[1])]
        return pd.DataFrame(Xt_arr, columns=cols, index=X.index)

    def fit_transform(
        self, X: pd.DataFrame, y: pd.Series, estimator_family: str = "tree"
    ) -> pd.DataFrame:
        """Fit and transform in one step."""
        self.fit(X, y, estimator_family=estimator_family)
        return self.transform(X)

    def export(self, out_dir: str) -> PipelineSummary:
        """Export fitted pipeline and metadata."""
        if self.pipeline_ is None:
            raise RuntimeError("Nothing to export. Fit a pipeline first.")
        os.makedirs(out_dir, exist_ok=True)
        joblib.dump(self.pipeline_, os.path.join(out_dir, "pipeline.joblib"))
        meta = {
            "summary": asdict(self.summary_) if self.summary_ else {},
            "config": self.cfg.model_dump(),
            "estimator_family": self.estimator_family_,
            "task": self.task_.value if self.task_ else None,
        }
        with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        if self.feature_names_:
            with open(os.path.join(out_dir, "feature_names.txt"), "w", encoding="utf-8") as f:
                for n in self.feature_names_:
                    f.write(n + "\n")
        if self.summary_:
            self.summary_.artifacts_path = out_dir
        return self.summary_ or PipelineSummary(feature_names=[], n_features_out=0, steps=[])

    # ---------- Internals ----------
    def _build_pipeline(self, X: pd.DataFrame, y: pd.Series, estimator_family: str) -> Pipeline:
        """Build feature engineering pipeline."""
        cfg = self.cfg
        task = detect_task(y)
        self.task_ = task
        
        # Optional: Setup caching with joblib.Memory
        memory = None
        if cfg.cache_dir:
            from joblib import Memory
            memory = Memory(location=cfg.cache_dir, verbose=0)
            logger.info(f"Caching enabled: {cfg.cache_dir}")

        # Detect column types with robust handling of edge cases
        num_cols = []
        cat_cols = []
        dt_cols = []
        
        for col in X.columns:
            col_series = X[col]
            col_dtype = col_series.dtype
            
            # Check for datetime first
            if pd.api.types.is_datetime64_any_dtype(col_series):
                dt_cols.append(col)
                continue
            
            # CRITICAL: Explicitly reject categorical dtype - even if it has numeric codes
            # This prevents Bug #1 (categorical columns causing skew computation errors)
            if isinstance(col_series.dtype, CategoricalDtype):
                cat_cols.append(col)
                continue
            
            # Check for object/string types - always treat as categorical or try conversion
            if pd.api.types.is_object_dtype(col_series) or pd.api.types.is_string_dtype(col_series):
                # Try to convert entire column to numeric (not just non-null values)
                try:
                    # Test conversion on ALL values including nulls
                    test_series = pd.to_numeric(col_series, errors='raise')
                    # If successful and we have enough valid data, treat as numeric
                    valid_ratio = test_series.notna().sum() / len(col_series)
                    if valid_ratio > 0.5:
                        num_cols.append(col)
                        continue
                except (ValueError, TypeError):
                    # Cannot convert to numeric - definitely categorical
                    pass
                cat_cols.append(col)
                continue
            
            # Column claims to be numeric dtype - but verify it's actually numeric
            if pd.api.types.is_numeric_dtype(col_series):
                # Double-check: try converting dropna values to float
                try:
                    non_null = col_series.dropna()
                    if len(non_null) > 0:
                        # Attempt conversion to verify it's truly numeric
                        _ = pd.to_numeric(non_null, errors='raise')
                        # Also check the actual values aren't strings masquerading as numeric
                        if non_null.dtype == object:
                            # It's object dtype - need to verify each value
                            sample = non_null.head(min(100, len(non_null)))
                            for val in sample:
                                if isinstance(val, str):
                                    raise ValueError(f"Found string value '{val}' in supposedly numeric column")
                    num_cols.append(col)
                except (ValueError, TypeError) as e:
                    # Column claims numeric dtype but contains non-numeric values
                    logger.warning(f"Column '{col}' has numeric dtype but validation failed: {e}. Treating as categorical.")
                    cat_cols.append(col)
                continue
            
            # Unknown/unsupported dtype - treat as categorical for safety
            logger.debug(f"Column '{col}' has unknown dtype {col_dtype}, treating as categorical")
            cat_cols.append(col)
        
        # Simple heuristic text columns: object with long strings
        text_cols = [c for c in cat_cols if X[c].astype(str).str.len().mean() >= 15]
        cat_cols = [c for c in cat_cols if c not in text_cols]

        # Cardinality per categorical
        card = {c: int(X[c].nunique(dropna=True)) for c in cat_cols}

        low_cat = [c for c in cat_cols if card[c] <= cfg.low_cardinality_max]
        mid_cat = [
            c for c in cat_cols if cfg.low_cardinality_max < card[c] <= cfg.mid_cardinality_max
        ]
        high_cat = [c for c in cat_cols if card[c] > cfg.mid_cardinality_max]

        # Numeric skew mask - with additional safety checks
        skew_map = {}
        for c in num_cols:
            try:
                col_data = X[c].dropna()
                # Extra safety: ensure column is not categorical dtype and has numeric values
                if len(col_data) == 0:
                    skew_map[c] = 0.0
                elif isinstance(col_data.dtype, CategoricalDtype):
                    # Should never happen after our filtering above, but be defensive
                    logger.warning(f"Column '{c}' is categorical but was in num_cols. Skipping skew computation.")
                    skew_map[c] = 0.0
                else:
                    # Convert to float to ensure numeric before skew computation
                    numeric_data = pd.to_numeric(col_data, errors='coerce')
                    if numeric_data.notna().sum() > 0:
                        skew_map[c] = float(numeric_data.skew())
                    else:
                        skew_map[c] = 0.0
            except (TypeError, ValueError, AttributeError) as e:
                # Defensive: if skew computation fails for any reason, default to 0
                logger.warning(f"Skew computation failed for column '{c}': {e}. Using 0.0.")
                skew_map[c] = 0.0
        
        skew_mask = [abs(skew_map[c]) >= cfg.skew_threshold for c in num_cols]

        # Outlier check
        def outlier_share(s: pd.Series) -> float:
            x = s.dropna().astype(float)
            if x.empty:
                return 0.0
            q1, q3 = np.percentile(x, [25, 75])
            iqr = q3 - q1
            if iqr == 0:
                return 0.0
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            return float(((x < lower) | (x > upper)).mean())

        heavy_outliers = any(outlier_share(X[c]) > cfg.outlier_share_threshold for c in num_cols)

        # Transformers per block
        num_missing_rate = float(X[num_cols].isna().mean().mean()) if num_cols else 0.0
        num_imputer = choose_numeric_imputer(num_missing_rate, len(num_cols), X.shape[0], cfg)

        steps_num: list[tuple[str, Any]] = [
            ("convert", NumericConverter(columns=num_cols)),  # Ensure numeric conversion
            ("impute", num_imputer)
        ]
        if any(skew_mask):
            steps_num.append(("yeojohnson", SkewedPowerTransformer(num_cols, skew_mask)))
        
        # Optional winsorization before scaling
        if cfg.winsorize:
            from .transformers import WinsorizerTransformer
            steps_num.append(("winsorize", WinsorizerTransformer(
                percentiles=cfg.clip_percentiles,
                columns=num_cols,
            )))
        
        scaler = choose_scaler(estimator_family, heavy_outliers, cfg)
        if scaler is not None:
            steps_num.append(("scale", scaler))
        num_pipe = Pipeline(steps=steps_num)

        # Categorical pipelines
        from .encoders import make_ohe
        cat_low_pipe = Pipeline(
            steps=[
                ("rare", RareCategoryGrouper(min_freq=cfg.rare_level_threshold)),
                ("impute", categorical_imputer(cfg)),
                ("ohe", make_ohe(
                    min_frequency=cfg.rare_level_threshold,
                    handle_unknown=cfg.ohe_handle_unknown,
                )),
            ]
        )
        # Mid-card TE (if enabled)
        # CRITICAL: Use OutOfFoldTargetEncoder for proper leakage-free training
        # NOTE: cols=None because ColumnTransformer already selects mid_cat columns
        if cfg.use_target_encoding and mid_cat:
            from .encoders import LeaveOneOutTargetEncoder
            if cfg.use_leave_one_out_te:
                te = LeaveOneOutTargetEncoder(
                    cols=None,  # Let ColumnTransformer handle column selection
                    smoothing=cfg.target_encoding_smoothing,
                    noise_std=cfg.target_encoding_noise,
                    random_state=cfg.random_state,
                    task=task.value,
                )
            else:
                # Use OutOfFoldTargetEncoder for training to prevent leakage
                te = OutOfFoldTargetEncoder(
                    cols=None,  # Let ColumnTransformer handle column selection
                    cv=cfg.cv_strategy,
                    n_splits=cfg.cv_n_splits,
                    shuffle=cfg.cv_shuffle,
                    random_state=cfg.cv_random_state or cfg.random_state,
                    smoothing=cfg.te_smoothing,
                    noise_std=cfg.te_noise,
                    prior_strategy=cfg.te_prior,
                    task=task.value,
                    raise_on_target_in_transform=cfg.raise_on_target_in_transform,
                )
            cat_mid_pipe = Pipeline(steps=[("impute", categorical_imputer(cfg)), ("te", te)])
        elif cfg.use_frequency_encoding and mid_cat:
            # Alternative: Use FrequencyEncoder
            freq_enc = FrequencyEncoder(cols=None, unseen_value=0.0)  # Let ColumnTransformer handle column selection
            cat_mid_pipe = Pipeline(steps=[("impute", categorical_imputer(cfg)), ("freq", freq_enc)])
        elif cfg.use_count_encoding and mid_cat:
            # Alternative: Use CountEncoder
            count_enc = CountEncoder(cols=None, unseen_value=0.0, normalize=False)  # Let ColumnTransformer handle column selection
            cat_mid_pipe = Pipeline(steps=[("impute", categorical_imputer(cfg)), ("count", count_enc)])
        else:
            # Fallback to hashing if TE disabled
            cat_mid_pipe = Pipeline(
                steps=[
                    ("impute", categorical_imputer(cfg)),
                    ("rare", RareCategoryGrouper(min_freq=cfg.rare_level_threshold)),
                    ("hash", HashingEncoder(n_features=cfg.hashing_n_features_tabular, seed=cfg.random_state)),
                ]
            )
        # High-card hashing
        cat_high_pipe = Pipeline(
            steps=[
                ("impute", categorical_imputer(cfg)),
                ("rare", RareCategoryGrouper(min_freq=cfg.rare_level_threshold)),
                (
                    "hash",
                    HashingEncoder(
                        n_features=cfg.hashing_n_features_tabular, seed=cfg.random_state
                    ),
                ),
            ]
        )

        # Text - Custom selector for text columns (using module-level class)
        text_transformers = []
        for c in text_cols:
            svd_k = (
                None
                if estimator_family.lower() in {"linear", "svm"}
                else cfg.svd_components_for_trees
            )
            text_transformers.append(
                (
                    f"text_{c}", 
                    build_text_pipeline(
                        c, 
                        cfg.tfidf_max_features, 
                        svd_k,
                        use_hashing=cfg.text_use_hashing,
                        hashing_features=cfg.text_hashing_features,
                        char_ngrams=cfg.text_char_ngrams,
                    ), 
                    c
                )
            )

        # Datetime expansion with optional Fourier & holiday features
        if dt_cols:
            from .time_series import FourierFeatures, HolidayFeatures
            dt_steps = [("base", DateTimeFeatures(dt_cols))]
            # Add Fourier features if enabled
            if cfg.use_fourier and cfg.time_column:
                for col in dt_cols:
                    dt_steps.append(
                        (f"fourier_{col}", FourierFeatures(column=col, orders=cfg.fourier_orders))
                    )
            # Add holiday features if enabled
            if cfg.holiday_country and cfg.time_column:
                for col in dt_cols:
                    dt_steps.append(
                        (f"holiday_{col}", HolidayFeatures(column=col, country_code=cfg.holiday_country))
                    )
            dt_pipe = Pipeline(steps=dt_steps) if len(dt_steps) > 0 else DateTimeFeatures(dt_cols)
        else:
            dt_pipe = "drop"

        transformers: list[tuple[str, Any, Any]] = []
        if num_cols:
            transformers.append(("num", num_pipe, num_cols))
        if low_cat:
            transformers.append(("cat_low", cat_low_pipe, low_cat))
        if mid_cat:
            transformers.append(("cat_mid", cat_mid_pipe, mid_cat))
        if high_cat:
            transformers.append(("cat_high", cat_high_pipe, high_cat))
        if dt_cols:
            transformers.append(("dt", dt_pipe, dt_cols))
        for name, pipe, col in text_transformers:
            # Wrap text column selection with custom selector
            transformers.append(
                (
                    name,
                    Pipeline([("select", TextColumnSelector(col)), ("text", pipe)]),
                    [col],
                )
            )

        # ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=transformers, remainder="drop", sparse_threshold=0.3
        )

        # Build final pipeline with optional steps
        pipe_steps = []
        
        # STEP 1: Schema validation (FIRST step before any transformation)
        if cfg.validate_schema:
            schema_validator = SchemaValidator(
                enabled=True,
                coerce=cfg.schema_coerce,
                strict=False,  # Use warnings instead of errors for robustness
                schema_path=cfg.schema_path,
            )
            pipe_steps.append(("schema_validator", schema_validator))
            logger.debug("Schema validation enabled as first pipeline step")
        
        # STEP 2: Main preprocessing
        pipe_steps.append(("preprocess", preprocessor))
        
        # STEP 3: Optional dimensionality reducer
        if cfg.reducer_kind:
            from .transformers import DimensionalityReducer
            pipe_steps.append((
                "reducer",
                DimensionalityReducer(
                    kind=cfg.reducer_kind,
                    max_components=cfg.reducer_components,
                    variance=cfg.reducer_variance,
                    random_state=cfg.random_state,
                )
            ))
        
        # STEP 4: Final safety check
        pipe_steps.append(("ensure_numeric", EnsureNumericOutput()))
        
        # Build pipeline with optional caching
        pipe = Pipeline(steps=pipe_steps, memory=memory)
        return pipe

    def _get_feature_names(self, X: pd.DataFrame) -> list[str]:
        """Get feature names from fitted pipeline by performing dummy transform."""
        names: list[str] = []
        if self.pipeline_ is None:
            return names

        # Perform dummy transform on small sample to get exact output shape
        sample = X.head(min(5, len(X)))
        Xt = self.pipeline_.transform(sample)
        Xt_arr = Xt.toarray() if hasattr(Xt, "toarray") else np.asarray(Xt)
        n_features_total = Xt_arr.shape[1]

        pre: ColumnTransformer = self.pipeline_.named_steps["preprocess"]

        for name, trans, cols in pre.transformers_:
            if name == "remainder":
                continue

            colnames = [str(c) for c in cols] if isinstance(cols, list) else [str(cols)]

            # Try to get feature names from transformer
            if hasattr(trans, "get_feature_names_out"):
                try:
                    fn = trans.get_feature_names_out(colnames)
                    names.extend([str(x) for x in fn])
                    continue
                except Exception:
                    pass

            # Fallback: infer from actual transformer output
            try:
                # Extract this transformer's output to count features
                col_indices = [i for i, col in enumerate(X.columns) if col in cols]
                if col_indices:
                    sample_subset = sample.iloc[:, col_indices]
                    trans_output = trans.transform(sample_subset)
                    trans_arr = trans_output.toarray() if hasattr(trans_output, "toarray") else np.asarray(trans_output)
                    n_features_actual = trans_arr.shape[1]
                else:
                    n_features_actual = len(colnames)
                
                # Generate names based on actual output
                if name.startswith("text_") or name == "cat_high":
                    names.extend([f"{name}__feat_{i}" for i in range(n_features_actual)])
                elif n_features_actual == len(colnames):
                    names.extend([f"{name}__{c}" for c in colnames])
                else:
                    names.extend([f"{name}__feat_{i}" for i in range(n_features_actual)])
                    
            except Exception as e:
                # Ultimate fallback: use column names
                logger.warning(f"Failed to infer feature names for {name}: {e}. Using fallback.")
                names.extend([f"{name}__{c}" for c in colnames])

        # Ensure name count matches actual output
        if len(names) != n_features_total:
            logger.warning(
                f"Feature name count mismatch: {len(names)} names vs {n_features_total} actual. "
                "Using generic names."
            )
            names = [f"feature_{i}" for i in range(n_features_total)]

        return names
