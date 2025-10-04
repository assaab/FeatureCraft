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
from .explainability import PipelineExplainer
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
from .exceptions import PipelineNotFittedError, SecurityError, InputValidationError, ExportError

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
        self.explainer_: PipelineExplainer | None = None
        self.explanation_: Any | None = None  # PipelineExplanation from explainability module

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
            path: Path to reference dataset (must be under workspace or allowed directories)
            
        Returns:
            Reference DataFrame
            
        Raises:
            SecurityError: If path attempts directory traversal
            FileNotFoundError: If file doesn't exist
        """
        from pathlib import Path
        
        # Resolve to absolute path and check for traversal
        ref_path = Path(path).resolve()
        workspace = Path.cwd().resolve()
        
        # Allow paths under workspace, artifacts dir, or tmp
        allowed_dirs = [
            workspace,
            Path("/tmp"),
            Path(self.cfg.artifacts_dir).resolve()
        ]
        
        # Cross-platform path validation using relative_to (works on Windows with different drives)
        path_allowed = False
        for allowed_dir in allowed_dirs:
            try:
                # Try to get relative path - raises ValueError if not under allowed_dir
                _ = ref_path.relative_to(allowed_dir)
                path_allowed = True
                break
            except ValueError:
                # Path not under this allowed directory, try next
                continue
        
        if not path_allowed:
            raise SecurityError(
                f"Path outside allowed directories. Use paths under workspace, artifacts, or /tmp.",
                provided_path=path,
                resolved_path=str(ref_path),
                allowed_dirs=[str(d) for d in allowed_dirs]
            )
        
        if not ref_path.exists():
            raise FileNotFoundError(f"Reference data not found: {path}")
        
        if ref_path.suffix == ".parquet":
            return pd.read_parquet(ref_path)
        else:
            return pd.read_csv(ref_path)
    
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

        # Store training columns for validation
        self._training_columns = list(X.columns)
        
        self.estimator_family_ = estimator_family
        self.pipeline_ = self._build_pipeline(X, y, estimator_family)
        self.pipeline_.fit(X, y)
        self.feature_names_ = self._get_feature_names(X)
        self.summary_ = PipelineSummary(
            feature_names=self.feature_names_ or [],
            n_features_out=len(self.feature_names_ or []),
            steps=[name for name, _ in self.pipeline_.steps],
        )
        
        # Update explanation with final feature count
        if self.explanation_:
            self.explanation_.n_features_out = len(self.feature_names_ or [])
            self.explanation_.summary["n_features_out"] = len(self.feature_names_ or [])
        
        # Auto-print explanation if configured
        if self.cfg.explain_transformations and self.cfg.explain_auto_print:
            console.print()  # Blank line before explanation
            self.print_explanation()
        
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted pipeline.
        
        Args:
            X: Input DataFrame to transform
            
        Returns:
            Transformed DataFrame with feature names
            
        Raises:
            PipelineNotFittedError: If pipeline not fitted
            InputValidationError: If input schema doesn't match training data
        """
        if self.pipeline_ is None:
            raise PipelineNotFittedError(
                "Cannot transform: pipeline not fitted. Call fit() first.",
                operation="transform"
            )
        
        # Validate input schema if enabled
        if self.cfg.validate_schema and hasattr(self, '_training_columns'):
            self._validate_transform_input(X)
        
        Xt = self.pipeline_.transform(X)
        Xt_arr = Xt.toarray() if hasattr(Xt, "toarray") else np.asarray(Xt)
        cols = self.feature_names_ or [f"f_{i}" for i in range(Xt_arr.shape[1])]
        return pd.DataFrame(Xt_arr, columns=cols, index=X.index)
    
    def _validate_transform_input(self, X: pd.DataFrame) -> None:
        """Validate that transform input matches training schema.
        
        Args:
            X: Input DataFrame to validate
            
        Raises:
            InputValidationError: If schema doesn't match
        """
        if not hasattr(self, '_training_columns'):
            return
        
        missing_cols = set(self._training_columns) - set(X.columns)
        if missing_cols:
            raise InputValidationError(
                f"Missing columns in transform input: {missing_cols}",
                missing_columns=list(missing_cols),
                expected_columns=self._training_columns
            )
        
        extra_cols = set(X.columns) - set(self._training_columns)
        if extra_cols:
            logger.warning(f"Extra columns in transform input (will be ignored): {extra_cols}")

    def fit_transform(
        self, X: pd.DataFrame, y: pd.Series, estimator_family: str = "tree"
    ) -> pd.DataFrame:
        """Fit and transform in one step."""
        self.fit(X, y, estimator_family=estimator_family)
        return self.transform(X)

    def export(self, out_dir: str) -> PipelineSummary:
        """Export fitted pipeline and metadata to disk.
        
        Args:
            out_dir: Directory path to save pipeline artifacts
            
        Returns:
            PipelineSummary with export metadata
            
        Raises:
            PipelineNotFittedError: If pipeline not fitted
            ExportError: If export fails
            
        Security Warning:
            The exported pipeline uses pickle serialization (via joblib).
            **Only load pipeline files from trusted sources.**
            
            Loading untrusted pickles can execute arbitrary code (CWE-502).
            
            For production use with untrusted pipelines, consider:
            - ONNX export (for supported models)
            - JSON/YAML config + retrain pattern
            - Containerization with read-only filesystem
            - Use load_pipeline() with checksum verification
        """
        if self.pipeline_ is None:
            raise PipelineNotFittedError(
                "Cannot export: pipeline not fitted. Call fit() first.",
                operation="export"
            )
        
        try:
            os.makedirs(out_dir, exist_ok=True)
            
            # Serialize pipeline and compute checksum for integrity
            import hashlib
            pipeline_path = os.path.join(out_dir, "pipeline.joblib")
            pipeline_bytes = joblib.dumps(self.pipeline_)
            checksum = hashlib.sha256(pipeline_bytes).hexdigest()
            
            # Write pipeline file
            with open(pipeline_path, "wb") as f:
                f.write(pipeline_bytes)
            
            # Write checksum file for verification
            with open(os.path.join(out_dir, "pipeline.sha256"), "w") as f:
                f.write(f"{checksum}  pipeline.joblib\n")
            
            logger.info(f"Pipeline exported with SHA256: {checksum[:16]}...")
            
        except Exception as e:
            raise ExportError(
                f"Failed to export pipeline: {e}",
                output_directory=out_dir
            ) from e
        
        meta = {
            "summary": asdict(self.summary_) if self.summary_ else {},
            "config": self.cfg.model_dump(),
            "estimator_family": self.estimator_family_,
            "pipeline_checksum_sha256": checksum,
            "task": self.task_.value if self.task_ else None,
        }
        with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        if self.feature_names_:
            with open(os.path.join(out_dir, "feature_names.txt"), "w", encoding="utf-8") as f:
                for n in self.feature_names_:
                    f.write(n + "\n")
        
        # Export explanation if available
        if self.explanation_:
            explanation_md_path = os.path.join(out_dir, "explanation.md")
            with open(explanation_md_path, "w", encoding="utf-8") as f:
                f.write(self.explanation_.to_markdown())
            logger.info(f"Saved pipeline explanation to {explanation_md_path}")
            
            explanation_json_path = os.path.join(out_dir, "explanation.json")
            with open(explanation_json_path, "w", encoding="utf-8") as f:
                f.write(self.explanation_.to_json())
            logger.debug(f"Saved pipeline explanation JSON to {explanation_json_path}")
        
        if self.summary_:
            self.summary_.artifacts_path = out_dir
        return self.summary_ or PipelineSummary(feature_names=[], n_features_out=0, steps=[])
    
    def get_explanation(self) -> Any:
        """Get the pipeline explanation object.
        
        Returns:
            PipelineExplanation object with details about all transformations
            
        Raises:
            RuntimeError: If pipeline has not been fitted yet
            
        Example:
            >>> afe = AutoFeatureEngineer()
            >>> afe.fit(X_train, y_train)
            >>> explanation = afe.get_explanation()
            >>> explanation.print_console()
        """
        if self.explanation_ is None:
            raise RuntimeError(
                "No explanation available. Fit the pipeline first with explain_transformations=True."
            )
        return self.explanation_
    
    def print_explanation(self, console: Optional[Console] = None) -> None:
        """Print pipeline explanation to console.
        
        Args:
            console: Optional Rich Console instance for custom formatting
            
        Raises:
            RuntimeError: If pipeline has not been fitted yet
            
        Example:
            >>> afe = AutoFeatureEngineer()
            >>> afe.fit(X_train, y_train)
            >>> afe.print_explanation()
        """
        explanation = self.get_explanation()
        explanation.print_console(console=console)
    
    def save_explanation(self, path: str, format: str = "markdown") -> None:
        """Save pipeline explanation to file.
        
        Args:
            path: Output file path
            format: Output format - 'markdown', 'md', 'json'
            
        Raises:
            RuntimeError: If pipeline has not been fitted yet
            ValueError: If format is not supported
            
        Example:
            >>> afe = AutoFeatureEngineer()
            >>> afe.fit(X_train, y_train)
            >>> afe.save_explanation("pipeline_explanation.md")
            >>> afe.save_explanation("pipeline_explanation.json", format="json")
        """
        explanation = self.get_explanation()
        
        format_lower = format.lower()
        if format_lower in ("markdown", "md"):
            with open(path, "w", encoding="utf-8") as f:
                f.write(explanation.to_markdown())
            logger.info(f"Saved explanation (markdown) to {path}")
        elif format_lower == "json":
            with open(path, "w", encoding="utf-8") as f:
                f.write(explanation.to_json())
            logger.info(f"Saved explanation (JSON) to {path}")
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'markdown', 'md', or 'json'.")

    # ---------- Internals ----------
    def _build_pipeline(self, X: pd.DataFrame, y: pd.Series, estimator_family: str) -> Pipeline:
        """Build feature engineering pipeline."""
        cfg = self.cfg
        task = detect_task(y)
        self.task_ = task
        
        # Initialize explainer
        self.explainer_ = PipelineExplainer(enabled=cfg.explain_transformations)
        
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
        
        # Explain column classification
        self.explainer_.explain_column_classification(
            num_cols=num_cols,
            cat_cols=cat_cols,
            dt_cols=dt_cols,
            text_cols=text_cols,
            low_cat=low_cat,
            mid_cat=mid_cat,
            high_cat=high_cat,
            card=card,
            low_threshold=cfg.low_cardinality_max,
            mid_threshold=cfg.mid_cardinality_max,
        )

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
        
        # Explain numeric imputation
        if num_cols:
            imputer_name = type(num_imputer).__name__
            if num_missing_rate <= cfg.numeric_simple_impute_max:
                reason = (
                    f"Using median imputation because missing rate ({num_missing_rate:.1%}) is low "
                    f"(<= {cfg.numeric_simple_impute_max:.1%}). Simple strategies work well for low missingness."
                )
            elif num_missing_rate <= cfg.numeric_advanced_impute_max:
                if len(num_cols) <= 100 and X.shape[0] <= 200_000:
                    reason = (
                        f"Using KNN imputation because missing rate ({num_missing_rate:.1%}) is moderate "
                        f"and dataset size is manageable ({len(num_cols)} features, {X.shape[0]} rows). "
                        "KNN can capture local patterns for better imputation."
                    )
                else:
                    reason = (
                        f"Using iterative imputation because missing rate ({num_missing_rate:.1%}) is moderate "
                        f"but dataset is large ({len(num_cols)} features, {X.shape[0]} rows). "
                        "Iterative imputation scales better than KNN."
                    )
            else:
                reason = (
                    f"Falling back to median imputation despite high missing rate ({num_missing_rate:.1%}) "
                    f"(> {cfg.numeric_advanced_impute_max:.1%}). Advanced methods may not be reliable with this much missingness."
                )
            
            self.explainer_.explain_imputation(
                strategy_name=imputer_name,
                columns=num_cols,
                missing_rate=num_missing_rate,
                reason=reason,
                config_params={
                    "numeric_simple_impute_max": cfg.numeric_simple_impute_max,
                    "numeric_advanced_impute_max": cfg.numeric_advanced_impute_max,
                },
                add_indicators=True,  # SimpleImputer with add_indicator=True
            )

        steps_num: list[tuple[str, Any]] = [
            ("convert", NumericConverter(columns=num_cols)),  # Ensure numeric conversion
            ("impute", num_imputer)
        ]
        if any(skew_mask):
            skewed_cols = [c for c, mask in zip(num_cols, skew_mask) if mask]
            skewed_info = {c: f"{skew_map[c]:.2f}" for c in skewed_cols[:10]}
            
            self.explainer_.explain_transformation(
                transform_name="Yeo-Johnson Power Transform",
                columns=skewed_cols,
                reason=(
                    f"Applying Yeo-Johnson transformation to {len(skewed_cols)} skewed numeric features "
                    f"(|skewness| >= {cfg.skew_threshold}). This normalizes distributions and can improve "
                    "model performance for linear models and neural networks."
                ),
                details={
                    "n_features": len(skewed_cols),
                    "skewness_threshold": cfg.skew_threshold,
                    "sample_skewness": skewed_info,
                },
                config_params={"skew_threshold": cfg.skew_threshold},
                recommendation=(
                    "Power transforms are most beneficial for linear models. "
                    "Tree-based models are generally robust to skewness."
                ),
            )
            
            steps_num.append(("yeojohnson", SkewedPowerTransformer(num_cols, skew_mask)))
        
        # Optional winsorization before scaling
        if cfg.winsorize:
            from .transformers import WinsorizerTransformer
            
            self.explainer_.explain_transformation(
                transform_name="Winsorization (Outlier Clipping)",
                columns=num_cols,
                reason=(
                    f"Clipping extreme values to {cfg.clip_percentiles[0]:.1%} and {cfg.clip_percentiles[1]:.1%} "
                    "percentiles to reduce the impact of outliers. This is especially useful before scaling."
                ),
                details={
                    "lower_percentile": cfg.clip_percentiles[0],
                    "upper_percentile": cfg.clip_percentiles[1],
                },
                config_params={"winsorize": cfg.winsorize, "clip_percentiles": cfg.clip_percentiles},
                recommendation="Winsorization is a robust alternative to removing outliers completely.",
            )
            
            steps_num.append(("winsorize", WinsorizerTransformer(
                percentiles=cfg.clip_percentiles,
                columns=num_cols,
            )))
        
        scaler = choose_scaler(estimator_family, heavy_outliers, cfg)
        if scaler is not None:
            scaler_name = type(scaler).__name__
            
            # Determine reason for scaler choice
            if heavy_outliers and cfg.scaler_robust_if_outliers:
                reason = (
                    f"Using RobustScaler because heavy outliers detected (>{cfg.outlier_share_threshold:.1%} "
                    f"of values are outliers). RobustScaler uses median and IQR, making it robust to outliers."
                )
            else:
                reason = (
                    f"Using {scaler_name} for {estimator_family} estimator family. "
                )
                if estimator_family.lower() in {"linear", "svm"}:
                    reason += "Linear models and SVMs benefit from standardized features with mean=0, std=1."
                elif estimator_family.lower() in {"knn", "nn"}:
                    reason += "Distance-based models require scaled features to prevent dominance by large-magnitude features."
                elif estimator_family.lower() in {"tree", "gbm"}:
                    reason += "Tree-based models don't require scaling but it was explicitly configured."
            
            self.explainer_.explain_scaling(
                scaler_name=scaler_name,
                columns=num_cols,
                reason=reason,
                details={
                    "estimator_family": estimator_family,
                    "heavy_outliers": heavy_outliers,
                    "outlier_threshold": cfg.outlier_share_threshold,
                },
                config_params={
                    f"scaler_{estimator_family.lower()}": cfg.scaler_tree if estimator_family.lower() == "tree" else cfg.scaler_linear,
                    "scaler_robust_if_outliers": cfg.scaler_robust_if_outliers,
                },
            )
            
            steps_num.append(("scale", scaler))
        elif num_cols:
            # Explain why no scaling
            reason = (
                f"No scaling applied for {estimator_family} estimator family. "
            )
            if estimator_family.lower() in {"tree", "gbm"}:
                reason += "Tree-based models are scale-invariant and don't require feature scaling."
            
            self.explainer_.explain_scaling(
                scaler_name="None (No Scaling)",
                columns=num_cols,
                reason=reason,
                details={"estimator_family": estimator_family},
                config_params={f"scaler_{estimator_family.lower()}": "none"},
            )
        
        num_pipe = Pipeline(steps=steps_num)

        # Categorical pipelines
        from .encoders import make_ohe
        
        # Explain low cardinality encoding
        if low_cat:
            self.explainer_.explain_encoding(
                strategy_name="One-Hot Encoding (OHE)",
                columns=low_cat,
                reason=(
                    f"Using one-hot encoding for {len(low_cat)} low-cardinality categorical features. "
                    f"OHE creates binary columns for each category, which works well when cardinality is low "
                    f"(<= {cfg.low_cardinality_max} unique values)."
                ),
                details={
                    "n_columns": len(low_cat),
                    "rare_grouping_threshold": cfg.rare_level_threshold,
                    "handle_unknown": cfg.ohe_handle_unknown,
                },
                config_params={
                    "low_cardinality_max": cfg.low_cardinality_max,
                    "rare_level_threshold": cfg.rare_level_threshold,
                    "ohe_handle_unknown": cfg.ohe_handle_unknown,
                },
                recommendation=(
                    f"Rare categories (<{cfg.rare_level_threshold:.1%} frequency) will be grouped into 'Other' "
                    "to prevent overfitting on rare values."
                ),
            )
        
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
                
                self.explainer_.explain_encoding(
                    strategy_name="Leave-One-Out Target Encoding",
                    columns=mid_cat,
                    reason=(
                        f"Using leave-one-out target encoding for {len(mid_cat)} medium-cardinality features. "
                        "This replaces categories with target statistics computed excluding the current row, "
                        "preventing leakage while capturing the predictive relationship between category and target."
                    ),
                    details={
                        "n_columns": len(mid_cat),
                        "smoothing": cfg.target_encoding_smoothing,
                        "noise_std": cfg.target_encoding_noise,
                    },
                    config_params={
                        "use_target_encoding": cfg.use_target_encoding,
                        "use_leave_one_out_te": cfg.use_leave_one_out_te,
                        "target_encoding_smoothing": cfg.target_encoding_smoothing,
                        "target_encoding_noise": cfg.target_encoding_noise,
                    },
                    recommendation=(
                        "Target encoding is powerful for medium-cardinality features but requires careful "
                        "cross-validation to avoid overfitting."
                    ),
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
                
                self.explainer_.explain_encoding(
                    strategy_name="Out-of-Fold Target Encoding",
                    columns=mid_cat,
                    reason=(
                        f"Using out-of-fold target encoding for {len(mid_cat)} medium-cardinality features. "
                        f"This uses {cfg.cv_n_splits}-fold cross-validation to encode categories with target statistics, "
                        "preventing leakage and providing robust encodings."
                    ),
                    details={
                        "n_columns": len(mid_cat),
                        "cv_strategy": cfg.cv_strategy,
                        "n_splits": cfg.cv_n_splits,
                        "smoothing": cfg.te_smoothing,
                        "noise_std": cfg.te_noise,
                        "prior_strategy": cfg.te_prior,
                    },
                    config_params={
                        "use_target_encoding": cfg.use_target_encoding,
                        "cv_strategy": cfg.cv_strategy,
                        "cv_n_splits": cfg.cv_n_splits,
                        "te_smoothing": cfg.te_smoothing,
                        "te_noise": cfg.te_noise,
                    },
                    recommendation=(
                        "Out-of-fold target encoding is the gold standard for preventing leakage. "
                        "Higher smoothing values add more regularization."
                    ),
                )
            cat_mid_pipe = Pipeline(steps=[("impute", categorical_imputer(cfg)), ("te", te)])
        elif cfg.use_frequency_encoding and mid_cat:
            # Alternative: Use FrequencyEncoder
            freq_enc = FrequencyEncoder(cols=None, unseen_value=0.0)  # Let ColumnTransformer handle column selection
            
            self.explainer_.explain_encoding(
                strategy_name="Frequency Encoding",
                columns=mid_cat,
                reason=(
                    f"Using frequency encoding for {len(mid_cat)} medium-cardinality features. "
                    "Each category is replaced with its frequency (proportion) in the training data. "
                    "This is simpler than target encoding and doesn't use target information."
                ),
                details={"n_columns": len(mid_cat), "unseen_value": 0.0},
                config_params={"use_frequency_encoding": cfg.use_frequency_encoding},
                recommendation="Frequency encoding is a safe alternative when you want to avoid target encoding.",
            )
            
            cat_mid_pipe = Pipeline(steps=[("impute", categorical_imputer(cfg)), ("freq", freq_enc)])
        elif cfg.use_count_encoding and mid_cat:
            # Alternative: Use CountEncoder
            count_enc = CountEncoder(cols=None, unseen_value=0.0, normalize=False)  # Let ColumnTransformer handle column selection
            
            self.explainer_.explain_encoding(
                strategy_name="Count Encoding",
                columns=mid_cat,
                reason=(
                    f"Using count encoding for {len(mid_cat)} medium-cardinality features. "
                    "Each category is replaced with its absolute count in the training data."
                ),
                details={"n_columns": len(mid_cat), "unseen_value": 0.0, "normalize": False},
                config_params={"use_count_encoding": cfg.use_count_encoding},
                recommendation="Count encoding preserves the absolute frequency information unlike frequency encoding.",
            )
            
            cat_mid_pipe = Pipeline(steps=[("impute", categorical_imputer(cfg)), ("count", count_enc)])
        else:
            # Fallback to hashing if TE disabled
            if mid_cat:
                self.explainer_.explain_encoding(
                    strategy_name="Feature Hashing",
                    columns=mid_cat,
                    reason=(
                        f"Using feature hashing for {len(mid_cat)} medium-cardinality features "
                        f"(target encoding is disabled). Hashing projects categories into {cfg.hashing_n_features_tabular} "
                        "dimensions using a hash function, providing a memory-efficient encoding."
                    ),
                    details={
                        "n_columns": len(mid_cat),
                        "n_hash_features": cfg.hashing_n_features_tabular,
                    },
                    config_params={
                        "hashing_n_features_tabular": cfg.hashing_n_features_tabular,
                        "use_target_encoding": cfg.use_target_encoding,
                    },
                    recommendation="Consider enabling target encoding for potentially better performance.",
                )
            
            cat_mid_pipe = Pipeline(
                steps=[
                    ("impute", categorical_imputer(cfg)),
                    ("rare", RareCategoryGrouper(min_freq=cfg.rare_level_threshold)),
                    ("hash", HashingEncoder(n_features=cfg.hashing_n_features_tabular, seed=cfg.random_state)),
                ]
            )
        # High-card hashing
        if high_cat:
            self.explainer_.explain_encoding(
                strategy_name="Feature Hashing (High Cardinality)",
                columns=high_cat,
                reason=(
                    f"Using feature hashing for {len(high_cat)} high-cardinality features "
                    f"(>{cfg.mid_cardinality_max} unique values). Hashing prevents dimension explosion "
                    f"by projecting categories into {cfg.hashing_n_features_tabular} dimensions."
                ),
                details={
                    "n_columns": len(high_cat),
                    "n_hash_features": cfg.hashing_n_features_tabular,
                    "rare_grouping_threshold": cfg.rare_level_threshold,
                },
                config_params={
                    "hashing_n_features_tabular": cfg.hashing_n_features_tabular,
                    "mid_cardinality_max": cfg.mid_cardinality_max,
                },
                recommendation=(
                    "For very high cardinality features (like IDs), consider whether they should be "
                    "included at all, as they may not generalize well."
                ),
            )
        
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
        if text_cols:
            text_method = "Hashing Vectorizer" if cfg.text_use_hashing else "TF-IDF"
            svd_k = (
                None
                if estimator_family.lower() in {"linear", "svm"}
                else cfg.svd_components_for_trees
            )
            
            details = {
                "n_columns": len(text_cols),
                "method": text_method,
            }
            
            if cfg.text_use_hashing:
                details["n_features"] = cfg.text_hashing_features
            else:
                details["max_features"] = cfg.tfidf_max_features
            
            if svd_k:
                details["svd_components"] = svd_k
            
            reason = (
                f"Processing {len(text_cols)} text columns using {text_method}. "
            )
            if cfg.text_use_hashing:
                reason += f"Hashing vectorizer provides memory-efficient text encoding with {cfg.text_hashing_features} features."
            else:
                reason += f"TF-IDF captures term importance across documents with up to {cfg.tfidf_max_features} features."
            
            if svd_k:
                reason += f" Applying SVD dimensionality reduction to {svd_k} components for tree models."
            
            self.explainer_.explain_text_processing(
                columns=text_cols,
                method=text_method,
                details=details,
                config_params={
                    "text_use_hashing": cfg.text_use_hashing,
                    "text_hashing_features": cfg.text_hashing_features,
                    "tfidf_max_features": cfg.tfidf_max_features,
                    "svd_components_for_trees": svd_k,
                    "text_char_ngrams": cfg.text_char_ngrams,
                },
            )
            
            for c in text_cols:
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
            
            features_generated = [
                "year", "quarter", "month", "weekday", "hour", "is_weekend",
                "month_sin/cos", "weekday_sin/cos", "hour_sin/cos"
            ]
            
            # Add Fourier features if enabled
            if cfg.use_fourier and cfg.time_column:
                features_generated.append(f"fourier (orders: {cfg.fourier_orders})")
                for col in dt_cols:
                    dt_steps.append(
                        (f"fourier_{col}", FourierFeatures(column=col, orders=cfg.fourier_orders))
                    )
            # Add holiday features if enabled
            if cfg.holiday_country and cfg.time_column:
                features_generated.append(f"holidays ({cfg.holiday_country})")
                for col in dt_cols:
                    dt_steps.append(
                        (f"holiday_{col}", HolidayFeatures(column=col, country_code=cfg.holiday_country))
                    )
            
            self.explainer_.explain_datetime_processing(
                columns=dt_cols,
                features_generated=features_generated,
                config_params={
                    "use_fourier": cfg.use_fourier,
                    "fourier_orders": cfg.fourier_orders if cfg.use_fourier else None,
                    "holiday_country": cfg.holiday_country,
                },
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
            
            self.explainer_.explain_validation(
                validation_type="Schema Validation",
                reason=(
                    "Validating input data schema before transformation to detect data drift and type errors. "
                    f"Schema coercion is {'enabled' if cfg.schema_coerce else 'disabled'}."
                ),
                config_params={
                    "validate_schema": cfg.validate_schema,
                    "schema_coerce": cfg.schema_coerce,
                    "schema_path": cfg.schema_path,
                },
            )
        
        # STEP 2: Main preprocessing
        pipe_steps.append(("preprocess", preprocessor))
        
        # STEP 3: Optional dimensionality reducer
        if cfg.reducer_kind:
            from .transformers import DimensionalityReducer
            
            n_comp = cfg.reducer_components or "auto"
            reason = (
                f"Applying {cfg.reducer_kind.upper()} dimensionality reduction to reduce feature space. "
            )
            if cfg.reducer_kind == "pca":
                if cfg.reducer_variance:
                    reason += f"Keeping components that explain {cfg.reducer_variance:.1%} of variance."
                else:
                    reason += f"Reducing to {n_comp} components."
            elif cfg.reducer_kind == "svd":
                reason += f"Using truncated SVD to extract {n_comp} latent features."
            elif cfg.reducer_kind == "umap":
                reason += f"Using UMAP for non-linear dimensionality reduction to {n_comp} components."
            
            self.explainer_.explain_dimensionality_reduction(
                method=cfg.reducer_kind,
                n_components=cfg.reducer_components or 0,
                reason=reason,
                config_params={
                    "reducer_kind": cfg.reducer_kind,
                    "reducer_components": cfg.reducer_components,
                    "reducer_variance": cfg.reducer_variance,
                },
            )
            
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
        
        # Finalize explanation with summary
        self.explainer_.set_summary(
            estimator_family=estimator_family,
            task_type=task.value,
            n_features_in=X.shape[1],
            n_features_out=0,  # Will be updated after fit
        )
        
        # Store explanation
        self.explanation_ = self.explainer_.get_explanation()
        
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
