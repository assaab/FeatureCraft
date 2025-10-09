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
    BinaryEncoder,
    CatBoostEncoder,
    CountEncoder,
    EntityEmbeddingsEncoder,
    FrequencyEncoder,
    HashingEncoder,
    KFoldTargetEncoder,
    OrdinalEncoder,
    OutOfFoldTargetEncoder,
    RareCategoryGrouper,
    WoEEncoder,
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
from .transformers import (
    DateTimeFeatures, 
    EnsureNumericOutput, 
    NumericConverter, 
    SkewedPowerTransformer,
    MathematicalTransformer,
    BinningTransformer,
    AutoBinningSelector,
)
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
    """Main class for automatic feature engineering with AI-powered optimization.
    
    Now supports intelligent feature engineering using AI (LLMs) to analyze
    your data and recommend optimal strategies, preventing feature explosion
    and reducing training time while maintaining performance.
    
    Key Features:
    - AI-powered feature engineering recommendations
    - Adaptive strategy selection based on dataset characteristics
    - Intelligent feature selection to prevent overfitting
    - Reduced training time with smarter feature creation
    
    Usage:
        # Standard mode (uses heuristics)
        afe = AutoFeatureEngineer()
        afe.fit(X, y)
        
        # AI-powered mode (requires API key)
        afe = AutoFeatureEngineer(
            use_ai_advisor=True,
            ai_api_key="your-openai-key"
        )
        afe.fit(X, y)
    """

    def __init__(
        self, 
        config: FeatureCraftConfig | None = None,
        use_ai_advisor: bool = False,
        ai_api_key: Optional[str] = None,
        ai_model: str = "gpt-4o-mini",
        ai_provider: str = "openai",
        time_budget: str = "balanced",
    ) -> None:
        """Initialize with optional config and AI advisor.
        
        Args:
            config: Feature engineering configuration
            use_ai_advisor: Enable AI-powered recommendations (requires API key)
            ai_api_key: API key for LLM provider (OpenAI, Anthropic)
            ai_model: Model name (e.g., 'gpt-4o-mini', 'claude-3-sonnet')
            ai_provider: LLM provider ('openai', 'anthropic')
            time_budget: Time budget ('fast', 'balanced', 'thorough')
        """
        self.cfg = config or FeatureCraftConfig()
        self.insights_: DatasetInsights | None = None
        self.pipeline_: Pipeline | None = None
        self.summary_: PipelineSummary | None = None
        self.feature_names_: list[str] | None = None
        self.estimator_family_: str = "tree"
        self.task_: TaskType | None = None
        self.explainer_: PipelineExplainer | None = None
        self.explanation_: Any | None = None  # PipelineExplanation from explainability module
        
        # AI advisor integration
        self.use_ai_advisor = use_ai_advisor
        self.ai_planner_: Any | None = None  # FeatureEngineeringPlanner
        self.ai_strategy_: Any | None = None  # FeatureStrategy
        
        if use_ai_advisor:
            try:
                from .ai import FeatureEngineeringPlanner
                self.ai_planner_ = FeatureEngineeringPlanner(
                    use_ai=True,
                    api_key=ai_api_key,
                    model=ai_model,
                    provider=ai_provider,
                    time_budget=time_budget,
                    base_config=self.cfg,
                    verbose=(self.cfg.verbosity >= 1),
                )
                logger.info("✓ AI-powered feature engineering enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize AI advisor: {e}")
                logger.info("Continuing with heuristic-based feature engineering")
                self.use_ai_advisor = False

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
        
        # AI-powered optimization: Get intelligent recommendations before building pipeline
        if self.use_ai_advisor and self.ai_planner_ is not None:
            try:
                # Analyze dataset first (if not already done)
                if self.insights_ is None:
                    # Combine X and y for analysis
                    df_combined = pd.concat([X, y], axis=1)
                    self.insights_ = self.analyze(df_combined, y.name or "target")
                
                # Get AI recommendations - CRITICAL: Pass current config to avoid using stale base_config
                plan = self.ai_planner_.create_plan(
                    X=X,
                    y=y,
                    insights=self.insights_,
                    estimator_family=estimator_family,
                    current_config=self.cfg,  # Pass current config instead of using stale base_config
                )
                
                # Apply optimized configuration
                self.cfg = plan.config
                self.ai_strategy_ = plan.strategy
                
                logger.info(f"✓ Applied AI-optimized configuration (estimated {plan.strategy.estimated_feature_count} features)")
            
            except RuntimeError as e:
                # AI feature engineering failed - re-raise with clear message
                if "AI-powered feature engineering failed" in str(e):
                    console.print(f"\n[red]❌ AI feature engineering failed: {e}[/red]")
                    console.print("[yellow]💡 Tip: Use use_ai_advisor=False for heuristic-based feature engineering[/yellow]")
                else:
                    console.print(f"\n[red]❌ AI optimization failed: {e}[/red]")
                    console.print("[yellow]💡 Tip: Check API key and model configuration[/yellow]")
                raise
            except Exception as e:
                # Re-raise all exceptions - do NOT fall back to local feature creation
                logger.error(f"AI optimization failed with unexpected error: {e}")
                console.print(f"\n[red]❌ AI feature engineering failed: {e}[/red]")
                console.print("[yellow]💡 Tip: Use use_ai_advisor=False for heuristic-based feature engineering[/yellow]")
                raise RuntimeError(f"AI feature engineering failed: {str(e)}") from e
        
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
        
        # Optional binning/discretization BEFORE power transforms
        # This allows linear models to learn thresholds and non-linear patterns
        if cfg.binning_enabled and num_cols:
            if cfg.binning_strategy == "auto":
                # Use AutoBinningSelector for intelligent strategy selection
                binning_transformer = AutoBinningSelector(
                    columns=cfg.binning_columns or num_cols,
                    n_bins=cfg.binning_n_bins,
                    encode=cfg.binning_encode,
                    prefer_supervised=cfg.binning_prefer_supervised,
                    skewness_threshold=cfg.binning_skewness_threshold,
                    random_state=cfg.random_state,
                )
                strategy_desc = "Auto (adaptive per column)"
            else:
                # Use specific strategy for all columns
                binning_transformer = BinningTransformer(
                    columns=cfg.binning_columns or num_cols,
                    strategy=cfg.binning_strategy,
                    n_bins=cfg.binning_n_bins,
                    encode=cfg.binning_encode,
                    custom_bins=cfg.binning_custom_bins,
                    handle_unknown=cfg.binning_handle_unknown,
                    subsample=cfg.binning_subsample,
                    random_state=cfg.random_state,
                )
                strategy_desc = cfg.binning_strategy
            
            # Fit binning to determine actual columns binned
            binning_transformer.fit(X[num_cols], y)
            
            # Explain binning transformation
            binning_cols = cfg.binning_columns or num_cols
            self.explainer_.explain_transformation(
                transform_name="Binning/Discretization",
                columns=binning_cols,
                reason=(
                    f"Converting {len(binning_cols)} continuous features into {cfg.binning_n_bins} discrete bins "
                    f"using {strategy_desc} strategy. This enables linear models to learn non-linear patterns "
                    "and threshold effects (e.g., 'age > 65 → high risk'). "
                    f"Output encoding: {cfg.binning_encode}."
                ),
                details={
                    "n_columns": len(binning_cols),
                    "strategy": cfg.binning_strategy,
                    "n_bins": cfg.binning_n_bins,
                    "encode": cfg.binning_encode,
                    "prefer_supervised": cfg.binning_prefer_supervised if cfg.binning_strategy == "auto" else None,
                },
                config_params={
                    "binning_enabled": cfg.binning_enabled,
                    "binning_strategy": cfg.binning_strategy,
                    "binning_n_bins": cfg.binning_n_bins,
                    "binning_encode": cfg.binning_encode,
                    "binning_prefer_supervised": cfg.binning_prefer_supervised,
                },
                recommendation=(
                    "Binning is especially useful for linear models (logistic regression, linear regression) "
                    "and can sometimes benefit tree-based models by reducing overfitting on continuous values."
                ),
            )
            
            steps_num.append(("binning", binning_transformer))
        
        # Mathematical transforms: Use new MathematicalTransformer with intelligent auto-selection
        # This replaces the old SkewedPowerTransformer with more comprehensive transform options
        if cfg.transform_strategy != "none" and num_cols:
            # Determine columns to transform
            transform_cols = cfg.transform_columns or num_cols
            
            # Create MathematicalTransformer with unified interface
            math_transformer = MathematicalTransformer(
                columns=transform_cols,
                strategy=cfg.transform_strategy,
                log_shift=cfg.log_shift,
                sqrt_handle_negatives=cfg.sqrt_handle_negatives,
                reciprocal_epsilon=cfg.reciprocal_epsilon,
                exponential_type=cfg.exponential_transform_type,
                boxcox_lambda=cfg.boxcox_lambda,
                skew_threshold=cfg.skew_threshold,
            )
            
            # Fit transformer to determine strategies per column
            math_transformer.fit(X[transform_cols], y)
            
            # Get selected strategies for explanation
            strategies = math_transformer.get_strategies()
            skewness_values = math_transformer.get_skewness()
            
            # Group columns by transform type for explanation
            strategy_groups: dict[str, list[str]] = {}
            for col, strategy in strategies.items():
                if strategy != "none":
                    if strategy not in strategy_groups:
                        strategy_groups[strategy] = []
                    strategy_groups[strategy].append(col)
            
            # Build comprehensive explanation
            if strategy_groups:
                transform_summary = []
                for strategy, cols in strategy_groups.items():
                    n_cols = len(cols)
                    sample_cols = cols[:5]
                    sample_skew = {c: f"{skewness_values.get(c, 0):.2f}" for c in sample_cols}
                    
                    strategy_name_map = {
                        "log": "Log transform: log(x + shift)",
                        "log1p": "Log1p transform: log(1 + x)",
                        "sqrt": "Square root transform: sqrt(x)",
                        "box_cox": "Box-Cox transform (optimized lambda)",
                        "yeo_johnson": "Yeo-Johnson transform (handles negatives)",
                        "reciprocal": "Reciprocal transform: 1/x",
                        "exponential": f"Exponential transform: {cfg.exponential_transform_type}",
                    }
                    
                    transform_summary.append(
                        f"  • {strategy_name_map.get(strategy, strategy)}: {n_cols} columns "
                        f"(sample: {', '.join(sample_cols)})"
                    )
                
                total_transformed = sum(len(cols) for cols in strategy_groups.values())
                
                if cfg.transform_strategy == "auto":
                    reason = (
                        f"Applying intelligent auto-selected mathematical transforms to {total_transformed}/{len(transform_cols)} "
                        f"numeric features based on data characteristics (skewness, range, zeros, negatives). "
                        f"Transforms normalize distributions and improve model performance.\n\n"
                        f"Selected transforms:\n" + "\n".join(transform_summary)
                    )
                else:
                    reason = (
                        f"Applying {cfg.transform_strategy} transform to {total_transformed} numeric features "
                        f"based on configuration setting (transform_strategy='{cfg.transform_strategy}').\n\n"
                        f"Transformed columns:\n" + "\n".join(transform_summary)
                    )
                
                self.explainer_.explain_transformation(
                    transform_name=f"Mathematical Transforms ({cfg.transform_strategy})",
                    columns=list(strategies.keys()),
                    reason=reason,
                    details={
                        "strategy": cfg.transform_strategy,
                        "n_transformed": total_transformed,
                        "n_total": len(transform_cols),
                        "strategy_breakdown": {k: len(v) for k, v in strategy_groups.items()},
                        "sample_skewness": {c: f"{skewness_values.get(c, 0):.2f}" for c in list(strategies.keys())[:10]},
                    },
                    config_params={
                        "transform_strategy": cfg.transform_strategy,
                        "skew_threshold": cfg.skew_threshold,
                        "log_shift": cfg.log_shift,
                        "sqrt_handle_negatives": cfg.sqrt_handle_negatives,
                        "reciprocal_epsilon": cfg.reciprocal_epsilon,
                        "exponential_transform_type": cfg.exponential_transform_type,
                        "boxcox_lambda": cfg.boxcox_lambda,
                    },
                    recommendation=(
                        "Mathematical transforms are most beneficial for linear models and neural networks. "
                        "Tree-based models are generally robust to skewness but may benefit from log/log1p for "
                        "very large magnitude differences. Auto-selection intelligently chooses the best transform "
                        "per column based on data characteristics."
                    ),
                )
                
                steps_num.append(("math_transform", math_transformer))
            else:
                # All columns had skewness below threshold - no transform needed
                logger.debug(
                    f"No transforms applied: all {len(transform_cols)} columns have "
                    f"|skewness| < {cfg.skew_threshold}"
                )
        
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
            from .text import build_comprehensive_text_pipeline
            
            # Determine which NLP features are enabled
            text_method = "Hashing Vectorizer" if cfg.text_use_hashing else "TF-IDF"
            svd_k = (
                None
                if estimator_family.lower() in {"linear", "svm"}
                else cfg.svd_components_for_trees
            )
            
            # Build comprehensive feature list
            enabled_features = []
            if cfg.text_extract_statistics:
                enabled_features.append("text statistics (char/word/sentence counts, avg_word_length, etc.)")
            if cfg.text_extract_linguistic:
                enabled_features.append("linguistic features (stopwords, punctuation, uppercase ratio)")
            if cfg.text_extract_sentiment:
                enabled_features.append(f"sentiment analysis ({cfg.text_sentiment_method})")
            if cfg.text_use_word_embeddings:
                enabled_features.append(f"word embeddings ({cfg.text_embedding_method}, {cfg.text_embedding_dims}D)")
            if cfg.text_use_sentence_embeddings:
                enabled_features.append(f"sentence embeddings ({cfg.text_sentence_model})")
            if cfg.text_extract_ner:
                enabled_features.append(f"named entity recognition ({cfg.text_ner_model})")
            if cfg.text_use_topic_modeling:
                enabled_features.append(f"topic modeling (LDA with {cfg.text_topic_n_topics} topics)")
            if cfg.text_extract_readability:
                enabled_features.append("readability scores (Flesch-Kincaid, SMOG, etc.)")
            
            enabled_features.append(f"{text_method} vectorization")
            
            details = {
                "n_columns": len(text_cols),
                "base_method": text_method,
                "enabled_features": enabled_features,
            }
            
            if cfg.text_use_hashing:
                details["n_features"] = cfg.text_hashing_features
            else:
                details["max_features"] = cfg.tfidf_max_features
            
            if svd_k:
                details["svd_components"] = svd_k
            
            # Build comprehensive reason string
            reason = (
                f"Processing {len(text_cols)} text columns with comprehensive NLP feature engineering. "
                f"Enabled features: {', '.join(enabled_features)}. "
            )
            
            if cfg.text_use_hashing:
                reason += f"Hashing vectorizer provides memory-efficient encoding with {cfg.text_hashing_features} features. "
            else:
                reason += f"TF-IDF captures term importance with up to {cfg.tfidf_max_features} features. "
            
            if svd_k:
                reason += f"Applying SVD dimensionality reduction to {svd_k} components for tree models."
            
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
                    "text_extract_statistics": cfg.text_extract_statistics,
                    "text_extract_sentiment": cfg.text_extract_sentiment,
                    "text_use_word_embeddings": cfg.text_use_word_embeddings,
                    "text_use_sentence_embeddings": cfg.text_use_sentence_embeddings,
                    "text_extract_ner": cfg.text_extract_ner,
                    "text_use_topic_modeling": cfg.text_use_topic_modeling,
                    "text_extract_readability": cfg.text_extract_readability,
                },
            )
            
            # Build comprehensive text pipeline for each text column
            for c in text_cols:
                text_pipeline = build_comprehensive_text_pipeline(
                    column_name=c,
                    cfg=cfg,
                )
                
                text_transformers.append(
                    (
                        f"text_{c}", 
                        text_pipeline,
                        c
                    )
                )

        # Datetime expansion with comprehensive feature engineering
        if dt_cols:
            from .time_series import FourierFeatures, HolidayFeatures
            
            # Build comprehensive DateTimeFeatures transformer with config options
            dt_base = DateTimeFeatures(
                dt_cols,
                extract_basic=cfg.dt_extract_basic,
                extract_cyclical=cfg.dt_extract_cyclical,
                extract_boolean_flags=cfg.dt_extract_boolean_flags,
                extract_season=cfg.dt_extract_season,
                extract_business=cfg.dt_extract_business,
                extract_relative=cfg.dt_extract_relative,
                reference_date=cfg.dt_reference_date,
                business_hour_start=cfg.dt_business_hour_start,
                business_hour_end=cfg.dt_business_hour_end,
            )
            dt_steps = [("base", dt_base)]
            
            # Track features generated for explainability
            features_generated = []
            if cfg.dt_extract_basic:
                features_generated.append("basic (year, month, day, day_of_week, week_of_year, quarter, day_of_year, hour, minute, second)")
            if cfg.dt_extract_cyclical:
                features_generated.append("cyclical (month_sin/cos, day_of_week_sin/cos, day_of_year_sin/cos, hour_sin/cos)")
            if cfg.dt_extract_boolean_flags:
                features_generated.append("boolean flags (is_weekend, is_month_start/end, is_quarter_start/end, is_year_start/end)")
            if cfg.dt_extract_season:
                features_generated.append("season (0=winter, 1=spring, 2=summer, 3=fall)")
            if cfg.dt_extract_business:
                features_generated.append(f"business logic (is_business_hour {cfg.dt_business_hour_start}-{cfg.dt_business_hour_end}, business_days_in_month)")
            if cfg.dt_extract_relative and cfg.dt_reference_date:
                features_generated.append(f"relative time (days/weeks/months since {cfg.dt_reference_date})")
            
            # Add Fourier features if enabled
            if cfg.use_fourier and cfg.time_column:
                features_generated.append(f"fourier (orders: {cfg.fourier_orders})")
                for col in dt_cols:
                    dt_steps.append(
                        (f"fourier_{col}", FourierFeatures(column=col, orders=cfg.fourier_orders))
                    )
            
            # Add holiday features if enabled
            if cfg.holiday_country and cfg.time_column:
                features_generated.append(f"holidays ({cfg.holiday_country}: is_holiday, days_to_holiday, days_from_holiday)")
                for col in dt_cols:
                    dt_steps.append(
                        (f"holiday_{col}", HolidayFeatures(
                            column=col, 
                            country_code=cfg.holiday_country,
                            extract_days_to=True,
                            extract_days_from=True,
                        ))
                    )
            
            # Explain datetime processing to user
            self.explainer_.explain_datetime_processing(
                columns=dt_cols,
                features_generated=features_generated,
                config_params={
                    "extract_basic": cfg.dt_extract_basic,
                    "extract_cyclical": cfg.dt_extract_cyclical,
                    "extract_boolean_flags": cfg.dt_extract_boolean_flags,
                    "extract_season": cfg.dt_extract_season,
                    "extract_business": cfg.dt_extract_business,
                    "extract_relative": cfg.dt_extract_relative,
                    "reference_date": cfg.dt_reference_date,
                    "business_hour_start": cfg.dt_business_hour_start,
                    "business_hour_end": cfg.dt_business_hour_end,
                    "use_fourier": cfg.use_fourier,
                    "fourier_orders": cfg.fourier_orders if cfg.use_fourier else None,
                    "holiday_country": cfg.holiday_country,
                },
            )
            
            dt_pipe = Pipeline(steps=dt_steps) if len(dt_steps) > 0 else dt_base
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
        
        # STEP 3: Feature Interactions (optional, after preprocessing)
        if cfg.interactions_enabled:
            from .interactions import build_interaction_pipeline
            from sklearn.pipeline import FeatureUnion
            
            logger.info("Building feature interaction pipeline...")
            
            # Build interaction transformers
            interaction_transformers = build_interaction_pipeline(cfg, X)
            
            if interaction_transformers:
                # Create a FeatureUnion to combine all interaction types
                interaction_union = FeatureUnion(
                    transformer_list=interaction_transformers,
                    n_jobs=1,  # Keep sequential for stability
                )
                
                # Wrap in a pipeline step
                pipe_steps.append(("interactions", interaction_union))
                
                # Explain interaction feature engineering
                enabled_interactions = []
                if cfg.interactions_use_arithmetic:
                    enabled_interactions.append(
                        f"Arithmetic ({', '.join(cfg.interactions_arithmetic_ops)})"
                    )
                if cfg.interactions_use_polynomial:
                    degree_str = "interaction-only" if cfg.interactions_polynomial_interaction_only else f"degree-{cfg.interactions_polynomial_degree}"
                    enabled_interactions.append(f"Polynomial ({degree_str})")
                if cfg.interactions_use_ratios:
                    enabled_interactions.append("Ratios & Proportions")
                if cfg.interactions_use_products:
                    enabled_interactions.append(f"{cfg.interactions_product_n_way}-way Products")
                if cfg.interactions_use_categorical_numeric:
                    enabled_interactions.append(f"Categorical×Numeric ({cfg.interactions_cat_num_strategy})")
                if cfg.interactions_use_binned:
                    enabled_interactions.append(f"Binned Interactions ({cfg.interactions_n_bins} bins)")
                
                self.explainer_.explain_transformation(
                    transform_name="Feature Interactions",
                    columns=num_cols + cat_cols,  # Interactions use both numeric and categorical
                    reason=(
                        "Creating feature interactions to capture non-linear relationships and "
                        "cross-feature patterns that linear models cannot learn directly. "
                        f"Enabled interaction types: {', '.join(enabled_interactions)}."
                    ),
                    details={
                        "interaction_types": enabled_interactions,
                        "arithmetic_ops": cfg.interactions_arithmetic_ops,
                        "polynomial_degree": cfg.interactions_polynomial_degree,
                        "cat_num_strategy": cfg.interactions_cat_num_strategy,
                        "n_bins": cfg.interactions_n_bins,
                    },
                    config_params={
                        "interactions_enabled": cfg.interactions_enabled,
                        "arithmetic": cfg.interactions_use_arithmetic,
                        "polynomial": cfg.interactions_use_polynomial,
                        "polynomial_degree": cfg.interactions_polynomial_degree if cfg.interactions_use_polynomial else None,
                        "ratios": cfg.interactions_use_ratios,
                        "products": cfg.interactions_use_products,
                        "categorical_numeric": cfg.interactions_use_categorical_numeric,
                        "binned": cfg.interactions_use_binned,
                    },
                )
                
                logger.info(f"Feature interactions enabled: {', '.join(enabled_interactions)}")
            else:
                logger.warning("Feature interactions enabled but no transformers were built")
        
        # STEP 4: Optional MI-based feature selection (after interactions)
        # This reduces feature count to prevent overfitting and improve performance
        if cfg.use_mi and cfg.mi_top_k:
            from .selection import MutualInfoSelector
            
            # Determine task type for MI computation
            task_str = "classification" if task == TaskType.CLASSIFICATION else "regression"
            
            mi_selector = MutualInfoSelector(
                k=cfg.mi_top_k,
                task=task_str,
                random_state=cfg.random_state,
            )
            
            # Add to pipeline
            pipe_steps.append(("mi_selection", mi_selector))
            
            # Explain MI selection
            self.explainer_.explain_feature_selection(
                method="Mutual Information (MI)",
                target_features=cfg.mi_top_k,
                reason=(
                    f"Selecting top {cfg.mi_top_k} features by mutual information score to prevent overfitting "
                    "and reduce computational cost. MI captures both linear and non-linear relationships between "
                    "features and target, making it a robust feature selection method."
                ),
                config_params={
                    "use_mi": cfg.use_mi,
                    "mi_top_k": cfg.mi_top_k,
                },
            )
            
            logger.info(f"MI feature selection enabled: keeping top {cfg.mi_top_k} features")
        
        # STEP 5: Optional WoE/IV-based feature selection (binary classification only)
        if cfg.use_woe_selection and task == TaskType.CLASSIFICATION:
            from .selection import WOEIVSelector
            
            woe_selector = WOEIVSelector(
                threshold=cfg.woe_iv_threshold,
                random_state=cfg.random_state,
            )
            
            pipe_steps.append(("woe_selection", woe_selector))
            
            self.explainer_.explain_feature_selection(
                method="Weight of Evidence (WoE/IV)",
                target_features=None,  # Threshold-based, not fixed count
                reason=(
                    f"Selecting features with Information Value (IV) >= {cfg.woe_iv_threshold} using WoE encoding. "
                    "Features with low IV have weak predictive power and are removed."
                ),
                config_params={
                    "use_woe_selection": cfg.use_woe_selection,
                    "woe_iv_threshold": cfg.woe_iv_threshold,
                },
            )
            
            logger.info(f"WoE/IV feature selection enabled with threshold {cfg.woe_iv_threshold}")
        
        # STEP 6: Optional dimensionality reducer
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
        
        # STEP 7: Final safety check
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
        """Get feature names from ALL fitted pipeline steps."""
        names: list[str] = []
        if self.pipeline_ is None:
            return names

        # Perform dummy transform on small sample to get exact output shape
        sample = X.head(min(5, len(X)))
        Xt = self.pipeline_.transform(sample)
        Xt_arr = Xt.toarray() if hasattr(Xt, "toarray") else np.asarray(Xt)
        n_features_total = Xt_arr.shape[1]

        # Collect names from EACH pipeline step that produces features
        for step_name, step_transformer in self.pipeline_.steps:
            if step_name == "schema_validator":
                continue  # Doesn't produce features
            
            if step_name == "preprocess":
                # Handle ColumnTransformer (existing logic)
                names.extend(self._get_names_from_column_transformer(step_transformer, sample, X))
            
            elif step_name == "interactions":
                # Handle FeatureUnion (NEW: this was missing!)
                names.extend(self._get_names_from_feature_union(step_transformer, sample))
            
            elif step_name == "mi_selection" or step_name == "woe_selection":
                # Handle feature selection - filters existing features
                if hasattr(step_transformer, 'get_feature_names_out'):
                    try:
                        selected_names = step_transformer.get_feature_names_out(names)
                        names = list(selected_names)
                        logger.debug(f"{step_name}: Selected {len(names)} features")
                    except Exception as e:
                        logger.warning(f"Failed to get feature names from {step_name}: {e}")
                elif hasattr(step_transformer, 'selected_features_'):
                    # Use selected_features_ attribute
                    names = [n for n in names if n in step_transformer.selected_features_]
                    logger.debug(f"{step_name}: Filtered to {len(names)} selected features")
            
            elif step_name == "reducer":
                # Handle dimensionality reducer
                names.extend(self._get_names_from_reducer(step_transformer, names))

        # Ensure name count matches actual output
        if len(names) != n_features_total:
            logger.warning(
                f"Feature name count mismatch: {len(names)} names vs {n_features_total} actual. "
                f"Collected from pipeline steps, but got {len(names)} != {n_features_total}. "
                "Using generic names as fallback."
            )
            names = [f"feature_{i}" for i in range(n_features_total)]

        return names

    def _get_names_from_column_transformer(
        self, 
        column_transformer: ColumnTransformer,
        sample: pd.DataFrame,
        X: pd.DataFrame
    ) -> list[str]:
        """Extract feature names from ColumnTransformer (preprocess step).
        
        Args:
            column_transformer: The ColumnTransformer instance
            sample: Small sample of data for transformation
            X: Original input DataFrame
            
        Returns:
            List of feature names from this transformer
        """
        names: list[str] = []
        
        for name, trans, cols in column_transformer.transformers_:
            if name == "remainder":
                continue

            colnames = [str(c) for c in cols] if isinstance(cols, list) else [str(cols)]

            # Try to get feature names from transformer
            if hasattr(trans, "get_feature_names_out"):
                try:
                    fn = trans.get_feature_names_out(colnames)
                    names.extend([str(x) for x in fn])
                    continue
                except Exception as e:
                    logger.debug(f"get_feature_names_out failed for {name}: {e}")

            # Fallback: infer from actual transformer output
            try:
                # Extract this transformer's output to count features
                col_indices = [i for i, col in enumerate(X.columns) if col in cols]
                if col_indices:
                    sample_subset = sample.iloc[:, col_indices]
                    # Check if transformer is fitted before calling transform
                    # Most sklearn transformers have fitted attributes like 'n_features_in_' when fitted
                    if hasattr(trans, 'n_features_in_'):
                        trans_output = trans.transform(sample_subset)
                        trans_arr = trans_output.toarray() if hasattr(trans_output, "toarray") else np.asarray(trans_output)
                        n_features_actual = trans_arr.shape[1]
                    else:
                        # Fallback to using column names if transformer not fitted
                        n_features_actual = len(colnames)
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
        
        return names

    def _get_names_from_feature_union(
        self,
        feature_union: FeatureUnion,
        sample: pd.DataFrame
    ) -> list[str]:
        """Extract feature names from FeatureUnion (interactions step).
        
        Args:
            feature_union: The FeatureUnion instance (contains interaction transformers)
            sample: Small sample of data for transformation
            
        Returns:
            List of feature names from all transformers in the union
        """
        names: list[str] = []
        
        # FeatureUnion has transformer_list: List[Tuple[str, transformer]]
        for trans_name, transformer in feature_union.transformer_list:
            # Try to get feature names directly
            if hasattr(transformer, 'get_feature_names_out'):
                try:
                    trans_names = transformer.get_feature_names_out()
                    names.extend([str(n) for n in trans_names])
                    logger.debug(f"Got {len(trans_names)} feature names from {trans_name}")
                    continue
                except Exception as e:
                    logger.debug(f"get_feature_names_out failed for {trans_name}: {e}")
            
            # Fallback: transform sample and count features
            try:
                # Check if transformer is fitted before calling transform
                if hasattr(transformer, 'n_features_in_'):
                    trans_output = transformer.transform(sample)
                    trans_arr = trans_output.toarray() if hasattr(trans_output, "toarray") else np.asarray(trans_output)
                    n_feats = trans_arr.shape[1]
                    fallback_names = [f"{trans_name}__feat_{i}" for i in range(n_feats)]
                    names.extend(fallback_names)
                    logger.warning(
                        f"{trans_name} doesn't support get_feature_names_out properly. "
                        f"Using fallback names for {n_feats} features."
                    )
                else:
                    # Transformer not fitted, skip fallback
                    logger.debug(f"Skipping fallback for unfitted transformer {trans_name}")
                    continue
            except Exception as e:
                logger.error(f"Failed to get names from {trans_name}: {e}")
        
        return names

    def _get_names_from_reducer(
        self,
        reducer: BaseEstimator,
        input_names: list[str]
    ) -> list[str]:
        """Extract feature names from dimensionality reducer.
        
        Args:
            reducer: The dimensionality reducer (PCA, etc.)
            input_names: Names of features before reduction
            
        Returns:
            List of feature names after reduction
        """
        names: list[str] = []
        
        # Try to get feature names
        if hasattr(reducer, 'get_feature_names_out'):
            try:
                names = [str(n) for n in reducer.get_feature_names_out(input_names)]
                return names
            except Exception as e:
                logger.debug(f"get_feature_names_out failed for reducer: {e}")
        
        # Fallback: generate generic names based on n_components
        if hasattr(reducer, 'n_components_'):
            n_components = reducer.n_components_
        elif hasattr(reducer, 'n_components'):
            n_components = reducer.n_components
        else:
            # Can't determine, return empty
            logger.warning("Could not determine number of components from reducer")
            return []
        
        # Generate names like "pca_0", "pca_1", etc.
        reducer_name = reducer.__class__.__name__.lower()
        names = [f"{reducer_name}_{i}" for i in range(n_components)]
        
        return names
