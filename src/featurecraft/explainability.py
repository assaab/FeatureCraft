"""Explainability and transparency module for FeatureCraft.

This module provides detailed explanations of transformation decisions made during
feature engineering pipeline construction.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .logging import get_logger

logger = get_logger(__name__)


class DecisionCategory(str, Enum):
    """Category of transformation decision."""
    
    COLUMN_CLASSIFICATION = "column_classification"
    ENCODING = "encoding"
    IMPUTATION = "imputation"
    SCALING = "scaling"
    TRANSFORMATION = "transformation"
    SELECTION = "selection"
    DIMENSIONALITY_REDUCTION = "dimensionality_reduction"
    TEXT_PROCESSING = "text_processing"
    DATETIME_PROCESSING = "datetime_processing"
    VALIDATION = "validation"


@dataclass
class TransformationExplanation:
    """Explanation for a single transformation decision.
    
    Attributes:
        category: Category of the decision
        operation: Name of the operation/transformer being applied
        columns: List of columns affected
        reason: Human-readable explanation of WHY this decision was made
        details: Additional details (statistics, thresholds, etc.)
        recommendation: Optional recommendation for the user
        config_used: Configuration parameters that influenced this decision
    """
    
    category: DecisionCategory
    operation: str
    columns: list[str]
    reason: str
    details: dict[str, Any] = field(default_factory=dict)
    recommendation: Optional[str] = None
    config_used: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "category": self.category.value,
            "operation": self.operation,
            "columns": self.columns,
            "reason": self.reason,
            "details": self.details,
            "recommendation": self.recommendation,
            "config_used": self.config_used,
        }


@dataclass
class PipelineExplanation:
    """Complete explanation of all pipeline decisions.
    
    Attributes:
        explanations: List of all transformation explanations
        summary: High-level summary of the pipeline
        estimator_family: Target estimator family
        task_type: Classification or regression
        n_features_in: Number of input features
        n_features_out: Number of output features
    """
    
    explanations: list[TransformationExplanation] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    estimator_family: Optional[str] = None
    task_type: Optional[str] = None
    n_features_in: int = 0
    n_features_out: int = 0
    
    def add_explanation(self, explanation: TransformationExplanation) -> None:
        """Add an explanation to the collection."""
        self.explanations.append(explanation)
        logger.debug(f"Added explanation: {explanation.operation} -> {explanation.reason}")
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "summary": self.summary,
            "estimator_family": self.estimator_family,
            "task_type": self.task_type,
            "n_features_in": self.n_features_in,
            "n_features_out": self.n_features_out,
            "explanations": [exp.to_dict() for exp in self.explanations],
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
    
    def to_markdown(self) -> str:
        """Convert to markdown format."""
        lines = ["# FeatureCraft Pipeline Explanation\n"]
        
        # Summary section
        lines.append("## Summary\n")
        lines.append(f"- **Task Type**: {self.task_type or 'Unknown'}")
        lines.append(f"- **Estimator Family**: {self.estimator_family or 'Unknown'}")
        lines.append(f"- **Input Features**: {self.n_features_in}")
        lines.append(f"- **Output Features**: {self.n_features_out}")
        lines.append("")
        
        # Group explanations by category
        by_category: dict[DecisionCategory, list[TransformationExplanation]] = {}
        for exp in self.explanations:
            if exp.category not in by_category:
                by_category[exp.category] = []
            by_category[exp.category].append(exp)
        
        # Generate sections for each category
        category_names = {
            DecisionCategory.COLUMN_CLASSIFICATION: "Column Classification",
            DecisionCategory.ENCODING: "Encoding Strategy",
            DecisionCategory.IMPUTATION: "Missing Value Imputation",
            DecisionCategory.SCALING: "Feature Scaling",
            DecisionCategory.TRANSFORMATION: "Feature Transformation",
            DecisionCategory.SELECTION: "Feature Selection",
            DecisionCategory.DIMENSIONALITY_REDUCTION: "Dimensionality Reduction",
            DecisionCategory.TEXT_PROCESSING: "Text Processing",
            DecisionCategory.DATETIME_PROCESSING: "DateTime Processing",
            DecisionCategory.VALIDATION: "Validation",
        }
        
        for category in DecisionCategory:
            if category not in by_category:
                continue
            
            lines.append(f"## {category_names.get(category, category.value)}\n")
            
            for exp in by_category[category]:
                lines.append(f"### {exp.operation}\n")
                lines.append(f"**Columns**: {', '.join(exp.columns) if exp.columns else 'All'}\n")
                lines.append(f"**Reason**: {exp.reason}\n")
                
                if exp.details:
                    lines.append("**Details**:")
                    for key, value in exp.details.items():
                        lines.append(f"- {key}: {value}")
                    lines.append("")
                
                if exp.config_used:
                    lines.append("**Configuration Used**:")
                    for key, value in exp.config_used.items():
                        lines.append(f"- `{key}`: {value}")
                    lines.append("")
                
                if exp.recommendation:
                    lines.append(f"💡 **Recommendation**: {exp.recommendation}\n")
                
                lines.append("")
        
        return "\n".join(lines)
    
    def print_console(self, console: Optional[Console] = None) -> None:
        """Print explanation to console using rich formatting."""
        import sys
        import platform
        
        if console is None:
            console = Console()
        
        # Detect if we can safely use emojis (check for Windows or limited encoding)
        use_emojis = True
        try:
            # Check if stdout can handle UTF-8
            encoding = getattr(sys.stdout, 'encoding', 'utf-8') or 'utf-8'
            if platform.system() == 'Windows' and encoding.lower() not in ['utf-8', 'utf8']:
                use_emojis = False
        except:
            use_emojis = False
        
        # Header
        console.print("\n")
        console.print(
            Panel.fit(
                "[bold cyan]FeatureCraft Pipeline Explanation[/bold cyan]\n"
                "Understanding what's happening under the hood",
                border_style="cyan"
            )
        )
        console.print()
        
        # Summary table
        summary_table = Table(title="Pipeline Summary", show_header=True, header_style="bold magenta")
        summary_table.add_column("Attribute", style="cyan")
        summary_table.add_column("Value", style="green")
        
        summary_table.add_row("Task Type", self.task_type or "Unknown")
        summary_table.add_row("Estimator Family", self.estimator_family or "Unknown")
        summary_table.add_row("Input Features", str(self.n_features_in))
        summary_table.add_row("Output Features", str(self.n_features_out))
        
        console.print(summary_table)
        console.print()
        
        # Group by category and display
        by_category: dict[DecisionCategory, list[TransformationExplanation]] = {}
        for exp in self.explanations:
            if exp.category not in by_category:
                by_category[exp.category] = []
            by_category[exp.category].append(exp)
        
        # Define emoji icons and ASCII fallbacks
        if use_emojis:
            category_icons = {
                DecisionCategory.COLUMN_CLASSIFICATION: "🔍",
                DecisionCategory.ENCODING: "🔢",
                DecisionCategory.IMPUTATION: "🔧",
                DecisionCategory.SCALING: "📏",
                DecisionCategory.TRANSFORMATION: "⚡",
                DecisionCategory.SELECTION: "✂️",
                DecisionCategory.DIMENSIONALITY_REDUCTION: "📉",
                DecisionCategory.TEXT_PROCESSING: "📝",
                DecisionCategory.DATETIME_PROCESSING: "📅",
                DecisionCategory.VALIDATION: "✅",
            }
            tip_icon = "💡"
        else:
            # ASCII fallbacks for Windows/limited encoding
            category_icons = {
                DecisionCategory.COLUMN_CLASSIFICATION: "[*]",
                DecisionCategory.ENCODING: "[#]",
                DecisionCategory.IMPUTATION: "[+]",
                DecisionCategory.SCALING: "[=]",
                DecisionCategory.TRANSFORMATION: "[~]",
                DecisionCategory.SELECTION: "[>]",
                DecisionCategory.DIMENSIONALITY_REDUCTION: "[<]",
                DecisionCategory.TEXT_PROCESSING: "[T]",
                DecisionCategory.DATETIME_PROCESSING: "[D]",
                DecisionCategory.VALIDATION: "[OK]",
            }
            tip_icon = "[!]"
        
        category_names = {
            DecisionCategory.COLUMN_CLASSIFICATION: "Column Classification",
            DecisionCategory.ENCODING: "Encoding Strategy",
            DecisionCategory.IMPUTATION: "Missing Value Imputation",
            DecisionCategory.SCALING: "Feature Scaling",
            DecisionCategory.TRANSFORMATION: "Feature Transformation",
            DecisionCategory.SELECTION: "Feature Selection",
            DecisionCategory.DIMENSIONALITY_REDUCTION: "Dimensionality Reduction",
            DecisionCategory.TEXT_PROCESSING: "Text Processing",
            DecisionCategory.DATETIME_PROCESSING: "DateTime Processing",
            DecisionCategory.VALIDATION: "Validation",
        }
        
        for category in DecisionCategory:
            if category not in by_category:
                continue
            
            icon = category_icons.get(category, "*")
            category_name = category_names.get(category, category.value)
            
            console.print(f"\n[bold yellow]{icon} {category_name}[/bold yellow]")
            # Use ASCII hyphen instead of box-drawing character for Windows compatibility
            console.print("-" * 80)
            
            for exp in by_category[category]:
                # Operation name
                console.print(f"\n[bold white]  > {exp.operation}[/bold white]")
                
                # Columns
                if exp.columns:
                    cols_display = ", ".join(exp.columns[:5])
                    if len(exp.columns) > 5:
                        cols_display += f" ... ({len(exp.columns)} total)"
                    console.print(f"    [dim]Columns:[/dim] {cols_display}")
                
                # Reason
                console.print(f"    [green]Reason:[/green] {exp.reason}")
                
                # Details
                if exp.details:
                    for key, value in list(exp.details.items())[:5]:  # Show first 5 details
                        console.print(f"    [dim]  - {key}:[/dim] {value}")
                
                # Recommendation
                if exp.recommendation:
                    console.print(f"    [yellow]{tip_icon} Tip:[/yellow] {exp.recommendation}")
        
        console.print("\n")


class PipelineExplainer:
    """Collects and manages pipeline explanations during construction.
    
    This class is used internally by AutoFeatureEngineer to build up
    explanations as the pipeline is constructed.
    """
    
    def __init__(self, enabled: bool = True):
        """Initialize explainer.
        
        Args:
            enabled: Whether explanations are enabled
        """
        self.enabled = enabled
        self.explanation = PipelineExplanation()
    
    def explain_column_classification(
        self,
        num_cols: list[str],
        cat_cols: list[str],
        dt_cols: list[str],
        text_cols: list[str],
        low_cat: list[str],
        mid_cat: list[str],
        high_cat: list[str],
        card: dict[str, int],
        low_threshold: int,
        mid_threshold: int,
    ) -> None:
        """Explain how columns were classified."""
        if not self.enabled:
            return
        
        # Numeric columns
        if num_cols:
            self.explanation.add_explanation(
                TransformationExplanation(
                    category=DecisionCategory.COLUMN_CLASSIFICATION,
                    operation="Numeric Column Detection",
                    columns=num_cols,
                    reason=f"Identified {len(num_cols)} columns as numeric based on dtype and value inspection",
                    details={
                        "n_columns": len(num_cols),
                        "column_names": num_cols[:10],  # Show first 10
                    },
                )
            )
        
        # Categorical columns
        if cat_cols:
            self.explanation.add_explanation(
                TransformationExplanation(
                    category=DecisionCategory.COLUMN_CLASSIFICATION,
                    operation="Categorical Column Detection",
                    columns=cat_cols,
                    reason=f"Identified {len(cat_cols)} columns as categorical (non-numeric, short strings)",
                    details={
                        "n_columns": len(cat_cols),
                        "column_names": cat_cols[:10],
                    },
                )
            )
        
        # Low cardinality
        if low_cat:
            self.explanation.add_explanation(
                TransformationExplanation(
                    category=DecisionCategory.COLUMN_CLASSIFICATION,
                    operation="Low Cardinality Classification",
                    columns=low_cat,
                    reason=f"Classified {len(low_cat)} categorical columns as low cardinality (<={low_threshold} unique values). "
                           "These will be one-hot encoded.",
                    details={
                        "n_columns": len(low_cat),
                        "cardinalities": {col: card[col] for col in low_cat[:10]},
                        "threshold": low_threshold,
                    },
                    config_used={"low_cardinality_max": low_threshold},
                )
            )
        
        # Mid cardinality
        if mid_cat:
            avg_card = sum(card[c] for c in mid_cat) / len(mid_cat)
            self.explanation.add_explanation(
                TransformationExplanation(
                    category=DecisionCategory.COLUMN_CLASSIFICATION,
                    operation="Medium Cardinality Classification",
                    columns=mid_cat,
                    reason=f"Classified {len(mid_cat)} categorical columns as medium cardinality "
                           f"({low_threshold} < unique <= {mid_threshold}). "
                           "These will use target encoding or frequency encoding.",
                    details={
                        "n_columns": len(mid_cat),
                        "avg_cardinality": round(avg_card, 1),
                        "cardinalities": {col: card[col] for col in mid_cat[:10]},
                        "threshold_range": f"{low_threshold} to {mid_threshold}",
                    },
                    config_used={
                        "low_cardinality_max": low_threshold,
                        "mid_cardinality_max": mid_threshold,
                    },
                )
            )
        
        # High cardinality
        if high_cat:
            avg_card = sum(card[c] for c in high_cat) / len(high_cat)
            self.explanation.add_explanation(
                TransformationExplanation(
                    category=DecisionCategory.COLUMN_CLASSIFICATION,
                    operation="High Cardinality Classification",
                    columns=high_cat,
                    reason=f"Classified {len(high_cat)} categorical columns as high cardinality (>{mid_threshold} unique values). "
                           "These will use hashing encoding to avoid dimension explosion.",
                    details={
                        "n_columns": len(high_cat),
                        "avg_cardinality": round(avg_card, 1),
                        "cardinalities": {col: card[col] for col in high_cat[:10]},
                        "threshold": mid_threshold,
                    },
                    config_used={"mid_cardinality_max": mid_threshold},
                    recommendation="Consider if these high-cardinality features are truly informative, "
                                 "or if they might be IDs that should be dropped.",
                )
            )
        
        # Text columns
        if text_cols:
            self.explanation.add_explanation(
                TransformationExplanation(
                    category=DecisionCategory.COLUMN_CLASSIFICATION,
                    operation="Text Column Detection",
                    columns=text_cols,
                    reason=f"Identified {len(text_cols)} columns as text (average string length >=15 characters). "
                           "These will be processed with TF-IDF or text hashing.",
                    details={
                        "n_columns": len(text_cols),
                        "column_names": text_cols,
                    },
                )
            )
        
        # DateTime columns
        if dt_cols:
            self.explanation.add_explanation(
                TransformationExplanation(
                    category=DecisionCategory.COLUMN_CLASSIFICATION,
                    operation="DateTime Column Detection",
                    columns=dt_cols,
                    reason=f"Identified {len(dt_cols)} datetime columns. These will be expanded into "
                           "cyclical features (month, day, hour, etc.) to capture temporal patterns.",
                    details={
                        "n_columns": len(dt_cols),
                        "column_names": dt_cols,
                    },
                )
            )
    
    def explain_imputation(
        self,
        strategy_name: str,
        columns: list[str],
        missing_rate: float,
        reason: str,
        config_params: dict[str, Any],
        add_indicators: bool = False,
    ) -> None:
        """Explain imputation strategy choice."""
        if not self.enabled:
            return
        
        details = {
            "missing_rate": f"{missing_rate:.2%}",
            "strategy": strategy_name,
        }
        
        recommendation = None
        if missing_rate > 0.3:
            recommendation = (
                "High missing rate detected (>30%). Consider investigating why data is missing "
                "or using domain knowledge to impute values."
            )
        
        if add_indicators:
            details["missing_indicators"] = "Added binary indicators for missingness pattern"
            reason += " Binary missing indicators will capture the missingness pattern."
        
        self.explanation.add_explanation(
            TransformationExplanation(
                category=DecisionCategory.IMPUTATION,
                operation=f"{strategy_name} Imputation",
                columns=columns,
                reason=reason,
                details=details,
                config_used=config_params,
                recommendation=recommendation,
            )
        )
    
    def explain_encoding(
        self,
        strategy_name: str,
        columns: list[str],
        reason: str,
        details: dict[str, Any],
        config_params: dict[str, Any],
        recommendation: Optional[str] = None,
    ) -> None:
        """Explain encoding strategy choice."""
        if not self.enabled:
            return
        
        self.explanation.add_explanation(
            TransformationExplanation(
                category=DecisionCategory.ENCODING,
                operation=strategy_name,
                columns=columns,
                reason=reason,
                details=details,
                config_used=config_params,
                recommendation=recommendation,
            )
        )
    
    def explain_scaling(
        self,
        scaler_name: str,
        columns: list[str],
        reason: str,
        details: dict[str, Any],
        config_params: dict[str, Any],
    ) -> None:
        """Explain scaling strategy choice."""
        if not self.enabled:
            return
        
        self.explanation.add_explanation(
            TransformationExplanation(
                category=DecisionCategory.SCALING,
                operation=scaler_name,
                columns=columns,
                reason=reason,
                details=details,
                config_used=config_params,
            )
        )
    
    def explain_transformation(
        self,
        transform_name: str,
        columns: list[str],
        reason: str,
        details: dict[str, Any],
        config_params: dict[str, Any],
        recommendation: Optional[str] = None,
    ) -> None:
        """Explain feature transformation choice."""
        if not self.enabled:
            return
        
        self.explanation.add_explanation(
            TransformationExplanation(
                category=DecisionCategory.TRANSFORMATION,
                operation=transform_name,
                columns=columns,
                reason=reason,
                details=details,
                config_used=config_params,
                recommendation=recommendation,
            )
        )
    
    def explain_text_processing(
        self,
        columns: list[str],
        method: str,
        details: dict[str, Any],
        config_params: dict[str, Any],
    ) -> None:
        """Explain text processing strategy."""
        if not self.enabled:
            return
        
        reason = f"Processing text columns using {method} to convert text into numeric features."
        
        self.explanation.add_explanation(
            TransformationExplanation(
                category=DecisionCategory.TEXT_PROCESSING,
                operation=f"Text Vectorization ({method})",
                columns=columns,
                reason=reason,
                details=details,
                config_used=config_params,
            )
        )
    
    def explain_datetime_processing(
        self,
        columns: list[str],
        features_generated: list[str],
        config_params: dict[str, Any],
    ) -> None:
        """Explain datetime feature engineering."""
        if not self.enabled:
            return
        
        reason = (
            "Extracting datetime features including cyclical encodings (sin/cos) for "
            "month, weekday, and hour to capture periodic patterns."
        )
        
        self.explanation.add_explanation(
            TransformationExplanation(
                category=DecisionCategory.DATETIME_PROCESSING,
                operation="DateTime Feature Extraction",
                columns=columns,
                reason=reason,
                details={
                    "features_per_column": features_generated,
                    "cyclical_encoding": "Using sine/cosine transformation for cyclical features",
                },
                config_used=config_params,
                recommendation="Cyclical encoding preserves the periodic nature of time features "
                             "(e.g., December and January are close).",
            )
        )
    
    def explain_dimensionality_reduction(
        self,
        method: str,
        n_components: int,
        reason: str,
        config_params: dict[str, Any],
    ) -> None:
        """Explain dimensionality reduction."""
        if not self.enabled:
            return
        
        self.explanation.add_explanation(
            TransformationExplanation(
                category=DecisionCategory.DIMENSIONALITY_REDUCTION,
                operation=f"{method.upper()} Dimensionality Reduction",
                columns=[],  # Applied to all features
                reason=reason,
                details={
                    "method": method,
                    "n_components": n_components,
                },
                config_used=config_params,
            )
        )
    
    def explain_validation(
        self,
        validation_type: str,
        reason: str,
        config_params: dict[str, Any],
    ) -> None:
        """Explain validation steps."""
        if not self.enabled:
            return
        
        self.explanation.add_explanation(
            TransformationExplanation(
                category=DecisionCategory.VALIDATION,
                operation=validation_type,
                columns=[],
                reason=reason,
                details={},
                config_used=config_params,
            )
        )
    
    def set_summary(
        self,
        estimator_family: str,
        task_type: str,
        n_features_in: int,
        n_features_out: int,
    ) -> None:
        """Set pipeline summary information."""
        self.explanation.estimator_family = estimator_family
        self.explanation.task_type = task_type
        self.explanation.n_features_in = n_features_in
        self.explanation.n_features_out = n_features_out
        self.explanation.summary = {
            "estimator_family": estimator_family,
            "task_type": task_type,
            "n_features_in": n_features_in,
            "n_features_out": n_features_out,
        }
    
    def explain(
        self,
        pipeline,
        X,
        task: str = 'classification'
    ) -> PipelineExplanation:
        """Explain a fitted pipeline.

        Args:
            pipeline: Fitted sklearn pipeline to explain
            X: Input data sample for analysis
            task: Task type ('classification' or 'regression')

        Returns:
            PipelineExplanation object with analysis results
        """
        from sklearn.utils.metaestimators import _BaseComposition
        from sklearn.pipeline import Pipeline

        # Reset explanation for new analysis
        self.explanation = PipelineExplanation()

        # Analyze pipeline steps
        if hasattr(pipeline, 'steps'):
            for step_name, step_transformer in pipeline.steps:
                self._analyze_step(step_name, step_transformer, X)

        # Set summary info
        self.set_summary(
            estimator_family=self._detect_estimator_family(pipeline),
            task_type=task,
            n_features_in=X.shape[1],
            n_features_out=X.shape[1]  # For now, assume no feature reduction
        )

        return self.get_explanation()

    def _analyze_step(self, step_name: str, transformer, X):
        """Analyze a single pipeline step."""
        # This is a simplified analysis - in a real implementation,
        # you'd want more sophisticated pipeline inspection
        try:
            # Try to get feature names if available
            if hasattr(transformer, 'get_feature_names_out'):
                try:
                    feature_names = transformer.get_feature_names_out()
                    n_features = len(feature_names)
                except:
                    n_features = X.shape[1]
            else:
                n_features = X.shape[1]

            self.explanation.add_explanation(
                TransformationExplanation(
                    category=DecisionCategory.TRANSFORMATION,
                    operation=step_name,
                    columns=[],  # Would need more analysis to determine this
                    reason=f"Pipeline step: {step_name}",
                    details={
                        "transformer_type": type(transformer).__name__,
                        "n_features": n_features,
                    },
                )
            )
        except Exception as e:
            logger.warning(f"Could not analyze step {step_name}: {e}")

    def _detect_estimator_family(self, pipeline) -> str:
        """Detect the estimator family from pipeline."""
        try:
            # Get the final estimator
            if hasattr(pipeline, 'steps') and pipeline.steps:
                final_step = pipeline.steps[-1][1]
                estimator_name = type(final_step).__name__.lower()

                if any(keyword in estimator_name for keyword in ['forest', 'tree', 'xgb', 'lgb']):
                    return 'tree'
                elif any(keyword in estimator_name for keyword in ['linear', 'logistic', 'ridge', 'lasso']):
                    return 'linear'
                else:
                    return 'other'
        except:
            pass
        return 'unknown'

    def get_explanation(self) -> PipelineExplanation:
        """Get the complete pipeline explanation."""
        return self.explanation

