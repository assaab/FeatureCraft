"""Reporter Module: Generate reports and export artifacts."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
import joblib
import pandas as pd

from ..logging import get_logger
from .config import AgentConfig
from .types import AgentResult, DatasetFingerprint

logger = get_logger(__name__)


class ArtifactStore:
    """Manages artifact storage and retrieval."""
    
    def __init__(self, root_dir: str):
        """Initialize artifact store.
        
        Args:
            root_dir: Root directory for artifacts
        """
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
    
    def save(self, filename: str, data: Any) -> None:
        """Save artifact.
        
        Args:
            filename: Filename
            data: Data to save (dict will be saved as JSON)
        """
        filepath = self.root_dir / filename
        
        if isinstance(data, dict):
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2, default=str)
        else:
            with open(filepath, "w") as f:
                f.write(str(data))
        
        logger.debug(f"Saved artifact: {filepath}")
    
    def save_pipeline(self, filename: str, pipeline: Any) -> None:
        """Save pipeline.
        
        Args:
            filename: Filename
            pipeline: Pipeline to save
        """
        filepath = self.root_dir / filename
        joblib.dump(pipeline, filepath)
        logger.debug(f"Saved pipeline: {filepath}")
    
    def load(self, filename: str) -> Any:
        """Load artifact.
        
        Args:
            filename: Filename
            
        Returns:
            Loaded data
        """
        filepath = self.root_dir / filename
        
        if filepath.suffix == ".json":
            with open(filepath, "r") as f:
                return json.load(f)
        else:
            with open(filepath, "r") as f:
                return f.read()


class Reporter:
    """Generate reports and export artifacts."""
    
    def __init__(
        self,
        config: AgentConfig,
        artifact_store: ArtifactStore,
    ):
        """Initialize reporter.
        
        Args:
            config: Agent configuration
            artifact_store: Artifact store
        """
        self.config = config
        self.artifact_store = artifact_store
    
    def export_artifacts(self, result: AgentResult) -> None:
        """Export all artifacts.
        
        Args:
            result: Agent result
        """
        logger.info("Exporting artifacts")
        
        # Save fingerprint
        if self.config.generate_json_artifacts:
            fingerprint_dict = self._fingerprint_to_dict(result.fingerprint)
            self.artifact_store.save("fingerprint.json", fingerprint_dict)
        
        # Save baselines
        if self.config.generate_json_artifacts:
            baselines_dict = {
                "raw": result.baseline_raw.to_dict(),
                "auto": result.baseline_auto.to_dict(),
            }
            self.artifact_store.save("baselines.json", baselines_dict)
        
        # Save best result
        if self.config.generate_json_artifacts:
            self.artifact_store.save("best_result.json", result.best_result.to_dict())
        
        # Save best pipeline
        self.artifact_store.save_pipeline("best_pipeline.joblib", result.best_pipeline)
        
        # Save ablation results
        if result.ablation_results and self.config.generate_json_artifacts:
            ablation_dict = {
                "operation_impacts": result.ablation_results.operation_impacts,
                "family_impacts": result.ablation_results.family_impacts,
            }
            self.artifact_store.save("ablation_results.json", ablation_dict)
        
        # Save importance scores
        if result.importance_scores and self.config.generate_json_artifacts:
            self.artifact_store.save("importance_scores.json", result.importance_scores)
        
        # Save leakage report
        if result.leakage_report and self.config.generate_json_artifacts:
            self.artifact_store.save("leakage_report.json", result.leakage_report)
        
        # Save ledger
        if result.ledger and self.config.generate_json_artifacts:
            self.artifact_store.save("ledger.json", result.ledger.to_dict())
        
        logger.info(f"[OK] Artifacts exported to: {self.artifact_store.root_dir}")
    
    def generate_report(
        self,
        result: AgentResult,
        format: str = "markdown",
    ) -> str:
        """Generate report.
        
        Args:
            result: Agent result
            format: Report format (markdown, html, json)
            
        Returns:
            Report path
        """
        logger.info(f"Generating {format} report")
        
        if format == "markdown" and self.config.generate_markdown_report:
            return self._generate_markdown_report(result)
        elif format == "html" and self.config.generate_html_report:
            return self._generate_html_report(result)
        elif format == "json" and self.config.generate_json_artifacts:
            return self._generate_json_report(result)
        else:
            logger.warning(f"Report format {format} not enabled or unsupported")
            return ""
    
    def _generate_markdown_report(self, result: AgentResult) -> str:
        """Generate Markdown report."""
        lines = [
            "# FeatureCraft Agent Report",
            "",
            f"**Run ID:** `{result.run_id}`",
            f"**Task:** {result.fingerprint.task_type}",
            f"**Dataset:** {result.fingerprint.n_rows} rows x {result.fingerprint.n_cols} columns",
            "",
            "## [RESULTS]",
            "",
            f"- **Best Score:** {result.best_score:.4f} +/- {result.best_result.cv_score_std:.4f}",
            f"- **Baseline (raw):** {result.baseline_raw.cv_score_mean:.4f}",
            f"- **Baseline (auto):** {result.baseline_auto.cv_score_mean:.4f}",
            f"- **Improvement:** {result.improvement_pct:+.1f}%",
            "",
            "## [FINGERPRINT] Dataset Fingerprint",
            "",
            f"- **Numeric columns:** {result.fingerprint.n_numeric}",
            f"- **Categorical columns:** {result.fingerprint.n_categorical}",
            f"- **Text columns:** {result.fingerprint.n_text}",
            f"- **Datetime columns:** {result.fingerprint.n_datetime}",
            f"- **Class balance:** {result.fingerprint.is_imbalanced and 'Imbalanced' or 'Balanced'}",
            "",
        ]
        
        # Ablation results
        if result.ablation_results:
            lines.extend([
                "## [ABLATION] Ablation Study",
                "",
                "Operation impacts (removing each operation):",
                "",
            ])
            
            for op, impact in result.ablation_results.operation_impacts.items():
                lines.append(f"- **{op}:** {impact:+.4f}")
            
            lines.append("")
        
        # Feature importance
        if result.importance_scores:
            lines.extend([
                "## [FEATURES] Top Feature Importances",
                "",
            ])
            
            top_features = list(result.importance_scores.items())[:10]
            for feat, imp in top_features:
                lines.append(f"- `{feat}`: {imp:.4f}")
            
            lines.append("")
        
        # Leakage check
        if result.leakage_report:
            lines.extend([
                "## [LEAKAGE] Leakage Check",
                "",
                f"- **Has Leakage:** {result.leakage_report.get('has_leakage', False)}",
                f"- **High PSI Features:** {len(result.leakage_report.get('high_psi_features', []))}",
                "",
            ])
        
        # Strategy details
        lines.extend([
            "## [STRATEGY] Strategy",
            "",
            f"```",
            f"{result.best_strategy.reasoning}",
            f"```",
            "",
            "---",
            "",
            f"*Generated by FeatureCraft Agent v1.0.0*",
        ])
        
        report_text = "\n".join(lines)
        
        # Save report
        self.artifact_store.save("report.md", report_text)
        
        report_path = str(self.artifact_store.root_dir / "report.md")
        logger.info(f"[OK] Markdown report saved: {report_path}")
        return report_path
    
    def _generate_html_report(self, result: AgentResult) -> str:
        """Generate HTML report."""
        # Simple HTML wrapper around markdown
        md_report = self._generate_markdown_report(result)
        
        try:
            import markdown
            
            with open(self.artifact_store.root_dir / "report.md", "r") as f:
                md_text = f.read()
            
            html_body = markdown.markdown(md_text, extensions=['tables', 'fenced_code'])
            
            html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>FeatureCraft Agent Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 1200px; margin: 40px auto; padding: 20px; }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        code {{ background: #f4f4f4; padding: 2px 6px; border-radius: 3px; }}
        pre {{ background: #f4f4f4; padding: 15px; border-radius: 5px; overflow-x: auto; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #3498db; color: white; }}
    </style>
</head>
<body>
{html_body}
</body>
</html>
"""
            
            self.artifact_store.save("report.html", html)
            
            report_path = str(self.artifact_store.root_dir / "report.html")
            logger.info(f"[OK] HTML report saved: {report_path}")
            return report_path
            
        except ImportError:
            logger.warning("markdown package not installed, skipping HTML report")
            return ""
    
    def _generate_json_report(self, result: AgentResult) -> str:
        """Generate JSON report."""
        report_dict = {
            "run_id": result.run_id,
            "fingerprint": self._fingerprint_to_dict(result.fingerprint),
            "best_result": result.best_result.to_dict(),
            "baseline_raw": result.baseline_raw.to_dict(),
            "baseline_auto": result.baseline_auto.to_dict(),
            "improvement_pct": result.improvement_pct,
            "strategy": {
                "reasoning": result.best_strategy.reasoning,
                "estimated_feature_count": result.best_strategy.estimated_feature_count,
                "risk_level": result.best_strategy.risk_level,
            },
        }
        
        if result.ablation_results:
            report_dict["ablation"] = {
                "operation_impacts": result.ablation_results.operation_impacts,
            }
        
        if result.importance_scores:
            report_dict["top_importances"] = dict(
                list(result.importance_scores.items())[:20]
            )
        
        if result.leakage_report:
            report_dict["leakage"] = result.leakage_report
        
        self.artifact_store.save("report.json", report_dict)
        
        report_path = str(self.artifact_store.root_dir / "report.json")
        logger.info(f"[OK] JSON report saved: {report_path}")
        return report_path
    
    def _fingerprint_to_dict(self, fp: DatasetFingerprint) -> Dict[str, Any]:
        """Convert fingerprint to dictionary."""
        return {
            "n_rows": fp.n_rows,
            "n_cols": fp.n_cols,
            "task_type": str(fp.task_type),
            "n_numeric": fp.n_numeric,
            "n_categorical": fp.n_categorical,
            "n_text": fp.n_text,
            "n_datetime": fp.n_datetime,
            "is_imbalanced": fp.is_imbalanced,
            "is_time_series": fp.is_time_series,
            "estimated_feature_count": fp.estimated_feature_count_baseline,
            "estimated_memory_gb": fp.estimated_memory_gb,
        }

