"""Configuration for FeatureCraft Agent."""

from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field


class ComputeBudget(BaseModel):
    """Resource limits for agent execution."""
    
    max_wall_time_minutes: int = Field(
        default=60,
        ge=1,
        le=1440,
        description="Total time budget in minutes"
    )
    
    max_pipelines: int = Field(
        default=50,
        ge=1,
        le=500,
        description="Max number of pipelines to evaluate"
    )
    
    max_fit_time_seconds: int = Field(
        default=300,
        ge=10,
        le=3600,
        description="Per-pipeline fit time limit"
    )
    
    max_memory_gb: float = Field(
        default=16.0,
        ge=0.5,
        le=256.0,
        description="Memory limit in GB"
    )
    
    early_stop_patience: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Stop if no improvement in N pipelines"
    )
    
    n_bayesian_trials: int = Field(
        default=30,
        ge=5,
        le=500,
        description="Bayesian optimization trials"
    )
    
    enable_bayesian: bool = Field(
        default=True,
        description="Enable Bayesian optimization stage"
    )
    
    enable_shap: bool = Field(
        default=False,
        description="Enable SHAP analysis (expensive)"
    )
    
    shap_sample_size: int = Field(
        default=1000,
        ge=100,
        le=10000,
        description="Sample size for SHAP computation"
    )
    
    @classmethod
    def from_preset(cls, preset: Literal["fast", "balanced", "thorough"]) -> ComputeBudget:
        """Create budget from preset."""
        presets = {
            "fast": cls(
                max_wall_time_minutes=15,
                max_pipelines=20,
                max_fit_time_seconds=120,
                n_bayesian_trials=10,
                enable_bayesian=False,
                enable_shap=False,
            ),
            "balanced": cls(
                max_wall_time_minutes=60,
                max_pipelines=50,
                max_fit_time_seconds=300,
                n_bayesian_trials=30,
                enable_bayesian=True,
                enable_shap=False,
            ),
            "thorough": cls(
                max_wall_time_minutes=180,
                max_pipelines=100,
                max_fit_time_seconds=600,
                n_bayesian_trials=50,
                enable_bayesian=True,
                enable_shap=True,
            ),
        }
        return presets[preset]
    
    def has_budget_for_bayesian(self) -> bool:
        """Check if budget allows Bayesian optimization."""
        return self.enable_bayesian
    
    def has_budget_for_shap(self) -> bool:
        """Check if budget allows SHAP analysis."""
        return self.enable_shap
    
    def stage_budget(self, stage: int) -> ComputeBudget:
        """Get budget for a specific stage."""
        # Allocate budget proportionally
        stage_fractions = {
            1: 0.05,  # Inspect
            2: 0.05,  # Strategize
            3: 0.10,  # Baselines
            4: 0.25,  # Candidates
            5: 0.40,  # Optimize
            6: 0.15,  # Report
        }
        
        fraction = stage_fractions.get(stage, 0.10)
        
        return ComputeBudget(
            max_wall_time_minutes=int(self.max_wall_time_minutes * fraction),
            max_pipelines=int(self.max_pipelines * fraction),
            max_fit_time_seconds=self.max_fit_time_seconds,
            max_memory_gb=self.max_memory_gb,
            early_stop_patience=self.early_stop_patience,
            n_bayesian_trials=self.n_bayesian_trials,
            enable_bayesian=self.enable_bayesian,
            enable_shap=self.enable_shap,
            shap_sample_size=self.shap_sample_size,
        )


class AgentConfig(BaseModel):
    """Configuration for FeatureCraft Agent."""
    
    model_config = {"extra": "forbid", "validate_assignment": True}
    
    # === Task Configuration ===
    estimator_family: Literal["tree", "linear", "svm", "catboost", "knn", "nn"] = Field(
        default="tree",
        description="Estimator family for feature engineering"
    )
    
    primary_metric: str = Field(
        default="auto",
        description="Primary metric (auto, logloss, roc_auc, rmse, mae, r2)"
    )
    
    # === Budget ===
    time_budget: Literal["fast", "balanced", "thorough"] = Field(
        default="balanced",
        description="Time budget preset"
    )
    
    # === CV Strategy ===
    n_cv_folds: int = Field(
        default=5,
        ge=2,
        le=20,
        description="Number of CV folds"
    )
    
    cv_shuffle: bool = Field(
        default=True,
        description="Shuffle data in CV splits"
    )
    
    # === Optimization ===
    baseline_improvement_threshold: float = Field(
        default=1.01,
        ge=1.0,
        le=2.0,
        description="Min improvement over baseline to keep candidate (1.01 = 1%)"
    )
    
    greedy_improvement_threshold: float = Field(
        default=1.005,
        ge=1.0,
        le=2.0,
        description="Min improvement in greedy search to keep operation (1.005 = 0.5%)"
    )
    
    # === Feature Engineering Strategy ===
    strategy_variants: list[str] = Field(
        default_factory=lambda: ["conservative", "balanced", "aggressive"],
        description="Strategy variants to try"
    )
    
    # === Outputs ===
    output_dir: str = Field(
        default="artifacts",
        description="Directory for artifacts and outputs"
    )
    
    generate_html_report: bool = Field(
        default=True,
        description="Generate HTML report"
    )
    
    generate_markdown_report: bool = Field(
        default=True,
        description="Generate Markdown report"
    )
    
    generate_json_artifacts: bool = Field(
        default=True,
        description="Generate JSON artifacts"
    )
    
    # === Random Seed ===
    random_seed: int = Field(
        default=42,
        ge=0,
        description="Random seed for reproducibility"
    )
    
    # === Optional Overrides ===
    entity_column: Optional[str] = Field(
        default=None,
        description="Entity/group column for GroupKFold"
    )
    
    time_column: Optional[str] = Field(
        default=None,
        description="Time column for time series"
    )
    
    # === Advanced ===
    use_cache: bool = Field(
        default=True,
        description="Cache intermediate artifacts"
    )
    
    verbose: bool = Field(
        default=True,
        description="Verbose output"
    )
    
    def get_budget(self) -> ComputeBudget:
        """Get compute budget from time_budget preset."""
        return ComputeBudget.from_preset(self.time_budget)

