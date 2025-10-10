"""Feature Engineering Planner - Orchestrates AI-driven feature engineering.

This module provides a high-level planner that coordinates between
data analysis, AI recommendations, and pipeline execution.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Tuple

import pandas as pd
from rich.console import Console

from ..config import FeatureCraftConfig
from ..logging import get_logger
from ..types import DatasetInsights
from .advisor import AIFeatureAdvisor, FeatureStrategy

logger = get_logger(__name__)
console = Console()


@dataclass
class FeaturePlan:
    """Complete feature engineering plan."""
    
    strategy: FeatureStrategy
    config: FeatureCraftConfig
    insights: DatasetInsights
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "strategy": asdict(self.strategy),
            "config": self.config.model_dump(),
            "metadata": self.metadata,
        }


class FeatureEngineeringPlanner:
    """High-level planner for AI-driven feature engineering.
    
    This class orchestrates the entire feature engineering process:
    1. Analyze dataset characteristics
    2. Get AI recommendations
    3. Apply smart optimizations
    4. Configure pipeline optimally
    
    Usage:
        planner = FeatureEngineeringPlanner(
            use_ai=True,
            api_key="your-key",
            time_budget="balanced"
        )
        
        plan = planner.create_plan(X, y, insights)
        optimized_config = plan.config
        
        # Use with AutoFeatureEngineer
        afe = AutoFeatureEngineer(config=optimized_config)
        afe.fit(X, y)
    """
    
    def __init__(
        self,
        use_ai: bool = True,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        provider: str = "openai",
        time_budget: str = "balanced",  # 'fast', 'balanced', 'thorough'
        base_config: Optional[FeatureCraftConfig] = None,
        verbose: bool = True,
    ):
        """Initialize Feature Engineering Planner.
        
        Args:
            use_ai: Enable AI recommendations (requires API key)
            api_key: API key for LLM provider
            model: Model name (e.g., 'gpt-4o-mini', 'claude-3-sonnet')
            provider: LLM provider ('openai', 'anthropic')
            time_budget: Time budget ('fast', 'balanced', 'thorough')
            base_config: Base configuration (uses defaults if None)
            verbose: Print recommendations and explanations
        """
        self.use_ai = use_ai
        self.time_budget = time_budget
        self.base_config = base_config or FeatureCraftConfig()
        self.verbose = verbose
        
        # Initialize AI advisor
        self.advisor = AIFeatureAdvisor(
            api_key=api_key,
            model=model,
            provider=provider,
            enable_ai=use_ai,
            require_client=False,  # Allow graceful degradation if client init fails
        )
        
        if verbose:
            mode = "AI-Powered" if use_ai and self.advisor.client else "Heuristic-Based"
            console.print(f"[cyan]✓ Feature Engineering Planner initialized ({mode} mode)[/cyan]")
    
    def create_plan(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        insights: DatasetInsights,
        estimator_family: str = "tree",
        current_config: Optional[FeatureCraftConfig] = None,
    ) -> FeaturePlan:
        """Create comprehensive feature engineering plan.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            insights: Dataset insights
            estimator_family: Estimator family ('tree', 'linear', etc.)
            current_config: Current config to use (defaults to base_config if not provided)
        
        Returns:
            FeaturePlan with optimized strategy and configuration
        """
        if self.verbose:
            console.print("\n[bold yellow]🧠 Creating AI-Powered Feature Engineering Plan...[/bold yellow]")
        
        # Get AI recommendations (will throw error if AI fails)
        try:
            strategy = self.advisor.recommend_strategy(
                X=X,
                y=y,
                insights=insights,
                estimator_family=estimator_family,
                time_budget=self.time_budget,
            )
        except RuntimeError as e:
            if self.verbose:
                console.print(f"\n[red]❌ AI feature engineering failed: {e}[/red]")
            raise RuntimeError(f"AI-powered feature engineering failed: {e}") from e
        
        # Apply strategy to config (use current_config if provided, otherwise base_config)
        config_to_use = current_config if current_config is not None else self.base_config
        optimized_config = self.advisor.apply_strategy(config_to_use, strategy)
        
        # Print strategy if verbose
        if self.verbose:
            self.advisor.print_strategy(strategy)
        
        # Create plan
        plan = FeaturePlan(
            strategy=strategy,
            config=optimized_config,
            insights=insights,
            metadata={
                "estimator_family": estimator_family,
                "time_budget": self.time_budget,
                "ai_enabled": self.use_ai and self.advisor.client is not None,
                "planner_version": "1.0.0",
            }
        )
        
        return plan
    
    def quick_plan(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        estimator_family: str = "tree",
        skip_insights: bool = False,
    ) -> FeaturePlan:
        """Create quick plan without full dataset analysis.
        
        Useful when you want fast recommendations without running full EDA.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            estimator_family: Estimator family
            skip_insights: Skip detailed profiling (faster but less accurate)
        
        Returns:
            FeaturePlan with recommendations
        """
        from ..insights import analyze_dataset, detect_task
        
        # Quick or full analysis
        if skip_insights:
            # Minimal insights
            from ..types import DatasetInsights, TaskType
            task = detect_task(y)
            insights = DatasetInsights(
                n_rows=len(X),
                n_cols=len(X.columns),
                task=task,
                target_name=y.name or "target",
                profiles=[],
                summary={
                    "n_numeric": len(X.select_dtypes(include=['number']).columns),
                    "n_categorical": len(X.select_dtypes(include=['object', 'category']).columns),
                }
            )
        else:
            # Full analysis
            insights = analyze_dataset(X, y, y.name or "target", self.base_config)
        
        return self.create_plan(X, y, insights, estimator_family)
    
    def compare_strategies(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        insights: DatasetInsights,
        estimator_families: list[str] = None,
    ) -> Dict[str, FeaturePlan]:
        """Compare strategies for different estimator families.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            insights: Dataset insights
            estimator_families: List of estimator families to compare
        
        Returns:
            Dict mapping estimator family to FeaturePlan
        """
        if estimator_families is None:
            estimator_families = ["tree", "linear"]
        
        plans = {}
        for family in estimator_families:
            if self.verbose:
                console.print(f"\n[cyan]→ Planning for {family} estimator...[/cyan]")
            plan = self.create_plan(X, y, insights, family)
            plans[family] = plan
        
        return plans
    
    def export_plan(self, plan: FeaturePlan, filepath: str):
        """Export plan to JSON file.
        
        Args:
            plan: FeaturePlan to export
            filepath: Path to save JSON file
        """
        import json
        from pathlib import Path
        
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(plan.to_dict(), f, indent=2, default=str)
        
        if self.verbose:
            console.print(f"[green]✓ Plan exported to {filepath}[/green]")
    
    @staticmethod
    def load_plan(filepath: str) -> Dict[str, Any]:
        """Load plan from JSON file.
        
        Args:
            filepath: Path to JSON file
        
        Returns:
            Plan dictionary
        """
        import json
        with open(filepath, 'r') as f:
            return json.load(f)

