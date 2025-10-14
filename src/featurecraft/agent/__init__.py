"""FeatureCraft Agent: Intelligent Autonomous Feature Engineering System.

This module implements an AI-driven agent that orchestrates end-to-end automated
feature engineering for tabular ML tasks. The agent analyzes datasets, selects
optimal strategies, builds pipelines, evaluates performance, and exports
production-ready artifacts.

Key Components:
- Inspector: Dataset fingerprinting and quality checks
- Strategist: Heuristic-based strategy selection
- Composer: Converts strategies to sklearn pipelines
- Evaluator: CV scoring, baselines, ablation studies
- Optimizer: Multi-stage search and refinement
- Reporter: Generates reports and artifacts

Usage:
    from featurecraft.agent import FeatureCraftAgent, AgentConfig
    
    agent = FeatureCraftAgent(config=AgentConfig(
        estimator_family="tree",
        primary_metric="roc_auc",
        time_budget_minutes=45
    ))
    
    result = agent.run(X=X, y=y, target_name="target")
    print(f"Best score: {result.best_score:.4f}")
    pipeline = result.load_pipeline()
"""

from .agent import FeatureCraftAgent
from .config import AgentConfig, ComputeBudget
from .types import (
    DatasetFingerprint,
    AgentResult,
    EvaluationResult,
    Candidate,
    AblationResults,
)

__all__ = [
    "FeatureCraftAgent",
    "AgentConfig",
    "ComputeBudget",
    "DatasetFingerprint",
    "AgentResult",
    "EvaluationResult",
    "Candidate",
    "AblationResults",
]

