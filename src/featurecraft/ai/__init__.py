"""AI-powered feature engineering module for FeatureCraft.

This module provides LLM-powered feature planning, validation, and execution
capabilities to automate and enhance feature engineering workflows.

Main components:
- Planner: Generate feature engineering plans from natural language
- Validator: Validate plans for safety (leakage, schema, time-ordering)
- Executor: Execute plans to generate features
- Providers: LLM provider abstractions (OpenAI, Anthropic, local)
- RAG: Retrieval-Augmented Generation for domain knowledge
- Pruning: LLM-guided feature pruning with statistical gates
- Ablation: Automated ablation studies
- Executors: Pandas, Spark, Ray execution engines

Example:
    >>> from featurecraft.ai import plan_features, execute_plan
    >>> plan = plan_features(df, target="churn", nl_intent="Create RFM features")
    >>> df_transformed = execute_plan(plan, df)
"""

from .schemas import FeatureSpec, FeaturePlan, AIBudget, AICallMetadata
from .planner import plan_features, LLMPlanner
from .executor import execute_plan, PlanExecutor
from .validator import validate_plan, PolicyValidator
from .providers import LLMProvider, get_provider
from .pruning import FeaturePruner, FeatureRanking, PruningResult
from .ablation import AblationRunner, AblationStudy, AblationResult
from .rag import RAGRetriever, RAGIndex, build_index, load_documents
from .executors import PandasExecutor, SparkExecutor, RayExecutor

__all__ = [
    # Schemas
    "FeatureSpec",
    "FeaturePlan",
    "AIBudget",
    "AICallMetadata",
    # Core functions
    "plan_features",
    "execute_plan",
    "validate_plan",
    # Classes - Phase 1
    "LLMPlanner",
    "PlanExecutor",
    "PolicyValidator",
    "LLMProvider",
    "get_provider",
    # Classes - Phase 2
    "FeaturePruner",
    "FeatureRanking",
    "PruningResult",
    "AblationRunner",
    "AblationStudy",
    "AblationResult",
    "RAGRetriever",
    "RAGIndex",
    "build_index",
    "load_documents",
    "PandasExecutor",
    "SparkExecutor",
    "RayExecutor",
]

