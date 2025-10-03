"""LLM-powered feature engineering planner."""

from __future__ import annotations

import json
import time
from datetime import datetime
from typing import Any, Literal

import pandas as pd

from .providers import BaseLLMProvider, get_provider
from .rag.retriever import RAGRetriever
from .schemas import (
    AIBudget,
    AICallMetadata,
    DatasetContext,
    FeaturePlan,
    FeatureSpec,
)
from .validator import PolicyValidator, ValidationResult
from ..logging import get_logger

logger = get_logger(__name__)


# ============================================================================
# Prompts & Tools
# ============================================================================

SYSTEM_PROMPT = """You are an expert feature engineering advisor for tabular machine learning.

Your task is to generate feature engineering plans as valid JSON conforming to the FeatureSpec schema.

CRITICAL RULES:
1. NEVER reference the target column in any feature computation
2. For time-series data, ONLY use past data (lags, rolling windows with closed='left')
3. Use out-of-fold (OOF) target encoding to prevent leakage
4. Prefer interpretable features over complex transformations
5. Consider the estimator family (tree vs linear) when choosing encoding strategies
6. Respect budget constraints (max features, max time)

FEATURE TYPES AVAILABLE:
- Aggregations: rolling_mean, rolling_sum, rolling_std, rolling_min, rolling_max
- Temporal: lag, diff, pct_change, ewm, expanding_mean
- Cardinality: nunique, count
- Encodings: target_encode, frequency_encode, count_encode, ohe, label_encode, hash_encode
- Binning: quantile_bin, custom_bin, kmeans_bin
- Interactions: multiply, divide, add, subtract, ratio
- Domain: recency, frequency, monetary, rfm_score

RESPONSE FORMAT:
Return ONLY valid JSON with this structure:
{
  "version": "1.0",
  "dataset_id": "<dataset_name>",
  "task": "classification" or "regression",
  "estimator_family": "tree",
  "constraints": {...},
  "budget": {...},
  "candidates": [
    {
      "name": "feature_name",
      "type": "rolling_mean",
      "source_col": "column_name",
      "window": "30d",
      "key_col": "customer_id",
      "time_col": "date",
      "params": {},
      "rationale": "Why this feature is useful",
      "safety_tags": ["no_target_ref", "time_safe"],
      "priority": 1
    }
  ],
  "rationale": "Overall plan explanation"
}

NO markdown, NO code blocks, ONLY JSON.
"""

FUNCTION_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_schema",
            "description": "Get DataFrame schema with column names and dtypes",
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset_id": {"type": "string", "description": "Dataset identifier"}
                },
                "required": ["dataset_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_stats",
            "description": "Get summary statistics for specified columns",
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset_id": {"type": "string"},
                    "columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Column names to get stats for",
                    },
                },
                "required": ["dataset_id"],
            },
        },
    },
]


# ============================================================================
# LLM Planner
# ============================================================================

class LLMPlanner:
    """LLM-powered feature engineering planner.
    
    This class uses an LLM to generate feature engineering plans based on
    dataset context and user intent.
    
    Example:
        >>> planner = LLMPlanner(provider="openai", model="gpt-4o")
        >>> plan = planner.plan(
        ...     df=train_df,
        ...     target="churn",
        ...     nl_intent="Create customer behavior features"
        ... )
    """
    
    def __init__(
        self,
        provider: BaseLLMProvider | str = "openai",
        model: str | None = None,
        validator: PolicyValidator | None = None,
        budget: AIBudget | None = None,
        rag_retriever: RAGRetriever | None = None,
        enable_rag: bool = False,
    ):
        """Initialize planner.
        
        Args:
            provider: LLM provider (instance or name like "openai", "anthropic")
            model: Model name (optional, uses provider default)
            validator: Policy validator (optional, creates default if None)
            budget: AI budget constraints (optional)
            rag_retriever: RAG retriever for domain knowledge (optional)
            enable_rag: Enable RAG-augmented planning
        """
        if isinstance(provider, str):
            self.provider = get_provider(provider, model=model)
        else:
            self.provider = provider
        
        self.validator = validator or PolicyValidator()
        self.budget = budget or AIBudget()
        self.rag_retriever = rag_retriever
        self.enable_rag = enable_rag and rag_retriever is not None
        self.call_history: list[AICallMetadata] = []
    
    def plan(
        self,
        df: pd.DataFrame,
        target: str,
        task: Literal["classification", "regression"] | None = None,
        nl_intent: str | None = None,
        estimator_family: str = "tree",
        time_col: str | None = None,
        key_col: str | None = None,
        constraints: dict[str, Any] | None = None,
        max_features: int | None = None,
        validate: bool = True,
    ) -> FeaturePlan:
        """Generate feature engineering plan from dataset and intent.
        
        Args:
            df: Input DataFrame
            target: Target column name
            task: Task type (auto-detected if None)
            nl_intent: Natural language intent/description
            estimator_family: Target estimator family (tree, linear, etc.)
            time_col: Time column for time-series
            key_col: Entity key column (customer_id, etc.)
            constraints: Additional constraints (leakage_blocklist, etc.)
            max_features: Maximum number of features to generate
            validate: Run validation after generation
            
        Returns:
            FeaturePlan (validated if validate=True)
            
        Raises:
            ValueError: If plan generation or validation fails
        """
        logger.info(f"🤖 Generating feature plan for target '{target}'")
        
        # Build dataset context
        context = DatasetContext(
            df=df,
            target=target,
            task=task or self._detect_task(df[target]),
            time_col=time_col,
            key_col=key_col,
        )
        
        # Build constraints
        full_constraints = constraints or {}
        full_constraints.update({
            "time_aware": time_col is not None,
            "time_col": time_col,
            "key_col": key_col,
        })
        
        # Build budget
        budget_dict = {
            "max_features": max_features or self.budget.max_features,
            "max_time_seconds": self.budget.max_time_seconds,
        }
        
        # Build user prompt
        user_prompt = self._build_prompt(
            context=context,
            nl_intent=nl_intent,
            estimator_family=estimator_family,
            constraints=full_constraints,
            budget=budget_dict,
        )
        
        # Call LLM
        start_time = time.time()
        
        try:
            response = self.provider.call(
                prompt=user_prompt,
                system_prompt=SYSTEM_PROMPT,
                response_format={"type": "json_object"},
                temperature=self.budget.deterministic_seed is not None and 0.0 or 0.3,
            )
            
            latency_ms = int((time.time() - start_time) * 1000)
            
            # Parse response
            plan_dict = json.loads(response["content"])
            plan = FeaturePlan.from_dict(plan_dict)
            
            # Log telemetry
            metadata = AICallMetadata(
                timestamp=datetime.utcnow().isoformat(),
                prompt_hash=self.provider.hash_string(user_prompt),
                response_hash=self.provider.hash_string(response["content"]),
                tokens_used=response.get("tokens_used", 0),
                latency_ms=latency_ms,
                provider=type(self.provider).__name__,
                model=response.get("model", "unknown"),
                cost_usd=self.provider.estimate_cost(response.get("tokens_used", 0)),
            )
            self.call_history.append(metadata)
            
            logger.info(
                f"✓ Generated plan with {len(plan.candidates)} features "
                f"({metadata.tokens_used} tokens, {latency_ms}ms, ${metadata.cost_usd:.4f})"
            )
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            raise ValueError(f"LLM returned invalid JSON: {e}")
        except Exception as e:
            logger.error(f"Failed to generate plan: {e}")
            raise ValueError(f"Plan generation failed: {e}")
        
        # Validate plan
        if validate:
            validation_result = self.validator.validate(plan, context)
            
            # Update metadata
            metadata.validator_status = "pass" if validation_result.is_valid else "fail"
            metadata.validator_errors = validation_result.errors
            
            plan.safety_summary = {
                "is_valid": validation_result.is_valid,
                "errors": validation_result.errors,
                "warnings": validation_result.warnings,
                "checks_passed": validation_result.checks_passed,
            }
            
            if not validation_result.is_valid:
                error_msg = f"Plan validation failed: {validation_result.errors}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            logger.info(f"✓ Plan validated successfully")
        
        return plan
    
    def _build_prompt(
        self,
        context: DatasetContext,
        nl_intent: str | None,
        estimator_family: str,
        constraints: dict[str, Any],
        budget: dict[str, Any],
    ) -> str:
        """Build user prompt for LLM."""
        # Build schema section
        schema_str = "\n".join(
            [f"  - {col}: {dtype}" for col, dtype in list(context.schema.items())[:50]]
        )
        
        # Build stats section (top 10 columns)
        stats_items = []
        for col, col_stats in list(context.stats.items())[:10]:
            stats_str = f"  - {col}: "
            if col_stats.get("mean") is not None:
                stats_str += f"mean={col_stats['mean']:.2f}, "
            stats_str += f"unique={col_stats['unique_count']}, missing={col_stats['missing_rate']:.1%}"
            stats_items.append(stats_str)
        
        stats_str = "\n".join(stats_items)
        
        # Build sample rows
        sample_str = json.dumps(context.sample_rows[:3], indent=2, default=str)
        
        # Retrieve relevant knowledge with RAG
        rag_context = ""
        if self.enable_rag and self.rag_retriever:
            query = f"Feature engineering for {context.task} task with {estimator_family} model"
            if nl_intent:
                query += f". User intent: {nl_intent}"
            
            try:
                rag_context = self.rag_retriever.retrieve(
                    query=query,
                    k=5,
                    mode="hybrid",
                )
                if rag_context:
                    rag_context = f"\n\nDOMAIN KNOWLEDGE (from knowledge base):\n{rag_context}\n"
            except Exception as e:
                logger.warning(f"RAG retrieval failed: {e}")
        
        # Build prompt
        prompt = f"""Generate a feature engineering plan for this dataset.

DATASET INFO:
- Target: {context.target}
- Task: {context.task}
- Rows: {len(context.df) if context.df is not None else 'unknown'}
- Columns: {len(context.schema)}
- Estimator family: {estimator_family}

SCHEMA (top 50):
{schema_str}

STATISTICS (top 10):
{stats_str}

SAMPLE ROWS (first 3):
{sample_str}

CONSTRAINTS:
{json.dumps(constraints, indent=2)}

BUDGET:
{json.dumps(budget, indent=2)}
{rag_context}
"""
        
        if nl_intent:
            prompt += f"\nUSER INTENT:\n{nl_intent}\n"
        
        prompt += """
TASK:
Generate a feature engineering plan with high-quality features that:
1. Are relevant for the task and estimator family
2. Do NOT leak target information
3. Use proper time-ordering if time-series
4. Respect the budget constraints
5. Are interpretable and well-documented
6. Consider domain knowledge from the knowledge base (if provided)

Return ONLY valid JSON (no markdown, no code blocks).
"""
        
        return prompt
    
    @staticmethod
    def _detect_task(y: pd.Series) -> Literal["classification", "regression"]:
        """Auto-detect task type from target."""
        if pd.api.types.is_numeric_dtype(y) and y.nunique() > 15:
            return "regression"
        return "classification"


# ============================================================================
# Convenience Functions
# ============================================================================

def plan_features(
    df: pd.DataFrame,
    target: str,
    task: Literal["classification", "regression"] | None = None,
    nl_intent: str | None = None,
    estimator_family: str = "tree",
    time_col: str | None = None,
    key_col: str | None = None,
    constraints: dict[str, Any] | None = None,
    max_features: int | None = None,
    provider: str = "openai",
    model: str | None = None,
    validate: bool = True,
) -> FeaturePlan:
    """Generate feature engineering plan (convenience function).
    
    Args:
        df: Input DataFrame
        target: Target column name
        task: Task type (auto-detected if None)
        nl_intent: Natural language description of desired features
        estimator_family: Target estimator (tree, linear, svm, knn, nn)
        time_col: Time column for time-series features
        key_col: Entity key (customer_id, user_id, etc.)
        constraints: Additional constraints
        max_features: Max features to generate
        provider: LLM provider (openai, anthropic, mock)
        model: Model name (optional)
        validate: Validate plan after generation
        
    Returns:
        FeaturePlan ready for execution
        
    Example:
        >>> plan = plan_features(
        ...     df=train_df,
        ...     target="churn",
        ...     nl_intent="Create RFM features for customer churn prediction",
        ...     time_col="transaction_date",
        ...     key_col="customer_id"
        ... )
        >>> print(f"Generated {len(plan.candidates)} features")
    """
    planner = LLMPlanner(provider=provider, model=model)
    
    return planner.plan(
        df=df,
        target=target,
        task=task,
        nl_intent=nl_intent,
        estimator_family=estimator_family,
        time_col=time_col,
        key_col=key_col,
        constraints=constraints,
        max_features=max_features,
        validate=validate,
    )

