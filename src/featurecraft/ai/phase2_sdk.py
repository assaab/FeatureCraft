"""Phase 2 SDK for advanced AI-powered feature engineering.

This module provides high-level functions for RAG-augmented planning,
feature pruning, ablation studies, and distributed execution.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

from .ablation import AblationRunner, AblationStudy
from .executors import PandasExecutor, RayExecutor, SparkExecutor
from .planner import LLMPlanner
from .pruning import FeaturePruner, PruningResult
from .rag.retriever import RAGRetriever
from .schemas import FeaturePlan
from ..logging import get_logger

logger = get_logger(__name__)


def plan_with_rag(
    df: pd.DataFrame,
    target: str,
    nl_intent: str | None = None,
    knowledge_dirs: list[str] | None = None,
    provider: str = "openai",
    embedder: str = "sentence_transformers",
    **kwargs,
) -> FeaturePlan:
    """Plan features with RAG-augmented context.
    
    Args:
        df: Input DataFrame
        target: Target column
        nl_intent: Natural language intent
        knowledge_dirs: Directories containing domain knowledge
        provider: LLM provider
        embedder: RAG embedder provider
        **kwargs: Additional arguments for plan_features
        
    Returns:
        FeaturePlan with RAG-augmented suggestions
        
    Example:
        >>> plan = plan_with_rag(
        ...     df=train_df,
        ...     target="churn",
        ...     nl_intent="Customer retention features",
        ...     knowledge_dirs=["knowledge_base/", "artifacts/"]
        ... )
    """
    logger.info("Planning features with RAG augmentation...")
    
    # Initialize RAG retriever
    rag_retriever = RAGRetriever(
        embedder=embedder,
        knowledge_dirs=knowledge_dirs or [],
    )
    
    # Initialize planner with RAG
    planner = LLMPlanner(
        provider=provider,
        rag_retriever=rag_retriever,
        enable_rag=True,
    )
    
    # Plan features
    plan = planner.plan(
        df=df,
        target=target,
        nl_intent=nl_intent,
        **kwargs,
    )
    
    return plan


def prune_features(
    plan: FeaturePlan,
    X_train: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    target_n_features: int | None = None,
    X_val: pd.DataFrame | None = None,
    y_val: pd.Series | np.ndarray | None = None,
    provider: str = "openai",
    **kwargs,
) -> PruningResult:
    """Prune features using LLM guidance and statistical gates.
    
    Args:
        plan: Feature plan to prune
        X_train: Training features
        y_train: Training target
        target_n_features: Target number of features
        X_val: Validation features
        y_val: Validation target
        provider: LLM provider
        **kwargs: Additional pruning parameters
        
    Returns:
        PruningResult with selected features
        
    Example:
        >>> result = prune_features(
        ...     plan=feature_plan,
        ...     X_train=X_train,
        ...     y_train=y_train,
        ...     target_n_features=50
        ... )
        >>> selected_features = result.selected_features
    """
    logger.info(f"Pruning features from plan with {len(plan.candidates)} candidates...")
    
    pruner = FeaturePruner(provider=provider, **kwargs)
    
    result = pruner.prune(
        plan=plan,
        X_train=X_train,
        y_train=y_train,
        target_n_features=target_n_features,
        X_val=X_val,
        y_val=y_val,
    )
    
    return result


def run_ablation_study(
    plan: FeaturePlan,
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    estimator: Any,
    strategies: list[str] = None,
    scoring: str = "roc_auc",
    cv: int = 5,
    **kwargs,
) -> AblationStudy:
    """Run automated ablation study on feature plan.
    
    Args:
        plan: Feature plan to ablate
        X: Input features
        y: Target variable
        estimator: Sklearn-compatible estimator
        strategies: Ablation strategies (on_off, window, encoding, interaction)
        scoring: Scoring metric
        cv: Cross-validation folds
        **kwargs: Additional ablation parameters
        
    Returns:
        AblationStudy with results
        
    Example:
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> study = run_ablation_study(
        ...     plan=feature_plan,
        ...     X=X_train,
        ...     y=y_train,
        ...     estimator=RandomForestClassifier(),
        ...     strategies=["on_off", "window"]
        ... )
        >>> print(f"Best score: {study.best_result.score:.4f}")
    """
    logger.info("Running ablation study...")
    
    runner = AblationRunner(
        estimator=estimator,
        scoring=scoring,
        cv=cv,
        **kwargs,
    )
    
    study = runner.run_ablation(
        plan=plan,
        X=X,
        y=y,
        strategies=strategies,
    )
    
    return study


def execute_distributed(
    plan: FeaturePlan,
    data: pd.DataFrame | Any,
    engine: Literal["pandas", "spark", "ray"] = "pandas",
    return_original: bool = False,
    **kwargs,
) -> pd.DataFrame | Any:
    """Execute feature plan with distributed execution engine.
    
    Args:
        plan: Feature plan to execute
        data: Input data (DataFrame for pandas/ray, Spark DF for spark)
        engine: Execution engine (pandas, spark, ray)
        return_original: Include original columns
        **kwargs: Engine-specific parameters
        
    Returns:
        DataFrame with generated features
        
    Example:
        >>> # Pandas (single machine)
        >>> df_features = execute_distributed(
        ...     plan=plan,
        ...     data=df,
        ...     engine="pandas"
        ... )
        
        >>> # Ray (distributed)
        >>> import ray
        >>> ray.init()
        >>> df_features = execute_distributed(
        ...     plan=plan,
        ...     data=df,
        ...     engine="ray"
        ... )
        
        >>> # Spark (cluster)
        >>> from pyspark.sql import SparkSession
        >>> spark = SparkSession.builder.getOrCreate()
        >>> sdf = spark.createDataFrame(df)
        >>> sdf_features = execute_distributed(
        ...     plan=plan,
        ...     data=sdf,
        ...     engine="spark"
        ... )
    """
    logger.info(f"Executing plan with {engine} engine...")
    
    if engine == "pandas":
        executor = PandasExecutor(**kwargs)
        return executor.execute(plan, data, return_original)
    
    elif engine == "ray":
        executor = RayExecutor(**kwargs)
        return executor.execute(plan, data, return_original)
    
    elif engine == "spark":
        executor = SparkExecutor(**kwargs)
        return executor.execute(plan, data, return_original)
    
    else:
        raise ValueError(f"Unknown engine: {engine}. Use pandas, spark, or ray.")


def full_pipeline(
    df: pd.DataFrame,
    target: str,
    X_train: pd.DataFrame | None = None,
    y_train: pd.Series | np.ndarray | None = None,
    nl_intent: str | None = None,
    knowledge_dirs: list[str] | None = None,
    enable_rag: bool = True,
    enable_pruning: bool = True,
    enable_ablation: bool = False,
    target_n_features: int | None = 50,
    executor_engine: Literal["pandas", "spark", "ray"] = "pandas",
    provider: str = "openai",
    **kwargs,
) -> tuple[FeaturePlan, pd.DataFrame, dict[str, Any]]:
    """Run full Phase 2 pipeline: plan + prune + execute.
    
    Args:
        df: Input DataFrame
        target: Target column
        X_train: Training features (for pruning)
        y_train: Training target (for pruning)
        nl_intent: Natural language intent
        knowledge_dirs: Knowledge directories for RAG
        enable_rag: Enable RAG augmentation
        enable_pruning: Enable feature pruning
        enable_ablation: Enable ablation study
        target_n_features: Target number of features after pruning
        executor_engine: Execution engine
        provider: LLM provider
        **kwargs: Additional parameters
        
    Returns:
        Tuple of (final_plan, features_df, metadata)
        
    Example:
        >>> plan, df_features, metadata = full_pipeline(
        ...     df=train_df,
        ...     target="churn",
        ...     X_train=X_train,
        ...     y_train=y_train,
        ...     nl_intent="Customer churn prediction features",
        ...     enable_rag=True,
        ...     enable_pruning=True,
        ...     target_n_features=50
        ... )
    """
    logger.info("🚀 Running full Phase 2 pipeline...")
    
    metadata = {}
    
    # Step 1: Plan with RAG
    if enable_rag:
        plan = plan_with_rag(
            df=df,
            target=target,
            nl_intent=nl_intent,
            knowledge_dirs=knowledge_dirs,
            provider=provider,
            **kwargs.get("plan_kwargs", {}),
        )
    else:
        from .planner import plan_features
        plan = plan_features(
            df=df,
            target=target,
            nl_intent=nl_intent,
            provider=provider,
            **kwargs.get("plan_kwargs", {}),
        )
    
    metadata["n_features_planned"] = len(plan.candidates)
    
    # Step 2: Prune features
    if enable_pruning and X_train is not None and y_train is not None:
        pruning_result = prune_features(
            plan=plan,
            X_train=X_train,
            y_train=y_train,
            target_n_features=target_n_features,
            provider=provider,
            **kwargs.get("pruning_kwargs", {}),
        )
        plan = pruning_result.pruned_plan
        metadata["n_features_pruned"] = len(plan.candidates)
        metadata["pruning_result"] = pruning_result
    
    # Step 3: Execute
    df_features = execute_distributed(
        plan=plan,
        data=df,
        engine=executor_engine,
        return_original=False,
        **kwargs.get("executor_kwargs", {}),
    )
    
    metadata["n_features_generated"] = df_features.shape[1]
    
    # Optional: Ablation study
    if enable_ablation and X_train is not None and y_train is not None:
        estimator = kwargs.get("ablation_estimator")
        if estimator:
            ablation_study = run_ablation_study(
                plan=plan,
                X=X_train,
                y=y_train,
                estimator=estimator,
                **kwargs.get("ablation_kwargs", {}),
            )
            metadata["ablation_study"] = ablation_study
    
    logger.info(f"✓ Pipeline complete: {metadata['n_features_generated']} features generated")
    
    return plan, df_features, metadata

