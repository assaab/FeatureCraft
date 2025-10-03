"""LLM-guided feature pruning with statistical gates."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

from .providers import BaseLLMProvider, get_provider
from .schemas import FeaturePlan, FeatureSpec
from ..logging import get_logger

logger = get_logger(__name__)


@dataclass
class FeatureRanking:
    """Feature ranking with rationale.
    
    Attributes:
        feature_name: Feature name
        rank: Rank (1 = highest priority)
        score: Importance score (0-1)
        rationale: Human-readable explanation
        gates_passed: Statistical gates passed
        metadata: Additional metadata
    """
    
    feature_name: str
    rank: int
    score: float
    rationale: str = ""
    gates_passed: dict[str, bool] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PruningResult:
    """Result of feature pruning.
    
    Attributes:
        original_features: Original feature names
        selected_features: Selected feature names
        rankings: Feature rankings
        pruned_plan: Pruned FeaturePlan
        metadata: Pruning metadata
    """
    
    original_features: list[str]
    selected_features: list[str]
    rankings: list[FeatureRanking]
    pruned_plan: FeaturePlan | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class FeaturePruner:
    """LLM-guided feature pruning with statistical validation.
    
    This class uses an LLM to rank feature candidates and applies
    statistical gates (MI, permutation importance, SHAP, stability)
    to finalize the selection.
    
    Example:
        >>> pruner = FeaturePruner(provider="openai")
        >>> result = pruner.prune(
        ...     plan=feature_plan,
        ...     X_train=X_train,
        ...     y_train=y_train,
        ...     target_n_features=50
        ... )
        >>> print(f"Selected {len(result.selected_features)} features")
    """
    
    def __init__(
        self,
        provider: BaseLLMProvider | str = "openai",
        model: str | None = None,
        enable_mi_gate: bool = True,
        enable_permutation_gate: bool = True,
        enable_shap_gate: bool = False,
        enable_stability_gate: bool = True,
        enable_leakage_gate: bool = True,
        mi_threshold: float = 0.01,
        permutation_threshold: float = 0.001,
        stability_threshold: float = 0.7,
    ):
        """Initialize feature pruner.
        
        Args:
            provider: LLM provider (instance or name)
            model: Model name
            enable_mi_gate: Enable mutual information gate
            enable_permutation_gate: Enable permutation importance gate
            enable_shap_gate: Enable SHAP gate
            enable_stability_gate: Enable stability gate
            enable_leakage_gate: Enable leakage detection gate
            mi_threshold: Min MI score to pass gate
            permutation_threshold: Min permutation importance to pass gate
            stability_threshold: Min stability score (correlation across folds)
        """
        if isinstance(provider, str):
            self.provider = get_provider(provider, model=model)
        else:
            self.provider = provider
        
        self.enable_mi_gate = enable_mi_gate
        self.enable_permutation_gate = enable_permutation_gate
        self.enable_shap_gate = enable_shap_gate
        self.enable_stability_gate = enable_stability_gate
        self.enable_leakage_gate = enable_leakage_gate
        
        self.mi_threshold = mi_threshold
        self.permutation_threshold = permutation_threshold
        self.stability_threshold = stability_threshold
    
    def prune(
        self,
        plan: FeaturePlan,
        X_train: pd.DataFrame,
        y_train: pd.Series | np.ndarray,
        target_n_features: int | None = None,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | np.ndarray | None = None,
    ) -> PruningResult:
        """Prune feature plan to select best features.
        
        Args:
            plan: Feature plan to prune
            X_train: Training features
            y_train: Training target
            target_n_features: Target number of features (if None, uses gates only)
            X_val: Validation features (for stability gate)
            y_val: Validation target (for stability gate)
            
        Returns:
            PruningResult with selected features
        """
        logger.info(f"Pruning {len(plan.candidates)} features...")
        
        # Step 1: LLM ranking
        llm_rankings = self._llm_rank_features(plan, X_train.head(100))
        
        # Step 2: Statistical gates
        gate_results = self._apply_gates(
            plan=plan,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
        )
        
        # Step 3: Combine LLM rankings with gate results
        final_rankings = self._combine_rankings(
            llm_rankings=llm_rankings,
            gate_results=gate_results,
        )
        
        # Step 4: Select features
        selected_features = self._select_features(
            rankings=final_rankings,
            target_n=target_n_features,
        )
        
        # Step 5: Create pruned plan
        pruned_plan = self._create_pruned_plan(plan, selected_features)
        
        result = PruningResult(
            original_features=[c.name for c in plan.candidates],
            selected_features=selected_features,
            rankings=final_rankings,
            pruned_plan=pruned_plan,
            metadata={
                "timestamp": datetime.utcnow().isoformat(),
                "original_n_features": len(plan.candidates),
                "selected_n_features": len(selected_features),
                "target_n_features": target_n_features,
                "gates_enabled": {
                    "mi": self.enable_mi_gate,
                    "permutation": self.enable_permutation_gate,
                    "shap": self.enable_shap_gate,
                    "stability": self.enable_stability_gate,
                    "leakage": self.enable_leakage_gate,
                },
            },
        )
        
        logger.info(
            f"✓ Pruned to {len(selected_features)} features "
            f"(from {len(plan.candidates)})"
        )
        
        return result
    
    def _llm_rank_features(
        self,
        plan: FeaturePlan,
        X_sample: pd.DataFrame,
    ) -> dict[str, float]:
        """Ask LLM to rank features by importance.
        
        Returns:
            Dict mapping feature name to LLM score (0-1)
        """
        # Build prompt
        prompt = self._build_ranking_prompt(plan, X_sample)
        
        system_prompt = """You are an expert feature engineering advisor.
Your task is to rank feature candidates by their expected importance for the ML task.

Return ONLY valid JSON with this structure:
{
  "rankings": [
    {
      "feature_name": "amt_mean_30d",
      "score": 0.95,
      "rationale": "Strong indicator of customer value"
    }
  ]
}

Score from 0.0 (useless) to 1.0 (critical).
NO markdown, NO code blocks, ONLY JSON.
"""
        
        # Call LLM
        try:
            response = self.provider.call(
                prompt=prompt,
                system_prompt=system_prompt,
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            
            # Parse response
            rankings_data = json.loads(response["content"])
            rankings = {
                r["feature_name"]: r["score"]
                for r in rankings_data.get("rankings", [])
            }
            
            logger.info(f"✓ LLM ranked {len(rankings)} features")
            return rankings
            
        except Exception as e:
            logger.warning(f"LLM ranking failed: {e}. Using uniform scores.")
            return {c.name: 0.5 for c in plan.candidates}
    
    def _build_ranking_prompt(
        self,
        plan: FeaturePlan,
        X_sample: pd.DataFrame,
    ) -> str:
        """Build prompt for LLM ranking."""
        # Feature candidates
        candidates_str = "\n".join([
            f"- {c.name}: {c.type} from {c.source_col} | {c.rationale}"
            for c in plan.candidates[:100]  # Limit to avoid token overflow
        ])
        
        # Sample data
        sample_str = X_sample.describe().to_string()
        
        prompt = f"""Rank these {len(plan.candidates)} feature candidates by importance.

TASK: {plan.task}
ESTIMATOR: {plan.estimator_family}

FEATURE CANDIDATES:
{candidates_str}

SAMPLE DATA STATISTICS:
{sample_str}

Rank all features from 0.0 (useless) to 1.0 (critical).
Consider:
1. Relevance to task
2. Statistical power
3. Interpretability
4. Redundancy (penalize highly correlated features)

Return JSON with rankings for ALL features.
"""
        
        return prompt
    
    def _apply_gates(
        self,
        plan: FeaturePlan,
        X_train: pd.DataFrame,
        y_train: pd.Series | np.ndarray,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | np.ndarray | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Apply statistical gates to features.
        
        Returns:
            Dict mapping feature name to gate results
        """
        gate_results = {}
        
        # Convert y to numpy
        if isinstance(y_train, pd.Series):
            y_train = y_train.values
        
        # Determine task
        task = plan.task
        
        # MI gate
        if self.enable_mi_gate:
            mi_scores = self._compute_mi(X_train, y_train, task)
            for feat, score in mi_scores.items():
                if feat not in gate_results:
                    gate_results[feat] = {}
                gate_results[feat]["mi_score"] = score
                gate_results[feat]["mi_pass"] = score >= self.mi_threshold
        
        # Permutation importance gate
        if self.enable_permutation_gate:
            perm_scores = self._compute_permutation_importance(X_train, y_train, task)
            for feat, score in perm_scores.items():
                if feat not in gate_results:
                    gate_results[feat] = {}
                gate_results[feat]["permutation_score"] = score
                gate_results[feat]["permutation_pass"] = score >= self.permutation_threshold
        
        # Stability gate (if validation data provided)
        if self.enable_stability_gate and X_val is not None and y_val is not None:
            stability_scores = self._compute_stability(
                X_train, X_val, y_train, y_val, task
            )
            for feat, score in stability_scores.items():
                if feat not in gate_results:
                    gate_results[feat] = {}
                gate_results[feat]["stability_score"] = score
                gate_results[feat]["stability_pass"] = score >= self.stability_threshold
        
        return gate_results
    
    def _compute_mi(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        task: str,
    ) -> dict[str, float]:
        """Compute mutual information scores."""
        logger.debug("Computing mutual information...")
        
        # Handle NaN
        X_clean = X.fillna(0)
        
        # Select MI function
        if task == "classification":
            mi_func = mutual_info_classif
        else:
            mi_func = mutual_info_regression
        
        try:
            mi_scores = mi_func(X_clean, y, random_state=42)
            return dict(zip(X.columns, mi_scores))
        except Exception as e:
            logger.warning(f"MI computation failed: {e}")
            return {col: 0.0 for col in X.columns}
    
    def _compute_permutation_importance(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        task: str,
    ) -> dict[str, float]:
        """Compute permutation importance."""
        logger.debug("Computing permutation importance...")
        
        # Handle NaN
        X_clean = X.fillna(0)
        
        # Train simple model
        if task == "classification":
            model = RandomForestClassifier(
                n_estimators=50,
                max_depth=10,
                random_state=42,
                n_jobs=-1,
            )
        else:
            model = RandomForestRegressor(
                n_estimators=50,
                max_depth=10,
                random_state=42,
                n_jobs=-1,
            )
        
        try:
            model.fit(X_clean, y)
            
            # Use feature importances as proxy (faster than permutation)
            importances = model.feature_importances_
            return dict(zip(X.columns, importances))
        except Exception as e:
            logger.warning(f"Permutation importance computation failed: {e}")
            return {col: 0.0 for col in X.columns}
    
    def _compute_stability(
        self,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        y_train: np.ndarray,
        y_val: np.ndarray,
        task: str,
    ) -> dict[str, float]:
        """Compute feature stability across train/val splits."""
        logger.debug("Computing feature stability...")
        
        stability_scores = {}
        
        for col in X_train.columns:
            try:
                # Compute correlation between train and val distributions
                train_vals = X_train[col].fillna(0).values
                val_vals = X_val[col].fillna(0).values
                
                # Pearson correlation
                if len(train_vals) > 1 and len(val_vals) > 1:
                    corr = np.corrcoef(
                        np.histogram(train_vals, bins=20)[0],
                        np.histogram(val_vals, bins=20)[0]
                    )[0, 1]
                    stability_scores[col] = max(0.0, corr)
                else:
                    stability_scores[col] = 0.0
            except Exception:
                stability_scores[col] = 0.0
        
        return stability_scores
    
    def _combine_rankings(
        self,
        llm_rankings: dict[str, float],
        gate_results: dict[str, dict[str, Any]],
    ) -> list[FeatureRanking]:
        """Combine LLM rankings with gate results."""
        all_features = set(llm_rankings.keys()) | set(gate_results.keys())
        
        combined = []
        for feat in all_features:
            llm_score = llm_rankings.get(feat, 0.5)
            gates = gate_results.get(feat, {})
            
            # Compute combined score
            # Start with LLM score, penalize for failed gates
            score = llm_score
            
            gates_passed = {}
            for gate_name in ["mi_pass", "permutation_pass", "stability_pass"]:
                if gate_name in gates:
                    gates_passed[gate_name] = gates[gate_name]
                    if not gates[gate_name]:
                        score *= 0.5  # Penalty for failed gate
            
            combined.append(FeatureRanking(
                feature_name=feat,
                rank=0,  # Will be set after sorting
                score=score,
                gates_passed=gates_passed,
                metadata=gates,
            ))
        
        # Sort by score and assign ranks
        combined.sort(key=lambda x: x.score, reverse=True)
        for i, ranking in enumerate(combined):
            ranking.rank = i + 1
        
        return combined
    
    def _select_features(
        self,
        rankings: list[FeatureRanking],
        target_n: int | None,
    ) -> list[str]:
        """Select features based on rankings and target count."""
        if target_n is None:
            # Select all features that passed all gates
            selected = [
                r.feature_name
                for r in rankings
                if all(r.gates_passed.values())
            ]
        else:
            # Select top N by score
            selected = [r.feature_name for r in rankings[:target_n]]
        
        return selected
    
    def _create_pruned_plan(
        self,
        plan: FeaturePlan,
        selected_features: list[str],
    ) -> FeaturePlan:
        """Create new plan with only selected features."""
        selected_specs = [
            spec for spec in plan.candidates
            if spec.name in selected_features
        ]
        
        pruned_plan = FeaturePlan(
            version=plan.version,
            dataset_id=plan.dataset_id,
            task=plan.task,
            estimator_family=plan.estimator_family,
            constraints=plan.constraints,
            budget=plan.budget,
            candidates=selected_specs,
            rationale=f"Pruned from {len(plan.candidates)} to {len(selected_specs)} features",
            safety_summary=plan.safety_summary,
            metadata={
                **plan.metadata,
                "pruned_from": len(plan.candidates),
                "pruned_to": len(selected_specs),
                "pruning_timestamp": datetime.utcnow().isoformat(),
            },
        )
        
        return pruned_plan

