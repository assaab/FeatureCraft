"""Auto-ablation studies for feature engineering plans."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score

from .schemas import FeaturePlan, FeatureSpec
from ..logging import get_logger

logger = get_logger(__name__)


@dataclass
class AblationExperiment:
    """Single ablation experiment configuration.
    
    Attributes:
        experiment_id: Unique experiment ID
        ablation_type: Type of ablation (on_off, window, encoding, interaction)
        features_included: Features included in this experiment
        features_excluded: Features excluded in this experiment
        params_modified: Parameters modified from baseline
        metadata: Additional metadata
    """
    
    experiment_id: str
    ablation_type: Literal["on_off", "window", "encoding", "interaction"]
    features_included: list[str]
    features_excluded: list[str] = field(default_factory=list)
    params_modified: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AblationResult:
    """Result of single ablation experiment.
    
    Attributes:
        experiment: Experiment configuration
        score: Performance score (higher is better)
        score_std: Standard deviation of score
        training_time: Training time in seconds
        n_features: Number of features used
        metadata: Additional result metadata
    """
    
    experiment: AblationExperiment
    score: float
    score_std: float = 0.0
    training_time: float = 0.0
    n_features: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "experiment_id": self.experiment.experiment_id,
            "ablation_type": self.experiment.ablation_type,
            "features_included": self.experiment.features_included,
            "features_excluded": self.experiment.features_excluded,
            "score": self.score,
            "score_std": self.score_std,
            "training_time": self.training_time,
            "n_features": self.n_features,
            "metadata": self.metadata,
        }


@dataclass
class AblationStudy:
    """Complete ablation study results.
    
    Attributes:
        baseline_result: Baseline (all features) result
        ablation_results: List of ablation experiment results
        best_result: Best performing configuration
        insights: Key insights from the study
        metadata: Study metadata
    """
    
    baseline_result: AblationResult
    ablation_results: list[AblationResult]
    best_result: AblationResult | None = None
    insights: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "baseline": self.baseline_result.to_dict(),
            "ablations": [r.to_dict() for r in self.ablation_results],
            "best": self.best_result.to_dict() if self.best_result else None,
            "insights": self.insights,
            "metadata": self.metadata,
        }
    
    def save(self, path: str) -> None:
        """Save study results to JSON."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"✓ Saved ablation study to {path}")


class AblationRunner:
    """Runner for automated ablation studies.
    
    Supports multiple ablation strategies:
    - On/Off: Include/exclude each feature
    - Window: Vary time windows (e.g., 7d vs 30d)
    - Encoding: Vary encoding strategies
    - Interaction: Test feature interactions
    
    Example:
        >>> runner = AblationRunner(
        ...     estimator=LogisticRegression(),
        ...     scoring="roc_auc",
        ...     cv=5
        ... )
        >>> study = runner.run_ablation(
        ...     plan=feature_plan,
        ...     X=X_train,
        ...     y=y_train,
        ...     strategies=["on_off", "window"]
        ... )
    """
    
    def __init__(
        self,
        estimator: Any,
        scoring: str = "roc_auc",
        cv: int = 5,
        n_jobs: int = -1,
        early_stop_patience: int | None = None,
        early_stop_threshold: float = 0.01,
        max_experiments: int | None = None,
    ):
        """Initialize ablation runner.
        
        Args:
            estimator: Sklearn-compatible estimator
            scoring: Scoring metric
            cv: Cross-validation folds
            n_jobs: Parallel jobs
            early_stop_patience: Stop if no improvement after N experiments
            early_stop_threshold: Min improvement threshold for early stopping
            max_experiments: Maximum number of experiments to run
        """
        self.estimator = estimator
        self.scoring = scoring
        self.cv = cv
        self.n_jobs = n_jobs
        self.early_stop_patience = early_stop_patience
        self.early_stop_threshold = early_stop_threshold
        self.max_experiments = max_experiments
    
    def run_ablation(
        self,
        plan: FeaturePlan,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        strategies: list[Literal["on_off", "window", "encoding", "interaction"]] = None,
        executor: Callable | None = None,
    ) -> AblationStudy:
        """Run ablation study on feature plan.
        
        Args:
            plan: Feature plan to ablate
            X: Input features
            y: Target variable
            strategies: Ablation strategies to test
            executor: Feature executor function (if None, uses X directly)
            
        Returns:
            AblationStudy with results
        """
        strategies = strategies or ["on_off"]
        logger.info(f"Running ablation study with strategies: {strategies}")
        
        # Baseline evaluation
        baseline_result = self._evaluate_config(
            plan=plan,
            X=X,
            y=y,
            experiment_id="baseline",
        )
        
        logger.info(
            f"Baseline score: {baseline_result.score:.4f} ± {baseline_result.score_std:.4f}"
        )
        
        # Generate ablation experiments
        experiments = []
        for strategy in strategies:
            if strategy == "on_off":
                experiments.extend(self._generate_on_off_experiments(plan))
            elif strategy == "window":
                experiments.extend(self._generate_window_experiments(plan))
            elif strategy == "encoding":
                experiments.extend(self._generate_encoding_experiments(plan))
            elif strategy == "interaction":
                experiments.extend(self._generate_interaction_experiments(plan))
        
        # Limit experiments
        if self.max_experiments and len(experiments) > self.max_experiments:
            logger.info(f"Limiting to {self.max_experiments} experiments (from {len(experiments)})")
            experiments = experiments[:self.max_experiments]
        
        # Run experiments
        ablation_results = []
        best_score = baseline_result.score
        no_improvement_count = 0
        
        for i, experiment in enumerate(experiments):
            logger.info(f"Running experiment {i+1}/{len(experiments)}: {experiment.experiment_id}")
            
            # Create modified plan
            modified_plan = self._apply_experiment(plan, experiment)
            
            # Evaluate
            result = self._evaluate_config(
                plan=modified_plan,
                X=X,
                y=y,
                experiment_id=experiment.experiment_id,
            )
            result.experiment = experiment
            ablation_results.append(result)
            
            # Check for improvement
            if result.score > best_score + self.early_stop_threshold:
                best_score = result.score
                no_improvement_count = 0
                logger.info(f"✓ New best score: {result.score:.4f}")
            else:
                no_improvement_count += 1
            
            # Early stopping
            if self.early_stop_patience and no_improvement_count >= self.early_stop_patience:
                logger.info(f"Early stopping: no improvement after {no_improvement_count} experiments")
                break
        
        # Find best result
        best_result = max(ablation_results, key=lambda r: r.score)
        
        # Generate insights
        insights = self._generate_insights(baseline_result, ablation_results, best_result)
        
        study = AblationStudy(
            baseline_result=baseline_result,
            ablation_results=ablation_results,
            best_result=best_result,
            insights=insights,
            metadata={
                "timestamp": datetime.utcnow().isoformat(),
                "n_experiments": len(ablation_results),
                "strategies": strategies,
                "baseline_score": baseline_result.score,
                "best_score": best_result.score,
                "improvement": best_result.score - baseline_result.score,
            },
        )
        
        logger.info(
            f"✓ Ablation study complete: {len(ablation_results)} experiments, "
            f"best score: {best_result.score:.4f} "
            f"(+{best_result.score - baseline_result.score:.4f})"
        )
        
        return study
    
    def _generate_on_off_experiments(self, plan: FeaturePlan) -> list[AblationExperiment]:
        """Generate on/off ablation experiments (leave-one-out)."""
        experiments = []
        
        for feature in plan.candidates:
            # Leave-one-out
            experiment = AblationExperiment(
                experiment_id=f"loo_{feature.name}",
                ablation_type="on_off",
                features_included=[f.name for f in plan.candidates if f.name != feature.name],
                features_excluded=[feature.name],
            )
            experiments.append(experiment)
        
        # Also test feature groups (e.g., all rolling features, all lags)
        feature_types = {}
        for feature in plan.candidates:
            if feature.type not in feature_types:
                feature_types[feature.type] = []
            feature_types[feature.type].append(feature.name)
        
        for feat_type, feat_names in feature_types.items():
            if len(feat_names) > 1:
                # Exclude all features of this type
                experiment = AblationExperiment(
                    experiment_id=f"exclude_type_{feat_type}",
                    ablation_type="on_off",
                    features_included=[
                        f.name for f in plan.candidates if f.type != feat_type
                    ],
                    features_excluded=feat_names,
                    metadata={"excluded_type": feat_type},
                )
                experiments.append(experiment)
        
        return experiments
    
    def _generate_window_experiments(self, plan: FeaturePlan) -> list[AblationExperiment]:
        """Generate window size ablation experiments."""
        experiments = []
        
        # Find features with window parameters
        window_features = [f for f in plan.candidates if f.window]
        
        # Test alternative windows
        alt_windows = ["7d", "14d", "30d", "60d", "90d"]
        
        for feature in window_features:
            current_window = feature.window
            
            for alt_window in alt_windows:
                if alt_window != current_window:
                    experiment = AblationExperiment(
                        experiment_id=f"window_{feature.name}_{alt_window}",
                        ablation_type="window",
                        features_included=[f.name for f in plan.candidates],
                        params_modified={feature.name: {"window": alt_window}},
                        metadata={
                            "original_window": current_window,
                            "new_window": alt_window,
                        },
                    )
                    experiments.append(experiment)
        
        return experiments
    
    def _generate_encoding_experiments(self, plan: FeaturePlan) -> list[AblationExperiment]:
        """Generate encoding ablation experiments."""
        experiments = []
        
        # Find encoding features
        encoding_types = ["target_encode", "frequency_encode", "count_encode", "ohe"]
        encoding_features = [f for f in plan.candidates if f.type in encoding_types]
        
        # Test alternative encodings
        alt_encodings = ["target_encode", "frequency_encode", "ohe"]
        
        for feature in encoding_features:
            current_encoding = feature.type
            
            for alt_encoding in alt_encodings:
                if alt_encoding != current_encoding:
                    experiment = AblationExperiment(
                        experiment_id=f"encoding_{feature.name}_{alt_encoding}",
                        ablation_type="encoding",
                        features_included=[f.name for f in plan.candidates],
                        params_modified={feature.name: {"type": alt_encoding}},
                        metadata={
                            "original_encoding": current_encoding,
                            "new_encoding": alt_encoding,
                        },
                    )
                    experiments.append(experiment)
        
        return experiments
    
    def _generate_interaction_experiments(self, plan: FeaturePlan) -> list[AblationExperiment]:
        """Generate feature interaction experiments."""
        experiments = []
        
        # Test pairs of features
        numeric_features = [
            f for f in plan.candidates
            if f.type in ["rolling_mean", "rolling_sum", "lag", "recency", "frequency"]
        ]
        
        # Limit to top 10 features to avoid combinatorial explosion
        numeric_features = numeric_features[:10]
        
        for f1, f2 in combinations(numeric_features, 2):
            # Test multiply interaction
            experiment = AblationExperiment(
                experiment_id=f"interact_{f1.name}_{f2.name}",
                ablation_type="interaction",
                features_included=[f.name for f in plan.candidates],
                params_modified={
                    f"interact_{f1.name}_{f2.name}": {
                        "type": "multiply",
                        "source_cols": [f1.source_col, f2.source_col],
                    }
                },
                metadata={"interaction_type": "multiply"},
            )
            experiments.append(experiment)
        
        return experiments
    
    def _apply_experiment(
        self,
        plan: FeaturePlan,
        experiment: AblationExperiment,
    ) -> FeaturePlan:
        """Apply experiment modifications to plan."""
        # Filter features
        modified_candidates = [
            spec for spec in plan.candidates
            if spec.name in experiment.features_included
        ]
        
        # Apply parameter modifications
        for feat_name, params in experiment.params_modified.items():
            for spec in modified_candidates:
                if spec.name == feat_name:
                    for param, value in params.items():
                        setattr(spec, param, value)
        
        # Create new plan
        modified_plan = FeaturePlan(
            version=plan.version,
            dataset_id=plan.dataset_id,
            task=plan.task,
            estimator_family=plan.estimator_family,
            constraints=plan.constraints,
            budget=plan.budget,
            candidates=modified_candidates,
            rationale=f"Ablation: {experiment.experiment_id}",
            metadata=plan.metadata,
        )
        
        return modified_plan
    
    def _evaluate_config(
        self,
        plan: FeaturePlan,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        experiment_id: str,
    ) -> AblationResult:
        """Evaluate a configuration with cross-validation."""
        # Select features from X
        feature_names = [f.name for f in plan.candidates]
        available_features = [f for f in feature_names if f in X.columns]
        
        if not available_features:
            # Return zero score if no features available
            return AblationResult(
                experiment=AblationExperiment(
                    experiment_id=experiment_id,
                    ablation_type="on_off",
                    features_included=[],
                ),
                score=0.0,
                n_features=0,
            )
        
        X_subset = X[available_features].fillna(0)
        
        # Cross-validate
        start_time = time.time()
        
        try:
            scores = cross_val_score(
                self.estimator,
                X_subset,
                y,
                cv=self.cv,
                scoring=self.scoring,
                n_jobs=self.n_jobs,
            )
            
            training_time = time.time() - start_time
            
            return AblationResult(
                experiment=AblationExperiment(
                    experiment_id=experiment_id,
                    ablation_type="on_off",
                    features_included=available_features,
                ),
                score=scores.mean(),
                score_std=scores.std(),
                training_time=training_time,
                n_features=len(available_features),
            )
        except Exception as e:
            logger.warning(f"Evaluation failed for {experiment_id}: {e}")
            return AblationResult(
                experiment=AblationExperiment(
                    experiment_id=experiment_id,
                    ablation_type="on_off",
                    features_included=available_features,
                ),
                score=0.0,
                n_features=len(available_features),
            )
    
    def _generate_insights(
        self,
        baseline: AblationResult,
        ablations: list[AblationResult],
        best: AblationResult,
    ) -> dict[str, Any]:
        """Generate insights from ablation results."""
        insights = {}
        
        # Improvement over baseline
        insights["improvement_over_baseline"] = best.score - baseline.score
        insights["relative_improvement_pct"] = (
            (best.score - baseline.score) / baseline.score * 100
        )
        
        # Most impactful features (by leave-one-out drop)
        loo_results = [r for r in ablations if r.experiment.ablation_type == "on_off"]
        if loo_results:
            feature_impacts = []
            for result in loo_results:
                if result.experiment.features_excluded:
                    excluded_feat = result.experiment.features_excluded[0]
                    impact = baseline.score - result.score
                    feature_impacts.append((excluded_feat, impact))
            
            feature_impacts.sort(key=lambda x: abs(x[1]), reverse=True)
            insights["most_impactful_features"] = feature_impacts[:10]
        
        # Best ablation type
        ablation_scores_by_type = {}
        for result in ablations:
            abl_type = result.experiment.ablation_type
            if abl_type not in ablation_scores_by_type:
                ablation_scores_by_type[abl_type] = []
            ablation_scores_by_type[abl_type].append(result.score)
        
        insights["avg_score_by_ablation_type"] = {
            abl_type: np.mean(scores)
            for abl_type, scores in ablation_scores_by_type.items()
        }
        
        return insights

