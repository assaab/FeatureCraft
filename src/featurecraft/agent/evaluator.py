"""Evaluator Module: CV scoring, baselines, ablation studies."""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import BaseCrossValidator, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    log_loss,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    f1_score,
)
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from ..config import FeatureCraftConfig
from ..logging import get_logger
from ..types import TaskType
from ..drift import compute_psi
from .config import AgentConfig
from .types import DatasetFingerprint, EvaluationResult, AblationResults

logger = get_logger(__name__)


class Evaluator:
    """Pipeline evaluation with CV scoring and analysis."""
    
    def __init__(
        self,
        config: AgentConfig,
        cv_strategy: BaseCrossValidator,
    ):
        """Initialize evaluator.
        
        Args:
            config: Agent configuration
            cv_strategy: Cross-validation strategy
        """
        self.config = config
        self.cv_strategy = cv_strategy
        self.fc_config = FeatureCraftConfig(random_state=config.random_seed)
    
    def evaluate_pipeline(
        self,
        pipeline: Pipeline,
        X: pd.DataFrame,
        y: pd.Series,
        pipeline_id: Optional[str] = None,
        n_splits: Optional[int] = None,
    ) -> EvaluationResult:
        """Evaluate pipeline with cross-validation.
        
        Args:
            pipeline: Pipeline to evaluate
            X: Feature dataframe
            y: Target series
            pipeline_id: Optional pipeline identifier
            n_splits: Optional override for n_splits
            
        Returns:
            EvaluationResult
        """
        if pipeline_id is None:
            pipeline_id = f"pipeline_{int(time.time() * 1000)}"
        
        logger.info(f"Evaluating pipeline: {pipeline_id}")
        
        start_time = time.time()
        
        # Determine task and metrics
        task_type = self._detect_task(y)
        scoring = self._get_scoring(task_type)
        
        # Add final estimator for evaluation if not present
        # FeatureCraft pipelines are feature engineering only, so we need an estimator for evaluation
        if not any(step_name in ["estimator", "classifier", "regressor"] for step_name, _ in pipeline.steps):
            from ..agent.composer import Composer
            composer = Composer(self.config)
            task_name = "classification" if task_type == TaskType.CLASSIFICATION else "regression"
            pipeline = composer.add_evaluation_estimator(pipeline, task_name)

        # Run cross-validation
        try:
            cv_results = cross_validate(
                pipeline,
                X,
                y,
                cv=self.cv_strategy if n_splits is None else self._override_splits(n_splits),
                scoring=scoring,
                return_train_score=False,
                n_jobs=1,
                error_score="raise",
            )
            
            # Extract primary metric scores
            primary_metric = self._get_primary_metric(task_type)
            score_key = f"test_{primary_metric}"
            cv_scores = cv_results[score_key].tolist()
            cv_score_mean = float(np.mean(cv_scores))
            cv_score_std = float(np.std(cv_scores))
            
            # Collect all metrics
            metrics = {}
            for key, values in cv_results.items():
                if key.startswith("test_"):
                    metric_name = key.replace("test_", "")
                    metrics[metric_name] = float(np.mean(values))
            
            # Timing
            fit_time = float(np.mean(cv_results["fit_time"]))
            score_time = float(np.mean(cv_results["score_time"]))
            total_time = time.time() - start_time
            
            # Get feature count (fit pipeline once to get feature names)
            try:
                pipeline.fit(X.head(100), y.head(100))  # Quick fit on sample
                if hasattr(pipeline, "get_feature_names_out"):
                    feature_names = pipeline.get_feature_names_out()
                    n_features_out = len(feature_names)
                else:
                    n_features_out = 0
                    feature_names = None
            except:
                n_features_out = 0
                feature_names = None
            
            result = EvaluationResult(
                pipeline_id=pipeline_id,
                cv_score_mean=cv_score_mean,
                cv_score_std=cv_score_std,
                cv_scores=cv_scores,
                metrics=metrics,
                fit_time_seconds=fit_time,
                transform_time_seconds=score_time,
                total_time_seconds=total_time,
                n_features_out=n_features_out,
                feature_names_out=feature_names,
            )
            
            logger.info(
                f"[OK] {pipeline_id}: {primary_metric}={cv_score_mean:.4f} +/- {cv_score_std:.4f} "
                f"({total_time:.1f}s, {n_features_out} features)"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Pipeline evaluation failed: {e}")
            # Return dummy result with low score
            return EvaluationResult(
                pipeline_id=pipeline_id,
                cv_score_mean=0.0,
                cv_score_std=0.0,
                cv_scores=[0.0] * self.config.n_cv_folds,
                metrics={"error": str(e)},
                total_time_seconds=time.time() - start_time,
            )
    
    def compute_baseline(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        baseline_type: str,
    ) -> EvaluationResult:
        """Compute baseline score.
        
        Args:
            X: Feature dataframe
            y: Target series
            baseline_type: 'raw' or 'auto'
            
        Returns:
            EvaluationResult for baseline
        """
        logger.info(f"Computing {baseline_type} baseline")
        
        from ..pipeline import AutoFeatureEngineer
        
        if baseline_type == "raw":
            # Minimal preprocessing
            config = FeatureCraftConfig(
                random_state=self.config.random_seed,
                use_target_encoding=False,
                use_frequency_encoding=False,
                text_extract_sentiment=False,
                verbosity=0,  # Minimal output for baseline
            )
        else:  # auto
            # Default FeatureCraft
            config = FeatureCraftConfig(random_state=self.config.random_seed)
        
        afe = AutoFeatureEngineer(config=config)
        pipeline = Pipeline([("afe", afe)])
        
        return self.evaluate_pipeline(
            pipeline=pipeline,
            X=X,
            y=y,
            pipeline_id=f"baseline_{baseline_type}",
        )
    
    def ablation_study(
        self,
        pipeline: Pipeline,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> AblationResults:
        """Run ablation study on pipeline.
        
        Args:
            pipeline: Fitted pipeline
            X: Feature dataframe
            y: Target series
            
        Returns:
            AblationResults
        """
        logger.info("Running ablation study")
        
        # Evaluate full pipeline
        full_result = self.evaluate_pipeline(pipeline, X, y, pipeline_id="ablation_full")
        full_score = full_result.cv_score_mean
        
        operation_impacts = {}
        
        # Try removing each step
        for i, (step_name, step) in enumerate(pipeline.steps):
            if step_name in ["preprocessing", "ensure_numeric"]:
                # Skip essential steps
                continue
            
            # Create pipeline without this step
            reduced_steps = [
                (name, s) for j, (name, s) in enumerate(pipeline.steps) if j != i
            ]
            
            if len(reduced_steps) < 2:
                continue
            
            reduced_pipeline = Pipeline(reduced_steps)
            
            try:
                ablation_result = self.evaluate_pipeline(
                    reduced_pipeline, X, y, pipeline_id=f"ablation_without_{step_name}"
                )
                
                # Impact = full - reduced (positive means step helps)
                impact = full_score - ablation_result.cv_score_mean
                operation_impacts[step_name] = impact
                
                logger.info(f"  {step_name}: impact={impact:+.4f}")
                
            except Exception as e:
                logger.warning(f"Ablation failed for {step_name}: {e}")
        
        return AblationResults(
            operation_impacts=operation_impacts,
            family_impacts={},
            ablation_details=[],
        )
    
    def permutation_importance(
        self,
        pipeline: Pipeline,
        X: pd.DataFrame,
        y: pd.Series,
        n_repeats: int = 10,
    ) -> Dict[str, float]:
        """Compute permutation importance.
        
        Args:
            pipeline: Fitted pipeline
            X: Feature dataframe
            y: Target series
            n_repeats: Number of permutation repeats
            
        Returns:
            Dictionary of feature importances
        """
        logger.info("Computing permutation importance")
        
        try:
            # Fit pipeline
            pipeline.fit(X, y)
            
            # Transform data
            X_transformed = pipeline.transform(X)
            
            # Get feature names
            if hasattr(pipeline, "get_feature_names_out"):
                feature_names = pipeline.get_feature_names_out()
            else:
                feature_names = [f"feature_{i}" for i in range(X_transformed.shape[1])]
            
            # Create dummy estimator for permutation importance
            task_type = self._detect_task(y)
            if task_type == TaskType.CLASSIFICATION:
                estimator = RandomForestClassifier(
                    n_estimators=10, random_state=self.config.random_seed, max_depth=5
                )
            else:
                estimator = RandomForestRegressor(
                    n_estimators=10, random_state=self.config.random_seed, max_depth=5
                )
            
            estimator.fit(X_transformed, y)
            
            # Compute permutation importance
            perm_result = permutation_importance(
                estimator,
                X_transformed,
                y,
                n_repeats=n_repeats,
                random_state=self.config.random_seed,
                n_jobs=1,
            )
            
            # Create importance dict
            importances = {}
            for i, name in enumerate(feature_names):
                importances[name] = float(perm_result.importances_mean[i])
            
            # Sort by importance
            importances = dict(
                sorted(importances.items(), key=lambda x: abs(x[1]), reverse=True)
            )
            
            logger.info(f"Computed importance for {len(importances)} features")
            return importances
            
        except Exception as e:
            logger.error(f"Permutation importance failed: {e}")
            return {}
    
    def compute_shap(
        self,
        pipeline: Pipeline,
        X: pd.DataFrame,
        y: pd.Series,
        n_samples: int = 1000,
    ) -> Optional[Any]:
        """Compute SHAP values (budget-gated).
        
        Args:
            pipeline: Fitted pipeline
            X: Feature dataframe
            y: Target series
            n_samples: Sample size for SHAP
            
        Returns:
            SHAP values or None
        """
        try:
            import shap
            
            logger.info(f"Computing SHAP values on {n_samples} samples")
            
            # Fit pipeline
            pipeline.fit(X, y)
            X_transformed = pipeline.transform(X)
            
            # Sample for efficiency
            if len(X_transformed) > n_samples:
                indices = np.random.choice(len(X_transformed), n_samples, replace=False)
                X_sample = X_transformed[indices]
            else:
                X_sample = X_transformed
            
            # Create explainer
            task_type = self._detect_task(y)
            if task_type == TaskType.CLASSIFICATION:
                model = RandomForestClassifier(n_estimators=50, random_state=self.config.random_seed)
            else:
                model = RandomForestRegressor(n_estimators=50, random_state=self.config.random_seed)
            
            model.fit(X_transformed, y)
            
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
            
            logger.info("SHAP computation complete")
            return shap_values
            
        except ImportError:
            logger.warning("SHAP not installed, skipping")
            return None
        except Exception as e:
            logger.error(f"SHAP computation failed: {e}")
            return None
    
    def check_leakage(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
    ) -> Dict[str, Any]:
        """Check for data leakage.
        
        Args:
            X_train: Training features (transformed)
            X_test: Test features (transformed)
            y_train: Training target
            
        Returns:
            Leakage report
        """
        logger.info("Checking for leakage")
        
        leakage_report = {
            "has_leakage": False,
            "psi_scores": {},
            "high_psi_features": [],
        }
        
        try:
            # Convert to DataFrame if needed
            if not isinstance(X_train, pd.DataFrame):
                X_train = pd.DataFrame(X_train)
            if not isinstance(X_test, pd.DataFrame):
                X_test = pd.DataFrame(X_test)
            
            # Compute PSI for each feature
            for col in X_train.columns:
                if col in X_test.columns:
                    try:
                        psi = compute_psi(
                            X_train[col].values,
                            X_test[col].values,
                            buckets=10,
                        )
                        leakage_report["psi_scores"][str(col)] = float(psi)
                        
                        # PSI > 0.25 indicates significant drift (potential leakage)
                        if psi > 0.25:
                            leakage_report["high_psi_features"].append(str(col))
                            leakage_report["has_leakage"] = True
                    except:
                        pass
            
            if leakage_report["has_leakage"]:
                logger.warning(
                    f"[!] Potential leakage detected: {len(leakage_report['high_psi_features'])} "
                    f"features with high PSI"
                )
            else:
                logger.info("[OK] No leakage detected")
            
        except Exception as e:
            logger.error(f"Leakage check failed: {e}")
        
        return leakage_report
    
    # === Helper methods ===
    
    def _detect_task(self, y: pd.Series) -> TaskType:
        """Detect task type."""
        from ..insights import detect_task
        return detect_task(y)
    
    def _get_primary_metric(self, task_type: TaskType) -> str:
        """Get primary metric name."""
        if self.config.primary_metric != "auto":
            # For ROC-AUC, we need predict_proba, but our pipelines don't have final estimators
            # Fall back to accuracy for feature engineering pipeline evaluation
            if self.config.primary_metric == "roc_auc":
                return "accuracy"
            return self.config.primary_metric

        if task_type == TaskType.CLASSIFICATION:
            return "accuracy"  # Use accuracy since we don't have predict_proba
        else:
            return "neg_root_mean_squared_error"
    
    def _get_scoring(self, task_type: TaskType) -> Dict[str, str]:
        """Get scoring dictionary for cross_validate."""
        if task_type == TaskType.CLASSIFICATION:
            return {
                "accuracy": "accuracy",
                "precision": "precision_macro",
                "recall": "recall_macro",
                "f1": "f1_macro",
            }
        else:
            return {
                "neg_root_mean_squared_error": "neg_root_mean_squared_error",
                "neg_mean_absolute_error": "neg_mean_absolute_error",
                "r2": "r2",
            }
    
    def _override_splits(self, n_splits: int) -> BaseCrossValidator:
        """Create CV with overridden n_splits."""
        from sklearn.model_selection import KFold, StratifiedKFold
        
        if hasattr(self.cv_strategy, "n_splits"):
            # Clone with new n_splits
            cv_type = type(self.cv_strategy)
            if cv_type == StratifiedKFold:
                return StratifiedKFold(
                    n_splits=n_splits,
                    shuffle=True,
                    random_state=self.config.random_seed,
                )
            else:
                return KFold(
                    n_splits=n_splits,
                    shuffle=True,
                    random_state=self.config.random_seed,
                )
        
        return self.cv_strategy

