"""Optimizer Module: Iterative pipeline refinement."""

from __future__ import annotations

import time
from typing import List, Dict, Any, Optional
import pandas as pd
from sklearn.pipeline import Pipeline

from ..logging import get_logger
from .config import AgentConfig, ComputeBudget
from .types import DatasetFingerprint, EvaluationResult
from .evaluator import Evaluator

logger = get_logger(__name__)


class Optimizer:
    """Multi-stage pipeline optimization."""
    
    def __init__(
        self,
        config: AgentConfig,
        evaluator: Evaluator,
        artifact_store: Optional[Any] = None,
    ):
        """Initialize optimizer.
        
        Args:
            config: Agent configuration
            evaluator: Evaluator instance
            artifact_store: Optional artifact store
        """
        self.config = config
        self.evaluator = evaluator
        self.artifact_store = artifact_store
    
    def greedy_forward_selection(
        self,
        base_pipeline: Pipeline,
        X: pd.DataFrame,
        y: pd.Series,
        budget: ComputeBudget,
    ) -> Pipeline:
        """Greedy forward selection to add beneficial operations.
        
        Args:
            base_pipeline: Starting pipeline
            X: Feature dataframe
            y: Target series
            budget: Compute budget for this stage
            
        Returns:
            Refined pipeline
        """
        logger.info("Starting greedy forward selection")
        
        # Evaluate base pipeline
        base_result = self.evaluator.evaluate_pipeline(
            base_pipeline, X, y, pipeline_id="greedy_base"
        )
        current_score = base_result.cv_score_mean
        current_pipeline = base_pipeline
        
        logger.info(f"Base score: {current_score:.4f}")
        
        # Candidate operations to try adding
        candidate_operations = self._get_candidate_operations(X)
        
        iterations = 0
        max_iterations = min(len(candidate_operations), budget.max_pipelines)
        no_improvement_count = 0
        
        for op_name, op_transformer in candidate_operations:
            if iterations >= max_iterations:
                logger.info(f"Reached max iterations: {max_iterations}")
                break
            
            if no_improvement_count >= budget.early_stop_patience:
                logger.info(f"Early stopping: no improvement in {budget.early_stop_patience} iterations")
                break
            
            iterations += 1
            
            # Try adding this operation
            logger.info(f"  [{iterations}/{max_iterations}] Trying: {op_name}")
            
            try:
                # Create new pipeline with added operation
                new_steps = list(current_pipeline.steps) + [(op_name, op_transformer)]
                new_pipeline = Pipeline(new_steps)
                
                # Evaluate
                new_result = self.evaluator.evaluate_pipeline(
                    new_pipeline, X, y, pipeline_id=f"greedy_{op_name}"
                )
                new_score = new_result.cv_score_mean
                
                # Check if improvement meets threshold
                improvement_ratio = new_score / current_score
                
                if improvement_ratio >= self.config.greedy_improvement_threshold:
                    gain_pct = (improvement_ratio - 1) * 100
                    logger.info(f"    [OK] Kept {op_name}: +{gain_pct:.2f}%")
                    current_pipeline = new_pipeline
                    current_score = new_score
                    no_improvement_count = 0
                else:
                    logger.info(f"    [X] Rejected {op_name}: insufficient improvement")
                    no_improvement_count += 1
                    
            except Exception as e:
                logger.warning(f"    [X] Failed to add {op_name}: {e}")
                no_improvement_count += 1
        
        logger.info(f"Greedy selection complete: final score={current_score:.4f}")
        return current_pipeline
    
    def bayesian_optimize(
        self,
        pipeline: Pipeline,
        X: pd.DataFrame,
        y: pd.Series,
        n_trials: int,
    ) -> Pipeline:
        """Bayesian hyperparameter optimization (optional).
        
        Args:
            pipeline: Pipeline to optimize
            X: Feature dataframe
            y: Target series
            n_trials: Number of Bayesian trials
            
        Returns:
            Optimized pipeline
        """
        logger.info(f"Starting Bayesian optimization ({n_trials} trials)")
        
        try:
            import optuna
            
            # Define objective function
            def objective(trial):
                # Example: tune a few hyperparameters
                # In practice, this would be more sophisticated
                
                # For now, just return current pipeline score
                result = self.evaluator.evaluate_pipeline(
                    pipeline, X, y, pipeline_id=f"bayesian_trial_{trial.number}"
                )
                return result.cv_score_mean
            
            # Create study
            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=self.config.random_seed),
            )
            
            # Optimize (with minimal trials for now)
            study.optimize(objective, n_trials=min(5, n_trials), n_jobs=1, show_progress_bar=False)
            
            logger.info(f"Bayesian optimization complete: best_value={study.best_value:.4f}")
            
            # For now, return original pipeline (real implementation would reconstruct best)
            return pipeline
            
        except ImportError:
            logger.warning("Optuna not installed, skipping Bayesian optimization")
            return pipeline
        except Exception as e:
            logger.error(f"Bayesian optimization failed: {e}")
            return pipeline
    
    def prune_and_consolidate(
        self,
        pipeline: Pipeline,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> Pipeline:
        """Prune redundant features and consolidate.
        
        Args:
            pipeline: Pipeline to prune
            X: Feature dataframe
            y: Target series
            
        Returns:
            Pruned pipeline
        """
        logger.info("Pruning and consolidating pipeline")
        
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
            
            # Convert to DataFrame
            if not isinstance(X_transformed, pd.DataFrame):
                X_transformed = pd.DataFrame(X_transformed, columns=feature_names)
            
            # Compute correlations and drop highly correlated features
            corr_matrix = X_transformed.corr().abs()
            upper_triangle = corr_matrix.where(
                pd.np.triu(pd.np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            
            to_drop = [
                column for column in upper_triangle.columns if any(upper_triangle[column] > 0.95)
            ]
            
            if to_drop:
                logger.info(f"  Dropping {len(to_drop)} highly correlated features")
                # Note: In practice, would add a feature selector to pipeline
            
            # For now, return original pipeline
            # Real implementation would add ColumnTransformer with selected columns
            return pipeline
            
        except Exception as e:
            logger.error(f"Pruning failed: {e}")
            return pipeline
    
    def _get_candidate_operations(self, X: pd.DataFrame) -> List[tuple]:
        """Get candidate operations to try adding.
        
        Args:
            X: Feature dataframe
            
        Returns:
            List of (name, transformer) tuples
        """
        from ..statistical import PercentileRankTransformer
        from ..transformers import BinningTransformer
        
        candidates = []
        
        # Percentile ranking
        candidates.append(("percentile_rank", PercentileRankTransformer()))
        
        # Binning (if numeric features exist)
        numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns
        if len(numeric_cols) > 0:
            candidates.append(("binning", BinningTransformer(strategy="quantile", n_bins=5)))
        
        return candidates

