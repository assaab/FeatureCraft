"""Main FeatureCraft Agent orchestration class."""

from __future__ import annotations

import hashlib
import time
import uuid
from typing import Optional
import pandas as pd
from sklearn.model_selection import train_test_split
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from ..config import FeatureCraftConfig
from ..logging import get_logger
from .config import AgentConfig, ComputeBudget
from .types import (
    DatasetFingerprint,
    AgentResult,
    Candidate,
    Baselines,
    RunLedger,
)
from .inspector import Inspector
from .strategist import Strategist
from .composer import Composer
from .evaluator import Evaluator
from .optimizer import Optimizer
from .reporter import Reporter, ArtifactStore

logger = get_logger(__name__)
console = Console()


class FeatureCraftAgent:
    """Intelligent autonomous feature engineering agent.
    
    The agent orchestrates end-to-end automated feature engineering:
    1. Inspects dataset and generates fingerprint
    2. Generates candidate strategies based on heuristics
    3. Evaluates baselines and candidates
    4. Optimizes top candidates iteratively
    5. Exports best pipeline and reports
    
    Example:
        >>> agent = FeatureCraftAgent(config=AgentConfig(
        ...     estimator_family="tree",
        ...     primary_metric="roc_auc",
        ...     time_budget="balanced"
        ... ))
        >>> result = agent.run(X=X, y=y, target_name="target")
        >>> print(f"Best score: {result.best_score:.4f}")
        >>> pipeline = result.load_pipeline()
    """
    
    def __init__(self, config: Optional[AgentConfig] = None):
        """Initialize agent.
        
        Args:
            config: Agent configuration (uses defaults if not provided)
        """
        self.config = config or AgentConfig()
        self.budget = self.config.get_budget()
    
    def run(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        target_name: str,
        entity_column: Optional[str] = None,
        time_column: Optional[str] = None,
    ) -> AgentResult:
        """Run the agent on a dataset.
        
        Args:
            X: Feature dataframe (without target)
            y: Target series
            target_name: Name of target variable
            entity_column: Optional entity/group column for GroupKFold
            time_column: Optional time column for time series
            
        Returns:
            AgentResult with best pipeline and analysis
        """
        # Generate run ID
        run_id = f"run_{uuid.uuid4().hex[:8]}_{int(time.time())}"
        
        # Create artifact store
        artifact_dir = f"{self.config.output_dir}/{run_id}"
        artifact_store = ArtifactStore(artifact_dir)
        
        # Override entity/time columns if provided
        if entity_column:
            self.config.entity_column = entity_column
        if time_column:
            self.config.time_column = time_column
        
        console.print(f"\n[bold cyan][AGENT] FeatureCraft Agent Starting[/bold cyan]")
        console.print(f"[dim]Run ID: {run_id}[/dim]")
        console.print(f"[dim]Dataset: {X.shape[0]} rows x {X.shape[1]} columns[/dim]\n")
        
        start_time = time.time()
        
        try:
            # === Stage 1: Inspect ===
            with self._stage_progress("[INSPECT] Stage 1/6: Inspecting Dataset"):
                inspector = Inspector(FeatureCraftConfig(random_state=self.config.random_seed))
                fingerprint = inspector.fingerprint(X, y)
                issues = inspector.check_data_quality(X, y)
                
                # Check for critical issues
                if any(issue.severity == "ERROR" for issue in issues):
                    console.print(f"[bold red][ERROR] Critical data quality issues found[/bold red]")
                    for issue in issues:
                        if issue.severity == "ERROR":
                            console.print(f"  - {issue.message}")
                    raise ValueError("Critical data quality issues prevent agent execution")
                
                artifact_store.save("fingerprint.json", self._fingerprint_to_dict(fingerprint))
            
            console.print(f"[green][OK][/green] Task: {fingerprint.task_type}")
            console.print(f"[green][OK][/green] Features: {fingerprint.n_numeric} numeric, "
                         f"{fingerprint.n_categorical} categorical\n")
            
            # === Stage 2: Strategize ===
            with self._stage_progress("[STRATEGY] Stage 2/6: Generating Strategies"):
                strategist = Strategist(self.config)
                initial_strategies = strategist.generate_initial_strategies(
                    fingerprint=fingerprint,
                    estimator_family=self.config.estimator_family,
                    budget=self.budget,
                )
                cv_strategy = strategist.select_cv_strategy(fingerprint)
            
            console.print(f"[green][OK][/green] Generated {len(initial_strategies)} strategies\n")
            
            # === Stage 3: Evaluate Baselines ===
            with self._stage_progress("[BASELINE] Stage 3/6: Evaluating Baselines"):
                evaluator = Evaluator(self.config, cv_strategy)
                baseline_raw = evaluator.compute_baseline(X, y, "raw")
                baseline_auto = evaluator.compute_baseline(X, y, "auto")
                baselines = Baselines(raw=baseline_raw, auto=baseline_auto)
                
                artifact_store.save("baselines.json", {
                    "raw": baseline_raw.to_dict(),
                    "auto": baseline_auto.to_dict(),
                })
            
            console.print(f"[green][OK][/green] Baseline (raw): {baseline_raw.cv_score_mean:.4f}")
            console.print(f"[green][OK][/green] Baseline (auto): {baseline_auto.cv_score_mean:.4f}\n")
            
            # === Stage 4: Build & Evaluate Candidates ===
            with self._stage_progress("[PIPELINES] Stage 4/6: Evaluating Candidate Pipelines"):
                composer = Composer(self.config)
                candidates = []
                
                for strategy in initial_strategies:
                    pipeline = composer.build_pipeline(strategy, X, y, fingerprint)
                    result = evaluator.evaluate_pipeline(pipeline, X, y)
                    
                    # Check baseline threshold
                    if result.cv_score_mean >= baseline_auto.cv_score_mean * self.config.baseline_improvement_threshold:
                        candidate = Candidate(strategy=strategy, pipeline=pipeline, result=result)
                        candidates.append(candidate)
                        console.print(
                            f"  [green][OK][/green] {strategy.reasoning[:50]}...: "
                            f"{result.cv_score_mean:.4f}"
                        )
                    else:
                        console.print(
                            f"  [red][X][/red] {strategy.reasoning[:50]}...: "
                            f"{result.cv_score_mean:.4f} (below threshold)"
                        )
            
            if not candidates:
                console.print(f"\n[yellow][!] No pipelines beat baseline. Using auto baseline.[/yellow]")
                # Return baseline result
                return self._create_baseline_result(
                    run_id=run_id,
                    fingerprint=fingerprint,
                    baselines=baselines,
                    artifact_store=artifact_store,
                )
            
            console.print(f"\n[green][OK][/green] {len(candidates)} candidates passed threshold\n")
            
            # Sort candidates by score
            candidates.sort(key=lambda c: c.score, reverse=True)
            best_k = candidates[:3]
            
            # === Stage 5: Optimize ===
            with self._stage_progress("[OPTIMIZE] Stage 5/6: Iterative Optimization"):
                optimizer = Optimizer(self.config, evaluator, artifact_store)
                
                # Greedy forward selection
                console.print("  -> Greedy forward selection...")
                refined = []
                for candidate in best_k:
                    refined_pipeline = optimizer.greedy_forward_selection(
                        base_pipeline=candidate.pipeline,
                        X=X,
                        y=y,
                        budget=self.budget.stage_budget(5),
                    )
                    refined_result = evaluator.evaluate_pipeline(refined_pipeline, X, y)
                    refined.append(Candidate(
                        strategy=candidate.strategy,
                        pipeline=refined_pipeline,
                        result=refined_result,
                    ))
                
                refined.sort(key=lambda c: c.score, reverse=True)
                top_2 = refined[:2]
                
                # Bayesian optimization (if budget allows)
                if self.budget.has_budget_for_bayesian():
                    console.print("  -> Bayesian hyperparameter tuning...")
                    tuned = []
                    for candidate in top_2:
                        tuned_pipeline = optimizer.bayesian_optimize(
                            pipeline=candidate.pipeline,
                            X=X,
                            y=y,
                            n_trials=self.budget.n_bayesian_trials,
                        )
                        tuned_result = evaluator.evaluate_pipeline(tuned_pipeline, X, y)
                        tuned.append(Candidate(
                            strategy=candidate.strategy,
                            pipeline=tuned_pipeline,
                            result=tuned_result,
                        ))
                    top_2 = tuned
                
                # Pruning and consolidation
                console.print("  -> Pruning and consolidation...")
                final_candidates = []
                for candidate in top_2:
                    pruned_pipeline = optimizer.prune_and_consolidate(
                        candidate.pipeline, X, y
                    )
                    final_result = evaluator.evaluate_pipeline(
                        pruned_pipeline, X, y, n_splits=10
                    )
                    final_candidates.append(Candidate(
                        strategy=candidate.strategy,
                        pipeline=pruned_pipeline,
                        result=final_result,
                    ))
                
                final_candidates.sort(key=lambda c: c.score, reverse=True)
                best_candidate = final_candidates[0]
            
            console.print(f"\n[bold green][SUCCESS] Best Pipeline Found[/bold green]")
            console.print(f"  Score: {best_candidate.score:.4f} +/- {best_candidate.result.cv_score_std:.4f}")
            improvement = (best_candidate.score / baseline_auto.cv_score_mean - 1) * 100
            console.print(f"  Improvement: {improvement:+.1f}%\n")
            
            # === Stage 6: Report ===
            with self._stage_progress("[REPORT] Stage 6/6: Generating Reports"):
                # Ablation study
                ablation_results = evaluator.ablation_study(
                    best_candidate.pipeline, X, y
                )
                
                # Permutation importance
                importance_scores = evaluator.permutation_importance(
                    best_candidate.pipeline, X, y, n_repeats=5
                )
                
                # SHAP (if budget allows)
                shap_values = None
                if self.budget.has_budget_for_shap():
                    shap_values = evaluator.compute_shap(
                        best_candidate.pipeline, X, y, n_samples=self.budget.shap_sample_size
                    )
                
                # Leakage check
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=self.config.random_seed
                )
                best_candidate.pipeline.fit(X_train, y_train)
                X_train_t = best_candidate.pipeline.transform(X_train)
                X_test_t = best_candidate.pipeline.transform(X_test)
                leakage_report = evaluator.check_leakage(X_train_t, X_test_t, y_train)
                
                # Create ledger
                ledger = self._create_ledger(
                    run_id=run_id,
                    X=X,
                    y=y,
                    target_name=target_name,
                    fingerprint=fingerprint,
                    cv_strategy=cv_strategy,
                )
                
                # Create result
                result = AgentResult(
                    run_id=run_id,
                    fingerprint=fingerprint,
                    best_strategy=best_candidate.strategy,
                    best_pipeline=best_candidate.pipeline,
                    best_result=best_candidate.result,
                    baseline_raw=baseline_raw,
                    baseline_auto=baseline_auto,
                    ablation_results=ablation_results,
                    importance_scores=importance_scores,
                    shap_values=shap_values,
                    leakage_report=leakage_report,
                    all_candidates=final_candidates,
                    ledger=ledger,
                    artifact_dir=artifact_dir,
                )
                
                # Export artifacts
                reporter = Reporter(self.config, artifact_store)
                reporter.export_artifacts(result)
                
                # Generate reports
                if self.config.generate_markdown_report:
                    reporter.generate_report(result, format="markdown")
                
                if self.config.generate_html_report:
                    reporter.generate_report(result, format="html")
                
                if self.config.generate_json_artifacts:
                    reporter.generate_report(result, format="json")
            
            elapsed_time = time.time() - start_time
            
            console.print(f"\n[bold green][SUCCESS] Agent Completed Successfully[/bold green]")
            console.print(f"[dim]Total time: {elapsed_time:.1f}s[/dim]")
            console.print(f"[dim]Artifacts: {artifact_dir}[/dim]\n")
            
            return result
            
        except Exception as e:
            logger.error(f"Agent execution failed: {e}", exc_info=True)
            console.print(f"\n[bold red][ERROR] Agent Failed: {e}[/bold red]\n")
            raise
    
    def _stage_progress(self, description: str):
        """Context manager for stage progress."""
        from contextlib import contextmanager
        
        @contextmanager
        def progress_context():
            if self.config.verbose:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                ) as progress:
                    task = progress.add_task(description, total=None)
                    yield
                    progress.update(task, completed=True)
            else:
                yield
        
        return progress_context()
    
    def _create_ledger(
        self,
        run_id: str,
        X: pd.DataFrame,
        y: pd.Series,
        target_name: str,
        fingerprint: DatasetFingerprint,
        cv_strategy: any,
    ) -> RunLedger:
        """Create run ledger."""
        dataset_hash = hashlib.md5(
            f"{X.shape}_{list(X.columns)}".encode()
        ).hexdigest()[:16]
        
        cv_splits_hash = hashlib.md5(str(cv_strategy).encode()).hexdigest()[:16]
        
        return RunLedger(
            run_id=run_id,
            dataset_hash=dataset_hash,
            target_name=target_name,
            task_type=fingerprint.task_type,
            estimator_family=self.config.estimator_family,
            random_seed=self.config.random_seed,
            cv_splits_hash=cv_splits_hash,
            config_snapshot=self.config.model_dump(),
        )
    
    def _fingerprint_to_dict(self, fp: DatasetFingerprint) -> dict:
        """Convert fingerprint to dictionary."""
        return {
            "n_rows": fp.n_rows,
            "n_cols": fp.n_cols,
            "task_type": str(fp.task_type),
            "n_numeric": fp.n_numeric,
            "n_categorical": fp.n_categorical,
            "n_text": fp.n_text,
            "n_datetime": fp.n_datetime,
        }
    
    def _create_baseline_result(
        self,
        run_id: str,
        fingerprint: DatasetFingerprint,
        baselines: Baselines,
        artifact_store: ArtifactStore,
    ) -> AgentResult:
        """Create result when no candidates beat baseline."""
        from ..pipeline import AutoFeatureEngineer
        
        # Use auto baseline as "best"
        afe = AutoFeatureEngineer(
            config=FeatureCraftConfig(random_state=self.config.random_seed)
        )
        
        from ..ai.advisor import FeatureStrategy
        baseline_strategy = FeatureStrategy(
            reasoning="Auto baseline (no custom strategies beat threshold)"
        )
        
        return AgentResult(
            run_id=run_id,
            fingerprint=fingerprint,
            best_strategy=baseline_strategy,
            best_pipeline=afe,
            best_result=baselines.auto,
            baseline_raw=baselines.raw,
            baseline_auto=baselines.auto,
            artifact_dir=str(artifact_store.root_dir),
        )

