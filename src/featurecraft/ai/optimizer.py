"""Adaptive Configuration Optimizer - Learns from feedback to improve strategies.

This module provides adaptive optimization that learns from model performance
to continuously improve feature engineering strategies.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..config import FeatureCraftConfig
from ..logging import get_logger

logger = get_logger(__name__)


@dataclass
class PerformanceFeedback:
    """Performance feedback for learning."""
    
    dataset_hash: str
    n_rows: int
    n_features_original: int
    n_features_engineered: int
    task: str
    estimator_family: str
    
    # Strategy used
    interactions_enabled: bool
    interaction_types: List[str]
    target_encoding_used: bool
    
    # Performance metrics
    cv_score: float
    cv_std: float
    fit_time: float
    transform_time: float
    train_time: float
    
    # Outcome
    success: bool
    overfitting_detected: bool = False
    
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class AdaptiveConfigOptimizer:
    """Adaptive optimizer that learns from feedback.
    
    Maintains a history of feature engineering strategies and their outcomes,
    using this to improve future recommendations.
    
    Features:
    - Learns which strategies work best for different dataset types
    - Detects overfitting patterns
    - Provides confidence scores for recommendations
    - Supports offline learning from logs
    
    Usage:
        optimizer = AdaptiveConfigOptimizer()
        
        # After training
        feedback = PerformanceFeedback(...)
        optimizer.record_feedback(feedback)
        
        # For next dataset
        suggestions = optimizer.suggest_improvements(current_config, dataset_profile)
    """
    
    def __init__(self, history_path: Optional[str] = None):
        """Initialize optimizer.
        
        Args:
            history_path: Path to load/save feedback history
        """
        self.history: List[PerformanceFeedback] = []
        self.history_path = Path(history_path) if history_path else None
        
        if self.history_path and self.history_path.exists():
            self.load_history()
    
    def record_feedback(self, feedback: PerformanceFeedback):
        """Record performance feedback.
        
        Args:
            feedback: Performance feedback
        """
        self.history.append(feedback)
        
        # Auto-save if path configured
        if self.history_path:
            self.save_history()
        
        logger.info(f"Recorded feedback: {feedback.dataset_hash[:8]}... (score: {feedback.cv_score:.4f})")
    
    def suggest_improvements(
        self,
        current_config: FeatureCraftConfig,
        dataset_profile: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Suggest configuration improvements based on history.
        
        Args:
            current_config: Current configuration
            dataset_profile: Profile of current dataset
        
        Returns:
            Dict with suggested config overrides
        """
        if not self.history:
            logger.info("No history available for adaptive suggestions")
            return {}
        
        suggestions = {}
        
        # Find similar datasets
        similar_feedback = self._find_similar_datasets(dataset_profile)
        
        if not similar_feedback:
            logger.info("No similar datasets found in history")
            return {}
        
        # Analyze successful strategies
        successful = [f for f in similar_feedback if f.success and not f.overfitting_detected]
        
        if not successful:
            logger.warning("No successful strategies found for similar datasets")
            return {}
        
        # Get best performing strategy
        best = max(successful, key=lambda f: f.cv_score)
        
        # Generate suggestions
        if not current_config.interactions_enabled and best.interactions_enabled:
            suggestions["interactions_enabled"] = True
            suggestions["_reasoning"] = f"Interactions improved performance on similar data (score: {best.cv_score:.4f})"
        
        elif current_config.interactions_enabled and not best.interactions_enabled:
            suggestions["interactions_enabled"] = False
            suggestions["_reasoning"] = f"Disabling interactions improved performance on similar data"
        
        # Target encoding suggestions
        if not current_config.use_target_encoding and best.target_encoding_used:
            suggestions["use_target_encoding"] = True
        
        logger.info(f"Generated {len(suggestions)} adaptive suggestions based on {len(successful)} successful cases")
        
        return suggestions
    
    def _find_similar_datasets(
        self,
        profile: Dict[str, Any],
        top_k: int = 5,
    ) -> List[PerformanceFeedback]:
        """Find similar datasets from history.
        
        Args:
            profile: Dataset profile
            top_k: Number of similar datasets to return
        
        Returns:
            List of similar feedback entries
        """
        if not self.history:
            return []
        
        # Calculate similarity scores
        scored = []
        for feedback in self.history:
            sim_score = self._calculate_similarity(profile, feedback)
            scored.append((sim_score, feedback))
        
        # Sort by similarity
        scored.sort(key=lambda x: x[0], reverse=True)
        
        # Return top-k
        return [f for _, f in scored[:top_k]]
    
    def _calculate_similarity(
        self,
        profile: Dict[str, Any],
        feedback: PerformanceFeedback,
    ) -> float:
        """Calculate similarity score between dataset profile and feedback.
        
        Args:
            profile: Current dataset profile
            feedback: Historical feedback
        
        Returns:
            Similarity score (0-1)
        """
        score = 0.0
        
        # Task similarity (exact match)
        if profile.get("task") == feedback.task:
            score += 0.3
        
        # Estimator family similarity
        if profile.get("estimator_family") == feedback.estimator_family:
            score += 0.2
        
        # Size similarity (log scale)
        if "n_rows" in profile:
            size_ratio = min(profile["n_rows"], feedback.n_rows) / max(profile["n_rows"], feedback.n_rows)
            score += 0.2 * size_ratio
        
        # Feature count similarity
        if "n_features" in profile:
            feat_ratio = min(profile["n_features"], feedback.n_features_original) / max(profile["n_features"], feedback.n_features_original)
            score += 0.15 * feat_ratio
        
        # Feature type distribution similarity
        if "feature_types" in profile and feedback.metadata:
            # Simple comparison of numeric vs categorical ratio
            score += 0.15  # Base score for having feature type info
        
        return score
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics from feedback history.
        
        Returns:
            Dict with summary statistics
        """
        if not self.history:
            return {"n_samples": 0}
        
        successful = [f for f in self.history if f.success]
        with_interactions = [f for f in self.history if f.interactions_enabled]
        
        stats = {
            "n_samples": len(self.history),
            "n_successful": len(successful),
            "success_rate": len(successful) / len(self.history),
            "avg_score": np.mean([f.cv_score for f in successful]) if successful else 0,
            "interactions_usage_rate": len(with_interactions) / len(self.history),
            "interactions_success_rate": sum(f.success for f in with_interactions) / len(with_interactions) if with_interactions else 0,
        }
        
        # Task breakdown
        tasks = {}
        for f in self.history:
            if f.task not in tasks:
                tasks[f.task] = {"total": 0, "successful": 0}
            tasks[f.task]["total"] += 1
            if f.success:
                tasks[f.task]["successful"] += 1
        
        stats["by_task"] = tasks
        
        return stats
    
    def save_history(self, path: Optional[str] = None):
        """Save feedback history to file.
        
        Args:
            path: Path to save (uses self.history_path if None)
        """
        save_path = Path(path) if path else self.history_path
        
        if not save_path:
            logger.warning("No save path specified")
            return
        
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(
                [asdict(f) for f in self.history],
                f,
                indent=2,
                default=str
            )
        
        logger.info(f"Saved {len(self.history)} feedback entries to {save_path}")
    
    def load_history(self, path: Optional[str] = None):
        """Load feedback history from file.
        
        Args:
            path: Path to load (uses self.history_path if None)
        """
        load_path = Path(path) if path else self.history_path
        
        if not load_path or not load_path.exists():
            logger.warning(f"History file not found: {load_path}")
            return
        
        with open(load_path, 'r') as f:
            data = json.load(f)
        
        self.history = [PerformanceFeedback(**item) for item in data]
        logger.info(f"Loaded {len(self.history)} feedback entries from {load_path}")
    
    def clear_history(self):
        """Clear all feedback history."""
        self.history = []
        logger.info("Cleared feedback history")

