"""Telemetry and observability for AI-powered feature engineering."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from .schemas import AICallMetadata
from ..logging import get_logger

logger = get_logger(__name__)


# ============================================================================
# Telemetry Logger
# ============================================================================

class TelemetryLogger:
    """Logger for AI call metadata and observability.
    
    This class logs AI call metadata to a JSONL file for analysis,
    monitoring, and debugging.
    
    Example:
        >>> telemetry = TelemetryLogger("logs/ai_telemetry.jsonl")
        >>> telemetry.log(metadata)
        >>> stats = telemetry.get_stats()
        >>> print(f"Total cost: ${stats['total_cost_usd']:.2f}")
    """
    
    def __init__(self, log_path: str | None = None):
        """Initialize telemetry logger.
        
        Args:
            log_path: Path to JSONL log file (creates if doesn't exist)
        """
        self.log_path = log_path or "logs/ai_telemetry.jsonl"
        
        # Create directory if needed
        Path(self.log_path).parent.mkdir(parents=True, exist_ok=True)
    
    def log(self, metadata: AICallMetadata) -> None:
        """Log AI call metadata to file.
        
        Args:
            metadata: AI call metadata to log
        """
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(metadata.to_dict()) + "\n")
            
            logger.debug(f"Logged AI call: {metadata.tokens_used} tokens, ${metadata.cost_usd:.4f}")
        except Exception as e:
            logger.warning(f"Failed to log telemetry: {e}")
    
    def get_stats(self) -> dict[str, Any]:
        """Get aggregate statistics from telemetry log.
        
        Returns:
            Dict with aggregate stats (total_calls, total_tokens, total_cost, etc.)
            
        Example:
            >>> stats = telemetry.get_stats()
            >>> print(f"Total calls: {stats['total_calls']}")
            >>> print(f"Average latency: {stats['avg_latency_ms']}ms")
        """
        if not os.path.exists(self.log_path):
            return {
                "total_calls": 0,
                "total_tokens": 0,
                "total_cost_usd": 0.0,
                "avg_latency_ms": 0,
                "cache_hit_rate": 0.0,
            }
        
        calls = []
        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        calls.append(json.loads(line))
        except Exception as e:
            logger.warning(f"Failed to read telemetry log: {e}")
            return {}
        
        if not calls:
            return {"total_calls": 0}
        
        total_calls = len(calls)
        total_tokens = sum(c.get("tokens_used", 0) for c in calls)
        total_cost = sum(c.get("cost_usd", 0.0) for c in calls)
        total_latency = sum(c.get("latency_ms", 0) for c in calls)
        cache_hits = sum(1 for c in calls if c.get("cache_hit", False))
        
        # Validation stats
        pass_count = sum(1 for c in calls if c.get("validator_status") == "pass")
        fail_count = sum(1 for c in calls if c.get("validator_status") == "fail")
        
        return {
            "total_calls": total_calls,
            "total_tokens": total_tokens,
            "total_cost_usd": round(total_cost, 4),
            "avg_tokens_per_call": round(total_tokens / total_calls, 1),
            "avg_latency_ms": round(total_latency / total_calls, 1),
            "avg_cost_per_call_usd": round(total_cost / total_calls, 4),
            "cache_hit_rate": round(cache_hits / total_calls, 3) if total_calls > 0 else 0.0,
            "validation_pass_rate": round(pass_count / total_calls, 3) if total_calls > 0 else 0.0,
            "validation_fail_count": fail_count,
        }
    
    def get_recent_calls(self, n: int = 10) -> list[dict[str, Any]]:
        """Get N most recent calls.
        
        Args:
            n: Number of recent calls to return
            
        Returns:
            List of call metadata dicts
        """
        if not os.path.exists(self.log_path):
            return []
        
        calls = []
        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        calls.append(json.loads(line))
        except Exception as e:
            logger.warning(f"Failed to read telemetry log: {e}")
            return []
        
        return calls[-n:]
    
    def clear(self) -> None:
        """Clear telemetry log file."""
        if os.path.exists(self.log_path):
            os.remove(self.log_path)
            logger.info(f"Cleared telemetry log: {self.log_path}")


# ============================================================================
# Singleton Instance
# ============================================================================

_global_telemetry: TelemetryLogger | None = None


def get_telemetry_logger(log_path: str | None = None) -> TelemetryLogger:
    """Get global telemetry logger instance.
    
    Args:
        log_path: Path to log file (uses default if None)
        
    Returns:
        TelemetryLogger instance
        
    Example:
        >>> telemetry = get_telemetry_logger()
        >>> telemetry.log(metadata)
    """
    global _global_telemetry
    
    if _global_telemetry is None:
        _global_telemetry = TelemetryLogger(log_path)
    
    return _global_telemetry


def log_ai_call(metadata: AICallMetadata) -> None:
    """Log AI call metadata (convenience function).
    
    Args:
        metadata: AI call metadata
        
    Example:
        >>> metadata = AICallMetadata(...)
        >>> log_ai_call(metadata)
    """
    telemetry = get_telemetry_logger()
    telemetry.log(metadata)


def get_telemetry_stats() -> dict[str, Any]:
    """Get aggregate telemetry stats (convenience function).
    
    Returns:
        Dict with aggregate statistics
        
    Example:
        >>> stats = get_telemetry_stats()
        >>> print(f"Total AI cost: ${stats['total_cost_usd']}")
    """
    telemetry = get_telemetry_logger()
    return telemetry.get_stats()


def reset_telemetry() -> None:
    """Reset telemetry by clearing the log file.
    
    Example:
        >>> reset_telemetry()
        >>> stats = get_telemetry_stats()
        >>> assert stats['total_calls'] == 0
    """
    telemetry = get_telemetry_logger()
    if os.path.exists(telemetry.log_path):
        try:
            os.remove(telemetry.log_path)
            logger.info("Telemetry log reset")
        except Exception as e:
            logger.warning(f"Failed to reset telemetry: {e}")
