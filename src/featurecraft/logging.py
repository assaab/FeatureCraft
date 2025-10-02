"""Logging utilities for FeatureCraft."""

from __future__ import annotations

import logging


def get_logger(name: str | None = None) -> logging.Logger:
    """Get a logger instance with consistent formatting."""
    logger = logging.getLogger(name or "featurecraft")
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "[%(asctime)s] %(levelname)s %(name)s - %(message)s", "%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
