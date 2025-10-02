"""Dimensionality reduction utilities for FeatureCraft."""

from __future__ import annotations

from typing import Optional

from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA, TruncatedSVD

from .logging import get_logger

logger = get_logger(__name__)

# Try importing UMAP
try:
    from umap import UMAP

    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    logger.debug("UMAP not installed. Use: pip install umap-learn")


def build_reducer(
    kind: str,
    n_components: Optional[int] = None,
    variance: Optional[float] = None,
    random_state: int = 42,
) -> Optional[BaseEstimator]:
    """Build dimensionality reduction transformer.

    Args:
        kind: Reducer type: 'pca', 'svd', 'umap', or None
        n_components: Number of components
        variance: Explained variance ratio (PCA only, alternative to n_components)
        random_state: Random seed

    Returns:
        Reducer instance or None if disabled/invalid
    """
    if kind is None or kind.lower() == "none":
        return None

    kind = kind.lower()

    if kind == "pca":
        if variance is not None:
            logger.info(f"Creating PCA with variance threshold: {variance}")
            return PCA(n_components=variance, random_state=random_state)
        elif n_components is not None:
            logger.info(f"Creating PCA with {n_components} components")
            return PCA(n_components=n_components, random_state=random_state)
        else:
            logger.warning("PCA requested but no n_components or variance specified. Skipping.")
            return None

    elif kind == "svd":
        if n_components is None:
            logger.warning("SVD requested but no n_components specified. Skipping.")
            return None
        logger.info(f"Creating TruncatedSVD with {n_components} components")
        return TruncatedSVD(n_components=n_components, random_state=random_state)

    elif kind == "umap":
        if not HAS_UMAP:
            logger.warning(
                "UMAP requested but not installed. " "Install with: pip install umap-learn. Skipping."
            )
            return None
        if n_components is None:
            n_components = 50
            logger.info(f"UMAP: using default {n_components} components")
        logger.info(f"Creating UMAP with {n_components} components")
        return UMAP(n_components=n_components, random_state=random_state)

    else:
        logger.warning(f"Unknown reducer kind: {kind}. Skipping.")
        return None

