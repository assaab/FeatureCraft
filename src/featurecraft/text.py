"""Text processing utilities for FeatureCraft."""

from __future__ import annotations

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline

from .logging import get_logger

logger = get_logger(__name__)


class AdaptiveSVD(BaseEstimator, TransformerMixin):
    """SVD that adapts n_components to input feature count."""

    def __init__(self, n_components: int = 100, random_state: int | None = 42):
        self.n_components = n_components
        self.random_state = random_state
        self.svd_: TruncatedSVD | None = None
        self.n_components_: int = 0

    def fit(self, X, y=None):
        """Fit SVD with adaptive component count."""
        n_features = X.shape[1]
        # SVD requires n_components < min(n_samples, n_features)
        self.n_components_ = min(self.n_components, n_features - 1, X.shape[0] - 1)
        if self.n_components_ < 1:
            # Skip SVD if we can't reduce dimensionality
            self.svd_ = None
        else:
            self.svd_ = TruncatedSVD(
                n_components=self.n_components_, random_state=self.random_state
            )
            self.svd_.fit(X)
        return self

    def transform(self, X):
        """Transform using fitted SVD or pass-through."""
        if self.svd_ is None:
            # Return as-is if SVD wasn't applicable
            return X
        return self.svd_.transform(X)


def build_text_pipeline(
    column_name: str,
    max_features: int = 20000,
    svd_components: int | None = None,
    use_hashing: bool = False,
    hashing_features: int = 16384,
    char_ngrams: bool = False,
) -> Pipeline:
    """Build text processing pipeline.

    Args:
        column_name: Name of the text column (for reference)
        max_features: Maximum TF-IDF features
        svd_components: Target SVD components (adaptively adjusted to data)
        use_hashing: Use HashingVectorizer instead of TF-IDF
        hashing_features: Number of features for hashing
        char_ngrams: Use character n-grams instead of word n-grams
    """
    steps = []
    
    if use_hashing:
        # Memory-efficient hashing vectorizer
        if char_ngrams:
            vectorizer = HashingVectorizer(
                n_features=hashing_features,
                analyzer="char_wb",
                ngram_range=(3, 5),
                alternate_sign=False,
            )
        else:
            vectorizer = HashingVectorizer(
                n_features=hashing_features,
                ngram_range=(1, 2),
                alternate_sign=False,
            )
        steps.append(("hasher", vectorizer))
        logger.debug(f"Using HashingVectorizer with {hashing_features} features for '{column_name}'")
    else:
        # Standard TF-IDF
        if char_ngrams:
            vectorizer = TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                max_features=max_features,
            )
        else:
            vectorizer = TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=max_features,
            )
        steps.append(("tfidf", vectorizer))
        logger.debug(f"Using TF-IDF with max_features={max_features} for '{column_name}'")
    
    if svd_components:
        # Use AdaptiveSVD to handle cases where vectorizer produces fewer features
        steps.append(("svd", AdaptiveSVD(n_components=svd_components, random_state=42)))
    
    return Pipeline(steps=steps)
