"""Dimensionality reduction transformers for feature engineering.

This module provides comprehensive dimensionality reduction techniques for:
- Feature compression and decorrelation (PCA, SVD)
- Supervised dimensionality reduction (LDA)
- Non-linear manifold learning (t-SNE, UMAP)
- Deep learning-based reduction (Autoencoders)

Use Cases:
- Compress high-dimensional data
- Remove multicollinearity
- Visualize complex datasets
- Improve model performance by reducing noise
- Extract latent representations

Supported Methods:
- PCA: Linear, fast, decorrelates features
- LDA: Supervised, maximizes class separation
- t-SNE: Non-linear, preserves local structure, great for visualization
- UMAP: Non-linear, faster than t-SNE, preserves global structure
- Autoencoder: Deep learning, learns non-linear manifolds
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from .logging import get_logger

logger = get_logger(__name__)

# Optional dependencies
try:
    from umap import UMAP
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    logger.debug("UMAP not installed. Install with: pip install umap-learn")

try:
    import tensorflow as tf
    from tensorflow import keras
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False
    logger.debug("TensorFlow not installed. Install with: pip install tensorflow")


class DimensionalityReducer(BaseEstimator, TransformerMixin):
    """Unified dimensionality reduction transformer with multiple methods.
    
    Supports PCA, LDA, t-SNE, UMAP, and Autoencoders for reducing feature dimensions
    while preserving important information. Can be used for compression, visualization,
    and improving downstream model performance.
    
    Methods Comparison:
    
    | Method      | Supervised | Linear | Scalable | Use Case                    |
    |-------------|-----------|--------|----------|----------------------------|
    | PCA         | No        | Yes    | Yes      | Decorrelation, compression |
    | LDA         | Yes       | Yes    | Yes      | Class separation           |
    | t-SNE       | No        | No     | No       | Visualization (2D/3D)      |
    | UMAP        | No        | No     | Yes      | Faster t-SNE alternative   |
    | Autoencoder | No        | No     | Yes      | Non-linear compression     |
    
    Example:
        >>> # PCA for compression
        >>> reducer = DimensionalityReducer(
        ...     method='pca',
        ...     n_components=10,
        ...     explained_variance_threshold=0.95
        ... )
        >>> reducer.fit(X_train)
        >>> X_reduced = reducer.transform(X_test)
        
        >>> # LDA for supervised reduction
        >>> reducer = DimensionalityReducer(
        ...     method='lda',
        ...     n_components=2
        ... )
        >>> reducer.fit(X_train, y_train)
        >>> X_reduced = reducer.transform(X_test)
        
        >>> # t-SNE for visualization
        >>> reducer = DimensionalityReducer(
        ...     method='tsne',
        ...     n_components=2,
        ...     perplexity=30
        ... )
        >>> reducer.fit(X_train)
        >>> X_viz = reducer.transform(X_test)
    """
    
    def __init__(
        self,
        columns: Sequence[str] | None = None,
        method: Literal['pca', 'lda', 'tsne', 'umap', 'autoencoder'] = 'pca',
        n_components: int | None = None,
        keep_original: bool = False,
        scale_features: bool = True,
        
        # PCA specific
        pca_explained_variance_threshold: float | None = None,
        pca_whiten: bool = False,
        pca_svd_solver: str = 'auto',
        
        # LDA specific
        lda_solver: str = 'svd',
        lda_shrinkage: str | float | None = None,
        
        # t-SNE specific
        tsne_perplexity: float = 30.0,
        tsne_learning_rate: float | str = 'auto',
        tsne_metric: str = 'euclidean',
        tsne_init: str = 'pca',
        tsne_n_iter: int = 1000,
        
        # UMAP specific
        umap_n_neighbors: int = 15,
        umap_min_dist: float = 0.1,
        umap_metric: str = 'euclidean',
        
        # Autoencoder specific
        autoencoder_hidden_layers: list[int] | None = None,
        autoencoder_activation: str = 'relu',
        autoencoder_epochs: int = 50,
        autoencoder_batch_size: int = 32,
        autoencoder_dropout: float = 0.2,
        
        # General
        random_state: int = 42,
        feature_prefix: str = 'reduced',
        verbose: int = 0,
    ) -> None:
        """Initialize dimensionality reducer.
        
        Args:
            columns: Columns to use (None = all numeric)
            method: Reduction method - 'pca', 'lda', 'tsne', 'umap', 'autoencoder'
            n_components: Number of dimensions to reduce to
            keep_original: Keep original features alongside reduced features
            scale_features: Standardize features before reduction
            
            # PCA Parameters
            pca_explained_variance_threshold: Keep components until variance threshold (overrides n_components)
            pca_whiten: Whiten components (unit variance)
            pca_svd_solver: SVD solver - 'auto', 'full', 'arpack', 'randomized'
            
            # LDA Parameters
            lda_solver: Solver - 'svd', 'lsqr', 'eigen'
            lda_shrinkage: Shrinkage parameter (lsqr/eigen only)
            
            # t-SNE Parameters
            tsne_perplexity: Balance between local and global structure (5-50)
            tsne_learning_rate: Learning rate (typically 10-1000 or 'auto')
            tsne_metric: Distance metric
            tsne_init: Initialization - 'random', 'pca'
            tsne_n_iter: Number of iterations
            
            # UMAP Parameters
            umap_n_neighbors: Size of local neighborhood (2-100)
            umap_min_dist: Minimum distance between points (0-1)
            umap_metric: Distance metric
            
            # Autoencoder Parameters
            autoencoder_hidden_layers: List of hidden layer sizes (e.g., [128, 64])
            autoencoder_activation: Activation function - 'relu', 'tanh', 'sigmoid'
            autoencoder_epochs: Training epochs
            autoencoder_batch_size: Batch size
            autoencoder_dropout: Dropout rate
            
            # General
            random_state: Random seed
            feature_prefix: Prefix for reduced feature names
            verbose: Verbosity level (0=silent, 1=progress, 2=debug)
        """
        self.columns = columns
        self.method = method
        self.n_components = n_components
        self.keep_original = keep_original
        self.scale_features = scale_features
        
        # PCA
        self.pca_explained_variance_threshold = pca_explained_variance_threshold
        self.pca_whiten = pca_whiten
        self.pca_svd_solver = pca_svd_solver
        
        # LDA
        self.lda_solver = lda_solver
        self.lda_shrinkage = lda_shrinkage
        
        # t-SNE
        self.tsne_perplexity = tsne_perplexity
        self.tsne_learning_rate = tsne_learning_rate
        self.tsne_metric = tsne_metric
        self.tsne_init = tsne_init
        self.tsne_n_iter = tsne_n_iter
        
        # UMAP
        self.umap_n_neighbors = umap_n_neighbors
        self.umap_min_dist = umap_min_dist
        self.umap_metric = umap_metric
        
        # Autoencoder
        self.autoencoder_hidden_layers = autoencoder_hidden_layers or [128, 64]
        self.autoencoder_activation = autoencoder_activation
        self.autoencoder_epochs = autoencoder_epochs
        self.autoencoder_batch_size = autoencoder_batch_size
        self.autoencoder_dropout = autoencoder_dropout
        
        # General
        self.random_state = random_state
        self.feature_prefix = feature_prefix
        self.verbose = verbose
        
        # Fitted attributes
        self.columns_: list[str] = []
        self.scaler_: StandardScaler | None = None
        self.reducer_: Any = None
        self.n_components_: int = 0
        self.feature_names_out_: list[str] = []
        self.explained_variance_: np.ndarray | None = None
        self.explained_variance_ratio_: np.ndarray | None = None
    
    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "DimensionalityReducer":
        """Fit dimensionality reduction on data.
        
        Args:
            X: Input DataFrame
            y: Target (required for LDA, optional for others)
            
        Returns:
            Self
        """
        df = pd.DataFrame(X)
        
        # Determine columns to use
        if self.columns is None:
            self.columns_ = list(df.select_dtypes(include=[np.number]).columns)
        else:
            self.columns_ = [c for c in self.columns if c in df.columns]
        
        if not self.columns_:
            logger.warning("No numeric columns found for dimensionality reduction.")
            return self
        
        # Extract data
        X_data = df[self.columns_].values
        
        # Handle missing values
        if np.isnan(X_data).any():
            logger.warning("Missing values detected. Imputing with column means.")
            col_means = np.nanmean(X_data, axis=0)
            for i in range(X_data.shape[1]):
                X_data[np.isnan(X_data[:, i]), i] = col_means[i]
        
        # Scale features if requested
        if self.scale_features:
            self.scaler_ = StandardScaler()
            X_data = self.scaler_.fit_transform(X_data)
        
        # Validate n_components
        if self.n_components is None:
            # Auto-select based on method
            if self.method in ['tsne', 'umap']:
                self.n_components = 2  # Default for visualization
            elif self.method == 'pca' and self.pca_explained_variance_threshold is not None:
                # Will be determined during PCA fit
                self.n_components = min(X_data.shape)
            else:
                # Default to 10 or half the features, whichever is smaller
                self.n_components = min(10, X_data.shape[1] // 2)
        
        # Fit reduction method
        try:
            if self.method == 'pca':
                self._fit_pca(X_data)
            elif self.method == 'lda':
                if y is None:
                    raise ValueError("LDA requires target labels (y). Please provide y parameter.")
                self._fit_lda(X_data, y)
            elif self.method == 'tsne':
                self._fit_tsne(X_data)
            elif self.method == 'umap':
                self._fit_umap(X_data)
            elif self.method == 'autoencoder':
                self._fit_autoencoder(X_data)
            else:
                raise ValueError(
                    f"Unknown reduction method: '{self.method}'. "
                    f"Choose from: pca, lda, tsne, umap, autoencoder"
                )
        except Exception as e:
            logger.error(f"Dimensionality reduction fit failed: {e}")
            raise
        
        # Generate feature names
        self._generate_feature_names()
        
        logger.info(
            f"Fitted {self.method} reduction from {len(self.columns_)} to "
            f"{self.n_components_} components"
        )
        
        return self
    
    def _fit_pca(self, X: np.ndarray) -> None:
        """Fit PCA."""
        if self.pca_explained_variance_threshold is not None:
            # Use variance threshold
            self.reducer_ = PCA(
                n_components=self.pca_explained_variance_threshold,
                whiten=self.pca_whiten,
                svd_solver=self.pca_svd_solver,
                random_state=self.random_state,
            )
        else:
            # Use fixed number of components
            n_comp = min(self.n_components, X.shape[0], X.shape[1])
            self.reducer_ = PCA(
                n_components=n_comp,
                whiten=self.pca_whiten,
                svd_solver=self.pca_svd_solver,
                random_state=self.random_state,
            )
        
        self.reducer_.fit(X)
        self.n_components_ = self.reducer_.n_components_
        
        # Store variance information
        self.explained_variance_ = self.reducer_.explained_variance_
        self.explained_variance_ratio_ = self.reducer_.explained_variance_ratio_
        
        logger.info(
            f"PCA: {self.n_components_} components explain "
            f"{self.explained_variance_ratio_.sum():.2%} of variance"
        )
    
    def _fit_lda(self, X: np.ndarray, y: pd.Series) -> None:
        """Fit Linear Discriminant Analysis."""
        y_values = y.values if isinstance(y, pd.Series) else y
        
        # LDA n_components must be < n_classes
        n_classes = len(np.unique(y_values))
        n_comp = min(self.n_components, n_classes - 1, X.shape[1])
        
        if n_comp < 1:
            raise ValueError(
                f"LDA requires at least 2 classes. Found {n_classes} classes."
            )
        
        self.reducer_ = LinearDiscriminantAnalysis(
            n_components=n_comp,
            solver=self.lda_solver,
            shrinkage=self.lda_shrinkage,
        )
        
        self.reducer_.fit(X, y_values)
        self.n_components_ = self.reducer_.n_components or n_comp
        
        # Store variance information if available
        if hasattr(self.reducer_, 'explained_variance_ratio_'):
            self.explained_variance_ratio_ = self.reducer_.explained_variance_ratio_
            logger.info(
                f"LDA: {self.n_components_} components explain "
                f"{self.explained_variance_ratio_.sum():.2%} of variance"
            )
    
    def _fit_tsne(self, X: np.ndarray) -> None:
        """Fit t-SNE."""
        # t-SNE typically used for 2D/3D visualization
        n_comp = min(self.n_components, 3)
        
        # Validate perplexity
        max_perplexity = (X.shape[0] - 1) / 3
        perplexity = min(self.tsne_perplexity, max_perplexity)
        
        self.reducer_ = TSNE(
            n_components=n_comp,
            perplexity=perplexity,
            learning_rate=self.tsne_learning_rate,
            metric=self.tsne_metric,
            init=self.tsne_init,
            n_iter=self.tsne_n_iter,
            random_state=self.random_state,
            verbose=self.verbose,
        )
        
        # t-SNE doesn't have separate fit/transform
        # We'll store the data for transform
        self._X_fit = X
        self.n_components_ = n_comp
        
        logger.info(f"t-SNE configured with {n_comp} components (perplexity={perplexity})")
    
    def _fit_umap(self, X: np.ndarray) -> None:
        """Fit UMAP."""
        if not HAS_UMAP:
            raise ImportError(
                "UMAP not installed. Install with: pip install umap-learn"
            )
        
        n_comp = min(self.n_components, X.shape[1])
        
        # Validate n_neighbors
        n_neighbors = min(self.umap_n_neighbors, X.shape[0] - 1)
        
        self.reducer_ = UMAP(
            n_components=n_comp,
            n_neighbors=n_neighbors,
            min_dist=self.umap_min_dist,
            metric=self.umap_metric,
            random_state=self.random_state,
            verbose=self.verbose > 0,
        )
        
        self.reducer_.fit(X)
        self.n_components_ = n_comp
        
        logger.info(f"UMAP fitted with {n_comp} components")
    
    def _fit_autoencoder(self, X: np.ndarray) -> None:
        """Fit Autoencoder."""
        if not HAS_TENSORFLOW:
            raise ImportError(
                "TensorFlow not installed. Install with: pip install tensorflow"
            )
        
        n_features = X.shape[1]
        n_comp = min(self.n_components, n_features // 2)
        
        # Build encoder
        encoder_layers = []
        encoder_layers.append(keras.layers.Input(shape=(n_features,)))
        
        # Add hidden layers
        for units in self.autoencoder_hidden_layers:
            encoder_layers.append(keras.layers.Dense(units, activation=self.autoencoder_activation))
            encoder_layers.append(keras.layers.Dropout(self.autoencoder_dropout))
        
        # Bottleneck layer
        encoder_layers.append(keras.layers.Dense(n_comp, activation='linear', name='bottleneck'))
        
        # Build decoder (mirror of encoder)
        decoder_layers = []
        for units in reversed(self.autoencoder_hidden_layers):
            decoder_layers.append(keras.layers.Dense(units, activation=self.autoencoder_activation))
            decoder_layers.append(keras.layers.Dropout(self.autoencoder_dropout))
        
        decoder_layers.append(keras.layers.Dense(n_features, activation='linear'))
        
        # Complete autoencoder
        inputs = keras.layers.Input(shape=(n_features,))
        x = inputs
        for layer in encoder_layers[1:]:  # Skip input layer
            x = layer(x)
        encoded = x
        for layer in decoder_layers:
            x = layer(x)
        
        autoencoder = keras.Model(inputs, x, name='autoencoder')
        
        # Encoder model for transform
        encoder = keras.Model(inputs, encoded, name='encoder')
        
        # Compile and train
        autoencoder.compile(
            optimizer='adam',
            loss='mse',
        )
        
        if self.verbose > 0:
            logger.info("Training autoencoder...")
        
        autoencoder.fit(
            X, X,
            epochs=self.autoencoder_epochs,
            batch_size=self.autoencoder_batch_size,
            shuffle=True,
            verbose=self.verbose,
            validation_split=0.1,
        )
        
        self.reducer_ = encoder
        self.n_components_ = n_comp
        
        logger.info(f"Autoencoder trained with {n_comp} latent dimensions")
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data to reduced dimensions.
        
        Args:
            X: Input DataFrame
            
        Returns:
            DataFrame with reduced dimensions
        """
        df = pd.DataFrame(X).copy()
        
        if not self.columns_ or self.reducer_ is None:
            return df
        
        # Extract data
        X_data = df[self.columns_].values
        
        # Handle missing values
        if np.isnan(X_data).any():
            col_means = np.nanmean(X_data, axis=0)
            for i in range(X_data.shape[1]):
                X_data[np.isnan(X_data[:, i]), i] = col_means[i]
        
        # Scale if needed
        if self.scaler_ is not None:
            X_data = self.scaler_.transform(X_data)
        
        # Transform based on method
        if self.method == 'tsne':
            # t-SNE doesn't support transform, must fit_transform
            logger.warning("t-SNE doesn't support transform(). Using fit_transform() on new data.")
            X_reduced = self.reducer_.fit_transform(X_data)
        elif self.method == 'autoencoder':
            X_reduced = self.reducer_.predict(X_data, verbose=0)
        else:
            X_reduced = self.reducer_.transform(X_data)
        
        # Create output DataFrame
        if self.keep_original:
            # Keep original features
            result = df.copy()
        else:
            # Remove original features
            result = df.drop(columns=self.columns_)
        
        # Add reduced features
        for i in range(self.n_components_):
            result[f'{self.feature_prefix}_{i}'] = X_reduced[:, i]
        
        return result
    
    def _generate_feature_names(self) -> None:
        """Generate output feature names."""
        self.feature_names_out_ = [
            f'{self.feature_prefix}_{i}' for i in range(self.n_components_)
        ]
    
    def get_feature_names_out(self, input_features=None) -> list[str]:
        """Get output feature names.
        
        Returns:
            List of feature names
        """
        return self.feature_names_out_
    
    def get_reduction_info(self) -> dict[str, Any]:
        """Get dimensionality reduction metadata.
        
        Returns:
            Dict with reduction information
        """
        info = {
            'method': self.method,
            'n_components': self.n_components_,
            'n_features_in': len(self.columns_),
            'feature_names_in': self.columns_,
        }
        
        if self.explained_variance_ratio_ is not None:
            info['explained_variance_ratio'] = self.explained_variance_ratio_
            info['total_explained_variance'] = self.explained_variance_ratio_.sum()
        
        if self.explained_variance_ is not None:
            info['explained_variance'] = self.explained_variance_
        
        return info


class MultiMethodDimensionalityReducer(BaseEstimator, TransformerMixin):
    """Apply multiple dimensionality reduction methods simultaneously.
    
    Creates an ensemble of reduced representations using different methods,
    which can capture both linear and non-linear structures in the data.
    
    Use Cases:
    - Combine linear (PCA) and non-linear (UMAP) views
    - Create diverse features for ensemble models
    - Robust feature extraction
    
    Example:
        >>> reducer = MultiMethodDimensionalityReducer(
        ...     methods=['pca', 'umap'],
        ...     n_components_pca=10,
        ...     n_components_umap=5
        ... )
        >>> reducer.fit(X_train)
        >>> X_reduced = reducer.transform(X_test)
    """
    
    def __init__(
        self,
        columns: Sequence[str] | None = None,
        methods: list[str] | None = None,
        n_components_pca: int = 10,
        n_components_lda: int = 5,
        n_components_umap: int = 5,
        n_components_autoencoder: int = 8,
        scale_features: bool = True,
        keep_original: bool = False,
        random_state: int = 42,
    ) -> None:
        """Initialize multi-method dimensionality reducer.
        
        Args:
            columns: Columns to use (None = all numeric)
            methods: List of methods to apply (default: ['pca', 'umap'])
            n_components_pca: Components for PCA
            n_components_lda: Components for LDA
            n_components_umap: Components for UMAP
            n_components_autoencoder: Components for Autoencoder
            scale_features: Standardize features
            keep_original: Keep original features
            random_state: Random seed
        """
        self.columns = columns
        self.methods = methods or ['pca', 'umap']
        self.n_components_pca = n_components_pca
        self.n_components_lda = n_components_lda
        self.n_components_umap = n_components_umap
        self.n_components_autoencoder = n_components_autoencoder
        self.scale_features = scale_features
        self.keep_original = keep_original
        self.random_state = random_state
        
        # Fitted attributes
        self.reducers_: dict[str, DimensionalityReducer] = {}
    
    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "MultiMethodDimensionalityReducer":
        """Fit all reduction methods.
        
        Args:
            X: Input DataFrame
            y: Target (required for LDA)
            
        Returns:
            Self
        """
        for method in self.methods:
            # Determine parameters
            if method == 'pca':
                n_comp = self.n_components_pca
            elif method == 'lda':
                n_comp = self.n_components_lda
            elif method == 'umap':
                n_comp = self.n_components_umap
            elif method == 'autoencoder':
                n_comp = self.n_components_autoencoder
            elif method == 'tsne':
                n_comp = 2  # t-SNE typically 2D
            else:
                logger.warning(f"Unknown method: {method}. Skipping.")
                continue
            
            # Create and fit reducer
            try:
                reducer = DimensionalityReducer(
                    columns=self.columns,
                    method=method,
                    n_components=n_comp,
                    keep_original=False,  # We'll handle this at the ensemble level
                    scale_features=self.scale_features,
                    feature_prefix=method,
                    random_state=self.random_state,
                )
                
                reducer.fit(X, y)
                self.reducers_[method] = reducer
                
            except Exception as e:
                logger.warning(f"Failed to fit {method}: {e}. Skipping.")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform using all fitted methods.
        
        Args:
            X: Input DataFrame
            
        Returns:
            DataFrame with all reduced features
        """
        df = pd.DataFrame(X).copy()
        
        if self.keep_original:
            result = df.copy()
        else:
            # Start with non-numeric columns
            numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
            result = df.drop(columns=numeric_cols)
        
        # Apply each reducer
        for method, reducer in self.reducers_.items():
            try:
                reduced = reducer.transform(df)
                
                # Extract only the new features
                new_cols = [col for col in reduced.columns if col.startswith(method)]
                for col in new_cols:
                    result[col] = reduced[col]
                
            except Exception as e:
                logger.warning(f"Failed to transform with {method}: {e}")
        
        return result
    
    def get_feature_names_out(self, input_features=None) -> list[str]:
        """Get all output feature names.
        
        Returns:
            List of feature names from all methods
        """
        names = []
        for reducer in self.reducers_.values():
            names.extend(reducer.get_feature_names_out())
        return names


class AdaptiveDimensionalityReducer(BaseEstimator, TransformerMixin):
    """Automatically select optimal dimensionality reduction method and parameters.
    
    Analyzes data characteristics and intelligently chooses:
    - Best reduction method (PCA, LDA, UMAP, etc.)
    - Optimal number of components
    - Appropriate parameters
    
    Decision Rules:
    - High-dimensional + linear relationships -> PCA
    - Supervised task + class labels -> LDA
    - Non-linear manifold structure -> UMAP
    - Large dataset + need for speed -> PCA or UMAP
    - Small dataset + visualization -> t-SNE
    
    Example:
        >>> reducer = AdaptiveDimensionalityReducer(
        ...     target_variance=0.95,
        ...     max_components=50
        ... )
        >>> reducer.fit(X_train, y_train)
        >>> print(f"Selected: {reducer.selected_method_} with {reducer.n_components_} components")
        >>> X_reduced = reducer.transform(X_test)
    """
    
    def __init__(
        self,
        columns: Sequence[str] | None = None,
        prefer_method: str = 'auto',
        target_variance: float = 0.95,
        max_components: int | None = None,
        min_components: int = 2,
        optimize_components: bool = True,
        scale_features: bool = True,
        random_state: int = 42,
    ) -> None:
        """Initialize adaptive dimensionality reducer.
        
        Args:
            columns: Columns to use (None = all numeric)
            prefer_method: Preferred method - 'auto', 'pca', 'lda', 'umap', etc.
            target_variance: Target explained variance for PCA (0-1)
            max_components: Maximum components to consider
            min_components: Minimum components to keep
            optimize_components: Auto-optimize number of components
            scale_features: Standardize features
            random_state: Random seed
        """
        self.columns = columns
        self.prefer_method = prefer_method
        self.target_variance = target_variance
        self.max_components = max_components
        self.min_components = min_components
        self.optimize_components = optimize_components
        self.scale_features = scale_features
        self.random_state = random_state
        
        # Fitted attributes
        self.selected_method_: str = ''
        self.n_components_: int = 0
        self.reducer_: DimensionalityReducer | None = None
    
    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "AdaptiveDimensionalityReducer":
        """Fit by selecting optimal method and parameters.
        
        Args:
            X: Input DataFrame
            y: Target (used for LDA selection and optimization)
            
        Returns:
            Self
        """
        df = pd.DataFrame(X)
        
        # Determine columns
        if self.columns is None:
            columns = list(df.select_dtypes(include=[np.number]).columns)
        else:
            columns = [c for c in self.columns if c in df.columns]
        
        if not columns:
            logger.warning("No numeric columns for dimensionality reduction.")
            return self
        
        X_data = df[columns].values
        n_features = X_data.shape[1]
        
        # Determine max components
        if self.max_components is None:
            self.max_components = min(50, n_features // 2)
        
        # Select method
        if self.prefer_method == 'auto':
            self.selected_method_ = self._auto_select_method(X_data, y)
        else:
            self.selected_method_ = self.prefer_method
        
        # Determine number of components
        if self.optimize_components:
            self.n_components_ = self._optimize_components(
                X_data, y, self.selected_method_
            )
        else:
            self.n_components_ = min(self.max_components, n_features // 2)
        
        # Ensure min components
        self.n_components_ = max(self.n_components_, self.min_components)
        
        logger.info(
            f"Adaptive reduction selected: {self.selected_method_} "
            f"with {self.n_components_} components"
        )
        
        # Create and fit reducer
        self.reducer_ = DimensionalityReducer(
            columns=columns,
            method=self.selected_method_,
            n_components=self.n_components_,
            scale_features=self.scale_features,
            pca_explained_variance_threshold=(
                self.target_variance if self.selected_method_ == 'pca' else None
            ),
            random_state=self.random_state,
        )
        
        self.reducer_.fit(df, y)
        
        return self
    
    def _auto_select_method(self, X: np.ndarray, y: pd.Series | None) -> str:
        """Auto-select reduction method based on data.
        
        Args:
            X: Input data
            y: Target labels
            
        Returns:
            Selected method name
        """
        n_samples, n_features = X.shape
        
        # Rule 1: If supervised and have labels, prefer LDA
        if y is not None and len(np.unique(y)) > 1:
            n_classes = len(np.unique(y))
            if n_classes < n_features:
                logger.info("Auto-selected LDA (supervised with labels)")
                return 'lda'
        
        # Rule 2: High-dimensional linear data -> PCA
        if n_features > 50:
            logger.info("Auto-selected PCA (high-dimensional data)")
            return 'pca'
        
        # Rule 3: Small dataset + moderate features -> UMAP (if available)
        if n_samples < 5000 and n_features > 10:
            if HAS_UMAP:
                logger.info("Auto-selected UMAP (moderate size, non-linear)")
                return 'umap'
            else:
                logger.info("UMAP not available, falling back to PCA")
                return 'pca'
        
        # Rule 4: Default to PCA (fast, reliable)
        logger.info("Auto-selected PCA (default)")
        return 'pca'
    
    def _optimize_components(
        self,
        X: np.ndarray,
        y: pd.Series | None,
        method: str
    ) -> int:
        """Optimize number of components.
        
        Args:
            X: Input data
            y: Target labels
            method: Reduction method
            
        Returns:
            Optimal number of components
        """
        if method == 'pca':
            # Use explained variance
            pca = PCA(n_components=min(self.max_components, X.shape[1]), random_state=self.random_state)
            
            # Scale first
            if self.scale_features:
                scaler = StandardScaler()
                X = scaler.fit_transform(X)
            
            pca.fit(X)
            
            # Find components that explain target variance
            cumsum = np.cumsum(pca.explained_variance_ratio_)
            n_comp = int(np.argmax(cumsum >= self.target_variance) + 1)
            
            logger.info(
                f"PCA: {n_comp} components explain {cumsum[n_comp-1]:.2%} variance"
            )
            
            return max(n_comp, self.min_components)
        
        elif method == 'lda':
            # LDA limited by n_classes - 1
            if y is None:
                return self.max_components
            n_classes = len(np.unique(y))
            return min(n_classes - 1, self.max_components)
        
        else:
            # For other methods, use heuristic
            return min(self.max_components, X.shape[1] // 3)
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform using selected method.
        
        Args:
            X: Input DataFrame
            
        Returns:
            DataFrame with reduced dimensions
        """
        if self.reducer_ is None:
            return pd.DataFrame(X).copy()
        
        return self.reducer_.transform(X)
    
    def get_feature_names_out(self, input_features=None) -> list[str]:
        """Get output feature names.
        
        Returns:
            List of feature names
        """
        if self.reducer_ is None:
            return []
        return self.reducer_.get_feature_names_out()


def build_dimensionality_reducer(
    method: str = 'auto',
    n_components: int | None = None,
    target_variance: float = 0.95,
    **kwargs
) -> DimensionalityReducer | AdaptiveDimensionalityReducer:
    """Build a dimensionality reduction pipeline with sensible defaults.
    
    Args:
        method: Reduction method - 'auto', 'pca', 'lda', 'tsne', 'umap', 'autoencoder', 'multi'
        n_components: Number of components
        target_variance: Target explained variance for PCA (0-1)
        **kwargs: Additional parameters for the transformer
        
    Returns:
        Dimensionality reducer
        
    Example:
        >>> # Auto-select optimal method
        >>> reducer = build_dimensionality_reducer(method='auto')
        
        >>> # PCA with 95% variance
        >>> reducer = build_dimensionality_reducer(
        ...     method='pca',
        ...     target_variance=0.95
        ... )
        
        >>> # UMAP for visualization
        >>> reducer = build_dimensionality_reducer(
        ...     method='umap',
        ...     n_components=2
        ... )
    """
    if method == 'auto':
        return AdaptiveDimensionalityReducer(
            target_variance=target_variance,
            max_components=n_components,
            **kwargs
        )
    
    elif method == 'multi':
        return MultiMethodDimensionalityReducer(
            n_components_pca=n_components or 10,
            n_components_umap=n_components or 5,
            **kwargs
        )
    
    else:
        return DimensionalityReducer(
            method=method,
            n_components=n_components,
            pca_explained_variance_threshold=(
                target_variance if method == 'pca' and n_components is None else None
            ),
            **kwargs
        )


# Legacy function for backward compatibility
def build_reducer(
    kind: str,
    n_components: int | None = None,
    variance: float | None = None,
    random_state: int = 42,
) -> BaseEstimator | None:
    """Build dimensionality reduction transformer (legacy function).
    
    This function is maintained for backward compatibility.
    Use build_dimensionality_reducer() for new code.
    
    Args:
        kind: Reducer type: 'pca', 'svd', 'umap', or None
        n_components: Number of components
        variance: Explained variance ratio (PCA only)
        random_state: Random seed
        
    Returns:
        Reducer instance or None
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
            logger.warning("PCA requested but no n_components or variance specified.")
            return None
    
    elif kind == "svd":
        if n_components is None:
            logger.warning("SVD requested but no n_components specified.")
            return None
        logger.info(f"Creating TruncatedSVD with {n_components} components")
        return TruncatedSVD(n_components=n_components, random_state=random_state)
    
    elif kind == "umap":
        if not HAS_UMAP:
            logger.warning("UMAP requested but not installed. Install with: pip install umap-learn")
            return None
        if n_components is None:
            n_components = 50
        logger.info(f"Creating UMAP with {n_components} components")
        return UMAP(n_components=n_components, random_state=random_state)
    
    else:
        logger.warning(f"Unknown reducer kind: {kind}")
        return None
