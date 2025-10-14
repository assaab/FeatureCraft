"""Clustering-based feature transformers for unsupervised feature extraction.

This module provides clustering transformers that extract meaningful features from raw data
using unsupervised learning algorithms. These features can capture complex patterns and
group structures that may not be obvious from the original features.

Clustering Features Include:
- Cluster membership (hard assignments)
- Distances to cluster centroids
- Cluster probabilities (soft assignments for applicable methods)
- Outlier detection flags
- Cluster density metrics

Supported Algorithms:
- K-Means: Fast, scalable clustering for spherical clusters
- DBSCAN: Density-based clustering for arbitrary shapes and outlier detection
- Gaussian Mixture: Probabilistic clustering with soft assignments
- Hierarchical: Tree-based clustering for nested group structures
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from .logging import get_logger

logger = get_logger(__name__)


class ClusteringFeatureExtractor(BaseEstimator, TransformerMixin):
    """Unified clustering feature extractor with multiple algorithms.
    
    Extracts clustering-based features using various algorithms (K-Means, DBSCAN, GMM, Hierarchical).
    This transformer creates new features based on cluster membership, distances, probabilities,
    and outlier detection.
    
    Use Cases:
    - Customer segmentation (K-Means)
    - Anomaly detection (DBSCAN outlier flags)
    - Probabilistic grouping (Gaussian Mixture)
    - Hierarchical taxonomies (Hierarchical clustering)
    - Feature engineering for downstream models
    
    Example:
        >>> # K-Means clustering with 5 clusters
        >>> cfe = ClusteringFeatureExtractor(
        ...     method='kmeans',
        ...     n_clusters=5,
        ...     extract_distance=True,
        ...     extract_cluster_id=True
        ... )
        >>> cfe.fit(X_train)
        >>> X_transformed = cfe.transform(X_test)
        
        >>> # DBSCAN for anomaly detection
        >>> cfe = ClusteringFeatureExtractor(
        ...     method='dbscan',
        ...     eps=0.5,
        ...     min_samples=5,
        ...     extract_outlier_flag=True
        ... )
    """
    
    def __init__(
        self,
        columns: Sequence[str] | None = None,
        method: Literal['kmeans', 'dbscan', 'gmm', 'hierarchical'] = 'kmeans',
        n_clusters: int = 5,
        extract_cluster_id: bool = True,
        extract_distance: bool = True,
        extract_probabilities: bool = False,
        extract_outlier_flag: bool = False,
        extract_density: bool = False,
        scale_features: bool = True,
        # K-Means specific
        kmeans_init: str = 'k-means++',
        kmeans_max_iter: int = 300,
        kmeans_n_init: int = 10,
        # DBSCAN specific
        dbscan_eps: float = 0.5,
        dbscan_min_samples: int = 5,
        dbscan_metric: str = 'euclidean',
        # Gaussian Mixture specific
        gmm_covariance_type: str = 'full',
        gmm_max_iter: int = 100,
        gmm_n_init: int = 1,
        # Hierarchical specific
        hierarchical_linkage: str = 'ward',
        hierarchical_distance_threshold: float | None = None,
        # General
        random_state: int = 42,
        feature_prefix: str = 'cluster',
    ) -> None:
        """Initialize clustering feature extractor.
        
        Args:
            columns: Columns to use for clustering (None = all numeric)
            method: Clustering algorithm - 'kmeans', 'dbscan', 'gmm', 'hierarchical'
            n_clusters: Number of clusters (not used for DBSCAN)
            extract_cluster_id: Extract cluster membership as a feature
            extract_distance: Extract distance to cluster centroid(s)
            extract_probabilities: Extract cluster probabilities (GMM only)
            extract_outlier_flag: Extract outlier/noise point flag (DBSCAN)
            extract_density: Extract local density metrics
            scale_features: Standardize features before clustering
            
            # K-Means Parameters
            kmeans_init: Initialization method - 'k-means++', 'random'
            kmeans_max_iter: Maximum iterations
            kmeans_n_init: Number of initializations
            
            # DBSCAN Parameters
            dbscan_eps: Maximum distance between samples in same neighborhood
            dbscan_min_samples: Minimum samples in neighborhood for core point
            dbscan_metric: Distance metric - 'euclidean', 'manhattan', 'cosine'
            
            # Gaussian Mixture Parameters
            gmm_covariance_type: Covariance type - 'full', 'tied', 'diag', 'spherical'
            gmm_max_iter: Maximum EM iterations
            gmm_n_init: Number of initializations
            
            # Hierarchical Parameters
            hierarchical_linkage: Linkage criterion - 'ward', 'complete', 'average', 'single'
            hierarchical_distance_threshold: Distance threshold for automatic cluster count
            
            # General
            random_state: Random seed for reproducibility
            feature_prefix: Prefix for generated feature names
        """
        self.columns = columns
        self.method = method
        self.n_clusters = n_clusters
        self.extract_cluster_id = extract_cluster_id
        self.extract_distance = extract_distance
        self.extract_probabilities = extract_probabilities
        self.extract_outlier_flag = extract_outlier_flag
        self.extract_density = extract_density
        self.scale_features = scale_features
        
        # K-Means
        self.kmeans_init = kmeans_init
        self.kmeans_max_iter = kmeans_max_iter
        self.kmeans_n_init = kmeans_n_init
        
        # DBSCAN
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.dbscan_metric = dbscan_metric
        
        # GMM
        self.gmm_covariance_type = gmm_covariance_type
        self.gmm_max_iter = gmm_max_iter
        self.gmm_n_init = gmm_n_init
        
        # Hierarchical
        self.hierarchical_linkage = hierarchical_linkage
        self.hierarchical_distance_threshold = hierarchical_distance_threshold
        
        # General
        self.random_state = random_state
        self.feature_prefix = feature_prefix
        
        # Fitted attributes
        self.columns_: list[str] = []
        self.scaler_: StandardScaler | None = None
        self.clusterer_: Any = None
        self.cluster_centers_: np.ndarray | None = None
        self.n_clusters_: int = 0
        self.feature_names_out_: list[str] = []
    
    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "ClusteringFeatureExtractor":
        """Fit clustering algorithm on data.
        
        Args:
            X: Input DataFrame
            y: Target (unused, for sklearn compatibility)
            
        Returns:
            Self
        """
        df = pd.DataFrame(X)
        
        # Determine columns to use
        if self.columns is None:
            # Use all numeric columns
            self.columns_ = list(df.select_dtypes(include=[np.number]).columns)
        else:
            self.columns_ = [c for c in self.columns if c in df.columns]
        
        if not self.columns_:
            logger.warning("No numeric columns found for clustering. Skipping.")
            return self
        
        # Extract data for clustering
        X_cluster = df[self.columns_].values
        
        # Handle missing values
        if np.isnan(X_cluster).any():
            logger.warning("Missing values detected in clustering features. Imputing with column means.")
            col_means = np.nanmean(X_cluster, axis=0)
            for i in range(X_cluster.shape[1]):
                X_cluster[np.isnan(X_cluster[:, i]), i] = col_means[i]
        
        # Scale features if requested
        if self.scale_features:
            self.scaler_ = StandardScaler()
            X_cluster = self.scaler_.fit_transform(X_cluster)
        
        # Fit clustering algorithm
        try:
            if self.method == 'kmeans':
                self._fit_kmeans(X_cluster)
            elif self.method == 'dbscan':
                self._fit_dbscan(X_cluster)
            elif self.method == 'gmm':
                self._fit_gmm(X_cluster)
            elif self.method == 'hierarchical':
                self._fit_hierarchical(X_cluster)
            else:
                raise ValueError(
                    f"Unknown clustering method: '{self.method}'. "
                    f"Choose from: kmeans, dbscan, gmm, hierarchical"
                )
        except Exception as e:
            logger.error(f"Clustering fit failed: {e}")
            raise
        
        # Generate feature names
        self._generate_feature_names()
        
        logger.info(
            f"Fitted {self.method} clustering on {len(self.columns_)} features. "
            f"Generated {len(self.feature_names_out_)} cluster features."
        )
        
        return self
    
    def _fit_kmeans(self, X: np.ndarray) -> None:
        """Fit K-Means clustering."""
        self.clusterer_ = KMeans(
            n_clusters=self.n_clusters,
            init=self.kmeans_init,
            max_iter=self.kmeans_max_iter,
            n_init=self.kmeans_n_init,
            random_state=self.random_state,
        )
        self.clusterer_.fit(X)
        self.cluster_centers_ = self.clusterer_.cluster_centers_
        self.n_clusters_ = self.n_clusters
    
    def _fit_dbscan(self, X: np.ndarray) -> None:
        """Fit DBSCAN clustering."""
        self.clusterer_ = DBSCAN(
            eps=self.dbscan_eps,
            min_samples=self.dbscan_min_samples,
            metric=self.dbscan_metric,
            n_jobs=-1,
        )
        labels = self.clusterer_.fit_predict(X)
        
        # DBSCAN uses -1 for outliers
        unique_labels = set(labels)
        unique_labels.discard(-1)  # Remove outlier label
        self.n_clusters_ = len(unique_labels)
        
        # Compute pseudo-centroids for non-outlier clusters
        if self.n_clusters_ > 0:
            centroids = []
            for label in sorted(unique_labels):
                cluster_points = X[labels == label]
                centroid = cluster_points.mean(axis=0)
                centroids.append(centroid)
            self.cluster_centers_ = np.array(centroids)
        else:
            self.cluster_centers_ = None
    
    def _fit_gmm(self, X: np.ndarray) -> None:
        """Fit Gaussian Mixture Model."""
        self.clusterer_ = GaussianMixture(
            n_components=self.n_clusters,
            covariance_type=self.gmm_covariance_type,
            max_iter=self.gmm_max_iter,
            n_init=self.gmm_n_init,
            random_state=self.random_state,
        )
        self.clusterer_.fit(X)
        self.cluster_centers_ = self.clusterer_.means_
        self.n_clusters_ = self.n_clusters
    
    def _fit_hierarchical(self, X: np.ndarray) -> None:
        """Fit Hierarchical clustering."""
        if self.hierarchical_distance_threshold is not None:
            # Automatic cluster count based on distance threshold
            self.clusterer_ = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=self.hierarchical_distance_threshold,
                linkage=self.hierarchical_linkage,
            )
        else:
            # Fixed cluster count
            self.clusterer_ = AgglomerativeClustering(
                n_clusters=self.n_clusters,
                linkage=self.hierarchical_linkage,
            )
        
        labels = self.clusterer_.fit_predict(X)
        self.n_clusters_ = len(np.unique(labels))
        
        # Compute centroids
        centroids = []
        for label in range(self.n_clusters_):
            cluster_points = X[labels == label]
            centroid = cluster_points.mean(axis=0)
            centroids.append(centroid)
        self.cluster_centers_ = np.array(centroids)
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data by extracting clustering features.
        
        Args:
            X: Input DataFrame
            
        Returns:
            DataFrame with original features and cluster features
        """
        df = pd.DataFrame(X).copy()
        
        if not self.columns_ or self.clusterer_ is None:
            # No clustering performed
            return df
        
        # Extract data for clustering
        X_cluster = df[self.columns_].values
        
        # Handle missing values (same as fit)
        if np.isnan(X_cluster).any():
            col_means = np.nanmean(X_cluster, axis=0)
            for i in range(X_cluster.shape[1]):
                X_cluster[np.isnan(X_cluster[:, i]), i] = col_means[i]
        
        # Scale if needed
        if self.scaler_ is not None:
            X_cluster = self.scaler_.transform(X_cluster)
        
        # Extract features based on method
        if self.method == 'kmeans':
            self._transform_kmeans(df, X_cluster)
        elif self.method == 'dbscan':
            self._transform_dbscan(df, X_cluster)
        elif self.method == 'gmm':
            self._transform_gmm(df, X_cluster)
        elif self.method == 'hierarchical':
            self._transform_hierarchical(df, X_cluster)
        
        return df
    
    def _transform_kmeans(self, df: pd.DataFrame, X: np.ndarray) -> None:
        """Extract K-Means features."""
        # Cluster ID
        if self.extract_cluster_id:
            cluster_ids = self.clusterer_.predict(X)
            df[f'{self.feature_prefix}_id'] = cluster_ids
        
        # Distance to closest centroid
        if self.extract_distance:
            distances = self.clusterer_.transform(X)
            df[f'{self.feature_prefix}_distance'] = distances.min(axis=1)
            
            # Distance to each centroid (optional, can create many features)
            if self.n_clusters_ <= 10:  # Only for small cluster counts
                for i in range(self.n_clusters_):
                    df[f'{self.feature_prefix}_distance_to_{i}'] = distances[:, i]
    
    def _transform_dbscan(self, df: pd.DataFrame, X: np.ndarray) -> None:
        """Extract DBSCAN features."""
        labels = self.clusterer_.fit_predict(X)
        
        # Cluster ID
        if self.extract_cluster_id:
            df[f'{self.feature_prefix}_id'] = labels
        
        # Outlier flag (-1 indicates noise points in DBSCAN)
        if self.extract_outlier_flag:
            df[f'{self.feature_prefix}_is_outlier'] = (labels == -1).astype(int)
        
        # Distance to nearest cluster centroid (for non-outliers)
        if self.extract_distance and self.cluster_centers_ is not None:
            distances = np.full(X.shape[0], np.nan)
            
            for i, point in enumerate(X):
                if labels[i] != -1:
                    # Distance to own cluster centroid
                    centroid_idx = labels[i]
                    if centroid_idx < len(self.cluster_centers_):
                        distances[i] = np.linalg.norm(point - self.cluster_centers_[centroid_idx])
                else:
                    # Distance to nearest centroid for outliers
                    if len(self.cluster_centers_) > 0:
                        dists_to_all = np.linalg.norm(
                            point - self.cluster_centers_, axis=1
                        )
                        distances[i] = dists_to_all.min()
            
            df[f'{self.feature_prefix}_distance'] = distances
    
    def _transform_gmm(self, df: pd.DataFrame, X: np.ndarray) -> None:
        """Extract Gaussian Mixture features."""
        # Cluster ID (hard assignment)
        if self.extract_cluster_id:
            cluster_ids = self.clusterer_.predict(X)
            df[f'{self.feature_prefix}_id'] = cluster_ids
        
        # Cluster probabilities (soft assignment)
        if self.extract_probabilities:
            probabilities = self.clusterer_.predict_proba(X)
            for i in range(self.n_clusters_):
                df[f'{self.feature_prefix}_prob_{i}'] = probabilities[:, i]
            
            # Max probability (confidence)
            df[f'{self.feature_prefix}_max_prob'] = probabilities.max(axis=1)
            
            # Entropy (uncertainty)
            # Clip probabilities to avoid log(0) and ensure non-negative entropy
            clipped_probs = np.clip(probabilities, 1e-15, 1.0)
            entropy = -np.sum(clipped_probs * np.log(clipped_probs), axis=1)
            df[f'{self.feature_prefix}_entropy'] = entropy
        
        # Distance to cluster centers
        if self.extract_distance:
            distances_to_all = np.array([
                np.linalg.norm(X - center, axis=1)
                for center in self.cluster_centers_
            ]).T
            
            df[f'{self.feature_prefix}_distance'] = distances_to_all.min(axis=1)
    
    def _transform_hierarchical(self, df: pd.DataFrame, X: np.ndarray) -> None:
        """Extract Hierarchical clustering features."""
        # Note: AgglomerativeClustering doesn't have predict(), need to refit
        labels = self.clusterer_.fit_predict(X)
        
        # Cluster ID
        if self.extract_cluster_id:
            df[f'{self.feature_prefix}_id'] = labels
        
        # Distance to cluster centroid
        if self.extract_distance and self.cluster_centers_ is not None:
            distances = np.full(X.shape[0], np.nan)
            
            for i, point in enumerate(X):
                cluster_id = labels[i]
                if cluster_id < len(self.cluster_centers_):
                    distances[i] = np.linalg.norm(point - self.cluster_centers_[cluster_id])
            
            df[f'{self.feature_prefix}_distance'] = distances
    
    def _generate_feature_names(self) -> None:
        """Generate output feature names based on enabled features."""
        names = []
        
        if self.extract_cluster_id:
            names.append(f'{self.feature_prefix}_id')
        
        if self.extract_distance:
            names.append(f'{self.feature_prefix}_distance')
            
            # Additional distance features for K-Means
            if self.method == 'kmeans' and self.n_clusters_ <= 10:
                for i in range(self.n_clusters_):
                    names.append(f'{self.feature_prefix}_distance_to_{i}')
        
        if self.extract_probabilities and self.method == 'gmm':
            for i in range(self.n_clusters_):
                names.append(f'{self.feature_prefix}_prob_{i}')
            names.append(f'{self.feature_prefix}_max_prob')
            names.append(f'{self.feature_prefix}_entropy')
        
        if self.extract_outlier_flag and self.method == 'dbscan':
            names.append(f'{self.feature_prefix}_is_outlier')
        
        self.feature_names_out_ = names
    
    def get_feature_names_out(self, input_features=None) -> list[str]:
        """Get output feature names.
        
        Returns:
            List of feature names
        """
        return self.feature_names_out_
    
    def get_cluster_info(self) -> dict[str, Any]:
        """Get clustering metadata and statistics.
        
        Returns:
            Dict with cluster information
        """
        info = {
            'method': self.method,
            'n_clusters': self.n_clusters_,
            'n_features': len(self.columns_),
            'feature_names': self.columns_,
        }
        
        if self.cluster_centers_ is not None:
            info['cluster_centers'] = self.cluster_centers_
        
        if self.method == 'kmeans' and hasattr(self.clusterer_, 'inertia_'):
            info['inertia'] = self.clusterer_.inertia_
        
        if self.method == 'gmm' and hasattr(self.clusterer_, 'lower_bound_'):
            info['log_likelihood'] = self.clusterer_.lower_bound_
        
        return info


class MultiMethodClusteringExtractor(BaseEstimator, TransformerMixin):
    """Apply multiple clustering algorithms simultaneously for ensemble features.
    
    This transformer applies multiple clustering methods (K-Means, DBSCAN, GMM) to the same
    data and combines their outputs, creating a richer feature set that captures different
    clustering perspectives.
    
    Use Case:
    - When cluster structure is ambiguous
    - For robust feature extraction
    - To capture both global (K-Means) and local (DBSCAN) patterns
    - For model ensembles that benefit from diverse features
    
    Example:
        >>> mce = MultiMethodClusteringExtractor(
        ...     methods=['kmeans', 'dbscan', 'gmm'],
        ...     n_clusters_kmeans=5,
        ...     n_clusters_gmm=3,
        ...     dbscan_eps=0.5
        ... )
        >>> mce.fit(X_train)
        >>> X_transformed = mce.transform(X_test)
    """
    
    def __init__(
        self,
        columns: Sequence[str] | None = None,
        methods: list[str] = None,
        n_clusters_kmeans: int = 5,
        n_clusters_gmm: int = 5,
        n_clusters_hierarchical: int = 5,
        dbscan_eps: float = 0.5,
        dbscan_min_samples: int = 5,
        scale_features: bool = True,
        extract_cluster_id: bool = True,
        extract_distance: bool = True,
        extract_probabilities: bool = True,
        random_state: int = 42,
    ) -> None:
        """Initialize multi-method clustering extractor.
        
        Args:
            columns: Columns to use (None = all numeric)
            methods: List of methods to apply (default: ['kmeans', 'gmm'])
            n_clusters_kmeans: Number of clusters for K-Means
            n_clusters_gmm: Number of components for GMM
            n_clusters_hierarchical: Number of clusters for Hierarchical
            dbscan_eps: Epsilon for DBSCAN
            dbscan_min_samples: Min samples for DBSCAN
            scale_features: Standardize features before clustering
            extract_cluster_id: Extract cluster IDs
            extract_distance: Extract distances
            extract_probabilities: Extract probabilities (GMM)
            random_state: Random seed
        """
        self.columns = columns
        self.methods = methods or ['kmeans', 'gmm']
        self.n_clusters_kmeans = n_clusters_kmeans
        self.n_clusters_gmm = n_clusters_gmm
        self.n_clusters_hierarchical = n_clusters_hierarchical
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.scale_features = scale_features
        self.extract_cluster_id = extract_cluster_id
        self.extract_distance = extract_distance
        self.extract_probabilities = extract_probabilities
        self.random_state = random_state
        
        # Fitted attributes
        self.extractors_: dict[str, ClusteringFeatureExtractor] = {}
    
    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "MultiMethodClusteringExtractor":
        """Fit all clustering methods.
        
        Args:
            X: Input DataFrame
            y: Target (unused)
            
        Returns:
            Self
        """
        for method in self.methods:
            # Determine parameters based on method
            if method == 'kmeans':
                n_clusters = self.n_clusters_kmeans
                prefix = 'kmeans'
            elif method == 'gmm':
                n_clusters = self.n_clusters_gmm
                prefix = 'gmm'
            elif method == 'hierarchical':
                n_clusters = self.n_clusters_hierarchical
                prefix = 'hierarchical'
            elif method == 'dbscan':
                n_clusters = 2  # Not used for DBSCAN
                prefix = 'dbscan'
            else:
                logger.warning(f"Unknown clustering method: {method}. Skipping.")
                continue
            
            # Create and fit extractor
            extractor = ClusteringFeatureExtractor(
                columns=self.columns,
                method=method,
                n_clusters=n_clusters,
                extract_cluster_id=self.extract_cluster_id,
                extract_distance=self.extract_distance,
                extract_probabilities=self.extract_probabilities and method == 'gmm',
                extract_outlier_flag=method == 'dbscan',
                scale_features=self.scale_features,
                dbscan_eps=self.dbscan_eps,
                dbscan_min_samples=self.dbscan_min_samples,
                random_state=self.random_state,
                feature_prefix=prefix,
            )
            
            try:
                extractor.fit(X, y)
                self.extractors_[method] = extractor
            except Exception as e:
                logger.warning(f"Failed to fit {method} clustering: {e}. Skipping.")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform using all fitted clustering methods.
        
        Args:
            X: Input DataFrame
            
        Returns:
            DataFrame with original features + all cluster features
        """
        df = pd.DataFrame(X).copy()
        
        for method, extractor in self.extractors_.items():
            try:
                df = extractor.transform(df)
            except Exception as e:
                logger.warning(f"Failed to transform with {method} clustering: {e}")
        
        return df
    
    def get_feature_names_out(self, input_features=None) -> list[str]:
        """Get all output feature names.
        
        Returns:
            List of feature names from all methods
        """
        names = []
        for extractor in self.extractors_.values():
            names.extend(extractor.get_feature_names_out())
        return names


class AdaptiveClusteringExtractor(BaseEstimator, TransformerMixin):
    """Automatically select optimal clustering method and parameters.
    
    This transformer analyzes the data characteristics and intelligently selects:
    - The best clustering algorithm (K-Means, DBSCAN, GMM, or Hierarchical)
    - Optimal number of clusters (using elbow method, silhouette, BIC)
    - Appropriate parameters for the chosen method
    
    Decision Rules:
    - High-dimensional data -> K-Means (fast, scalable)
    - Non-convex clusters or outliers -> DBSCAN
    - Overlapping clusters -> Gaussian Mixture
    - Hierarchical structure -> Hierarchical clustering
    
    Example:
        >>> ace = AdaptiveClusteringExtractor(
        ...     prefer_method='auto',
        ...     max_clusters=10,
        ...     optimize_k=True
        ... )
        >>> ace.fit(X_train)
        >>> print(f"Selected: {ace.selected_method_} with {ace.n_clusters_} clusters")
        >>> X_transformed = ace.transform(X_test)
    """
    
    def __init__(
        self,
        columns: Sequence[str] | None = None,
        prefer_method: str = 'auto',
        max_clusters: int = 10,
        optimize_k: bool = True,
        k_selection_method: str = 'silhouette',
        scale_features: bool = True,
        random_state: int = 42,
    ) -> None:
        """Initialize adaptive clustering extractor.
        
        Args:
            columns: Columns to use (None = all numeric)
            prefer_method: Preferred method - 'auto', 'kmeans', 'dbscan', 'gmm', 'hierarchical'
            max_clusters: Maximum number of clusters to try
            optimize_k: Optimize number of clusters automatically
            k_selection_method: Method for selecting K - 'silhouette', 'elbow', 'bic'
            scale_features: Standardize features
            random_state: Random seed
        """
        self.columns = columns
        self.prefer_method = prefer_method
        self.max_clusters = max_clusters
        self.optimize_k = optimize_k
        self.k_selection_method = k_selection_method
        self.scale_features = scale_features
        self.random_state = random_state
        
        # Fitted attributes
        self.selected_method_: str = ''
        self.n_clusters_: int = 0
        self.extractor_: ClusteringFeatureExtractor | None = None
    
    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "AdaptiveClusteringExtractor":
        """Fit by selecting optimal clustering method and parameters.
        
        Args:
            X: Input DataFrame
            y: Target (unused)
            
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
            logger.warning("No numeric columns for clustering.")
            return self
        
        X_data = df[columns].values
        
        # Handle missing values
        if np.isnan(X_data).any():
            col_means = np.nanmean(X_data, axis=0)
            for i in range(X_data.shape[1]):
                X_data[np.isnan(X_data[:, i]), i] = col_means[i]
        
        # Scale
        if self.scale_features:
            scaler = StandardScaler()
            X_data = scaler.fit_transform(X_data)
        
        # Select method
        if self.prefer_method == 'auto':
            self.selected_method_ = self._auto_select_method(X_data)
        else:
            self.selected_method_ = self.prefer_method
        
        # Optimize K if requested
        if self.optimize_k and self.selected_method_ in ['kmeans', 'gmm', 'hierarchical']:
            self.n_clusters_ = self._optimize_k(X_data, self.selected_method_)
        else:
            self.n_clusters_ = min(5, self.max_clusters)
        
        logger.info(
            f"Adaptive clustering selected: {self.selected_method_} "
            f"with {self.n_clusters_} clusters"
        )
        
        # Create and fit extractor
        self.extractor_ = ClusteringFeatureExtractor(
            columns=columns,
            method=self.selected_method_,
            n_clusters=self.n_clusters_,
            extract_cluster_id=True,
            extract_distance=True,
            extract_probabilities=True,
            scale_features=self.scale_features,
            random_state=self.random_state,
        )
        self.extractor_.fit(df, y)
        
        return self
    
    def _auto_select_method(self, X: np.ndarray) -> str:
        """Automatically select clustering method based on data characteristics.
        
        Args:
            X: Scaled input data
            
        Returns:
            Selected method name
        """
        n_samples, n_features = X.shape
        
        # Rule 1: Small dataset -> GMM (more flexible)
        if n_samples < 1000:
            return 'gmm'
        
        # Rule 2: High-dimensional -> K-Means (scalable)
        if n_features > 20:
            return 'kmeans'
        
        # Rule 3: Check for outliers using simple heuristic
        # If many points are far from mean, use DBSCAN
        distances_from_mean = np.linalg.norm(X - X.mean(axis=0), axis=1)
        outlier_ratio = np.mean(distances_from_mean > 3 * distances_from_mean.std())
        
        if outlier_ratio > 0.05:  # More than 5% outliers
            return 'dbscan'
        
        # Rule 4: Default to K-Means (fast, reliable)
        return 'kmeans'
    
    def _optimize_k(self, X: np.ndarray, method: str) -> int:
        """Optimize number of clusters.
        
        Args:
            X: Scaled input data
            method: Clustering method
            
        Returns:
            Optimal number of clusters
        """
        from sklearn.metrics import silhouette_score
        
        k_range = range(2, min(self.max_clusters + 1, X.shape[0] // 10))
        scores = []
        
        for k in k_range:
            try:
                if method == 'kmeans':
                    clusterer = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
                    labels = clusterer.fit_predict(X)
                elif method == 'gmm':
                    clusterer = GaussianMixture(n_components=k, random_state=self.random_state)
                    clusterer.fit(X)
                    labels = clusterer.predict(X)
                elif method == 'hierarchical':
                    clusterer = AgglomerativeClustering(n_clusters=k)
                    labels = clusterer.fit_predict(X)
                else:
                    return 5  # Default
                
                # Score based on selection method
                if self.k_selection_method == 'silhouette':
                    score = silhouette_score(X, labels, sample_size=min(1000, X.shape[0]))
                    scores.append(score)
                elif self.k_selection_method == 'bic' and method == 'gmm':
                    score = clusterer.bic(X)
                    scores.append(-score)  # Lower BIC is better
                else:
                    # Elbow method (inertia)
                    if hasattr(clusterer, 'inertia_'):
                        scores.append(-clusterer.inertia_)
                    else:
                        scores.append(0)
            
            except Exception as e:
                logger.warning(f"Failed to evaluate k={k}: {e}")
                scores.append(-np.inf)
        
        if not scores:
            return 5  # Default fallback
        
        # Find optimal K
        optimal_idx = np.argmax(scores)
        optimal_k = list(k_range)[optimal_idx]
        
        return optimal_k
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform using selected clustering method.
        
        Args:
            X: Input DataFrame
            
        Returns:
            DataFrame with cluster features
        """
        if self.extractor_ is None:
            return pd.DataFrame(X).copy()
        
        return self.extractor_.transform(X)
    
    def get_feature_names_out(self, input_features=None) -> list[str]:
        """Get output feature names.
        
        Returns:
            List of feature names
        """
        if self.extractor_ is None:
            return []
        return self.extractor_.get_feature_names_out()


def build_clustering_pipeline(
    method: str = 'auto',
    n_clusters: int = 5,
    extract_all: bool = True,
    **kwargs
) -> ClusteringFeatureExtractor | AdaptiveClusteringExtractor:
    """Build a clustering feature extraction pipeline with sensible defaults.
    
    Args:
        method: Clustering method - 'auto', 'kmeans', 'dbscan', 'gmm', 'hierarchical', 'multi'
        n_clusters: Number of clusters
        extract_all: Extract all available features (IDs, distances, probabilities)
        **kwargs: Additional parameters for the transformer
        
    Returns:
        Clustering transformer
        
    Example:
        >>> # Auto-select optimal clustering
        >>> pipeline = build_clustering_pipeline(method='auto')
        
        >>> # K-Means with 10 clusters
        >>> pipeline = build_clustering_pipeline(method='kmeans', n_clusters=10)
        
        >>> # Multi-method ensemble
        >>> pipeline = build_clustering_pipeline(method='multi')
    """
    if method == 'auto':
        return AdaptiveClusteringExtractor(
            max_clusters=n_clusters,
            optimize_k=True,
            **kwargs
        )
    
    elif method == 'multi':
        return MultiMethodClusteringExtractor(
            methods=['kmeans', 'gmm', 'dbscan'],
            n_clusters_kmeans=n_clusters,
            n_clusters_gmm=n_clusters,
            **kwargs
        )
    
    else:
        return ClusteringFeatureExtractor(
            method=method,
            n_clusters=n_clusters,
            extract_cluster_id=extract_all,
            extract_distance=extract_all,
            extract_probabilities=extract_all,
            extract_outlier_flag=extract_all,
            **kwargs
        )

