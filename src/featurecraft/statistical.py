"""Statistical feature transformers for FeatureCraft.

This module provides advanced statistical feature engineering capabilities including:
- Row-wise statistics (mean, std, min, max, median, skew, kurtosis)
- Percentile ranking within columns
- Z-score standardization
- Outlier detection and flagging (IQR method, Z-score method)
- Rolling statistics across rows
- Quantile-based features

These transformers are essential for:
- Cross-feature aggregations
- Outlier-aware modeling
- Normalized feature representations
- Distribution-based features
"""

from __future__ import annotations

from typing import List, Literal, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin

from .logging import get_logger

logger = get_logger(__name__)


class RowStatisticsTransformer(BaseEstimator, TransformerMixin):
    """Create statistical features across columns for each row.
    
    Computes row-wise statistics like mean, std, min, max across numeric features.
    This captures cross-feature patterns and can be especially useful for:
    - Data quality features (e.g., count of nulls per row)
    - Signal strength indicators (e.g., std across sensor readings)
    - Aggregated feature representations
    
    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from featurecraft.statistical import RowStatisticsTransformer
    >>> 
    >>> # Sample data
    >>> X = pd.DataFrame({
    ...     'feature1': [1, 2, 3, 4],
    ...     'feature2': [5, 6, 7, 8],
    ...     'feature3': [9, 10, 11, 12]
    ... })
    >>> 
    >>> # Create row statistics
    >>> transformer = RowStatisticsTransformer(
    ...     statistics=['mean', 'std', 'min', 'max'],
    ...     include_null_count=True
    ... )
    >>> X_transformed = transformer.fit_transform(X)
    >>> # Creates: row_mean, row_std, row_min, row_max, row_null_count
    
    Parameters
    ----------
    statistics : List[str], optional
        List of statistics to compute. Options:
        - 'mean': Row mean
        - 'std': Row standard deviation
        - 'min': Row minimum
        - 'max': Row maximum
        - 'median': Row median
        - 'sum': Row sum
        - 'range': Max - Min
        - 'skew': Row skewness
        - 'kurtosis': Row kurtosis
        Default: ['mean', 'std', 'min', 'max']
    numeric_cols : List[str], optional
        Specific columns to compute statistics over. If None, uses all numeric columns.
    include_null_count : bool, optional
        Add a feature counting null values per row. Default: True
    prefix : str, optional
        Prefix for output column names. Default: 'row'
    skip_single_value_rows : bool, optional
        Skip computation for rows with single non-null value (returns NaN). Default: False
    """
    
    def __init__(
        self,
        statistics: List[str] = None,
        numeric_cols: Optional[List[str]] = None,
        include_null_count: bool = True,
        prefix: str = "row",
        skip_single_value_rows: bool = False,
    ):
        if statistics is None:
            statistics = ['mean', 'std', 'min', 'max']
        self.statistics = statistics
        self.numeric_cols = numeric_cols
        self.include_null_count = include_null_count
        self.prefix = prefix
        self.skip_single_value_rows = skip_single_value_rows
        self.numeric_cols_: List[str] = []
        self.feature_names_: List[str] = []
    
    def fit(self, X: pd.DataFrame, y=None):
        """Learn numeric columns from training data.
        
        Parameters
        ----------
        X : pd.DataFrame
            Training data
        y : Ignored
            Not used, kept for sklearn compatibility
            
        Returns
        -------
        self
        """
        df = pd.DataFrame(X)
        
        if self.numeric_cols is not None:
            # Validate specified columns exist
            missing = [col for col in self.numeric_cols if col not in df.columns]
            if missing:
                raise ValueError(f"Specified numeric columns not found: {missing}")
            self.numeric_cols_ = self.numeric_cols
        else:
            # Auto-detect numeric columns
            self.numeric_cols_ = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not self.numeric_cols_:
            logger.warning("No numeric columns found for row statistics")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate row-wise statistical features.
        
        Parameters
        ----------
        X : pd.DataFrame
            Data to transform
            
        Returns
        -------
        pd.DataFrame
            Row statistics features
        """
        df = pd.DataFrame(X)
        out = pd.DataFrame(index=df.index)
        
        if not self.numeric_cols_:
            logger.warning("No numeric columns available, returning empty DataFrame")
            return out
        
        # Get numeric subset
        numeric_subset = df[self.numeric_cols_]
        
        # Compute statistics
        for stat in self.statistics:
            stat_lower = stat.lower()
            
            if stat_lower == 'mean':
                out[f'{self.prefix}_mean'] = numeric_subset.mean(axis=1, skipna=True)
            elif stat_lower == 'std':
                out[f'{self.prefix}_std'] = numeric_subset.std(axis=1, skipna=True)
            elif stat_lower == 'min':
                out[f'{self.prefix}_min'] = numeric_subset.min(axis=1, skipna=True)
            elif stat_lower == 'max':
                out[f'{self.prefix}_max'] = numeric_subset.max(axis=1, skipna=True)
            elif stat_lower == 'median':
                out[f'{self.prefix}_median'] = numeric_subset.median(axis=1, skipna=True)
            elif stat_lower == 'sum':
                out[f'{self.prefix}_sum'] = numeric_subset.sum(axis=1, skipna=True)
            elif stat_lower == 'range':
                out[f'{self.prefix}_range'] = numeric_subset.max(axis=1, skipna=True) - numeric_subset.min(axis=1, skipna=True)
            elif stat_lower == 'skew':
                out[f'{self.prefix}_skew'] = numeric_subset.apply(lambda x: x.skew(), axis=1)
            elif stat_lower == 'kurtosis':
                out[f'{self.prefix}_kurtosis'] = numeric_subset.apply(lambda x: x.kurtosis(), axis=1)
            else:
                logger.warning(f"Unknown statistic '{stat}', skipping")
        
        # Null count (across all columns, not just numeric)
        if self.include_null_count:
            out[f'{self.prefix}_null_count'] = df.isnull().sum(axis=1)
            out[f'{self.prefix}_non_null_count'] = df.notna().sum(axis=1)
        
        self.feature_names_ = list(out.columns)
        return out
    
    def get_feature_names_out(self, input_features=None):
        """Get output feature names.
        
        Returns
        -------
        List[str]
            Feature names
        """
        return self.feature_names_ if self.feature_names_ else []


class PercentileRankTransformer(BaseEstimator, TransformerMixin):
    """Transform features to their percentile ranks within each column.
    
    Converts each value to its percentile rank (0-100) within the column distribution.
    This is useful for:
    - Normalizing features with different scales
    - Handling outliers gracefully
    - Creating distribution-invariant features
    
    Examples
    --------
    >>> import pandas as pd
    >>> from featurecraft.statistical import PercentileRankTransformer
    >>> 
    >>> X = pd.DataFrame({
    ...     'score': [10, 20, 30, 40, 50],
    ...     'age': [25, 35, 45, 55, 65]
    ... })
    >>> 
    >>> transformer = PercentileRankTransformer()
    >>> X_transformed = transformer.fit_transform(X)
    >>> # Each value becomes its percentile rank (0-100)
    
    Parameters
    ----------
    columns : List[str], optional
        Columns to transform. If None, transforms all numeric columns.
    method : str, optional
        Method for computing percentile rank. Options:
        - 'average': Average rank of tied values
        - 'min': Minimum rank of tied values
        - 'max': Maximum rank of tied values
        - 'dense': Like 'min', but rank increases by 1 between groups
        - 'ordinal': All values get distinct rank
        Default: 'average'
    suffix : str, optional
        Suffix for transformed column names. Default: '_pct_rank'
    pct : bool, optional
        If True, returns percentile (0-100). If False, returns rank (0-1). Default: True
    """
    
    def __init__(
        self,
        columns: Optional[List[str]] = None,
        method: str = 'average',
        suffix: str = '_pct_rank',
        pct: bool = True,
    ):
        self.columns = columns
        self.method = method
        self.suffix = suffix
        self.pct = pct
        self.columns_: List[str] = []
        self.feature_names_: List[str] = []
    
    def fit(self, X: pd.DataFrame, y=None):
        """Learn columns from training data.
        
        Parameters
        ----------
        X : pd.DataFrame
            Training data
        y : Ignored
            Not used, kept for sklearn compatibility
            
        Returns
        -------
        self
        """
        df = pd.DataFrame(X)
        
        if self.columns is not None:
            missing = [col for col in self.columns if col not in df.columns]
            if missing:
                raise ValueError(f"Specified columns not found: {missing}")
            self.columns_ = self.columns
        else:
            self.columns_ = df.select_dtypes(include=[np.number]).columns.tolist()
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform features to percentile ranks.
        
        Parameters
        ----------
        X : pd.DataFrame
            Data to transform
            
        Returns
        -------
        pd.DataFrame
            Percentile rank features
        """
        df = pd.DataFrame(X)
        out = pd.DataFrame(index=df.index)
        
        for col in self.columns_:
            if col not in df.columns:
                logger.warning(f"Column '{col}' not found, skipping")
                continue
            
            # Compute percentile rank (always get fractional ranks 0-1)
            ranks = df[col].rank(method=self.method, pct=True)

            # Convert to percentage if requested (0-1 to 0-100)
            if self.pct:
                ranks = ranks * 100
            
            out[f'{col}{self.suffix}'] = ranks
        
        self.feature_names_ = list(out.columns)
        return out
    
    def get_feature_names_out(self, input_features=None):
        """Get output feature names."""
        return self.feature_names_ if self.feature_names_ else []


class ZScoreTransformer(BaseEstimator, TransformerMixin):
    """Compute Z-scores for features: (x - μ) / σ
    
    Standardizes features by removing mean and scaling to unit variance.
    This is similar to sklearn's StandardScaler but:
    - Preserves feature names
    - Allows selective column transformation
    - Can compute modified Z-scores using median/MAD
    
    Examples
    --------
    >>> import pandas as pd
    >>> from featurecraft.statistical import ZScoreTransformer
    >>> 
    >>> X = pd.DataFrame({
    ...     'feature1': [1, 2, 3, 4, 5],
    ...     'feature2': [10, 20, 30, 40, 50]
    ... })
    >>> 
    >>> transformer = ZScoreTransformer()
    >>> X_transformed = transformer.fit_transform(X)
    >>> # Each feature now has mean=0, std=1
    
    Parameters
    ----------
    columns : List[str], optional
        Columns to transform. If None, transforms all numeric columns.
    robust : bool, optional
        If True, use robust Z-score: (x - median) / MAD. Default: False
    suffix : str, optional
        Suffix for transformed column names. Default: '_zscore'
    clip : float, optional
        If provided, clip Z-scores to [-clip, clip]. Default: None
    """
    
    def __init__(
        self,
        columns: Optional[List[str]] = None,
        robust: bool = False,
        suffix: str = '_zscore',
        clip: Optional[float] = None,
    ):
        self.columns = columns
        self.robust = robust
        self.suffix = suffix
        self.clip = clip
        self.columns_: List[str] = []
        self.means_: dict = {}
        self.stds_: dict = {}
        self.medians_: dict = {}
        self.mads_: dict = {}
        self.feature_names_: List[str] = []
    
    def fit(self, X: pd.DataFrame, y=None):
        """Learn mean and std (or median and MAD) from training data.
        
        Parameters
        ----------
        X : pd.DataFrame
            Training data
        y : Ignored
            Not used, kept for sklearn compatibility
            
        Returns
        -------
        self
        """
        df = pd.DataFrame(X)
        
        if self.columns is not None:
            missing = [col for col in self.columns if col not in df.columns]
            if missing:
                raise ValueError(f"Specified columns not found: {missing}")
            self.columns_ = self.columns
        else:
            self.columns_ = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Compute statistics
        for col in self.columns_:
            if self.robust:
                # Robust statistics: median and MAD
                self.medians_[col] = df[col].median()
                self.mads_[col] = np.median(np.abs(df[col] - self.medians_[col]))
                # Avoid division by zero
                if self.mads_[col] == 0:
                    self.mads_[col] = 1.0
            else:
                # Standard statistics: mean and std
                self.means_[col] = df[col].mean()
                self.stds_[col] = df[col].std()
                # Avoid division by zero
                if self.stds_[col] == 0 or pd.isna(self.stds_[col]):
                    self.stds_[col] = 1.0
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform features to Z-scores.
        
        Parameters
        ----------
        X : pd.DataFrame
            Data to transform
            
        Returns
        -------
        pd.DataFrame
            Z-score features
        """
        df = pd.DataFrame(X)
        out = pd.DataFrame(index=df.index)
        
        for col in self.columns_:
            if col not in df.columns:
                logger.warning(f"Column '{col}' not found, skipping")
                continue
            
            if self.robust:
                # Robust Z-score: (x - median) / MAD
                z_scores = (df[col] - self.medians_[col]) / self.mads_[col]
            else:
                # Standard Z-score: (x - mean) / std
                z_scores = (df[col] - self.means_[col]) / self.stds_[col]
            
            # Clip if requested
            if self.clip is not None:
                z_scores = z_scores.clip(-self.clip, self.clip)
            
            out[f'{col}{self.suffix}'] = z_scores
        
        self.feature_names_ = list(out.columns)
        return out
    
    def get_feature_names_out(self, input_features=None):
        """Get output feature names."""
        return self.feature_names_ if self.feature_names_ else []


class OutlierDetector(BaseEstimator, TransformerMixin):
    """Detect and flag outliers using multiple methods.
    
    Creates binary flags indicating whether each value is an outlier.
    Supports multiple detection methods:
    - IQR (Interquartile Range): x < Q1 - k*IQR or x > Q3 + k*IQR
    - Z-score: |z| > threshold
    - Modified Z-score: |modified_z| > threshold (using median/MAD)
    - Isolation Forest: anomaly detection
    
    Examples
    --------
    >>> import pandas as pd
    >>> from featurecraft.statistical import OutlierDetector
    >>> 
    >>> X = pd.DataFrame({
    ...     'feature1': [1, 2, 3, 4, 100],  # 100 is an outlier
    ...     'feature2': [10, 20, 30, 40, 50]
    ... })
    >>> 
    >>> detector = OutlierDetector(method='iqr', k=1.5)
    >>> X_flags = detector.fit_transform(X)
    >>> # Creates binary flags: feature1_is_outlier, feature2_is_outlier
    
    Parameters
    ----------
    method : str, optional
        Outlier detection method:
        - 'iqr': IQR method (default)
        - 'zscore': Z-score method
        - 'modified_zscore': Modified Z-score (robust)
        - 'isolation_forest': Isolation Forest (sklearn)
        Default: 'iqr'
    columns : List[str], optional
        Columns to check for outliers. If None, uses all numeric columns.
    k : float, optional
        Multiplier for IQR method. Default: 1.5
    threshold : float, optional
        Threshold for Z-score methods. Default: 3.0
    suffix : str, optional
        Suffix for output column names. Default: '_is_outlier'
    return_scores : bool, optional
        If True, return outlier scores instead of binary flags. Default: False
    """
    
    def __init__(
        self,
        method: Literal['iqr', 'zscore', 'modified_zscore', 'isolation_forest'] = 'iqr',
        columns: Optional[List[str]] = None,
        k: float = 1.5,
        threshold: float = 3.0,
        suffix: str = '_is_outlier',
        return_scores: bool = False,
    ):
        self.method = method
        self.columns = columns
        self.k = k
        self.threshold = threshold
        self.suffix = suffix
        self.return_scores = return_scores
        self.columns_: List[str] = []
        self.q1_: dict = {}
        self.q3_: dict = {}
        self.iqr_: dict = {}
        self.means_: dict = {}
        self.stds_: dict = {}
        self.medians_: dict = {}
        self.mads_: dict = {}
        self.feature_names_: List[str] = []
    
    def fit(self, X: pd.DataFrame, y=None):
        """Learn outlier boundaries from training data.
        
        Parameters
        ----------
        X : pd.DataFrame
            Training data
        y : Ignored
            Not used, kept for sklearn compatibility
            
        Returns
        -------
        self
        """
        df = pd.DataFrame(X)
        
        if self.columns is not None:
            missing = [col for col in self.columns if col not in df.columns]
            if missing:
                raise ValueError(f"Specified columns not found: {missing}")
            self.columns_ = self.columns
        else:
            self.columns_ = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Learn statistics based on method
        for col in self.columns_:
            if self.method == 'iqr':
                self.q1_[col] = df[col].quantile(0.25)
                self.q3_[col] = df[col].quantile(0.75)
                self.iqr_[col] = self.q3_[col] - self.q1_[col]
            
            elif self.method == 'zscore':
                self.means_[col] = df[col].mean()
                self.stds_[col] = df[col].std()
                if self.stds_[col] == 0 or pd.isna(self.stds_[col]):
                    self.stds_[col] = 1.0
            
            elif self.method == 'modified_zscore':
                self.medians_[col] = df[col].median()
                self.mads_[col] = np.median(np.abs(df[col] - self.medians_[col]))
                if self.mads_[col] == 0:
                    self.mads_[col] = 1.0
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Detect outliers and create flags/scores.
        
        Parameters
        ----------
        X : pd.DataFrame
            Data to transform
            
        Returns
        -------
        pd.DataFrame
            Outlier flags or scores
        """
        df = pd.DataFrame(X)
        out = pd.DataFrame(index=df.index)
        
        for col in self.columns_:
            if col not in df.columns:
                logger.warning(f"Column '{col}' not found, skipping")
                continue
            
            if self.method == 'iqr':
                # IQR method
                lower_bound = self.q1_[col] - self.k * self.iqr_[col]
                upper_bound = self.q3_[col] + self.k * self.iqr_[col]
                
                if self.return_scores:
                    # Return distance from bounds
                    below_lower = np.maximum(0, lower_bound - df[col])
                    above_upper = np.maximum(0, df[col] - upper_bound)
                    out[f'{col}{self.suffix}'] = below_lower + above_upper
                else:
                    # Return binary flags
                    is_outlier = (df[col] < lower_bound) | (df[col] > upper_bound)
                    out[f'{col}{self.suffix}'] = is_outlier.astype(int)
            
            elif self.method == 'zscore':
                # Z-score method
                z_scores = np.abs((df[col] - self.means_[col]) / self.stds_[col])
                
                if self.return_scores:
                    out[f'{col}{self.suffix}'] = z_scores
                else:
                    out[f'{col}{self.suffix}'] = (z_scores > self.threshold).astype(int)
            
            elif self.method == 'modified_zscore':
                # Modified Z-score (robust)
                modified_z = np.abs((df[col] - self.medians_[col]) / self.mads_[col])
                
                if self.return_scores:
                    out[f'{col}{self.suffix}'] = modified_z
                else:
                    out[f'{col}{self.suffix}'] = (modified_z > self.threshold).astype(int)
            
            elif self.method == 'isolation_forest':
                # Note: Isolation Forest requires sklearn
                try:
                    from sklearn.ensemble import IsolationForest
                    
                    # Fit and predict on single column
                    iso_forest = IsolationForest(contamination=0.1, random_state=42)
                    predictions = iso_forest.fit_predict(df[[col]])
                    
                    if self.return_scores:
                        # Return anomaly scores
                        out[f'{col}{self.suffix}'] = -iso_forest.score_samples(df[[col]])
                    else:
                        # Return binary flags (1 = outlier, 0 = inlier)
                        out[f'{col}{self.suffix}'] = (predictions == -1).astype(int)
                
                except ImportError:
                    logger.error("sklearn required for isolation_forest method")
                    out[f'{col}{self.suffix}'] = 0
        
        self.feature_names_ = list(out.columns)
        return out
    
    def get_feature_names_out(self, input_features=None):
        """Get output feature names."""
        return self.feature_names_ if self.feature_names_ else []


class QuantileTransformer(BaseEstimator, TransformerMixin):
    """Create features based on quantile membership.
    
    Assigns each value to a quantile bin and creates:
    - Binary flags for each quantile
    - Quantile labels (0, 1, 2, ...)
    - Distance to quantile boundaries
    
    Useful for:
    - Binning continuous features
    - Creating distribution-based categories
    - Capturing non-linear relationships
    
    Examples
    --------
    >>> import pandas as pd
    >>> from featurecraft.statistical import QuantileTransformer
    >>> 
    >>> X = pd.DataFrame({
    ...     'income': [20000, 35000, 50000, 75000, 100000]
    ... })
    >>> 
    >>> transformer = QuantileTransformer(n_quantiles=4)
    >>> X_transformed = transformer.fit_transform(X)
    >>> # Creates: income_quantile (0, 1, 2, 3)
    
    Parameters
    ----------
    columns : List[str], optional
        Columns to transform. If None, uses all numeric columns.
    n_quantiles : int, optional
        Number of quantile bins. Default: 4 (quartiles)
    output_type : str, optional
        Type of output:
        - 'label': Quantile labels (0, 1, 2, ...)
        - 'onehot': One-hot encoded quantiles
        - 'both': Both labels and one-hot
        Default: 'label'
    suffix : str, optional
        Suffix for output column names. Default: '_quantile'
    """
    
    def __init__(
        self,
        columns: Optional[List[str]] = None,
        n_quantiles: int = 4,
        output_type: Literal['label', 'onehot', 'both'] = 'label',
        suffix: str = '_quantile',
    ):
        self.columns = columns
        self.n_quantiles = n_quantiles
        self.output_type = output_type
        self.suffix = suffix
        self.columns_: List[str] = []
        self.quantiles_: dict = {}
        self.feature_names_: List[str] = []
    
    def fit(self, X: pd.DataFrame, y=None):
        """Learn quantile boundaries from training data.
        
        Parameters
        ----------
        X : pd.DataFrame
            Training data
        y : Ignored
            Not used, kept for sklearn compatibility
            
        Returns
        -------
        self
        """
        df = pd.DataFrame(X)
        
        if self.columns is not None:
            missing = [col for col in self.columns if col not in df.columns]
            if missing:
                raise ValueError(f"Specified columns not found: {missing}")
            self.columns_ = self.columns
        else:
            self.columns_ = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Learn quantile boundaries
        for col in self.columns_:
            quantile_probs = np.linspace(0, 1, self.n_quantiles + 1)
            self.quantiles_[col] = df[col].quantile(quantile_probs).values
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform features to quantile-based features.
        
        Parameters
        ----------
        X : pd.DataFrame
            Data to transform
            
        Returns
        -------
        pd.DataFrame
            Quantile features
        """
        df = pd.DataFrame(X)
        out = pd.DataFrame(index=df.index)
        
        for col in self.columns_:
            if col not in df.columns:
                logger.warning(f"Column '{col}' not found, skipping")
                continue
            
            # Assign to quantile bins
            quantile_labels = pd.cut(
                df[col],
                bins=self.quantiles_[col],
                labels=False,
                include_lowest=True,
                duplicates='drop'
            )
            
            if self.output_type in ['label', 'both']:
                out[f'{col}{self.suffix}'] = quantile_labels
            
            if self.output_type in ['onehot', 'both']:
                # One-hot encode quantiles
                for q in range(self.n_quantiles):
                    out[f'{col}{self.suffix}_q{q}'] = (quantile_labels == q).astype(int)
        
        self.feature_names_ = list(out.columns)
        return out
    
    def get_feature_names_out(self, input_features=None):
        """Get output feature names."""
        return self.feature_names_ if self.feature_names_ else []


class TargetBasedFeaturesTransformer(BaseEstimator, TransformerMixin):
    """Create target-based features using out-of-fold encoding to prevent leakage.
    
    Generates statistical features based on target variable grouped by categorical
    or binned numeric features. Uses out-of-fold (OOF) encoding to prevent data leakage:
    - During training, computes statistics using other folds (not the current row's fold)
    - During inference, uses statistics computed on entire training set
    
    Features created:
    - Target mean per group
    - Target std per group
    - Target count per group
    - Target median per group
    - Target rank (within each group)
    - Target percentile (within each group)
    
    Examples
    --------
    >>> import pandas as pd
    >>> from featurecraft.statistical import TargetBasedFeaturesTransformer
    >>> 
    >>> X = pd.DataFrame({
    ...     'category': ['A', 'A', 'B', 'B', 'A', 'B'],
    ...     'value': [1, 2, 3, 4, 5, 6]
    ... })
    >>> y = pd.Series([10, 20, 15, 25, 30, 35])
    >>> 
    >>> transformer = TargetBasedFeaturesTransformer(
    ...     columns=['category'],
    ...     statistics=['mean', 'std', 'count'],
    ...     n_folds=3
    ... )
    >>> X_transformed = transformer.fit_transform(X, y)
    >>> # Creates: category_target_mean, category_target_std, category_target_count
    
    Parameters
    ----------
    columns : List[str], optional
        Columns to group by. If None, uses all categorical columns.
    statistics : List[str], optional
        Statistics to compute: 'mean', 'std', 'count', 'median', 'min', 'max',
        'rank', 'percentile'. Default: ['mean', 'std', 'count']
    n_folds : int, optional
        Number of folds for out-of-fold encoding. Default: 5
    random_state : int, optional
        Random seed for fold splitting. Default: 42
    smooth : float, optional
        Smoothing factor for mean (Bayesian smoothing). Default: 0.0
    min_samples_leaf : int, optional
        Minimum samples per group for statistics. Groups with fewer samples
        use global statistics. Default: 10
    """
    
    def __init__(
        self,
        columns: Optional[List[str]] = None,
        statistics: List[str] = None,
        n_folds: int = 5,
        random_state: int = 42,
        smooth: float = 0.0,
        min_samples_leaf: int = 10,
    ):
        if statistics is None:
            statistics = ['mean', 'std', 'count']
        self.columns = columns
        self.statistics = statistics
        self.n_folds = n_folds
        self.random_state = random_state
        self.smooth = smooth
        self.min_samples_leaf = min_samples_leaf
        
        self.columns_: List[str] = []
        self.global_target_mean_: float = 0.0
        self.global_target_std_: float = 0.0
        self.target_stats_: dict = {}
        self.feature_names_: List[str] = []
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Learn target statistics from training data using out-of-fold encoding.
        
        Parameters
        ----------
        X : pd.DataFrame
            Training features
        y : pd.Series
            Training target
            
        Returns
        -------
        self
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"X must be DataFrame, got {type(X).__name__}")
        if not isinstance(y, pd.Series):
            raise TypeError(f"y must be Series, got {type(y).__name__}")
        
        df = pd.DataFrame(X).copy()
        
        # Determine columns to use
        if self.columns is not None:
            missing = [col for col in self.columns if col not in df.columns]
            if missing:
                raise ValueError(f"Specified columns not found: {missing}")
            self.columns_ = self.columns
        else:
            # Use categorical columns
            self.columns_ = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if not self.columns_:
            logger.warning("TargetBasedFeaturesTransformer: No columns found")
            return self
        
        # Compute global statistics
        self.global_target_mean_ = y.mean()
        self.global_target_std_ = y.std()
        
        # Compute target statistics per group (for inference)
        for col in self.columns_:
            grouped = pd.DataFrame({col: df[col], 'target': y}).groupby(col)['target']
            
            self.target_stats_[col] = {}
            if 'mean' in self.statistics or 'rank' in self.statistics or 'percentile' in self.statistics:
                self.target_stats_[col]['mean'] = grouped.mean().to_dict()
            if 'std' in self.statistics:
                self.target_stats_[col]['std'] = grouped.std().fillna(0).to_dict()
            if 'count' in self.statistics:
                self.target_stats_[col]['count'] = grouped.count().to_dict()
            if 'median' in self.statistics:
                self.target_stats_[col]['median'] = grouped.median().to_dict()
            if 'min' in self.statistics:
                self.target_stats_[col]['min'] = grouped.min().to_dict()
            if 'max' in self.statistics:
                self.target_stats_[col]['max'] = grouped.max().to_dict()
        
        logger.info(
            f"TargetBasedFeaturesTransformer: Fitted on {len(self.columns_)} columns "
            f"with {len(self.statistics)} statistics"
        )
        
        return self
    
    def transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Transform by creating target-based features.
        
        Parameters
        ----------
        X : pd.DataFrame
            Data to transform
        y : pd.Series, optional
            Target values. If provided during training, uses out-of-fold encoding.
            If None (inference), uses statistics from fit().
            
        Returns
        -------
        pd.DataFrame
            Target-based features
        """
        df = pd.DataFrame(X).copy()
        out = pd.DataFrame(index=df.index)
        
        if not self.columns_:
            return out
        
        # Use out-of-fold encoding during training (if y is provided)
        if y is not None:
            from sklearn.model_selection import KFold
            kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
            
            for col in self.columns_:
                if col not in df.columns:
                    logger.warning(f"Column '{col}' not found, skipping")
                    continue
                
                # Initialize output columns
                for stat in self.statistics:
                    if stat in ['mean', 'std', 'count', 'median', 'min', 'max']:
                        out[f'{col}_target_{stat}'] = 0.0
                    elif stat == 'rank':
                        out[f'{col}_target_rank'] = 0.0
                    elif stat == 'percentile':
                        out[f'{col}_target_percentile'] = 0.0
                
                # Compute OOF statistics
                for train_idx, val_idx in kf.split(df):
                    train_data = pd.DataFrame({
                        'group': df[col].iloc[train_idx],
                        'target': y.iloc[train_idx]
                    })
                    val_groups = df[col].iloc[val_idx]
                    
                    # Compute statistics on training fold
                    grouped = train_data.groupby('group')['target']
                    
                    for stat in self.statistics:
                        if stat == 'mean':
                            stats_map = grouped.mean().to_dict()
                            out.loc[val_idx, f'{col}_target_mean'] = val_groups.map(stats_map).fillna(
                                self.global_target_mean_
                            )
                        elif stat == 'std':
                            stats_map = grouped.std().fillna(0).to_dict()
                            out.loc[val_idx, f'{col}_target_std'] = val_groups.map(stats_map).fillna(
                                self.global_target_std_
                            )
                        elif stat == 'count':
                            stats_map = grouped.count().to_dict()
                            out.loc[val_idx, f'{col}_target_count'] = val_groups.map(stats_map).fillna(0)
                        elif stat == 'median':
                            stats_map = grouped.median().to_dict()
                            out.loc[val_idx, f'{col}_target_median'] = val_groups.map(stats_map).fillna(
                                self.global_target_mean_
                            )
                        elif stat == 'min':
                            stats_map = grouped.min().to_dict()
                            out.loc[val_idx, f'{col}_target_min'] = val_groups.map(stats_map).fillna(
                                self.global_target_mean_
                            )
                        elif stat == 'max':
                            stats_map = grouped.max().to_dict()
                            out.loc[val_idx, f'{col}_target_max'] = val_groups.map(stats_map).fillna(
                                self.global_target_mean_
                            )
        
        else:
            # Inference: use statistics from fit()
            for col in self.columns_:
                if col not in df.columns:
                    logger.warning(f"Column '{col}' not found, skipping")
                    continue
                
                for stat in self.statistics:
                    if stat in ['mean', 'std', 'count', 'median', 'min', 'max']:
                        stats_map = self.target_stats_[col].get(stat, {})
                        default_val = self.global_target_mean_ if stat in ['mean', 'median', 'min', 'max'] else 0.0
                        out[f'{col}_target_{stat}'] = df[col].map(stats_map).fillna(default_val)
                    elif stat == 'rank':
                        # Rank within group (not available during inference, use mean instead)
                        stats_map = self.target_stats_[col].get('mean', {})
                        out[f'{col}_target_rank'] = df[col].map(stats_map).fillna(self.global_target_mean_)
                    elif stat == 'percentile':
                        # Percentile within group (not available during inference, use mean instead)
                        stats_map = self.target_stats_[col].get('mean', {})
                        out[f'{col}_target_percentile'] = df[col].map(stats_map).fillna(self.global_target_mean_)
        
        self.feature_names_ = list(out.columns)
        return out
    
    def get_feature_names_out(self, input_features=None):
        """Get output feature names."""
        return self.feature_names_ if self.feature_names_ else []


class MissingValuePatternsTransformer(BaseEstimator, TransformerMixin):
    """Extract features from missing value patterns.
    
    Creates advanced missing value features:
    - Missing count per row
    - Missing ratio per row
    - Missing indicator flags per column
    - Missing pattern clusters (groups rows with similar missingness)
    - Co-occurrence of missingness (which features tend to be missing together)
    
    These patterns can be highly predictive and reveal data quality issues.
    
    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from featurecraft.statistical import MissingValuePatternsTransformer
    >>> 
    >>> X = pd.DataFrame({
    ...     'a': [1, np.nan, 3, np.nan, 5],
    ...     'b': [np.nan, 2, 3, np.nan, 5],
    ...     'c': [1, 2, np.nan, np.nan, 5]
    ... })
    >>> 
    >>> transformer = MissingValuePatternsTransformer(
    ...     include_count=True,
    ...     include_ratio=True,
    ...     include_clusters=True,
    ...     n_clusters=2
    ... )
    >>> X_transformed = transformer.fit_transform(X)
    >>> # Creates: missing_count, missing_ratio, missing_cluster, etc.
    
    Parameters
    ----------
    include_count : bool, optional
        Include missing count per row. Default: True
    include_ratio : bool, optional
        Include missing ratio per row. Default: True
    include_indicators : bool, optional
        Include binary missing indicators per column. Default: True
    include_clusters : bool, optional
        Cluster rows by missing patterns. Default: True
    include_cooccurrence : bool, optional
        Include pairwise missing co-occurrence features. Default: False
    n_clusters : int, optional
        Number of clusters for missing patterns. Default: 3
    min_missing_rate : float, optional
        Minimum missing rate for column to include indicator. Default: 0.01
    """
    
    def __init__(
        self,
        include_count: bool = True,
        include_ratio: bool = True,
        include_indicators: bool = True,
        include_clusters: bool = True,
        include_cooccurrence: bool = False,
        n_clusters: int = 3,
        min_missing_rate: float = 0.01,
    ):
        self.include_count = include_count
        self.include_ratio = include_ratio
        self.include_indicators = include_indicators
        self.include_clusters = include_clusters
        self.include_cooccurrence = include_cooccurrence
        self.n_clusters = n_clusters
        self.min_missing_rate = min_missing_rate
        
        self.columns_with_missing_: List[str] = []
        self.n_features_: int = 0
        self.cluster_model_ = None
        self.cooccurrence_pairs_: List[tuple] = []
        self.feature_names_: List[str] = []
    
    def fit(self, X: pd.DataFrame, y=None):
        """Learn missing value patterns from training data.
        
        Parameters
        ----------
        X : pd.DataFrame
            Training data
        y : Ignored
            Not used, kept for sklearn compatibility
            
        Returns
        -------
        self
        """
        df = pd.DataFrame(X)
        self.n_features_ = df.shape[1]
        
        # Identify columns with missing values above threshold
        missing_rates = df.isnull().mean()
        self.columns_with_missing_ = missing_rates[
            missing_rates > self.min_missing_rate
        ].index.tolist()
        
        if not self.columns_with_missing_:
            logger.info("MissingValuePatternsTransformer: No columns with significant missing values")
            return self
        
        # Fit clustering model if requested
        if self.include_clusters and len(self.columns_with_missing_) > 1:
            from sklearn.cluster import KMeans
            
            # Create binary missing matrix
            missing_matrix = df[self.columns_with_missing_].isnull().astype(int)
            
            # Fit KMeans on missing patterns
            n_clusters = min(self.n_clusters, len(missing_matrix.drop_duplicates()))
            self.cluster_model_ = KMeans(
                n_clusters=n_clusters,
                random_state=42,
                n_init=10
            )
            self.cluster_model_.fit(missing_matrix)
        
        # Compute co-occurrence pairs if requested
        if self.include_cooccurrence and len(self.columns_with_missing_) > 1:
            missing_matrix = df[self.columns_with_missing_].isnull().astype(int)
            
            # Find pairs with significant co-occurrence
            from itertools import combinations
            for col1, col2 in combinations(self.columns_with_missing_, 2):
                # Compute co-occurrence rate
                both_missing = (missing_matrix[col1] & missing_matrix[col2]).sum()
                either_missing = (missing_matrix[col1] | missing_matrix[col2]).sum()
                
                if either_missing > 0:
                    cooccur_rate = both_missing / either_missing
                    # Keep pairs with strong co-occurrence (>50%)
                    if cooccur_rate > 0.5:
                        self.cooccurrence_pairs_.append((col1, col2))
        
        logger.info(
            f"MissingValuePatternsTransformer: Found {len(self.columns_with_missing_)} "
            f"columns with missing values, {len(self.cooccurrence_pairs_)} co-occurrence pairs"
        )
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform by creating missing value pattern features.
        
        Parameters
        ----------
        X : pd.DataFrame
            Data to transform
            
        Returns
        -------
        pd.DataFrame
            Missing value pattern features
        """
        df = pd.DataFrame(X)
        out = pd.DataFrame(index=df.index)
        
        # Missing count per row
        if self.include_count:
            out['missing_count'] = df.isnull().sum(axis=1)
        
        # Missing ratio per row
        if self.include_ratio:
            out['missing_ratio'] = df.isnull().sum(axis=1) / self.n_features_
        
        # Missing indicators per column
        if self.include_indicators and self.columns_with_missing_:
            for col in self.columns_with_missing_:
                if col in df.columns:
                    out[f'{col}_is_missing'] = df[col].isnull().astype(int)
        
        # Missing pattern clusters
        if self.include_clusters and self.cluster_model_ is not None and self.columns_with_missing_:
            missing_matrix = df[self.columns_with_missing_].isnull().astype(int)
            out['missing_cluster'] = self.cluster_model_.predict(missing_matrix)
        
        # Co-occurrence features
        if self.include_cooccurrence and self.cooccurrence_pairs_:
            for col1, col2 in self.cooccurrence_pairs_:
                if col1 in df.columns and col2 in df.columns:
                    both_missing = (df[col1].isnull() & df[col2].isnull()).astype(int)
                    out[f'{col1}_{col2}_both_missing'] = both_missing
        
        self.feature_names_ = list(out.columns)
        return out
    
    def get_feature_names_out(self, input_features=None):
        """Get output feature names."""
        return self.feature_names_ if self.feature_names_ else []


def build_statistical_pipeline(
    include_row_stats: bool = False,
    include_percentile_rank: bool = False,
    include_zscore: bool = False,
    include_outlier_detection: bool = False,
    include_quantiles: bool = False,
    **kwargs
) -> list:
    """Build a pipeline of statistical transformers.
    
    Convenience function to create a list of statistical transformers
    that can be used in a sklearn Pipeline.
    
    Parameters
    ----------
    include_row_stats : bool, optional
        Include row statistics transformer. Default: True
    include_percentile_rank : bool, optional
        Include percentile rank transformer. Default: False
    include_zscore : bool, optional
        Include Z-score transformer. Default: False
    include_outlier_detection : bool, optional
        Include outlier detection. Default: True
    include_quantiles : bool, optional
        Include quantile transformer. Default: False
    **kwargs : dict
        Additional parameters for transformers
        
    Returns
    -------
    list
        List of (name, transformer) tuples for Pipeline
        
    Examples
    --------
    >>> from sklearn.pipeline import Pipeline
    >>> from featurecraft.statistical import build_statistical_pipeline
    >>> 
    >>> transformers = build_statistical_pipeline(
    ...     include_row_stats=True,
    ...     include_outlier_detection=True
    ... )
    >>> pipeline = Pipeline(transformers)
    """
    transformers = []
    
    if include_row_stats:
        transformers.append(('row_stats', RowStatisticsTransformer(**kwargs.get('row_stats_params', {}))))
    
    if include_percentile_rank:
        transformers.append(('percentile_rank', PercentileRankTransformer(**kwargs.get('percentile_params', {}))))
    
    if include_zscore:
        transformers.append(('zscore', ZScoreTransformer(**kwargs.get('zscore_params', {}))))
    
    if include_outlier_detection:
        transformers.append(('outlier_detector', OutlierDetector(**kwargs.get('outlier_params', {}))))
    
    if include_quantiles:
        transformers.append(('quantiles', QuantileTransformer(**kwargs.get('quantile_params', {}))))
    
    return transformers

