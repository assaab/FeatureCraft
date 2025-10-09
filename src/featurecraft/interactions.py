"""Feature interaction utilities for FeatureCraft.

This module provides comprehensive feature interaction capabilities including:
- Arithmetic operations (addition, subtraction, multiplication, division)
- Polynomial features (squares, cubes, cross-products)
- Ratio and proportion features
- Product interactions (multi-way)
- Binned interactions (categorical × numeric)
- Categorical × numeric interactions
"""

from __future__ import annotations

from itertools import combinations
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PolynomialFeatures as SklearnPolynomial

from .logging import get_logger

logger = get_logger(__name__)


class ArithmeticInteractions(BaseEstimator, TransformerMixin):
    """Create arithmetic interactions between numeric features.
    
    Generates features through basic arithmetic operations:
    - Addition: A + B
    - Subtraction: A - B (and B - A)
    - Multiplication: A × B
    - Division: A / B (and B / A, with safe division)
    
    Useful for capturing:
    - Linear combinations
    - Differences that may be predictive
    - Products (e.g., price × quantity)
    - Ratios (e.g., revenue / cost)
    
    Parameters
    ----------
    operations : List[str], optional
        Operations to perform. Options: 'add', 'subtract', 'multiply', 'divide'.
        Default: ['multiply', 'divide']
    max_features : int, optional
        Maximum number of feature pairs to consider. Default: 100
    specific_pairs : List[Tuple[str, str]], optional
        Specific feature pairs to interact. If None, considers all pairs.
    symmetric : bool, optional
        If True, only create A op B (not B op A for commutative ops). Default: True
    """
    
    def __init__(
        self,
        operations: List[str] = None,
        max_features: int = 100,
        specific_pairs: Optional[List[Tuple[str, str]]] = None,
        symmetric: bool = True,
    ):
        if operations is None:
            operations = ['multiply', 'divide']
        self.operations = operations
        self.max_features = max_features
        self.specific_pairs = specific_pairs
        self.symmetric = symmetric
        self.feature_names_: List[str] = []
        self.numeric_cols_: List[str] = []
    
    def fit(self, X: pd.DataFrame, y=None):
        """Fit the transformer (learn numeric columns)."""
        if isinstance(X, pd.DataFrame):
            self.numeric_cols_ = X.select_dtypes(include=[np.number]).columns.tolist()
        else:
            # If array, create dummy column names
            self.numeric_cols_ = [f"x{i}" for i in range(X.shape[1])]
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate arithmetic interaction features."""
        df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        out = pd.DataFrame(index=df.index)
        
        # Determine which pairs to create
        if self.specific_pairs:
            pairs = self.specific_pairs
        else:
            # Get all numeric columns
            numeric_cols = [col for col in self.numeric_cols_ if col in df.columns]
            
            # Limit number of interactions
            if len(numeric_cols) > 2:
                # Create combinations
                all_pairs = list(combinations(numeric_cols, 2))
                
                # Limit to max_features pairs
                if len(all_pairs) > self.max_features:
                    # Take first max_features pairs
                    pairs = all_pairs[:self.max_features]
                else:
                    pairs = all_pairs
            else:
                pairs = list(combinations(numeric_cols, 2)) if len(numeric_cols) >= 2 else []
        
        # Generate interactions
        for col1, col2 in pairs:
            if col1 not in df.columns or col2 not in df.columns:
                continue
            
            # Addition
            if 'add' in self.operations:
                out[f"{col1}_add_{col2}"] = df[col1] + df[col2]
            
            # Subtraction
            if 'subtract' in self.operations:
                out[f"{col1}_sub_{col2}"] = df[col1] - df[col2]
                if not self.symmetric:
                    out[f"{col2}_sub_{col1}"] = df[col2] - df[col1]
            
            # Multiplication
            if 'multiply' in self.operations:
                out[f"{col1}_mul_{col2}"] = df[col1] * df[col2]
            
            # Division (with safe division)
            if 'divide' in self.operations:
                # Avoid division by zero
                out[f"{col1}_div_{col2}"] = df[col1] / df[col2].replace(0, np.nan)
                if not self.symmetric:
                    out[f"{col2}_div_{col1}"] = df[col2] / df[col1].replace(0, np.nan)
        
        self.feature_names_ = list(out.columns)
        return out
    
    def get_feature_names_out(self, input_features=None):
        """Get output feature names."""
        return self.feature_names_ if self.feature_names_ else []


class PolynomialInteractions(BaseEstimator, TransformerMixin):
    """Create polynomial and interaction features.
    
    Generates polynomial features including:
    - Single feature powers: x², x³, etc.
    - Cross-products: x₁ × x₂, x₁² × x₂, etc.
    
    Uses sklearn's PolynomialFeatures under the hood with smart defaults.
    
    Useful for:
    - Non-linear patterns
    - Modeling complex relationships
    - Revenue forecasting (price × quantity, etc.)
    
    Parameters
    ----------
    degree : int, optional
        Degree of polynomial features. Default: 2
    interaction_only : bool, optional
        If True, only interaction features (no x², x³). Default: False
    include_bias : bool, optional
        If True, include bias column (all 1s). Default: False
    max_features : int, optional
        Maximum number of input features to consider. Default: 10
        (polynomial features explode quickly with many features)
    """
    
    def __init__(
        self,
        degree: int = 2,
        interaction_only: bool = False,
        include_bias: bool = False,
        max_features: int = 10,
    ):
        self.degree = degree
        self.interaction_only = interaction_only
        self.include_bias = include_bias
        self.max_features = max_features
        self.poly_: Optional[SklearnPolynomial] = None
        self.numeric_cols_: List[str] = []
        self.feature_names_: List[str] = []
    
    def fit(self, X: pd.DataFrame, y=None):
        """Fit the polynomial transformer."""
        df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        
        # Get numeric columns
        if isinstance(df, pd.DataFrame):
            self.numeric_cols_ = df.select_dtypes(include=[np.number]).columns.tolist()
        else:
            self.numeric_cols_ = [f"x{i}" for i in range(df.shape[1])]
        
        # Limit to max_features to avoid explosion
        if len(self.numeric_cols_) > self.max_features:
            logger.warning(
                f"Too many numeric features ({len(self.numeric_cols_)}) for polynomial features. "
                f"Using top {self.max_features} features."
            )
            self.numeric_cols_ = self.numeric_cols_[:self.max_features]
        
        # Initialize sklearn's PolynomialFeatures
        self.poly_ = SklearnPolynomial(
            degree=self.degree,
            interaction_only=self.interaction_only,
            include_bias=self.include_bias,
        )
        
        # Fit on selected columns
        if len(self.numeric_cols_) > 0:
            X_subset = df[self.numeric_cols_] if isinstance(df, pd.DataFrame) else df[:, :len(self.numeric_cols_)]
            self.poly_.fit(X_subset)
            self.feature_names_ = self._generate_feature_names()
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate polynomial features."""
        df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        
        if self.poly_ is None or len(self.numeric_cols_) == 0:
            return pd.DataFrame(index=df.index)
        
        # Get subset of numeric columns
        available_cols = [col for col in self.numeric_cols_ if col in df.columns]
        
        if len(available_cols) == 0:
            return pd.DataFrame(index=df.index)
        
        X_subset = df[available_cols]
        
        # Transform
        X_poly = self.poly_.transform(X_subset)
        
        # Create DataFrame with proper names
        out = pd.DataFrame(
            X_poly,
            index=df.index,
            columns=self.feature_names_[:X_poly.shape[1]]
        )
        
        # Remove original features (they're already in the dataset)
        # Keep only the new polynomial/interaction features
        if not self.include_bias:
            # Skip first len(available_cols) columns (original features)
            out = out.iloc[:, len(available_cols):]
        
        return out
    
    def _generate_feature_names(self) -> List[str]:
        """Generate human-readable feature names."""
        if self.poly_ is None:
            return []
        
        # Get feature names from sklearn
        try:
            names = self.poly_.get_feature_names_out(self.numeric_cols_)
            # Clean up names (sklearn uses 'x0 x1' format)
            return [name.replace(' ', '_') for name in names]
        except Exception:
            # Fallback: generate simple names
            n_features = self.poly_.n_output_features_
            return [f"poly_{i}" for i in range(n_features)]
    
    def get_feature_names_out(self, input_features=None):
        """Get output feature names."""
        return self.feature_names_ if self.feature_names_ else []


class RatioFeatures(BaseEstimator, TransformerMixin):
    """Create ratio and proportion features from numeric columns.
    
    Generates domain-agnostic ratio features:
    - Simple ratios: A / B
    - Proportions: A / (A + B)
    - Log ratios: log(A / B)
    
    Useful for:
    - Conversion rates (clicks / impressions)
    - Market share (product_sales / total_sales)
    - Financial ratios (debt / income)
    
    Parameters
    ----------
    include_proportions : bool, optional
        Include A / (A + B) style proportions. Default: True
    include_log_ratios : bool, optional
        Include log(A / B) ratios. Default: False
    max_features : int, optional
        Maximum number of feature pairs. Default: 50
    specific_pairs : List[Tuple[str, str]], optional
        Specific pairs to create ratios for.
    """
    
    def __init__(
        self,
        include_proportions: bool = True,
        include_log_ratios: bool = False,
        max_features: int = 50,
        specific_pairs: Optional[List[Tuple[str, str]]] = None,
    ):
        self.include_proportions = include_proportions
        self.include_log_ratios = include_log_ratios
        self.max_features = max_features
        self.specific_pairs = specific_pairs
        self.feature_names_: List[str] = []
        self.numeric_cols_: List[str] = []
    
    def fit(self, X: pd.DataFrame, y=None):
        """Fit the transformer."""
        if isinstance(X, pd.DataFrame):
            self.numeric_cols_ = X.select_dtypes(include=[np.number]).columns.tolist()
        else:
            self.numeric_cols_ = [f"x{i}" for i in range(X.shape[1])]
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate ratio features."""
        df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        out = pd.DataFrame(index=df.index)
        
        # Determine pairs
        if self.specific_pairs:
            pairs = self.specific_pairs
        else:
            numeric_cols = [col for col in self.numeric_cols_ if col in df.columns]
            all_pairs = list(combinations(numeric_cols, 2))
            
            if len(all_pairs) > self.max_features:
                pairs = all_pairs[:self.max_features]
            else:
                pairs = all_pairs
        
        for col1, col2 in pairs:
            if col1 not in df.columns or col2 not in df.columns:
                continue
            
            # Simple ratio: A / B
            out[f"{col1}_ratio_{col2}"] = df[col1] / df[col2].replace(0, np.nan)
            
            # Proportion: A / (A + B)
            if self.include_proportions:
                total = df[col1] + df[col2]
                out[f"{col1}_prop_of_sum"] = df[col1] / total.replace(0, np.nan)
                out[f"{col2}_prop_of_sum"] = df[col2] / total.replace(0, np.nan)
            
            # Log ratio: log(A / B)
            if self.include_log_ratios:
                # Safe log ratio (handle zeros and negatives)
                ratio = df[col1] / df[col2].replace(0, np.nan)
                out[f"{col1}_log_ratio_{col2}"] = np.log(np.abs(ratio) + 1e-8)
        
        self.feature_names_ = list(out.columns)
        return out
    
    def get_feature_names_out(self, input_features=None):
        """Get output feature names."""
        return self.feature_names_ if self.feature_names_ else []


class ProductInteractions(BaseEstimator, TransformerMixin):
    """Create multi-way product interactions.
    
    Generates products of 2 or more features:
    - 2-way: A × B
    - 3-way: A × B × C
    - N-way: A × B × C × ... × N
    
    Useful for:
    - Multi-factor effects (price × quantity × discount)
    - Complex interactions in business metrics
    
    Parameters
    ----------
    n_way : int, optional
        Number of features to multiply together. Default: 3
    max_interactions : int, optional
        Maximum number of product interactions to create. Default: 20
    specific_tuples : List[Tuple[str, ...]], optional
        Specific feature tuples to multiply.
    """
    
    def __init__(
        self,
        n_way: int = 3,
        max_interactions: int = 20,
        specific_tuples: Optional[List[Tuple[str, ...]]] = None,
    ):
        self.n_way = n_way
        self.max_interactions = max_interactions
        self.specific_tuples = specific_tuples
        self.feature_names_: List[str] = []
        self.numeric_cols_: List[str] = []
    
    def fit(self, X: pd.DataFrame, y=None):
        """Fit the transformer."""
        if isinstance(X, pd.DataFrame):
            self.numeric_cols_ = X.select_dtypes(include=[np.number]).columns.tolist()
        else:
            self.numeric_cols_ = [f"x{i}" for i in range(X.shape[1])]
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate product interactions."""
        df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        out = pd.DataFrame(index=df.index)
        
        # Determine tuples
        if self.specific_tuples:
            tuples = self.specific_tuples
        else:
            numeric_cols = [col for col in self.numeric_cols_ if col in df.columns]
            
            if len(numeric_cols) < self.n_way:
                logger.warning(
                    f"Not enough numeric features ({len(numeric_cols)}) for {self.n_way}-way interactions"
                )
                return out
            
            all_tuples = list(combinations(numeric_cols, self.n_way))
            
            if len(all_tuples) > self.max_interactions:
                tuples = all_tuples[:self.max_interactions]
            else:
                tuples = all_tuples
        
        # Generate products
        for feature_tuple in tuples:
            if not all(col in df.columns for col in feature_tuple):
                continue
            
            # Start with first feature
            product = df[feature_tuple[0]].copy()
            
            # Multiply by remaining features
            for col in feature_tuple[1:]:
                product = product * df[col]
            
            # Create feature name
            feature_name = "_x_".join(feature_tuple)
            out[feature_name] = product
        
        self.feature_names_ = list(out.columns)
        return out
    
    def get_feature_names_out(self, input_features=None):
        """Get output feature names."""
        return self.feature_names_ if self.feature_names_ else []


class CategoricalNumericInteractions(BaseEstimator, TransformerMixin):
    """Create interactions between categorical and numeric features.
    
    For each (categorical, numeric) pair, creates:
    - Group statistics: mean, std, min, max per category
    - Deviations: (value - group_mean) / group_std
    
    Useful for:
    - Segment-specific patterns (age_group × income)
    - Category-conditional effects
    
    Parameters
    ----------
    strategy : str, optional
        Strategy for interactions: 'group_stats', 'deviation', 'both'.
        Default: 'both'
    max_pairs : int, optional
        Maximum number of (categorical, numeric) pairs. Default: 20
    specific_pairs : List[Tuple[str, str]], optional
        Specific (categorical, numeric) pairs.
    """
    
    def __init__(
        self,
        strategy: str = 'both',
        max_pairs: int = 20,
        specific_pairs: Optional[List[Tuple[str, str]]] = None,
    ):
        self.strategy = strategy
        self.max_pairs = max_pairs
        self.specific_pairs = specific_pairs
        self.feature_names_: List[str] = []
        self.group_stats_: dict = {}
        self.categorical_cols_: List[str] = []
        self.numeric_cols_: List[str] = []
    
    def fit(self, X: pd.DataFrame, y=None):
        """Fit the transformer (learn group statistics)."""
        df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X

        # Identify categorical and numeric columns
        if isinstance(df, pd.DataFrame):
            # Original categorical columns
            original_cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

            # Detect one-hot encoded columns (numeric columns with 0/1 values that likely represent categories)
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            one_hot_cols = []

            for col in numeric_cols:
                # Check if this looks like a one-hot encoded column
                unique_vals = df[col].dropna().unique()
                if (len(unique_vals) <= 3 and  # Few unique values (typically 0, 1, maybe NaN)
                    set(unique_vals).issubset({0, 1, np.nan}) and  # Only 0s and 1s
                    df[col].sum() > 0):  # Has at least some 1s
                    one_hot_cols.append(col)

            self.categorical_cols_ = original_cat_cols + one_hot_cols
            self.numeric_cols_ = [col for col in numeric_cols if col not in one_hot_cols]
        else:
            # Cannot determine types from array
            logger.warning("Array input not supported for CategoricalNumericInteractions")
            return self
        
        # Determine pairs
        pairs = self._get_pairs(df)
        
        # Compute group statistics for each pair
        for cat_col, num_col in pairs:
            group_key = f"{cat_col}___{num_col}"
            
            # Compute statistics
            grouped = df.groupby(cat_col)[num_col]
            
            self.group_stats_[group_key] = {
                'mean': grouped.mean().to_dict(),
                'std': grouped.std().to_dict(),
                'min': grouped.min().to_dict(),
                'max': grouped.max().to_dict(),
                'count': grouped.count().to_dict(),
            }
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate categorical-numeric interactions."""
        df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        out = pd.DataFrame(index=df.index)
        
        if not isinstance(df, pd.DataFrame):
            return out
        
        # Determine pairs
        pairs = self._get_pairs(df)
        
        for cat_col, num_col in pairs:
            if cat_col not in df.columns or num_col not in df.columns:
                continue
            
            group_key = f"{cat_col}___{num_col}"
            
            if group_key not in self.group_stats_:
                continue
            
            stats = self.group_stats_[group_key]
            
            # Collect new columns in a dictionary to avoid DataFrame fragmentation
            new_columns = {}

            # Map group statistics
            if self.strategy in ['group_stats', 'both']:
                new_columns[f"{cat_col}_{num_col}_mean"] = df[cat_col].map(stats['mean'])
                new_columns[f"{cat_col}_{num_col}_std"] = df[cat_col].map(stats['std'])
                new_columns[f"{cat_col}_{num_col}_min"] = df[cat_col].map(stats['min'])
                new_columns[f"{cat_col}_{num_col}_max"] = df[cat_col].map(stats['max'])

            # Compute deviations
            if self.strategy in ['deviation', 'both']:
                group_mean = df[cat_col].map(stats['mean'])
                group_std = df[cat_col].map(stats['std']).replace(0, 1)  # Avoid division by zero

                # Standardized deviation
                new_columns[f"{cat_col}_{num_col}_deviation"] = (df[num_col] - group_mean) / group_std

                # Raw deviation
                new_columns[f"{cat_col}_{num_col}_diff"] = df[num_col] - group_mean

            # Add all new columns at once using concat to avoid fragmentation
            if new_columns:
                out = pd.concat([out, pd.DataFrame(new_columns, index=df.index)], axis=1)
        
        self.feature_names_ = list(out.columns)
        return out
    
    def _get_pairs(self, df: pd.DataFrame) -> List[Tuple[str, str]]:
        """Get categorical-numeric pairs to process."""
        if self.specific_pairs:
            return self.specific_pairs
        
        # Get available columns
        cat_cols = [col for col in self.categorical_cols_ if col in df.columns]
        num_cols = [col for col in self.numeric_cols_ if col in df.columns]
        
        # Generate all pairs
        all_pairs = [(cat, num) for cat in cat_cols for num in num_cols]
        
        # Limit to max_pairs
        if len(all_pairs) > self.max_pairs:
            return all_pairs[:self.max_pairs]
        
        return all_pairs
    
    def get_feature_names_out(self, input_features=None):
        """Get output feature names."""
        return self.feature_names_ if self.feature_names_ else []


class BinnedInteractions(BaseEstimator, TransformerMixin):
    """Create interactions between binned numeric features.
    
    First bins numeric features into categories, then creates interactions
    between the binned features and other numeric features.
    
    Useful for:
    - Capturing non-linear effects through binning
    - Segment-specific patterns (age_bin × income)
    
    Parameters
    ----------
    n_bins : int, optional
        Number of bins for quantile binning. Default: 5
    features_to_bin : List[str], optional
        Specific features to bin. If None, bins all numeric features.
    max_bins : int, optional
        Maximum number of features to bin. Default: 5
    """
    
    def __init__(
        self,
        n_bins: int = 5,
        features_to_bin: Optional[List[str]] = None,
        max_bins: int = 5,
    ):
        self.n_bins = n_bins
        self.features_to_bin = features_to_bin
        self.max_bins = max_bins
        self.feature_names_: List[str] = []
        self.bin_edges_: dict = {}
        self.numeric_cols_: List[str] = []
    
    def fit(self, X: pd.DataFrame, y=None):
        """Fit the transformer (learn bin edges)."""
        df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        
        # Get numeric columns
        if isinstance(df, pd.DataFrame):
            self.numeric_cols_ = df.select_dtypes(include=[np.number]).columns.tolist()
        else:
            self.numeric_cols_ = [f"x{i}" for i in range(df.shape[1])]
        
        # Determine which features to bin
        if self.features_to_bin:
            to_bin = [col for col in self.features_to_bin if col in self.numeric_cols_]
        else:
            to_bin = self.numeric_cols_[:self.max_bins]
        
        # Learn bin edges for each feature
        for col in to_bin:
            if col not in df.columns:
                continue
            
            # Use quantile-based binning
            try:
                _, bin_edges = pd.qcut(
                    df[col],
                    q=self.n_bins,
                    labels=False,
                    retbins=True,
                    duplicates='drop'
                )
                self.bin_edges_[col] = bin_edges
            except (ValueError, TypeError) as e:
                logger.warning(f"Could not bin feature {col}: {e}")
                continue
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate binned interactions."""
        df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        out = pd.DataFrame(index=df.index)
        
        # Create binned features
        binned_df = pd.DataFrame(index=df.index)
        
        for col, bin_edges in self.bin_edges_.items():
            if col not in df.columns:
                continue
            
            # Bin the feature
            binned_col = f"{col}_binned"
            binned_df[binned_col] = pd.cut(
                df[col],
                bins=bin_edges,
                labels=False,
                include_lowest=True,
                duplicates='drop'
            ).fillna(-1).astype(int)
        
        # Now create interactions between binned and numeric features
        # Use CategoricalNumericInteractions
        if len(binned_df.columns) > 0:
            # Treat binned features as categorical
            cat_num_transformer = CategoricalNumericInteractions(
                strategy='group_stats',
                max_pairs=50
            )
            
            # Combine binned features with original numeric features
            combined = pd.concat([binned_df, df[self.numeric_cols_]], axis=1)
            
            # Fit and transform
            cat_num_transformer.fit(combined)
            out = cat_num_transformer.transform(combined)
        
        self.feature_names_ = list(out.columns)
        return out
    
    def get_feature_names_out(self, input_features=None):
        """Get output feature names."""
        return self.feature_names_ if self.feature_names_ else []


def build_interaction_pipeline(
    cfg: 'FeatureCraftConfig',
    X: pd.DataFrame,
) -> List[Tuple[str, BaseEstimator]]:
    """Build a comprehensive feature interaction pipeline.
    
    Args:
        cfg: FeatureCraftConfig instance
        X: Input DataFrame
    
    Returns:
        List of (name, transformer) tuples for use in ColumnTransformer
    """
    transformers = []
    
    # 1. Arithmetic Interactions
    if cfg.interactions_use_arithmetic:
        transformers.append((
            'arithmetic',
            ArithmeticInteractions(
                operations=cfg.interactions_arithmetic_ops,
                max_features=cfg.interactions_max_arithmetic_pairs,
                symmetric=True,
            )
        ))
    
    # 2. Polynomial Features
    if cfg.interactions_use_polynomial:
        transformers.append((
            'polynomial',
            PolynomialInteractions(
                degree=cfg.interactions_polynomial_degree,
                interaction_only=cfg.interactions_polynomial_interaction_only,
                include_bias=False,
                max_features=cfg.interactions_polynomial_max_features,
            )
        ))
    
    # 3. Ratio Features
    if cfg.interactions_use_ratios:
        transformers.append((
            'ratios',
            RatioFeatures(
                include_proportions=cfg.interactions_ratios_include_proportions,
                include_log_ratios=cfg.interactions_ratios_include_log,
                max_features=cfg.interactions_max_ratio_pairs,
            )
        ))
    
    # 4. Product Interactions (3-way)
    if cfg.interactions_use_products:
        transformers.append((
            'products',
            ProductInteractions(
                n_way=cfg.interactions_product_n_way,
                max_interactions=cfg.interactions_max_products,
            )
        ))
    
    # 5. Categorical × Numeric Interactions
    if cfg.interactions_use_categorical_numeric:
        transformers.append((
            'cat_num',
            CategoricalNumericInteractions(
                strategy=cfg.interactions_cat_num_strategy,
                max_pairs=cfg.interactions_max_cat_num_pairs,
            )
        ))
    
    # 6. Binned Interactions
    if cfg.interactions_use_binned:
        transformers.append((
            'binned',
            BinnedInteractions(
                n_bins=cfg.interactions_n_bins,
                max_bins=cfg.interactions_max_features_to_bin,
            )
        ))
    
    return transformers

