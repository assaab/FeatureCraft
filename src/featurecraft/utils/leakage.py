"""Leakage prevention utilities for FeatureCraft transformers."""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd


class LeakageGuardMixin:
    """Mixin to prevent target leakage during transform().
    
    This mixin should be inherited by transformers that accept y during fit()
    but must NOT receive y during transform() in production inference.
    
    Usage:
        class MyEncoder(BaseEstimator, TransformerMixin, LeakageGuardMixin):
            def __init__(self, raise_on_target_in_transform: bool = True):
                self.raise_on_target_in_transform = raise_on_target_in_transform
                
            def fit(self, X, y):
                # Fit logic here
                return self
                
            def transform(self, X, y=None):
                self.ensure_no_target_in_transform(y)  # Guard
                # Transform logic here
                return X_transformed
    
    Attributes:
        raise_on_target_in_transform: If True, raises ValueError when y is not None during transform()
    """
    
    raise_on_target_in_transform: bool = True
    
    def ensure_no_target_in_transform(self, y: Optional[Any]) -> None:
        """Ensure that target (y) is not provided during transform().
        
        This is a critical check to prevent label leakage in production pipelines.
        When raise_on_target_in_transform is True, raises an error if y is not None.
        
        Args:
            y: Target variable (should be None during transform)
            
        Raises:
            ValueError: If y is not None and raise_on_target_in_transform is True
            
        Example:
            >>> encoder = MyEncoder(raise_on_target_in_transform=True)
            >>> encoder.fit(X_train, y_train)
            >>> # This will succeed
            >>> X_test_transformed = encoder.transform(X_test)
            >>> # This will raise ValueError
            >>> X_test_transformed = encoder.transform(X_test, y_test)
        """
        if not getattr(self, 'raise_on_target_in_transform', True):
            # Guard disabled - allow y to be passed (for backwards compatibility)
            return
            
        if y is not None:
            # Check if y is empty/trivial (e.g., empty Series/array)
            if isinstance(y, (pd.Series, pd.DataFrame)):
                if len(y) == 0:
                    return  # Empty - not a leak
            elif hasattr(y, '__len__'):
                if len(y) == 0:
                    return  # Empty - not a leak
            
            # y is not None and not empty - this is a potential leakage
            transformer_name = self.__class__.__name__
            raise ValueError(
                f"{transformer_name}.transform() received target variable (y={type(y).__name__}). "
                f"This is a potential label leakage risk in production pipelines. "
                f"transform() should only receive features (X), never the target (y). "
                f"If this is intentional (e.g., for fit_transform during training), "
                f"set raise_on_target_in_transform=False or use fit_transform() instead. "
                f"To disable this check globally, set config.raise_on_target_in_transform=False."
            )


def ensure_no_target_in_transform(y: Optional[Any], transformer_name: str = "Transformer") -> None:
    """Standalone helper to ensure target is not passed to transform().
    
    This is a convenience function that can be used in transformers that don't inherit LeakageGuardMixin.
    
    Args:
        y: Target variable (should be None)
        transformer_name: Name of the transformer (for error message)
        
    Raises:
        ValueError: If y is not None
        
    Example:
        >>> def transform(self, X, y=None):
        ...     ensure_no_target_in_transform(y, self.__class__.__name__)
        ...     return self._transform_logic(X)
    """
    if y is not None:
        # Check if y is empty/trivial
        if isinstance(y, (pd.Series, pd.DataFrame)):
            if len(y) == 0:
                return
        elif hasattr(y, '__len__'):
            if len(y) == 0:
                return
        
        raise ValueError(
            f"{transformer_name}.transform() received target variable (y={type(y).__name__}). "
            f"This is a potential label leakage risk. "
            f"transform() should only receive features (X), never the target (y)."
        )

