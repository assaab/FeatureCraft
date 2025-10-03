"""Plan validation with safety checks for leakage prevention and schema compliance."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

from .schemas import DatasetContext, FeaturePlan, FeatureSpec, ValidationResult
from ..logging import get_logger

logger = get_logger(__name__)


# ============================================================================
# Policy Validator
# ============================================================================

class PolicyValidator:
    """Validator for feature engineering plans with safety checks.
    
    This validator ensures that generated feature plans are safe, valid,
    and free from common pitfalls like data leakage, schema violations,
    and time-ordering issues.
    
    Checks performed:
    - Leakage detection (target column references, future data)
    - Schema validation (column existence, type compatibility)
    - Time-ordering (lags, windows, cutoff times)
    - Feature name uniqueness
    - Parameter validity
    
    Example:
        >>> validator = PolicyValidator()
        >>> result = validator.validate(plan, context)
        >>> if not result.is_valid:
        ...     print(f"Validation failed: {result.errors}")
    """
    
    def __init__(
        self,
        strict_mode: bool = False,
        enable_leakage_checks: bool = True,
        enable_schema_checks: bool = True,
        enable_time_checks: bool = True,
    ):
        """Initialize validator.
        
        Args:
            strict_mode: If True, warnings are treated as errors
            enable_leakage_checks: Enable leakage detection
            enable_schema_checks: Enable schema validation
            enable_time_checks: Enable time-ordering checks
        """
        self.strict_mode = strict_mode
        self.enable_leakage_checks = enable_leakage_checks
        self.enable_schema_checks = enable_schema_checks
        self.enable_time_checks = enable_time_checks
    
    def validate(
        self,
        plan: FeaturePlan,
        context: DatasetContext | None = None,
    ) -> ValidationResult:
        """Validate feature plan.
        
        Args:
            plan: Feature plan to validate
            context: Dataset context (optional but recommended)
            
        Returns:
            ValidationResult with errors, warnings, and pass/fail status
        """
        errors: list[str] = []
        warnings: list[str] = []
        checks_passed: dict[str, bool] = {}
        
        # Check 1: Feature name uniqueness
        check_result = self._check_feature_names(plan)
        checks_passed["feature_names"] = check_result[0]
        errors.extend(check_result[1])
        
        # Check 2: Schema validation
        if self.enable_schema_checks and context:
            check_result = self._check_schema(plan, context)
            checks_passed["schema"] = check_result[0]
            errors.extend(check_result[1])
            warnings.extend(check_result[2])
        
        # Check 3: Leakage detection
        if self.enable_leakage_checks and context:
            check_result = self._check_leakage(plan, context)
            checks_passed["leakage"] = check_result[0]
            errors.extend(check_result[1])
            warnings.extend(check_result[2])
        
        # Check 4: Time-ordering validation
        if self.enable_time_checks:
            check_result = self._check_time_ordering(plan)
            checks_passed["time_ordering"] = check_result[0]
            errors.extend(check_result[1])
            warnings.extend(check_result[2])
        
        # Check 5: Parameter validation
        check_result = self._check_parameters(plan)
        checks_passed["parameters"] = check_result[0]
        errors.extend(check_result[1])
        warnings.extend(check_result[2])
        
        # Treat warnings as errors in strict mode
        if self.strict_mode and warnings:
            errors.extend([f"[STRICT] {w}" for w in warnings])
            warnings = []
        
        # Build result
        is_valid = len(errors) == 0
        
        result = ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            checks_passed=checks_passed,
            metadata={
                "total_features": len(plan.candidates),
                "checks_run": len(checks_passed),
                "strict_mode": self.strict_mode,
            },
        )
        
        if is_valid:
            logger.info(f"✓ Plan validation passed: {len(plan.candidates)} features")
        else:
            logger.error(f"✗ Plan validation failed: {len(errors)} errors")
        
        return result
    
    def _check_feature_names(
        self, plan: FeaturePlan
    ) -> tuple[bool, list[str]]:
        """Check feature name uniqueness and validity."""
        errors: list[str] = []
        
        # Check uniqueness
        names = [f.name for f in plan.candidates]
        duplicates = [name for name in names if names.count(name) > 1]
        if duplicates:
            errors.append(
                f"Duplicate feature names: {set(duplicates)}"
            )
        
        # Check name validity (alphanumeric + underscore)
        invalid_pattern = re.compile(r'^[a-zA-Z][a-zA-Z0-9_]*$')
        for feat in plan.candidates:
            if not invalid_pattern.match(feat.name):
                errors.append(
                    f"Invalid feature name '{feat.name}': must start with letter, "
                    "contain only alphanumeric and underscore"
                )
        
        return len(errors) == 0, errors
    
    def _check_schema(
        self, plan: FeaturePlan, context: DatasetContext
    ) -> tuple[bool, list[str], list[str]]:
        """Validate feature specs against dataset schema."""
        errors: list[str] = []
        warnings: list[str] = []
        
        available_cols = set(context.schema.keys())
        
        for feat in plan.candidates:
            # Check source columns exist
            source_cols = (
                [feat.source_col] if isinstance(feat.source_col, str)
                else feat.source_col
            )
            
            for col in source_cols:
                if col not in available_cols:
                    errors.append(
                        f"Feature '{feat.name}': source column '{col}' not found in schema"
                    )
            
            # Check key/time columns exist
            if feat.key_col and feat.key_col not in available_cols:
                errors.append(
                    f"Feature '{feat.name}': key_col '{feat.key_col}' not found"
                )
            
            if feat.time_col and feat.time_col not in available_cols:
                errors.append(
                    f"Feature '{feat.name}': time_col '{feat.time_col}' not found"
                )
            
            # Check type compatibility
            if feat.type in ["rolling_mean", "rolling_sum", "rolling_std", "lag"]:
                # These require numeric source columns
                for col in source_cols:
                    if col in context.schema:
                        dtype = context.schema[col]
                        if "int" not in dtype and "float" not in dtype:
                            warnings.append(
                                f"Feature '{feat.name}': '{col}' is {dtype}, "
                                f"but {feat.type} typically requires numeric column"
                            )
        
        return len(errors) == 0, errors, warnings
    
    def _check_leakage(
        self, plan: FeaturePlan, context: DatasetContext
    ) -> tuple[bool, list[str], list[str]]:
        """Detect potential data leakage."""
        errors: list[str] = []
        warnings: list[str] = []
        
        target = context.target
        blocklist = plan.constraints.get("leakage_blocklist", [])
        
        # Common leakage patterns
        leakage_patterns = [
            target.lower(),
            f"{target}_date",
            f"{target}_time",
            "prediction",
            "label",
            "ground_truth",
        ]
        
        # Add custom blocklist
        leakage_patterns.extend([b.lower() for b in blocklist])
        
        for feat in plan.candidates:
            # Check source columns
            source_cols = (
                [feat.source_col] if isinstance(feat.source_col, str)
                else feat.source_col
            )
            
            for col in source_cols:
                col_lower = col.lower()
                
                # Direct target reference
                if col_lower == target.lower():
                    errors.append(
                        f"Feature '{feat.name}': LEAKAGE - direct target reference '{col}'"
                    )
                
                # Suspicious column names
                for pattern in leakage_patterns:
                    if pattern in col_lower and col_lower != target.lower():
                        warnings.append(
                            f"Feature '{feat.name}': possible leakage - '{col}' "
                            f"matches pattern '{pattern}'"
                        )
            
            # Check for target encoding without out-of-fold protection
            if feat.type == "target_encode":
                if "out_of_fold" not in feat.safety_tags and "oof" not in feat.safety_tags:
                    warnings.append(
                        f"Feature '{feat.name}': target encoding without out-of-fold tag. "
                        "Ensure proper CV is used to prevent leakage."
                    )
            
            # Check for future data in time-series
            if context.time_col and feat.type in ["lag", "rolling_mean"]:
                if not feat.time_col:
                    warnings.append(
                        f"Feature '{feat.name}': time-dependent feature but no time_col specified"
                    )
        
        return len(errors) == 0, errors, warnings
    
    def _check_time_ordering(
        self, plan: FeaturePlan
    ) -> tuple[bool, list[str], list[str]]:
        """Validate time-ordering for temporal features."""
        errors: list[str] = []
        warnings: list[str] = []
        
        for feat in plan.candidates:
            # Check lag features have valid lag values
            if feat.type == "lag":
                lag = feat.params.get("lag", 1)
                if lag <= 0:
                    errors.append(
                        f"Feature '{feat.name}': lag must be positive, got {lag}"
                    )
            
            # Check window specifications
            if feat.window:
                if not self._is_valid_window(feat.window):
                    errors.append(
                        f"Feature '{feat.name}': invalid window '{feat.window}'. "
                        "Use format like '7d', '30d', '1h', '15m', '4w'"
                    )
            
            # Warn if time-dependent feature lacks time column
            time_dependent_types = [
                "rolling_mean", "rolling_sum", "lag", "diff",
                "ewm", "expanding_mean", "recency"
            ]
            if feat.type in time_dependent_types:
                if not feat.time_col and "time_safe" not in feat.safety_tags:
                    warnings.append(
                        f"Feature '{feat.name}': time-dependent type '{feat.type}' "
                        "but no time_col specified"
                    )
        
        return len(errors) == 0, errors, warnings
    
    def _check_parameters(
        self, plan: FeaturePlan
    ) -> tuple[bool, list[str], list[str]]:
        """Validate feature parameters."""
        errors: list[str] = []
        warnings: list[str] = []
        
        for feat in plan.candidates:
            # Check required parameters per feature type
            if feat.type in ["rolling_mean", "rolling_sum", "rolling_std"]:
                if not feat.window:
                    errors.append(
                        f"Feature '{feat.name}': type '{feat.type}' requires window parameter"
                    )
            
            # Check smoothing for target encoding
            if feat.type == "target_encode":
                smoothing = feat.params.get("smoothing", 0)
                if smoothing < 0:
                    errors.append(
                        f"Feature '{feat.name}': smoothing must be >= 0, got {smoothing}"
                    )
            
            # Check binning parameters
            if feat.type in ["quantile_bin", "custom_bin"]:
                n_bins = feat.params.get("n_bins", 5)
                if n_bins < 2:
                    errors.append(
                        f"Feature '{feat.name}': n_bins must be >= 2, got {n_bins}"
                    )
        
        return len(errors) == 0, errors, warnings
    
    @staticmethod
    def _is_valid_window(window: str) -> bool:
        """Check if window specification is valid."""
        # Match patterns like "7d", "30d", "1h", "15m", "4w", "2M"
        # Supports: d (days), h (hours), m (minutes), s (seconds), w (weeks), M (months), Y (years)
        pattern = re.compile(r'^\d+[dhmsSwMY]$')
        return bool(pattern.match(window))


# ============================================================================
# Convenience Functions
# ============================================================================

def validate_plan(
    plan: FeaturePlan,
    context: DatasetContext | None = None,
    strict_mode: bool = False,
) -> ValidationResult:
    """Validate feature plan (convenience function).
    
    Args:
        plan: Feature plan to validate
        context: Dataset context (optional)
        strict_mode: Treat warnings as errors
        
    Returns:
        ValidationResult
        
    Example:
        >>> result = validate_plan(plan, context)
        >>> if result.is_valid:
        ...     print("Plan is safe to execute")
        >>> else:
        ...     print(f"Validation errors: {result.errors}")
    """
    validator = PolicyValidator(strict_mode=strict_mode)
    return validator.validate(plan, context)


def detect_leakage_columns(
    df: pd.DataFrame,
    target: str,
    time_col: str | None = None,
) -> dict[str, str]:
    """Detect potential leakage columns in dataset.
    
    This is a heuristic-based detector that flags suspicious columns
    that might cause data leakage.
    
    Args:
        df: Input DataFrame
        target: Target column name
        time_col: Time column (if time-series)
        
    Returns:
        Dict mapping column name to risk level (high/medium/low)
        
    Example:
        >>> leakage = detect_leakage_columns(df, target="churn")
        >>> for col, risk in leakage.items():
        ...     if risk == "high":
        ...         print(f"Remove column: {col}")
    """
    leakage = {}
    
    target_lower = target.lower()
    
    for col in df.columns:
        if col == target:
            continue
        
        col_lower = col.lower()
        risk = None
        
        # High risk patterns
        high_risk_patterns = [
            f"{target_lower}_date",
            f"{target_lower}_time",
            f"{target_lower}_flag",
            "prediction",
            "label",
            "ground_truth",
            "actual",
        ]
        
        for pattern in high_risk_patterns:
            if pattern in col_lower:
                risk = "high"
                break
        
        # Medium risk: Perfect or near-perfect correlation
        if risk is None and pd.api.types.is_numeric_dtype(df[col]):
            if target in df.columns and pd.api.types.is_numeric_dtype(df[target]):
                try:
                    corr = abs(df[col].corr(df[target]))
                    if corr > 0.99:
                        risk = "high"
                    elif corr > 0.95:
                        risk = "medium"
                except Exception:
                    pass
        
        # Low risk: ID-like columns (high cardinality)
        if risk is None:
            if df[col].nunique() / len(df) > 0.95:
                risk = "low"
        
        if risk:
            leakage[col] = risk
    
    return leakage

