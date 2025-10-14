"""Inspector Module: Dataset fingerprinting and quality checks."""

from __future__ import annotations

import hashlib
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

from ..config import FeatureCraftConfig
from ..insights import detect_task, profile_columns
from ..logging import get_logger
from ..types import TaskType, Issue
from ..utils import (
    is_numeric_series,
    is_datetime_series,
    is_text_candidate,
    calc_cardinality,
    calc_missing_rate,
    calc_skewness,
    calc_outlier_share,
)
from .types import DatasetFingerprint

logger = get_logger(__name__)


class Inspector:
    """Dataset fingerprinting and quality analysis."""
    
    def __init__(self, config: FeatureCraftConfig):
        """Initialize inspector.
        
        Args:
            config: FeatureCraft configuration
        """
        self.config = config
    
    def fingerprint(self, X: pd.DataFrame, y: pd.Series) -> DatasetFingerprint:
        """Compute comprehensive dataset fingerprint.
        
        Args:
            X: Feature dataframe
            y: Target series
            
        Returns:
            DatasetFingerprint with all characteristics
        """
        logger.info(f"Computing dataset fingerprint for {X.shape[0]} rows x {X.shape[1]} cols")
        
        # Basic stats
        n_rows, n_cols = X.shape
        target_name = y.name or "target"
        task_type = detect_task(y)
        
        # Column type detection
        numeric_cols = [c for c in X.columns if is_numeric_series(X[c])]
        categorical_cols = [
            c for c in X.columns
            if not is_numeric_series(X[c])
            and not is_datetime_series(X[c])
            and not is_text_candidate(X[c])
        ]
        text_cols = [c for c in X.columns if is_text_candidate(X[c])]
        datetime_cols = [c for c in X.columns if is_datetime_series(X[c])]
        
        # Missing values
        missing_summary = {c: calc_missing_rate(X[c]) for c in X.columns}
        high_missing_cols = [c for c, rate in missing_summary.items() if rate > 0.3]
        
        # Constant columns
        constant_cols = [c for c in X.columns if X[c].nunique() <= 1]
        
        # Duplicate columns
        duplicate_col_groups = self._find_duplicate_columns(X)
        
        # Cardinality analysis
        cardinality_summary = {}
        low_card, mid_card, high_card, ultra_high_card = [], [], [], []
        rare_category_ratios = {}
        
        for col in categorical_cols:
            card = calc_cardinality(X[col])
            cardinality_summary[col] = card
            
            if card <= 10:
                low_card.append(col)
            elif card <= 50:
                mid_card.append(col)
            elif card <= 1000:
                high_card.append(col)
            else:
                ultra_high_card.append(col)
            
            # Rare category ratio
            vc = X[col].value_counts(normalize=True)
            rare_ratio = (vc < 0.01).sum() / len(vc) if len(vc) > 0 else 0
            rare_category_ratios[col] = rare_ratio
        
        # Numeric distributions
        skewness_summary = {}
        kurtosis_summary = {}
        heavily_skewed_cols = []
        outlier_share_summary = {}
        high_outlier_cols = []
        near_zero_variance_cols = []
        
        for col in numeric_cols:
            skew = calc_skewness(X[col])
            skewness_summary[col] = skew
            if abs(skew) > 1.5:
                heavily_skewed_cols.append(col)
            
            # Kurtosis
            try:
                kurt = float(X[col].dropna().kurtosis())
                kurtosis_summary[col] = kurt
            except:
                kurtosis_summary[col] = 0.0
            
            # Outliers
            outlier_share = calc_outlier_share(X[col])
            outlier_share_summary[col] = outlier_share
            if outlier_share > 0.05:
                high_outlier_cols.append(col)
            
            # Near-zero variance
            var = X[col].var()
            if var < 1e-4:
                near_zero_variance_cols.append(col)
        
        # Target analysis
        target_dtype = str(y.dtype)
        target_nunique = y.nunique()
        target_missing_rate = calc_missing_rate(y)
        
        class_balance = None
        minority_class_ratio = None
        is_imbalanced = False
        target_skewness = None
        
        if task_type == TaskType.CLASSIFICATION:
            vc = y.value_counts(normalize=True)
            class_balance = {str(k): float(v) for k, v in vc.items()}
            minority_class_ratio = float(vc.min()) if len(vc) > 0 else 0.0
            is_imbalanced = minority_class_ratio < 0.2
        else:
            target_skewness = calc_skewness(y)
        
        # Correlation analysis
        high_corr_pairs, multicollinear_groups, target_corr_top = self._analyze_correlations(
            X, y, numeric_cols
        )
        
        # Mutual information
        mi_scores, low_mi_cols = self._compute_mutual_info(X, y, task_type)
        
        # Time series detection
        time_column, is_time_series, time_granularity, time_gaps, has_regular_intervals = (
            self._detect_time_series(X, datetime_cols)
        )
        
        # Entity/group detection
        entity_columns = self._detect_entity_columns(X, categorical_cols)
        
        # Text analysis
        text_length_stats, text_vocab_size = self._analyze_text_columns(X, text_cols)
        
        # Leakage detection
        leakage_risk_cols, id_like_with_high_mi = self._detect_leakage_risks(
            X, y, numeric_cols, categorical_cols, mi_scores, high_corr_pairs
        )
        
        # Complexity estimates
        estimated_feature_count = self._estimate_feature_count(
            len(numeric_cols), len(categorical_cols), len(text_cols), cardinality_summary
        )
        estimated_memory_gb = self._estimate_memory(n_rows, estimated_feature_count)
        
        return DatasetFingerprint(
            n_rows=n_rows,
            n_cols=n_cols,
            n_features_original=n_cols,
            target_name=target_name,
            task_type=task_type,
            n_numeric=len(numeric_cols),
            n_categorical=len(categorical_cols),
            n_text=len(text_cols),
            n_datetime=len(datetime_cols),
            n_constant=len(constant_cols),
            n_duplicate=len(duplicate_col_groups),
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            text_cols=text_cols,
            datetime_cols=datetime_cols,
            missing_summary=missing_summary,
            high_missing_cols=high_missing_cols,
            constant_cols=constant_cols,
            duplicate_col_groups=duplicate_col_groups,
            cardinality_summary=cardinality_summary,
            low_cardinality_cols=low_card,
            mid_cardinality_cols=mid_card,
            high_cardinality_cols=high_card,
            ultra_high_cardinality_cols=ultra_high_card,
            rare_category_ratios=rare_category_ratios,
            skewness_summary=skewness_summary,
            kurtosis_summary=kurtosis_summary,
            heavily_skewed_cols=heavily_skewed_cols,
            outlier_share_summary=outlier_share_summary,
            high_outlier_cols=high_outlier_cols,
            near_zero_variance_cols=near_zero_variance_cols,
            target_dtype=target_dtype,
            target_nunique=target_nunique,
            target_missing_rate=target_missing_rate,
            class_balance=class_balance,
            minority_class_ratio=minority_class_ratio,
            is_imbalanced=is_imbalanced,
            target_skewness=target_skewness,
            high_correlation_pairs=high_corr_pairs,
            multicollinear_groups=multicollinear_groups,
            target_correlation_top=target_corr_top,
            mutual_info_scores=mi_scores,
            low_mi_cols=low_mi_cols,
            time_column=time_column,
            is_time_series=is_time_series,
            time_granularity=time_granularity,
            time_gaps=time_gaps,
            has_regular_intervals=has_regular_intervals,
            entity_columns=entity_columns,
            text_length_stats=text_length_stats,
            text_vocab_size=text_vocab_size,
            leakage_risk_cols=leakage_risk_cols,
            id_like_with_high_mi=id_like_with_high_mi,
            estimated_feature_count_baseline=estimated_feature_count,
            estimated_memory_gb=estimated_memory_gb,
        )
    
    def check_data_quality(self, X: pd.DataFrame, y: pd.Series) -> List[Issue]:
        """Check for data quality issues.
        
        Args:
            X: Feature dataframe
            y: Target series
            
        Returns:
            List of issues found
        """
        issues = []
        
        # Check for empty data
        if len(X) == 0:
            issues.append(Issue(
                severity="ERROR",
                code="EMPTY_DATA",
                column=None,
                message="Dataset is empty"
            ))
            return issues
        
        # Check target missing
        if y.isna().all():
            issues.append(Issue(
                severity="ERROR",
                code="TARGET_ALL_MISSING",
                column=y.name,
                message="Target variable is entirely missing"
            ))
        
        # Check high missingness
        for col in X.columns:
            missing_rate = calc_missing_rate(X[col])
            if missing_rate > 0.95:
                issues.append(Issue(
                    severity="WARN",
                    code="VERY_HIGH_MISSING",
                    column=col,
                    message=f"Column {col} has {missing_rate:.1%} missing values"
                ))
        
        return issues
    
    def estimate_leakage_risk(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Estimate leakage risk per feature.
        
        Args:
            X: Feature dataframe
            y: Target series
            
        Returns:
            Dictionary mapping column names to risk scores (0-1)
        """
        risk_scores = {}
        
        for col in X.columns:
            risk = 0.0
            
            # Check perfect correlation for numeric
            if is_numeric_series(X[col]) and is_numeric_series(y):
                corr = X[col].corr(y)
                if abs(corr) > 0.99:
                    risk += 0.5
            
            # Check suspicious names
            suspicious_terms = ["target", "label", "prediction", "outcome", "result"]
            if any(term in col.lower() for term in suspicious_terms):
                risk += 0.3
            
            # Check ID-like with high cardinality
            if calc_cardinality(X[col]) > 0.9 * len(X):
                risk += 0.2
            
            risk_scores[col] = min(risk, 1.0)
        
        return risk_scores
    
    # === Private helper methods ===
    
    def _find_duplicate_columns(self, X: pd.DataFrame) -> List[List[str]]:
        """Find groups of duplicate columns."""
        groups = []
        checked = set()
        
        for i, col1 in enumerate(X.columns):
            if col1 in checked:
                continue
            
            group = [col1]
            for col2 in X.columns[i+1:]:
                if col2 in checked:
                    continue
                
                try:
                    if X[col1].equals(X[col2]):
                        group.append(col2)
                        checked.add(col2)
                except:
                    pass
            
            if len(group) > 1:
                groups.append(group)
            checked.add(col1)
        
        return groups
    
    def _analyze_correlations(
        self, X: pd.DataFrame, y: pd.Series, numeric_cols: List[str]
    ) -> Tuple[List[Tuple[str, str, float]], List[List[str]], List[Tuple[str, float]]]:
        """Analyze correlations."""
        high_corr_pairs = []
        multicollinear_groups = []
        target_corr_top = []
        
        if len(numeric_cols) == 0:
            return high_corr_pairs, multicollinear_groups, target_corr_top
        
        # Pairwise correlations
        num_df = X[numeric_cols].fillna(X[numeric_cols].median())
        corr_matrix = num_df.corr()
        
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i+1:]:
                corr = corr_matrix.loc[col1, col2]
                if abs(corr) > 0.95:
                    high_corr_pairs.append((col1, col2, float(corr)))
        
        # Target correlations
        if is_numeric_series(y):
            target_corrs = []
            for col in numeric_cols:
                corr = X[col].corr(y)
                if not np.isnan(corr):
                    target_corrs.append((col, float(corr)))
            
            target_corrs.sort(key=lambda x: abs(x[1]), reverse=True)
            target_corr_top = target_corrs[:10]
        
        return high_corr_pairs, multicollinear_groups, target_corr_top
    
    def _compute_mutual_info(
        self, X: pd.DataFrame, y: pd.Series, task_type: TaskType
    ) -> Tuple[Dict[str, float], List[str]]:
        """Compute mutual information scores."""
        mi_scores = {}
        low_mi_cols = []
        
        # Prepare numeric features for MI
        numeric_cols = [c for c in X.columns if is_numeric_series(X[c])]
        
        if len(numeric_cols) == 0:
            return mi_scores, low_mi_cols
        
        X_num = X[numeric_cols].fillna(X[numeric_cols].median())
        
        try:
            if task_type == TaskType.CLASSIFICATION:
                mi = mutual_info_classif(X_num, y, random_state=self.config.random_state)
            else:
                mi = mutual_info_regression(X_num, y, random_state=self.config.random_state)
            
            for col, score in zip(numeric_cols, mi):
                mi_scores[col] = float(score)
                if score < 0.01:
                    low_mi_cols.append(col)
        except Exception as e:
            logger.warning(f"MI computation failed: {e}")
        
        return mi_scores, low_mi_cols
    
    def _detect_time_series(
        self, X: pd.DataFrame, datetime_cols: List[str]
    ) -> Tuple[Any, bool, Any, Any, bool]:
        """Detect time series characteristics."""
        time_column = None
        is_time_series = False
        time_granularity = None
        time_gaps = None
        has_regular_intervals = False
        
        if len(datetime_cols) > 0:
            # Take first datetime column
            time_column = datetime_cols[0]
            is_time_series = True
            
            # Compute gaps
            time_col = pd.to_datetime(X[time_column])
            sorted_times = time_col.sort_values()
            gaps = sorted_times.diff().dt.total_seconds()
            
            time_gaps = {
                "median_gap_seconds": float(gaps.median()),
                "mean_gap_seconds": float(gaps.mean()),
                "max_gap_seconds": float(gaps.max()),
            }
            
            # Check regularity
            gap_std = gaps.std()
            has_regular_intervals = gap_std < gaps.mean() * 0.1
        
        return time_column, is_time_series, time_granularity, time_gaps, has_regular_intervals
    
    def _detect_entity_columns(
        self, X: pd.DataFrame, categorical_cols: List[str]
    ) -> List[str]:
        """Detect entity/group columns."""
        entity_cols = []
        
        for col in categorical_cols:
            card = calc_cardinality(X[col])
            # Entity columns typically have moderate cardinality and repeated values
            if 10 < card < len(X) * 0.5:
                # Check for repeats
                vc = X[col].value_counts()
                if (vc > 1).sum() > len(vc) * 0.3:
                    entity_cols.append(col)
        
        return entity_cols
    
    def _analyze_text_columns(
        self, X: pd.DataFrame, text_cols: List[str]
    ) -> Tuple[Dict[str, Dict[str, float]], Dict[str, int]]:
        """Analyze text column statistics."""
        length_stats = {}
        vocab_size = {}
        
        for col in text_cols:
            text_series = X[col].astype(str)
            lengths = text_series.str.len()
            
            length_stats[col] = {
                "mean_len": float(lengths.mean()),
                "std_len": float(lengths.std()),
                "max_len": float(lengths.max()),
            }
            
            # Vocab size (rough estimate)
            vocab_size[col] = text_series.nunique()
        
        return length_stats, vocab_size
    
    def _detect_leakage_risks(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        numeric_cols: List[str],
        categorical_cols: List[str],
        mi_scores: Dict[str, float],
        high_corr_pairs: List[Tuple[str, str, float]]
    ) -> Tuple[List[str], List[str]]:
        """Detect potential leakage."""
        leakage_risk_cols = []
        id_like_with_high_mi = []
        
        # Check perfect correlations
        for col in numeric_cols:
            if is_numeric_series(y):
                corr = X[col].corr(y)
                if abs(corr) > 0.99:
                    leakage_risk_cols.append(col)
        
        # Check ID-like columns with high MI
        for col in categorical_cols:
            card = calc_cardinality(X[col])
            if card > 0.95 * len(X):  # ID-like
                mi = mi_scores.get(col, 0.0)
                if mi > 0.5:
                    id_like_with_high_mi.append(col)
        
        return leakage_risk_cols, id_like_with_high_mi
    
    def _estimate_feature_count(
        self,
        n_numeric: int,
        n_categorical: int,
        n_text: int,
        cardinality_summary: Dict[str, int]
    ) -> int:
        """Estimate output feature count."""
        count = n_numeric  # Numeric pass-through
        
        # Categorical encoding
        for card in cardinality_summary.values():
            if card <= 10:
                count += card  # OHE
            else:
                count += 1  # Target/freq encoding
        
        # Text features (rough estimate)
        count += n_text * 5  # Stats features per text col
        
        return count
    
    def _estimate_memory(self, n_rows: int, n_features: int) -> float:
        """Estimate memory usage in GB."""
        # Rough estimate: 8 bytes per float64 * n_rows * n_features
        bytes_estimate = 8 * n_rows * n_features * 3  # 3x for overhead
        gb = bytes_estimate / (1024 ** 3)
        return round(gb, 2)

