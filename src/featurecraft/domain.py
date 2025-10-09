"""Domain-specific feature transformers for FeatureCraft.

This module provides industry and domain-specific feature engineering capabilities:
- Finance: Technical indicators (RSI, MACD, Bollinger Bands), risk ratios (Sharpe, Sortino)
- E-commerce: RFM analysis (Recency, Frequency, Monetary), customer lifetime value
- Healthcare: Vital sign ratios, BMI, clinical scores
- NLP: Text statistics, part-of-speech features, readability scores
- Geospatial: Distance calculations, coordinate transformations, proximity features

These transformers enable domain expertise to be encoded as features.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .logging import get_logger

logger = get_logger(__name__)


class FinanceTechnicalIndicators(BaseEstimator, TransformerMixin):
    """Create technical indicators for financial time-series data.
    
    Computes common technical analysis indicators:
    - RSI (Relative Strength Index): Momentum oscillator
    - MACD (Moving Average Convergence Divergence): Trend indicator
    - Bollinger Bands: Volatility indicator
    - Moving averages (SMA, EMA)
    - Rate of Change (ROC)
    - Stochastic Oscillator
    
    Examples
    --------
    >>> import pandas as pd
    >>> from featurecraft.domain import FinanceTechnicalIndicators
    >>> 
    >>> # Stock price data
    >>> X = pd.DataFrame({
    ...     'close': [100, 102, 98, 105, 103, 107, 110],
    ...     'high': [102, 104, 100, 107, 105, 109, 112],
    ...     'low': [98, 100, 96, 103, 101, 105, 108]
    ... })
    >>> 
    >>> transformer = FinanceTechnicalIndicators(
    ...     indicators=['rsi', 'macd', 'bollinger'],
    ...     price_col='close'
    ... )
    >>> X_transformed = transformer.fit_transform(X)
    
    Parameters
    ----------
    indicators : List[str], optional
        List of indicators to compute:
        - 'rsi': Relative Strength Index
        - 'macd': MACD indicator
        - 'bollinger': Bollinger Bands
        - 'sma': Simple Moving Average
        - 'ema': Exponential Moving Average
        - 'roc': Rate of Change
        - 'stochastic': Stochastic Oscillator
        Default: ['rsi', 'macd']
    price_col : str, optional
        Column containing price data. Default: 'close'
    high_col : str, optional
        Column containing high prices (for stochastic). Default: 'high'
    low_col : str, optional
        Column containing low prices (for stochastic). Default: 'low'
    rsi_period : int, optional
        Period for RSI calculation. Default: 14
    macd_fast : int, optional
        Fast period for MACD. Default: 12
    macd_slow : int, optional
        Slow period for MACD. Default: 26
    macd_signal : int, optional
        Signal period for MACD. Default: 9
    bb_period : int, optional
        Period for Bollinger Bands. Default: 20
    bb_std : float, optional
        Standard deviations for Bollinger Bands. Default: 2.0
    """
    
    def __init__(
        self,
        indicators: List[str] = None,
        price_col: str = 'close',
        high_col: str = 'high',
        low_col: str = 'low',
        rsi_period: int = 14,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        bb_period: int = 20,
        bb_std: float = 2.0,
        sma_periods: List[int] = None,
        ema_periods: List[int] = None,
    ):
        if indicators is None:
            indicators = ['rsi', 'macd']
        if sma_periods is None:
            sma_periods = [20, 50]
        if ema_periods is None:
            ema_periods = [12, 26]
        
        self.indicators = indicators
        self.price_col = price_col
        self.high_col = high_col
        self.low_col = low_col
        self.rsi_period = rsi_period
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.sma_periods = sma_periods
        self.ema_periods = ema_periods
        self.feature_names_: List[str] = []
    
    def fit(self, X: pd.DataFrame, y=None):
        """Fit transformer (no-op for technical indicators)."""
        return self
    
    def _compute_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Compute Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _compute_macd(self, prices: pd.Series, fast: int, slow: int, signal: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Compute MACD, signal line, and histogram."""
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    def _compute_bollinger_bands(self, prices: pd.Series, period: int, std_dev: float) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Compute Bollinger Bands."""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, sma, lower_band
    
    def _compute_stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Compute Stochastic Oscillator."""
        lowest_low = low.rolling(window=period).min()
        highest_high = high.rolling(window=period).max()
        stoch = 100 * (close - lowest_low) / (highest_high - lowest_low).replace(0, np.nan)
        return stoch
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate technical indicator features.
        
        Parameters
        ----------
        X : pd.DataFrame
            Data with price columns
            
        Returns
        -------
        pd.DataFrame
            Technical indicator features
        """
        df = pd.DataFrame(X)
        out = pd.DataFrame(index=df.index)
        
        # Check if price column exists
        if self.price_col not in df.columns:
            logger.warning(f"Price column '{self.price_col}' not found")
            return out
        
        prices = df[self.price_col]
        
        # RSI
        if 'rsi' in self.indicators:
            out['rsi'] = self._compute_rsi(prices, self.rsi_period)
        
        # MACD
        if 'macd' in self.indicators:
            macd_line, signal_line, histogram = self._compute_macd(
                prices, self.macd_fast, self.macd_slow, self.macd_signal
            )
            out['macd_line'] = macd_line
            out['macd_signal'] = signal_line
            out['macd_histogram'] = histogram
        
        # Bollinger Bands
        if 'bollinger' in self.indicators:
            upper, middle, lower = self._compute_bollinger_bands(prices, self.bb_period, self.bb_std)
            out['bb_upper'] = upper
            out['bb_middle'] = middle
            out['bb_lower'] = lower
            out['bb_width'] = upper - lower
            out['bb_position'] = (prices - lower) / (upper - lower).replace(0, np.nan)
        
        # Simple Moving Averages
        if 'sma' in self.indicators:
            for period in self.sma_periods:
                out[f'sma_{period}'] = prices.rolling(window=period).mean()
        
        # Exponential Moving Averages
        if 'ema' in self.indicators:
            for period in self.ema_periods:
                out[f'ema_{period}'] = prices.ewm(span=period, adjust=False).mean()
        
        # Rate of Change
        if 'roc' in self.indicators:
            out['roc'] = prices.pct_change(periods=10) * 100
        
        # Stochastic Oscillator
        if 'stochastic' in self.indicators:
            if self.high_col in df.columns and self.low_col in df.columns:
                out['stochastic'] = self._compute_stochastic(
                    df[self.high_col], df[self.low_col], prices
                )
            else:
                logger.warning("High/Low columns not found for stochastic oscillator")
        
        self.feature_names_ = list(out.columns)
        return out
    
    def get_feature_names_out(self, input_features=None):
        """Get output feature names."""
        return self.feature_names_ if self.feature_names_ else []


class FinanceRiskMetrics(BaseEstimator, TransformerMixin):
    """Calculate financial risk metrics and ratios.
    
    Computes risk-adjusted return metrics:
    - Sharpe Ratio: (Return - Risk-Free Rate) / Volatility
    - Sortino Ratio: (Return - Risk-Free Rate) / Downside Deviation
    - Maximum Drawdown: Largest peak-to-trough decline
    - Volatility: Standard deviation of returns
    - Beta: Correlation with market
    
    Examples
    --------
    >>> import pandas as pd
    >>> from featurecraft.domain import FinanceRiskMetrics
    >>> 
    >>> X = pd.DataFrame({
    ...     'returns': [0.01, -0.02, 0.03, 0.01, -0.01],
    ...     'market_returns': [0.015, -0.015, 0.025, 0.012, -0.008]
    ... })
    >>> 
    >>> transformer = FinanceRiskMetrics(
    ...     returns_col='returns',
    ...     market_returns_col='market_returns'
    ... )
    >>> X_transformed = transformer.fit_transform(X)
    
    Parameters
    ----------
    returns_col : str, optional
        Column containing returns. Default: 'returns'
    market_returns_col : str, optional
        Column containing market returns (for beta). Default: None
    risk_free_rate : float, optional
        Annual risk-free rate. Default: 0.02
    window : int, optional
        Rolling window for calculations. Default: 252 (trading days in year)
    """
    
    def __init__(
        self,
        returns_col: str = 'returns',
        market_returns_col: Optional[str] = None,
        risk_free_rate: float = 0.02,
        window: int = 252,
    ):
        self.returns_col = returns_col
        self.market_returns_col = market_returns_col
        self.risk_free_rate = risk_free_rate
        self.window = window
        self.feature_names_: List[str] = []
    
    def fit(self, X: pd.DataFrame, y=None):
        """Fit transformer (no-op)."""
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate risk metric features."""
        df = pd.DataFrame(X)
        out = pd.DataFrame(index=df.index)
        
        if self.returns_col not in df.columns:
            logger.warning(f"Returns column '{self.returns_col}' not found")
            return out
        
        returns = df[self.returns_col]
        
        # Volatility (annualized)
        out['volatility'] = returns.rolling(window=self.window).std() * np.sqrt(252)
        
        # Sharpe Ratio
        mean_return = returns.rolling(window=self.window).mean() * 252
        out['sharpe_ratio'] = (mean_return - self.risk_free_rate) / out['volatility']
        
        # Sortino Ratio (using downside deviation)
        downside_returns = returns.where(returns < 0, 0)
        downside_std = downside_returns.rolling(window=self.window).std() * np.sqrt(252)
        out['sortino_ratio'] = (mean_return - self.risk_free_rate) / downside_std.replace(0, np.nan)
        
        # Maximum Drawdown
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        out['max_drawdown'] = drawdown.rolling(window=self.window).min()
        
        # Beta (if market returns provided)
        if self.market_returns_col and self.market_returns_col in df.columns:
            market_returns = df[self.market_returns_col]
            
            # Rolling covariance and variance
            covariance = returns.rolling(window=self.window).cov(market_returns)
            market_variance = market_returns.rolling(window=self.window).var()
            out['beta'] = covariance / market_variance.replace(0, np.nan)
        
        self.feature_names_ = list(out.columns)
        return out
    
    def get_feature_names_out(self, input_features=None):
        """Get output feature names."""
        return self.feature_names_ if self.feature_names_ else []


class EcommerceRFM(BaseEstimator, TransformerMixin):
    """Create RFM (Recency, Frequency, Monetary) features for e-commerce.
    
    RFM analysis segments customers based on:
    - Recency: Days since last purchase
    - Frequency: Number of purchases
    - Monetary: Total/average purchase value
    
    These are critical for customer segmentation, churn prediction, and CLV modeling.
    
    Examples
    --------
    >>> import pandas as pd
    >>> from featurecraft.domain import EcommerceRFM
    >>> 
    >>> X = pd.DataFrame({
    ...     'customer_id': [1, 1, 2, 2, 3],
    ...     'order_date': pd.to_datetime(['2024-01-01', '2024-02-01', 
    ...                                   '2024-01-15', '2024-03-01', '2024-03-15']),
    ...     'order_value': [100, 150, 200, 50, 300]
    ... })
    >>> 
    >>> transformer = EcommerceRFM(
    ...     customer_col='customer_id',
    ...     date_col='order_date',
    ...     monetary_col='order_value',
    ...     reference_date='2024-03-31'
    ... )
    >>> X_transformed = transformer.fit_transform(X)
    
    Parameters
    ----------
    customer_col : str
        Column containing customer ID
    date_col : str
        Column containing transaction date
    monetary_col : str
        Column containing transaction value
    reference_date : str or datetime, optional
        Reference date for recency calculation. If None, uses max date in data.
    create_scores : bool, optional
        Create RFM scores (1-5 quintiles). Default: True
    create_segments : bool, optional
        Create customer segments (Champion, Loyal, etc.). Default: False
    """
    
    def __init__(
        self,
        customer_col: str,
        date_col: str,
        monetary_col: str,
        reference_date: Optional[Union[str, datetime]] = None,
        create_scores: bool = True,
        create_segments: bool = False,
    ):
        self.customer_col = customer_col
        self.date_col = date_col
        self.monetary_col = monetary_col
        self.reference_date = reference_date
        self.create_scores = create_scores
        self.create_segments = create_segments
        self.rfm_data_: Optional[pd.DataFrame] = None
        self.feature_names_: List[str] = []
    
    def fit(self, X: pd.DataFrame, y=None):
        """Compute RFM metrics from transaction data."""
        df = pd.DataFrame(X).copy()
        
        # Validate columns
        required_cols = [self.customer_col, self.date_col, self.monetary_col]
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Required columns not found: {missing}")
        
        # Convert date column to datetime
        df[self.date_col] = pd.to_datetime(df[self.date_col])
        
        # Determine reference date
        if self.reference_date is None:
            ref_date = df[self.date_col].max()
        else:
            ref_date = pd.to_datetime(self.reference_date)
        
        # Compute RFM metrics per customer
        rfm = df.groupby(self.customer_col).agg({
            self.date_col: lambda x: (ref_date - x.max()).days,  # Recency
            self.monetary_col: ['count', 'sum', 'mean']  # Frequency & Monetary
        })
        
        # Flatten column names
        rfm.columns = ['recency', 'frequency', 'monetary_total', 'monetary_avg']
        rfm.reset_index(inplace=True)
        
        # Create RFM scores (quintiles: 1-5)
        if self.create_scores:
            # Recency: lower is better (reverse)
            rfm['recency_score'] = pd.qcut(rfm['recency'], q=5, labels=[5, 4, 3, 2, 1], duplicates='drop')
            # Frequency: higher is better
            rfm['frequency_score'] = pd.qcut(rfm['frequency'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5], duplicates='drop')
            # Monetary: higher is better
            rfm['monetary_score'] = pd.qcut(rfm['monetary_total'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5], duplicates='drop')
            
            # Convert to int
            rfm['recency_score'] = rfm['recency_score'].astype(int)
            rfm['frequency_score'] = rfm['frequency_score'].astype(int)
            rfm['monetary_score'] = rfm['monetary_score'].astype(int)
            
            # Combined RFM score
            rfm['rfm_score'] = rfm['recency_score'] * 100 + rfm['frequency_score'] * 10 + rfm['monetary_score']
        
        # Create customer segments
        if self.create_segments:
            rfm['segment'] = 'Other'
            
            # Define segments based on RFM scores
            if self.create_scores:
                # Champions: High R, F, M
                rfm.loc[(rfm['recency_score'] >= 4) & (rfm['frequency_score'] >= 4) & (rfm['monetary_score'] >= 4), 'segment'] = 'Champions'
                # Loyal: High F
                rfm.loc[(rfm['frequency_score'] >= 4) & (rfm['segment'] == 'Other'), 'segment'] = 'Loyal'
                # Big Spenders: High M
                rfm.loc[(rfm['monetary_score'] >= 4) & (rfm['segment'] == 'Other'), 'segment'] = 'Big Spenders'
                # At Risk: Low R, High F/M
                rfm.loc[(rfm['recency_score'] <= 2) & (rfm['frequency_score'] >= 3) & (rfm['segment'] == 'Other'), 'segment'] = 'At Risk'
                # Lost: Low R, F, M
                rfm.loc[(rfm['recency_score'] <= 2) & (rfm['frequency_score'] <= 2), 'segment'] = 'Lost'
        
        self.rfm_data_ = rfm
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Map RFM features to input data."""
        df = pd.DataFrame(X)
        
        if self.rfm_data_ is None:
            logger.warning("RFM data not fitted")
            return pd.DataFrame(index=df.index)
        
        # Merge RFM data with input
        out = df[[self.customer_col]].merge(
            self.rfm_data_,
            on=self.customer_col,
            how='left'
        )
        
        # Drop customer_col from output
        out = out.drop(columns=[self.customer_col])
        
        self.feature_names_ = list(out.columns)
        return out
    
    def get_feature_names_out(self, input_features=None):
        """Get output feature names."""
        return self.feature_names_ if self.feature_names_ else []


class HealthcareVitalSigns(BaseEstimator, TransformerMixin):
    """Create healthcare-specific features from vital signs.
    
    Computes clinical metrics and ratios:
    - BMI (Body Mass Index): weight / height²
    - MAP (Mean Arterial Pressure): DBP + 1/3(SBP - DBP)
    - Shock Index: HR / SBP
    - Pulse Pressure: SBP - DBP
    - Temperature in different units
    - Age-adjusted risk scores
    
    Examples
    --------
    >>> import pandas as pd
    >>> from featurecraft.domain import HealthcareVitalSigns
    >>> 
    >>> X = pd.DataFrame({
    ...     'weight_kg': [70, 80, 90],
    ...     'height_cm': [170, 175, 180],
    ...     'sbp': [120, 140, 130],
    ...     'dbp': [80, 90, 85],
    ...     'heart_rate': [70, 85, 75]
    ... })
    >>> 
    >>> transformer = HealthcareVitalSigns()
    >>> X_transformed = transformer.fit_transform(X)
    
    Parameters
    ----------
    weight_col : str, optional
        Weight column (kg). Default: 'weight_kg'
    height_col : str, optional
        Height column (cm). Default: 'height_cm'
    sbp_col : str, optional
        Systolic blood pressure. Default: 'sbp'
    dbp_col : str, optional
        Diastolic blood pressure. Default: 'dbp'
    heart_rate_col : str, optional
        Heart rate (bpm). Default: 'heart_rate'
    age_col : str, optional
        Age column for age-adjusted scores. Default: None
    """
    
    def __init__(
        self,
        weight_col: str = 'weight_kg',
        height_col: str = 'height_cm',
        sbp_col: str = 'sbp',
        dbp_col: str = 'dbp',
        heart_rate_col: str = 'heart_rate',
        age_col: Optional[str] = None,
    ):
        self.weight_col = weight_col
        self.height_col = height_col
        self.sbp_col = sbp_col
        self.dbp_col = dbp_col
        self.heart_rate_col = heart_rate_col
        self.age_col = age_col
        self.feature_names_: List[str] = []
    
    def fit(self, X: pd.DataFrame, y=None):
        """Fit transformer (no-op)."""
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate healthcare features."""
        df = pd.DataFrame(X)
        out = pd.DataFrame(index=df.index)
        
        # BMI: weight(kg) / (height(m))²
        if self.weight_col in df.columns and self.height_col in df.columns:
            height_m = df[self.height_col] / 100  # cm to m
            out['bmi'] = df[self.weight_col] / (height_m ** 2)
            
            # BMI categories
            out['bmi_underweight'] = (out['bmi'] < 18.5).astype(int)
            out['bmi_normal'] = ((out['bmi'] >= 18.5) & (out['bmi'] < 25)).astype(int)
            out['bmi_overweight'] = ((out['bmi'] >= 25) & (out['bmi'] < 30)).astype(int)
            out['bmi_obese'] = (out['bmi'] >= 30).astype(int)
        
        # Mean Arterial Pressure (MAP)
        if self.sbp_col in df.columns and self.dbp_col in df.columns:
            out['map'] = df[self.dbp_col] + (df[self.sbp_col] - df[self.dbp_col]) / 3
            
            # Pulse Pressure
            out['pulse_pressure'] = df[self.sbp_col] - df[self.dbp_col]
            
            # Hypertension flags
            out['hypertension_stage1'] = ((df[self.sbp_col] >= 130) | (df[self.dbp_col] >= 80)).astype(int)
            out['hypertension_stage2'] = ((df[self.sbp_col] >= 140) | (df[self.dbp_col] >= 90)).astype(int)
        
        # Shock Index: HR / SBP (normal < 0.7, >1.0 indicates shock)
        if self.heart_rate_col in df.columns and self.sbp_col in df.columns:
            out['shock_index'] = df[self.heart_rate_col] / df[self.sbp_col].replace(0, np.nan)
            out['shock_risk'] = (out['shock_index'] > 1.0).astype(int)
        
        # Heart rate categories
        if self.heart_rate_col in df.columns:
            out['hr_bradycardia'] = (df[self.heart_rate_col] < 60).astype(int)
            out['hr_normal'] = ((df[self.heart_rate_col] >= 60) & (df[self.heart_rate_col] <= 100)).astype(int)
            out['hr_tachycardia'] = (df[self.heart_rate_col] > 100).astype(int)
        
        # Age-adjusted features
        if self.age_col and self.age_col in df.columns:
            age = df[self.age_col]
            
            # Age-adjusted heart rate (max HR ≈ 220 - age)
            if self.heart_rate_col in df.columns:
                max_hr = 220 - age
                out['hr_percent_max'] = (df[self.heart_rate_col] / max_hr) * 100
        
        self.feature_names_ = list(out.columns)
        return out
    
    def get_feature_names_out(self, input_features=None):
        """Get output feature names."""
        return self.feature_names_ if self.feature_names_ else []


class NLPTextStatistics(BaseEstimator, TransformerMixin):
    """Extract statistical features from text data.
    
    Computes text-based features:
    - Character count, word count, sentence count
    - Average word length, sentence length
    - Punctuation counts
    - Uppercase/lowercase ratios
    - Special character counts
    - Readability scores (if textstat installed)
    - Part-of-speech counts (if spacy/nltk installed)
    
    Examples
    --------
    >>> import pandas as pd
    >>> from featurecraft.domain import NLPTextStatistics
    >>> 
    >>> X = pd.DataFrame({
    ...     'text': [
    ...         'Hello world!',
    ...         'This is a longer sentence with more words.',
    ...         'Short text.'
    ...     ]
    ... })
    >>> 
    >>> transformer = NLPTextStatistics(text_col='text')
    >>> X_transformed = transformer.fit_transform(X)
    
    Parameters
    ----------
    text_col : str
        Column containing text data
    include_readability : bool, optional
        Include readability scores (requires textstat). Default: True
    include_pos : bool, optional
        Include part-of-speech counts (requires spacy/nltk). Default: False
    pos_engine : str, optional
        POS tagging engine: 'spacy' or 'nltk'. Default: 'spacy'
    """
    
    def __init__(
        self,
        text_col: str,
        include_readability: bool = True,
        include_pos: bool = False,
        pos_engine: Literal['spacy', 'nltk'] = 'spacy',
    ):
        self.text_col = text_col
        self.include_readability = include_readability
        self.include_pos = include_pos
        self.pos_engine = pos_engine
        self.feature_names_: List[str] = []
    
    def fit(self, X: pd.DataFrame, y=None):
        """Fit transformer (no-op)."""
        return self
    
    def _compute_basic_stats(self, text: str) -> dict:
        """Compute basic text statistics."""
        if pd.isna(text) or text == '':
            return {
                'char_count': 0,
                'word_count': 0,
                'sentence_count': 0,
                'avg_word_length': 0,
                'punctuation_count': 0,
                'uppercase_count': 0,
                'digit_count': 0,
            }
        
        text = str(text)
        words = text.split()
        sentences = text.replace('!', '.').replace('?', '.').split('.')
        sentences = [s for s in sentences if s.strip()]
        
        return {
            'char_count': len(text),
            'word_count': len(words),
            'sentence_count': len(sentences),
            'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
            'avg_sentence_length': len(words) / len(sentences) if sentences else 0,
            'punctuation_count': sum(1 for c in text if c in '.,!?;:'),
            'uppercase_count': sum(1 for c in text if c.isupper()),
            'digit_count': sum(1 for c in text if c.isdigit()),
            'whitespace_ratio': sum(1 for c in text if c.isspace()) / len(text) if text else 0,
        }
    
    def _compute_readability(self, text: str) -> dict:
        """Compute readability scores using textstat."""
        try:
            import textstat
            
            if pd.isna(text) or text == '':
                return {}
            
            text = str(text)
            return {
                'flesch_reading_ease': textstat.flesch_reading_ease(text),
                'flesch_kincaid_grade': textstat.flesch_kincaid_grade(text),
                'automated_readability_index': textstat.automated_readability_index(text),
            }
        except ImportError:
            logger.warning("textstat not installed, skipping readability scores")
            return {}
        except:
            # Handle errors in textstat computation
            return {}
    
    def _compute_pos(self, text: str) -> dict:
        """Compute part-of-speech counts."""
        if pd.isna(text) or text == '':
            return {}
        
        text = str(text)
        
        try:
            if self.pos_engine == 'spacy':
                import spacy
                
                # Load model (you may need to download: python -m spacy download en_core_web_sm)
                try:
                    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
                except:
                    logger.warning("Spacy model not found. Run: python -m spacy download en_core_web_sm")
                    return {}
                
                doc = nlp(text)
                pos_counts = {}
                for token in doc:
                    pos = token.pos_
                    pos_counts[f'pos_{pos.lower()}_count'] = pos_counts.get(f'pos_{pos.lower()}_count', 0) + 1
                
                return pos_counts
            
            elif self.pos_engine == 'nltk':
                import nltk
                from nltk import pos_tag, word_tokenize
                
                try:
                    tokens = word_tokenize(text)
                    pos_tags = pos_tag(tokens)
                    
                    pos_counts = {}
                    for word, pos in pos_tags:
                        # Simplify POS tags
                        if pos.startswith('N'):
                            pos_type = 'noun'
                        elif pos.startswith('V'):
                            pos_type = 'verb'
                        elif pos.startswith('J'):
                            pos_type = 'adj'
                        elif pos.startswith('R'):
                            pos_type = 'adv'
                        else:
                            pos_type = 'other'
                        
                        pos_counts[f'pos_{pos_type}_count'] = pos_counts.get(f'pos_{pos_type}_count', 0) + 1
                    
                    return pos_counts
                except:
                    logger.warning("NLTK data not found. Run: nltk.download('punkt'); nltk.download('averaged_perceptron_tagger')")
                    return {}
        
        except ImportError:
            logger.warning(f"{self.pos_engine} not installed for POS tagging")
            return {}
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate text statistics features."""
        df = pd.DataFrame(X)
        
        if self.text_col not in df.columns:
            logger.warning(f"Text column '{self.text_col}' not found")
            return pd.DataFrame(index=df.index)
        
        # Compute features for each row
        features_list = []
        for text in df[self.text_col]:
            features = {}
            
            # Basic statistics
            features.update(self._compute_basic_stats(text))
            
            # Readability scores
            if self.include_readability:
                features.update(self._compute_readability(text))
            
            # POS counts
            if self.include_pos:
                features.update(self._compute_pos(text))
            
            features_list.append(features)
        
        # Convert to DataFrame
        out = pd.DataFrame(features_list, index=df.index)
        
        # Fill missing POS columns with 0
        out = out.fillna(0)
        
        self.feature_names_ = list(out.columns)
        return out
    
    def get_feature_names_out(self, input_features=None):
        """Get output feature names."""
        return self.feature_names_ if self.feature_names_ else []


class GeospatialFeatures(BaseEstimator, TransformerMixin):
    """Create geospatial features from coordinates.
    
    Computes distance and proximity features:
    - Haversine distance between coordinates
    - Distance to point of interest (POI)
    - Coordinate transformations (lat/lon to radians, cartesian)
    - Spatial binning (geohash-like)
    - Distance features for multiple POIs
    
    Examples
    --------
    >>> import pandas as pd
    >>> from featurecraft.domain import GeospatialFeatures
    >>> 
    >>> X = pd.DataFrame({
    ...     'latitude': [40.7128, 34.0522, 41.8781],
    ...     'longitude': [-74.0060, -118.2437, -87.6298]
    ... })
    >>> 
    >>> # Distance to NYC
    >>> transformer = GeospatialFeatures(
    ...     lat_col='latitude',
    ...     lon_col='longitude',
    ...     poi_coords=[(40.7128, -74.0060)]  # NYC
    ... )
    >>> X_transformed = transformer.fit_transform(X)
    
    Parameters
    ----------
    lat_col : str, optional
        Latitude column name. Default: 'latitude'
    lon_col : str, optional
        Longitude column name. Default: 'longitude'
    poi_coords : List[Tuple[float, float]], optional
        List of (lat, lon) coordinates for points of interest
    poi_names : List[str], optional
        Names for each POI (for feature naming)
    include_cartesian : bool, optional
        Include cartesian coordinate transformation. Default: False
    include_radians : bool, optional
        Include radian conversion of lat/lon. Default: False
    """
    
    def __init__(
        self,
        lat_col: str = 'latitude',
        lon_col: str = 'longitude',
        poi_coords: Optional[List[Tuple[float, float]]] = None,
        poi_names: Optional[List[str]] = None,
        include_cartesian: bool = False,
        include_radians: bool = False,
    ):
        self.lat_col = lat_col
        self.lon_col = lon_col
        self.poi_coords = poi_coords or []
        self.poi_names = poi_names
        self.include_cartesian = include_cartesian
        self.include_radians = include_radians
        self.feature_names_: List[str] = []
        
        # Validate POI names
        if self.poi_names and len(self.poi_names) != len(self.poi_coords):
            raise ValueError("Number of POI names must match number of POI coordinates")
    
    def fit(self, X: pd.DataFrame, y=None):
        """Fit transformer (no-op)."""
        return self
    
    def _haversine_distance(self, lat1: np.ndarray, lon1: np.ndarray, 
                           lat2: float, lon2: float) -> np.ndarray:
        """Compute Haversine distance in kilometers."""
        # Convert to radians
        lat1_rad = np.radians(lat1)
        lon1_rad = np.radians(lon1)
        lat2_rad = np.radians(lat2)
        lon2_rad = np.radians(lon2)
        
        # Haversine formula
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        # Radius of Earth in kilometers
        r = 6371
        
        return c * r
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate geospatial features."""
        df = pd.DataFrame(X)
        out = pd.DataFrame(index=df.index)
        
        # Check if coordinate columns exist
        if self.lat_col not in df.columns or self.lon_col not in df.columns:
            logger.warning(f"Coordinate columns '{self.lat_col}' or '{self.lon_col}' not found")
            return out
        
        lat = df[self.lat_col]
        lon = df[self.lon_col]
        
        # Radians conversion
        if self.include_radians:
            out['lat_rad'] = np.radians(lat)
            out['lon_rad'] = np.radians(lon)
        
        # Cartesian coordinates (for 3D distance calculations)
        if self.include_cartesian:
            lat_rad = np.radians(lat)
            lon_rad = np.radians(lon)
            
            out['x'] = np.cos(lat_rad) * np.cos(lon_rad)
            out['y'] = np.cos(lat_rad) * np.sin(lon_rad)
            out['z'] = np.sin(lat_rad)
        
        # Distance to POIs
        for i, (poi_lat, poi_lon) in enumerate(self.poi_coords):
            if self.poi_names:
                poi_name = self.poi_names[i]
            else:
                poi_name = f'poi_{i}'
            
            distance = self._haversine_distance(lat.values, lon.values, poi_lat, poi_lon)
            out[f'distance_to_{poi_name}_km'] = distance
        
        # Spatial binning (simple lat/lon rounding for grid cells)
        out['lat_bin'] = (lat // 1).astype(int)  # 1-degree bins
        out['lon_bin'] = (lon // 1).astype(int)
        
        self.feature_names_ = list(out.columns)
        return out
    
    def get_feature_names_out(self, input_features=None):
        """Get output feature names."""
        return self.feature_names_ if self.feature_names_ else []


def build_domain_pipeline(
    include_finance_technical: bool = False,
    include_finance_risk: bool = False,
    include_ecommerce_rfm: bool = False,
    include_healthcare: bool = False,
    include_nlp: bool = False,
    include_geospatial: bool = False,
    **kwargs
) -> list:
    """Build a pipeline of domain-specific transformers.
    
    Convenience function to create a list of domain transformers
    that can be used in a sklearn Pipeline.
    
    Parameters
    ----------
    include_finance_technical : bool, optional
        Include financial technical indicators. Default: False
    include_finance_risk : bool, optional
        Include financial risk metrics. Default: False
    include_ecommerce_rfm : bool, optional
        Include e-commerce RFM analysis. Default: False
    include_healthcare : bool, optional
        Include healthcare vital signs features. Default: False
    include_nlp : bool, optional
        Include NLP text statistics. Default: False
    include_geospatial : bool, optional
        Include geospatial features. Default: False
    **kwargs : dict
        Additional parameters for transformers
        
    Returns
    -------
    list
        List of (name, transformer) tuples for Pipeline
        
    Examples
    --------
    >>> from sklearn.pipeline import Pipeline
    >>> from featurecraft.domain import build_domain_pipeline
    >>> 
    >>> transformers = build_domain_pipeline(
    ...     include_finance_technical=True,
    ...     include_finance_risk=True
    ... )
    >>> pipeline = Pipeline(transformers)
    """
    transformers = []
    
    if include_finance_technical:
        transformers.append(('finance_technical', FinanceTechnicalIndicators(**kwargs.get('finance_technical_params', {}))))
    
    if include_finance_risk:
        transformers.append(('finance_risk', FinanceRiskMetrics(**kwargs.get('finance_risk_params', {}))))
    
    if include_ecommerce_rfm:
        transformers.append(('ecommerce_rfm', EcommerceRFM(**kwargs.get('ecommerce_rfm_params', {}))))
    
    if include_healthcare:
        transformers.append(('healthcare', HealthcareVitalSigns(**kwargs.get('healthcare_params', {}))))
    
    if include_nlp:
        transformers.append(('nlp', NLPTextStatistics(**kwargs.get('nlp_params', {}))))
    
    if include_geospatial:
        transformers.append(('geospatial', GeospatialFeatures(**kwargs.get('geospatial_params', {}))))
    
    return transformers

