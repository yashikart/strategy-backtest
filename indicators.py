"""
Technical Indicators Module for Options Trading Strategy

This module provides comprehensive technical indicators including:
- MACD (Moving Average Convergence Divergence)
- RSI (Relative Strength Index)
- ADX (Average Directional Index) 
- SuperTrend
- Bollinger Bands
- EMA Crossover (12/26)
- Stochastic Oscillator
- ATR (Average True Range)

Author: Strategy Backtest System
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict
import ta
import warnings
warnings.filterwarnings('ignore')

def calculate_ema(data: pd.Series, window: int) -> pd.Series:
    """Calculate Exponential Moving Average"""
    return data.ewm(span=window, adjust=False).mean()

def calculate_sma(data: pd.Series, window: int) -> pd.Series:
    """Calculate Simple Moving Average"""
    return data.rolling(window=window).mean()

def calculate_macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate MACD (Moving Average Convergence Divergence)
    
    Args:
        data: Price series (typically close price)
        fast: Fast EMA period (default: 12)
        slow: Slow EMA period (default: 26)
        signal: Signal line EMA period (default: 9)
    
    Returns:
        Tuple of (macd_line, signal_line, histogram)
    """
    ema_fast = calculate_ema(data, fast)
    ema_slow = calculate_ema(data, slow)
    
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal)
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram

def calculate_rsi(data: pd.Series, window: int = 14) -> pd.Series:
    """
    Calculate RSI (Relative Strength Index)
    
    Args:
        data: Price series (typically close price)
        window: Lookback period (default: 14)
    
    Returns:
        RSI values
    """
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate ADX (Average Directional Index)
    
    Args:
        high: High price series
        low: Low price series
        close: Close price series
        window: Lookback period (default: 14)
    
    Returns:
        Tuple of (adx, plus_di, minus_di)
    """
    # True Range
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    plus_dm = (high - high.shift()).where((high - high.shift()) > (low.shift() - low), 0)
    plus_dm = plus_dm.where(plus_dm > 0, 0)
    
    minus_dm = (low.shift() - low).where((low.shift() - low) > (high - high.shift()), 0)
    minus_dm = minus_dm.where(minus_dm > 0, 0)
    
    # Smoothed values
    atr = tr.ewm(alpha=1/window, adjust=False).mean()
    plus_dm_smooth = plus_dm.ewm(alpha=1/window, adjust=False).mean()
    minus_dm_smooth = minus_dm.ewm(alpha=1/window, adjust=False).mean()
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / atr
    minus_di = 100 * minus_dm_smooth / atr
    
    # ADX calculation
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.ewm(alpha=1/window, adjust=False).mean()
    
    return adx, plus_di, minus_di

def calculate_supertrend(high: pd.Series, low: pd.Series, close: pd.Series, 
                        atr_period: int = 10, multiplier: float = 3.0) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate SuperTrend indicator
    
    Args:
        high: High price series
        low: Low price series  
        close: Close price series
        atr_period: ATR calculation period (default: 10)
        multiplier: ATR multiplier (default: 3.0)
    
    Returns:
        Tuple of (supertrend, trend_direction)
    """
    # Calculate ATR
    atr = calculate_atr(high, low, close, atr_period)
    
    # Calculate basic bands
    hl_avg = (high + low) / 2
    upper_band = hl_avg + (multiplier * atr)
    lower_band = hl_avg - (multiplier * atr)
    
    # Initialize arrays
    supertrend = pd.Series(index=close.index, dtype=float)
    trend = pd.Series(index=close.index, dtype=int)
    
    # Calculate SuperTrend
    for i in range(1, len(close)):
        # Current values
        curr_close = close.iloc[i]
        prev_close = close.iloc[i-1]
        curr_upper = upper_band.iloc[i]
        curr_lower = lower_band.iloc[i]
        
        if i == 1:
            # Initialize first values
            if curr_close <= curr_lower:
                supertrend.iloc[i] = curr_upper
                trend.iloc[i] = -1
            else:
                supertrend.iloc[i] = curr_lower
                trend.iloc[i] = 1
        else:
            prev_supertrend = supertrend.iloc[i-1]
            prev_trend = trend.iloc[i-1]
            
            # Calculate final upper and lower bands
            if curr_upper < prev_supertrend or prev_close > prev_supertrend:
                final_upper = curr_upper
            else:
                final_upper = prev_supertrend
                
            if curr_lower > prev_supertrend or prev_close < prev_supertrend:
                final_lower = curr_lower
            else:
                final_lower = prev_supertrend
            
            # Determine trend and SuperTrend value
            if prev_trend == 1 and curr_close > final_lower:
                trend.iloc[i] = 1
                supertrend.iloc[i] = final_lower
            elif prev_trend == 1 and curr_close <= final_lower:
                trend.iloc[i] = -1
                supertrend.iloc[i] = final_upper
            elif prev_trend == -1 and curr_close < final_upper:
                trend.iloc[i] = -1
                supertrend.iloc[i] = final_upper
            else:
                trend.iloc[i] = 1
                supertrend.iloc[i] = final_lower
    
    return supertrend, trend

def calculate_bollinger_bands(data: pd.Series, window: int = 20, std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Bollinger Bands
    
    Args:
        data: Price series (typically close price)
        window: Moving average period (default: 20)
        std_dev: Standard deviation multiplier (default: 2.0)
    
    Returns:
        Tuple of (middle_band, upper_band, lower_band)
    """
    middle_band = calculate_sma(data, window)
    std = data.rolling(window=window).std()
    
    upper_band = middle_band + (std * std_dev)
    lower_band = middle_band - (std * std_dev)
    
    return middle_band, upper_band, lower_band

def calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, 
                        k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate Stochastic Oscillator
    
    Args:
        high: High price series
        low: Low price series
        close: Close price series
        k_period: %K period (default: 14)
        d_period: %D period (default: 3)
    
    Returns:
        Tuple of (%K, %D)
    """
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    
    k_percent = 100 * (close - lowest_low) / (highest_high - lowest_low)
    d_percent = k_percent.rolling(window=d_period).mean()
    
    return k_percent, d_percent

def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """
    Calculate ATR (Average True Range)
    
    Args:
        high: High price series
        low: Low price series
        close: Close price series
        window: Lookback period (default: 14)
    
    Returns:
        ATR values
    """
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=window).mean()
    
    return atr

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all technical indicators to the dataframe
    
    Args:
        df: DataFrame with OHLC data (columns: open, high, low, close)
    
    Returns:
        DataFrame with added technical indicators
    """
    result_df = df.copy()
    
    # Ensure we have the required columns
    required_cols = ['open', 'high', 'low', 'close']
    for col in required_cols:
        if col not in result_df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    print("Adding technical indicators...")
    
    # 1. MACD
    macd, macd_signal, macd_hist = calculate_macd(result_df['close'])
    result_df['macd'] = macd
    result_df['macd_signal'] = macd_signal
    result_df['macd_hist'] = macd_hist
    
    # 2. RSI
    result_df['rsi_14'] = calculate_rsi(result_df['close'], 14)
    
    # 3. ADX
    adx, plus_di, minus_di = calculate_adx(result_df['high'], result_df['low'], result_df['close'])
    result_df['adx_14'] = adx
    result_df['plus_di'] = plus_di
    result_df['minus_di'] = minus_di
    
    # 4. SuperTrend
    supertrend, trend = calculate_supertrend(result_df['high'], result_df['low'], result_df['close'])
    result_df['supertrend'] = supertrend
    result_df['supertrend_trend'] = trend
    
    # 5. Bollinger Bands
    bb_middle, bb_upper, bb_lower = calculate_bollinger_bands(result_df['close'])
    result_df['bb_middle'] = bb_middle
    result_df['bb_high'] = bb_upper
    result_df['bb_low'] = bb_lower
    
    # 6. EMA Crossover (12/26)
    result_df['ema_12'] = calculate_ema(result_df['close'], 12)
    result_df['ema_26'] = calculate_ema(result_df['close'], 26)
    result_df['ema_50'] = calculate_ema(result_df['close'], 50)
    
    # 7. Stochastic
    stoch_k, stoch_d = calculate_stochastic(result_df['high'], result_df['low'], result_df['close'])
    result_df['stoch_k'] = stoch_k
    result_df['stoch_d'] = stoch_d
    
    # 8. ATR
    result_df['atr_14'] = calculate_atr(result_df['high'], result_df['low'], result_df['close'])
    
    # Additional useful indicators
    # Simple Moving Averages
    result_df['sma_20'] = calculate_sma(result_df['close'], 20)
    result_df['sma_50'] = calculate_sma(result_df['close'], 50)
    
    # Volume-weighted indicators (if volume available)
    if 'volume' in result_df.columns:
        result_df['vwap'] = (result_df['close'] * result_df['volume']).rolling(20).sum() / result_df['volume'].rolling(20).sum()
    
    # Price momentum
    result_df['momentum_10'] = result_df['close'] / result_df['close'].shift(10) - 1
    result_df['momentum_20'] = result_df['close'] / result_df['close'].shift(20) - 1
    
    # Volatility measures
    result_df['volatility_20'] = result_df['close'].pct_change().rolling(20).std() * np.sqrt(252)
    
    print(f"Added {len([col for col in result_df.columns if col not in df.columns])} technical indicators")
    
    return result_df

def get_indicator_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get summary statistics for all indicators
    
    Args:
        df: DataFrame with indicators
    
    Returns:
        Summary statistics DataFrame
    """
    indicator_cols = [col for col in df.columns if col not in ['datetime', 'open', 'high', 'low', 'close', 'volume', 'signal']]
    
    summary = df[indicator_cols].describe()
    summary.loc['null_count'] = df[indicator_cols].isnull().sum()
    summary.loc['null_pct'] = (df[indicator_cols].isnull().sum() / len(df)) * 100
    
    return summary

def validate_indicators(df: pd.DataFrame) -> Dict[str, bool]:
    """
    Validate that indicators are calculated correctly
    
    Args:
        df: DataFrame with indicators
    
    Returns:
        Dictionary of validation results
    """
    validation_results = {}
    
    # Check if indicators exist
    expected_indicators = ['macd', 'rsi_14', 'adx_14', 'supertrend', 'bb_middle', 'ema_12', 'stoch_k', 'atr_14']
    
    for indicator in expected_indicators:
        validation_results[f'{indicator}_exists'] = indicator in df.columns
        
        if indicator in df.columns:
            # Check for reasonable values
            values = df[indicator].dropna()
            validation_results[f'{indicator}_has_values'] = len(values) > 0
            validation_results[f'{indicator}_not_all_nan'] = not df[indicator].isna().all()
            
            # Specific validations
            if indicator == 'rsi_14':
                validation_results[f'{indicator}_range_valid'] = ((values >= 0) & (values <= 100)).all()
            elif indicator == 'adx_14':
                validation_results[f'{indicator}_positive'] = (values >= 0).all()
            elif indicator == 'atr_14':
                validation_results[f'{indicator}_positive'] = (values >= 0).all()
    
    return validation_results

if __name__ == "__main__":
    # Test the indicators with sample data
    print("Testing technical indicators...")
    
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    np.random.seed(42)
    
    # Generate realistic OHLC data
    close_prices = 100 + np.cumsum(np.random.randn(100) * 0.02)
    high_prices = close_prices + np.random.uniform(0, 2, 100)
    low_prices = close_prices - np.random.uniform(0, 2, 100)
    open_prices = close_prices + np.random.uniform(-1, 1, 100)
    
    sample_data = pd.DataFrame({
        'datetime': dates,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': np.random.randint(1000, 10000, 100),
        'signal': np.random.choice(['Buy', 'Sell', 'Hold'], 100)
    })
    
    # Add indicators
    sample_with_indicators = add_indicators(sample_data)
    
    # Validate
    validation = validate_indicators(sample_with_indicators)
    
    print("\nValidation Results:")
    for key, value in validation.items():
        status = "✓" if value else "✗"
        print(f"{status} {key}: {value}")
    
    # Show summary
    print(f"\nDataFrame shape: {sample_with_indicators.shape}")
    print(f"Columns: {list(sample_with_indicators.columns)}")
    
    print("\nIndicators test completed successfully!")