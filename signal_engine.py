import pandas as pd
import numpy as np
from indicators import add_indicators
from typing import Dict, Tuple

class SignalEngine:
    """
    Composite Signal Engine that combines in-house signals with technical indicators
    using a weighted voting system.
    
    Voting Logic:
    - In-house signal: 30% weight
    - Technical indicators: 70% weight split across multiple indicators
    - MACD: 15%
    - RSI: 12% 
    - SuperTrend: 15%
    - EMA Crossover: 10%
    - Bollinger Bands: 10%
    - Stochastic: 5%
    - ADX: 8%
    - ATR Volatility: 5%
    """
    
    def __init__(self, 
                 inhouse_weight: float = 0.30,
                 macd_weight: float = 0.15,
                 rsi_weight: float = 0.12,
                 supertrend_weight: float = 0.15,
                 ema_weight: float = 0.10,
                 bb_weight: float = 0.10,
                 stoch_weight: float = 0.05,
                 adx_weight: float = 0.08,
                 atr_weight: float = 0.05):
        """
        Initialize the signal engine with indicator weights.
        
        Args:
            inhouse_weight: Weight for original in-house signal
            macd_weight: Weight for MACD signal
            rsi_weight: Weight for RSI signal
            supertrend_weight: Weight for SuperTrend signal
            ema_weight: Weight for EMA crossover signal
            bb_weight: Weight for Bollinger Bands signal
            stoch_weight: Weight for Stochastic signal
            adx_weight: Weight for ADX signal
            atr_weight: Weight for ATR volatility signal
        """
        self.weights = {
            'inhouse': inhouse_weight,
            'macd': macd_weight,
            'rsi': rsi_weight,
            'supertrend': supertrend_weight,
            'ema': ema_weight,
            'bb': bb_weight,
            'stoch': stoch_weight,
            'adx': adx_weight,
            'atr': atr_weight
        }
        
        # Ensure weights sum to 1
        total_weight = sum(self.weights.values())
        self.weights = {k: v/total_weight for k, v in self.weights.items()}
        
    def generate_macd_signal(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate MACD-based signals.
        Buy: MACD > Signal line and MACD histogram > 0
        Sell: MACD < Signal line and MACD histogram < 0
        """
        buy_condition = (df['macd'] > df['macd_signal']) & (df['macd_hist'] > 0)
        sell_condition = (df['macd'] < df['macd_signal']) & (df['macd_hist'] < 0)
        
        signals = pd.Series('Hold', index=df.index)
        signals[buy_condition] = 'Buy'
        signals[sell_condition] = 'Sell'
        return signals
    
    def generate_rsi_signal(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate RSI-based signals.
        Buy: RSI < 30 (oversold)
        Sell: RSI > 70 (overbought)
        """
        buy_condition = df['rsi_14'] < 30
        sell_condition = df['rsi_14'] > 70
        
        signals = pd.Series('Hold', index=df.index)
        signals[buy_condition] = 'Buy'
        signals[sell_condition] = 'Sell'
        return signals
    
    def generate_ema_signal(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate EMA crossover signals.
        Buy: EMA12 > EMA26 (golden cross)
        Sell: EMA12 < EMA26 (death cross)
        """
        buy_condition = df['ema_12'] > df['ema_26']
        sell_condition = df['ema_12'] < df['ema_26']
        
        signals = pd.Series('Hold', index=df.index)
        signals[buy_condition] = 'Buy'
        signals[sell_condition] = 'Sell'
        return signals
    
    def generate_bb_signal(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate Bollinger Bands signals.
        Buy: Price touches lower band
        Sell: Price touches upper band
        """
        buy_condition = df['close'] <= df['bb_low']
        sell_condition = df['close'] >= df['bb_high']
        
        signals = pd.Series('Hold', index=df.index)
        signals[buy_condition] = 'Buy'
        signals[sell_condition] = 'Sell'
        return signals
    
    def generate_stoch_signal(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate Stochastic signals.
        Buy: %K < 20 and %K > %D
        Sell: %K > 80 and %K < %D
        """
        buy_condition = (df['stoch_k'] < 20) & (df['stoch_k'] > df['stoch_d'])
        sell_condition = (df['stoch_k'] > 80) & (df['stoch_k'] < df['stoch_d'])
        
        signals = pd.Series('Hold', index=df.index)
        signals[buy_condition] = 'Buy'
        signals[sell_condition] = 'Sell'
        return signals
    
    def generate_adx_signal(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate ADX trend strength signals.
        Buy: ADX > 25 and EMA12 > EMA26 (strong uptrend)
        Sell: ADX > 25 and EMA12 < EMA26 (strong downtrend)
        """
        strong_trend = df['adx_14'] > 25
        buy_condition = strong_trend & (df['ema_12'] > df['ema_26'])
        sell_condition = strong_trend & (df['ema_12'] < df['ema_26'])
        
        signals = pd.Series('Hold', index=df.index)
        signals[buy_condition] = 'Buy'
        signals[sell_condition] = 'Sell'
        return signals
    
    def generate_supertrend_signal(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate SuperTrend-based signals.
        Buy: Price above SuperTrend and trend is up
        Sell: Price below SuperTrend and trend is down
        """
        buy_condition = (df['close'] > df['supertrend']) & (df['supertrend_trend'] == 1)
        sell_condition = (df['close'] < df['supertrend']) & (df['supertrend_trend'] == -1)
        
        signals = pd.Series('Hold', index=df.index)
        signals[buy_condition] = 'Buy'
        signals[sell_condition] = 'Sell'
        return signals
    
    def generate_atr_signal(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate ATR-based volatility signals.
        Buy: Low volatility with upward momentum
        Sell: High volatility with downward momentum
        """
        # Calculate ATR percentile
        atr_percentile = df['atr_14'].rolling(50).rank(pct=True)
        
        # Calculate price momentum
        momentum = df['close'].pct_change(5)
        
        # Low volatility + positive momentum = Buy
        buy_condition = (atr_percentile < 0.3) & (momentum > 0.01)
        # High volatility + negative momentum = Sell
        sell_condition = (atr_percentile > 0.7) & (momentum < -0.01)
        
        signals = pd.Series('Hold', index=df.index)
        signals[buy_condition] = 'Buy'
        signals[sell_condition] = 'Sell'
        return signals
    
    def signal_to_numeric(self, signal_series: pd.Series) -> pd.Series:
        """
        Convert signal strings to numeric values.
        Buy: +1, Sell: -1, Hold: 0
        """
        # Handle any NaN values or unexpected strings
        return signal_series.fillna('Hold').map({'Buy': 1, 'Sell': -1, 'Hold': 0}).fillna(0)
    
    def numeric_to_signal(self, numeric_series: pd.Series) -> pd.Series:
        """
        Convert numeric values back to signal strings.
        +1: Buy, -1: Sell, 0: Hold
        """
        def map_signal(x):
            # Handle NaN and non-numeric values
            if pd.isna(x) or not isinstance(x, (int, float)):
                return 'Hold'
            
            x = float(x)  # Ensure it's a float
            if x > 0.3:  # Lower threshold for Buy
                return 'Buy'
            elif x < -0.3:  # Lower threshold for Sell
                return 'Sell'
            else:
                return 'Hold'
        
        return numeric_series.apply(map_signal)
    
    def generate_composite_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate composite signals using weighted voting.
        
        Args:
            df: DataFrame with OHLC data and existing signals
            
        Returns:
            DataFrame with additional composite signal columns
        """
        # Add technical indicators if not present
        if 'ema_12' not in df.columns:
            df = add_indicators(df)
        
        result_df = df.copy()
        
        # Convert original signal to numeric
        inhouse_numeric = self.signal_to_numeric(df['signal'])
        
        # Generate individual indicator signals
        macd_signal = self.generate_macd_signal(df)
        rsi_signal = self.generate_rsi_signal(df)
        supertrend_signal = self.generate_supertrend_signal(df)
        ema_signal = self.generate_ema_signal(df)
        bb_signal = self.generate_bb_signal(df)
        stoch_signal = self.generate_stoch_signal(df)
        adx_signal = self.generate_adx_signal(df)
        atr_signal = self.generate_atr_signal(df)
        
        # Convert all signals to numeric
        macd_numeric = self.signal_to_numeric(macd_signal)
        rsi_numeric = self.signal_to_numeric(rsi_signal)
        supertrend_numeric = self.signal_to_numeric(supertrend_signal)
        ema_numeric = self.signal_to_numeric(ema_signal)
        bb_numeric = self.signal_to_numeric(bb_signal)
        stoch_numeric = self.signal_to_numeric(stoch_signal)
        adx_numeric = self.signal_to_numeric(adx_signal)
        atr_numeric = self.signal_to_numeric(atr_signal)
        
        # Calculate weighted composite score
        composite_score = (
            self.weights['inhouse'] * inhouse_numeric +
            self.weights['macd'] * macd_numeric +
            self.weights['rsi'] * rsi_numeric +
            self.weights['supertrend'] * supertrend_numeric +
            self.weights['ema'] * ema_numeric +
            self.weights['bb'] * bb_numeric +
            self.weights['stoch'] * stoch_numeric +
            self.weights['adx'] * adx_numeric +
            self.weights['atr'] * atr_numeric
        )
        
        # Convert composite score to final signal
        composite_signal = self.numeric_to_signal(composite_score)
        
        # Add all signals to result
        result_df['macd_signal'] = macd_signal
        result_df['rsi_signal'] = rsi_signal
        result_df['supertrend_signal'] = supertrend_signal
        result_df['ema_signal'] = ema_signal
        result_df['bb_signal'] = bb_signal
        result_df['stoch_signal'] = stoch_signal
        result_df['adx_signal'] = adx_signal
        result_df['atr_signal'] = atr_signal
        result_df['composite_score'] = composite_score
        result_df['composite_signal'] = composite_signal
        
        # Add signal strength and confidence metrics
        result_df['signal_strength'] = abs(composite_score)
        result_df['signal_confidence'] = self.calculate_signal_confidence(result_df)
        
        return result_df
    
    def calculate_signal_confidence(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate signal confidence based on indicator agreement.
        """
        signal_cols = ['macd_signal', 'rsi_signal', 'supertrend_signal', 'ema_signal', 
                      'bb_signal', 'stoch_signal', 'adx_signal', 'atr_signal']
        
        # Count how many indicators agree with composite signal
        confidence_scores = []
        
        for idx in df.index:
            composite = df.loc[idx, 'composite_signal']
            if composite == 'Hold':
                confidence_scores.append(0.5)  # Neutral confidence
                continue
            
            agreements = 0
            total_signals = 0
            
            for col in signal_cols:
                if col in df.columns:
                    signal_val = df.loc[idx, col]
                    if signal_val != 'Hold':
                        total_signals += 1
                        if signal_val == composite:
                            agreements += 1
            
            if total_signals > 0:
                confidence = agreements / total_signals
            else:
                confidence = 0.5
                
            confidence_scores.append(confidence)
        
        return pd.Series(confidence_scores, index=df.index)
    
    def get_signal_summary(self, df: pd.DataFrame) -> Dict:
        """
        Get summary statistics of all signals.
        
        Args:
            df: DataFrame with signals
            
        Returns:
            Dictionary with signal statistics
        """
        signal_cols = ['signal', 'composite_signal', 'macd_signal', 'rsi_signal', 
                      'supertrend_signal', 'ema_signal', 'bb_signal', 'stoch_signal', 
                      'adx_signal', 'atr_signal']
        
        summary = {}
        for col in signal_cols:
            if col in df.columns:
                summary[col] = df[col].value_counts().to_dict()
        
        # Calculate signal agreement metrics
        if 'composite_signal' in df.columns:
            # Agreement between composite and original
            agreement = (df['signal'] == df['composite_signal']).mean()
            summary['composite_inhouse_agreement'] = agreement
            
            # Signal strength distribution
            summary['composite_score_stats'] = {
                'mean': df['composite_score'].mean(),
                'std': df['composite_score'].std(),
                'min': df['composite_score'].min(),
                'max': df['composite_score'].max()
            }
        
        return summary

def test_signal_engine():
    """
    Test function for the signal engine.
    """
    print("Testing Signal Engine...")
    
    # Load data
    df = pd.read_csv('data/spot_with_signals_2023.csv')
    
    # Initialize signal engine
    engine = SignalEngine()
    
    # Generate composite signals
    print("Generating composite signals...")
    df_with_signals = engine.generate_composite_signal(df)
    
    # Get summary
    summary = engine.get_signal_summary(df_with_signals)
    
    print("\n=== Signal Summary ===")
    for signal_type, counts in summary.items():
        if isinstance(counts, dict) and 'Buy' in counts:
            print(f"{signal_type}: {counts}")
    
    print(f"\nComposite-Inhouse Agreement: {summary.get('composite_inhouse_agreement', 'N/A'):.3f}")
    print(f"Composite Score Stats: {summary.get('composite_score_stats', {})}")
    
    # Save results
    df_with_signals.to_csv('data/spot_with_composite_signals_2023.csv', index=False)
    print("\nResults saved to: data/spot_with_composite_signals_2023.csv")
    
    return df_with_signals

if __name__ == "__main__":
    test_signal_engine()