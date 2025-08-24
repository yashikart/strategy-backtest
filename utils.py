import pandas as pd
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')

class OptionsDataManager:
    """
    Manages options data loading and ATM options matching for backtesting.
    Handles large options datasets efficiently using chunking and caching.
    """
    
    def __init__(self, options_file_path: str):
        """
        Initialize the options data manager.
        
        Args:
            options_file_path: Path to the options CSV or parquet file
        """
        self.options_file_path = options_file_path
        self.options_cache = {}  # Cache for loaded options data
        self.expiry_cache = {}   # Cache for expiry dates
        
        # Determine file format
        self.is_parquet = options_file_path.endswith('.parquet')
        print(f"📁 Options data format: {'Parquet' if self.is_parquet else 'CSV'}")
        
    def get_nearest_expiry(self, timestamp: str) -> str:
        """
        Find the nearest expiry date for a given timestamp.
        
        Args:
            timestamp: Current timestamp
            
        Returns:
            Nearest expiry date string
        """
        if timestamp in self.expiry_cache:
            return self.expiry_cache[timestamp]
        
        # Parse timestamp
        current_date = pd.to_datetime(timestamp).date()
        
        # Load unique expiries (we'll do this once and cache)
        if not hasattr(self, '_expiry_dates'):
            print("📅 Loading expiry dates...")
            
            if self.is_parquet:
                # Load parquet file (much faster than CSV)
                df_sample = pd.read_parquet(self.options_file_path, columns=['expiry_date'])
                expiry_set = set(df_sample['expiry_date'].unique())
            else:
                # Load CSV with chunking
                chunk_iter = pd.read_csv(self.options_file_path, chunksize=10000, usecols=['expiry_date'])
                expiry_set = set()
                for chunk in tqdm(chunk_iter, desc="Loading expiry dates"):
                    expiry_set.update(chunk['expiry_date'].unique())
            
            self._expiry_dates = sorted([pd.to_datetime(exp).date() for exp in expiry_set])
            print(f"✅ Loaded {len(self._expiry_dates)} unique expiry dates")
        
        # Find nearest expiry after current date
        future_expiries = [exp for exp in self._expiry_dates if exp > current_date]
        
        if not future_expiries:
            # If no future expiries, use the last available
            nearest_expiry = self._expiry_dates[-1]
        else:
            nearest_expiry = min(future_expiries)
        
        nearest_expiry_str = nearest_expiry.strftime('%Y-%m-%d')
        self.expiry_cache[timestamp] = nearest_expiry_str
        return nearest_expiry_str
    
    def find_atm_options(self, timestamp: str, spot_price: float, option_type: str) -> Dict:
        """
        Find At-The-Money (ATM) options for a given timestamp and spot price.
        
        Args:
            timestamp: Current timestamp
            spot_price: Current spot price
            option_type: 'CE' for Call, 'PE' for Put
            
        Returns:
            Dictionary with ATM option details
        """
        cache_key = f"{timestamp}_{spot_price}_{option_type}"
        if cache_key in self.options_cache:
            return self.options_cache[cache_key]
        
        # Get nearest expiry
        nearest_expiry = self.get_nearest_expiry(timestamp)
        
        # Parse timestamp for matching
        target_datetime = pd.to_datetime(timestamp)
        
        # Load options data efficiently based on file format
        if self.is_parquet:
            # Load parquet with filtering (much faster)
            try:
                # Read parquet with filters for better performance
                df_options = pd.read_parquet(
                    self.options_file_path,
                    filters=[
                        ('expiry_date', '==', nearest_expiry),
                        ('option_type', '==', option_type)
                    ]
                )
                
                if not df_options.empty:
                    # Convert datetime and find time matches
                    df_options['datetime_parsed'] = pd.to_datetime(df_options['datetime'])
                    time_diff = abs(df_options['datetime_parsed'] - target_datetime)
                    time_mask = time_diff <= pd.Timedelta(minutes=1)
                    time_filtered = df_options[time_mask]
                    
                    if not time_filtered.empty:
                        # Find ATM strike
                        time_filtered['strike_diff'] = abs(time_filtered['strike_price'] - spot_price)
                        min_diff_idx = time_filtered['strike_diff'].idxmin()
                        best_option = time_filtered.loc[min_diff_idx].to_dict()
                        
            except Exception as e:
                print(f"⚠️ Parquet filtering failed: {e}. Using fallback method.")
                best_option = None
        
        else:
            # Fallback to CSV chunking method
            chunk_iter = pd.read_csv(self.options_file_path, chunksize=50000)
            
            for chunk in chunk_iter:
                # Filter by expiry and option type
                filtered = chunk[
                    (chunk['expiry_date'] == nearest_expiry) &
                    (chunk['option_type'] == option_type)
                ]
                
                if filtered.empty:
                    continue
                
                # Convert datetime strings to datetime objects for comparison
                filtered['datetime_parsed'] = pd.to_datetime(filtered['datetime'])
                
                # Find options within 1 minute of target time
                time_diff = abs(filtered['datetime_parsed'] - target_datetime)
                time_mask = time_diff <= pd.Timedelta(minutes=1)
                
                time_filtered = filtered[time_mask]
                
                if time_filtered.empty:
                    continue
                
                # Find ATM strike (closest to spot price)
                time_filtered['strike_diff'] = abs(time_filtered['strike_price'] - spot_price)
                
                # Get the option with minimum strike difference
                min_diff_idx = time_filtered['strike_diff'].idxmin()
                current_min_diff = time_filtered.loc[min_diff_idx, 'strike_diff']
                
                if current_min_diff < min_strike_diff:
                    min_strike_diff = current_min_diff
                    best_option = time_filtered.loc[min_diff_idx].to_dict()
        
        if best_option is None:
            # Fallback: create a synthetic option
            atm_strike = round(spot_price / 50) * 50  # Round to nearest 50
            best_option = {
                'strike_price': atm_strike,
                'option_type': option_type,
                'expiry_date': nearest_expiry,
                'close': max(spot_price * 0.02, 10),  # Estimate 2% of spot or min 10
                'datetime': timestamp,
                'ticker': f'NIFTY{nearest_expiry.replace("-", "")}{int(atm_strike)}{option_type}',
                'underlying_symbol': 'NIFTY'
            }
        
        self.options_cache[cache_key] = best_option
        return best_option

class PerformanceCalculator:
    """
    Calculates various performance metrics for backtesting.
    """
    
    @staticmethod
    def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.06) -> float:
        """
        Calculate Sharpe ratio.
        
        Args:
            returns: Series of returns
            risk_free_rate: Annual risk-free rate (default 6%)
            
        Returns:
            Sharpe ratio
        """
        if returns.std() == 0:
            return 0
        
        excess_returns = returns.mean() - risk_free_rate / 252  # Daily risk-free rate
        return excess_returns / returns.std() * np.sqrt(252)  # Annualized
    
    @staticmethod
    def calculate_max_drawdown(equity_curve: pd.Series) -> Tuple[float, int, int]:
        """
        Calculate maximum drawdown.
        
        Args:
            equity_curve: Series of equity values
            
        Returns:
            Tuple of (max_drawdown_pct, start_idx, end_idx)
        """
        if len(equity_curve) < 2:
            return 0.0, 0, 0
        
        # Calculate running maximum
        running_max = equity_curve.expanding().max()
        
        # Calculate drawdown
        drawdown = (equity_curve - running_max) / running_max
        
        # Find maximum drawdown
        max_dd_idx = drawdown.idxmin()
        max_drawdown = drawdown.min()
        
        # Find start of drawdown period (handle edge cases)
        try:
            start_subset = running_max[:max_dd_idx]
            if len(start_subset) > 0:
                start_idx = start_subset.idxmax()
            else:
                start_idx = equity_curve.index[0]
        except (ValueError, IndexError):
            start_idx = equity_curve.index[0]
        
        return max_drawdown, start_idx, max_dd_idx
    
    @staticmethod
    def calculate_win_rate(trades_df: pd.DataFrame) -> Dict:
        """
        Calculate win rate and other trade statistics.
        
        Args:
            trades_df: DataFrame with trade data including P&L
            
        Returns:
            Dictionary with trade statistics
        """
        if trades_df.empty:
            return {}
        
        winning_trades = trades_df[trades_df['pnl'] > 0]
        losing_trades = trades_df[trades_df['pnl'] < 0]
        
        return {
            'total_trades': len(trades_df),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(trades_df) if len(trades_df) > 0 else 0,
            'avg_win': winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0,
            'avg_loss': losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0,
            'max_win': trades_df['pnl'].max() if len(trades_df) > 0 else 0,
            'max_loss': trades_df['pnl'].min() if len(trades_df) > 0 else 0,
            'profit_factor': abs(winning_trades['pnl'].sum() / losing_trades['pnl'].sum()) if len(losing_trades) > 0 and losing_trades['pnl'].sum() != 0 else float('inf')
        }

class DataProcessor:
    """
    Utility functions for data processing and preparation.
    """
    
    @staticmethod
    def prepare_ml_features(df: pd.DataFrame, lookback_periods: List[int] = [5, 10, 20]) -> pd.DataFrame:
        """
        Prepare features for machine learning models.
        
        Args:
            df: DataFrame with OHLC and indicator data
            lookback_periods: List of lookback periods for features
            
        Returns:
            DataFrame with ML features
        """
        result_df = df.copy()
        
        # Price-based features
        result_df['returns'] = result_df['close'].pct_change()
        result_df['log_returns'] = np.log(result_df['close'] / result_df['close'].shift(1))
        
        # Volatility features
        for period in lookback_periods:
            result_df[f'volatility_{period}'] = result_df['returns'].rolling(period).std()
            result_df[f'avg_return_{period}'] = result_df['returns'].rolling(period).mean()
        
        # Technical indicator momentum
        if 'rsi_14' in result_df.columns:
            result_df['rsi_momentum'] = result_df['rsi_14'].diff()
        
        if 'macd' in result_df.columns:
            result_df['macd_momentum'] = result_df['macd'].diff()
        
        # Price position features
        if 'bb_high' in result_df.columns and 'bb_low' in result_df.columns:
            result_df['bb_position'] = (result_df['close'] - result_df['bb_low']) / (result_df['bb_high'] - result_df['bb_low'])
        
        # Volume features (if available)
        if 'volume' in result_df.columns:
            result_df['volume_sma_20'] = result_df['volume'].rolling(20).mean()
            result_df['volume_ratio'] = result_df['volume'] / result_df['volume_sma_20']
        
        # Lagged features
        lag_features = ['close', 'rsi_14', 'macd', 'ema_12', 'ema_26']
        for feature in lag_features:
            if feature in result_df.columns:
                for lag in [1, 2, 3, 5]:
                    result_df[f'{feature}_lag_{lag}'] = result_df[feature].shift(lag)
        
        return result_df
    
    @staticmethod
    def create_target_variables(df: pd.DataFrame, 
                               forward_periods: List[int] = [1, 3, 5]) -> pd.DataFrame:
        """
        Create target variables for ML models.
        
        Args:
            df: DataFrame with price data
            forward_periods: List of forward looking periods
            
        Returns:
            DataFrame with target variables
        """
        result_df = df.copy()
        
        for period in forward_periods:
            # Future returns
            result_df[f'future_return_{period}'] = result_df['close'].pct_change(period).shift(-period)
            
            # Binary classification targets
            result_df[f'price_up_{period}'] = (result_df[f'future_return_{period}'] > 0).astype(int)
            
            # Multi-class classification targets
            def classify_return(ret):
                if pd.isna(ret):
                    return np.nan
                elif ret > 0.01:  # > 1% up
                    return 2  # Strong Buy
                elif ret > 0:
                    return 1  # Buy
                elif ret > -0.01:  # > -1% down
                    return 0  # Hold
                else:
                    return -1  # Sell
            
            result_df[f'price_class_{period}'] = result_df[f'future_return_{period}'].apply(classify_return)
        
        return result_df
    
    @staticmethod
    def clean_data(df: pd.DataFrame, fill_method: str = 'forward') -> pd.DataFrame:
        """
        Clean and prepare data for analysis.
        
        Args:
            df: Input DataFrame
            fill_method: Method to fill missing values ('forward', 'backward', 'interpolate')
            
        Returns:
            Cleaned DataFrame
        """
        result_df = df.copy()
        
        # Remove infinite values
        result_df = result_df.replace([np.inf, -np.inf], np.nan)
        
        # Fill missing values
        if fill_method == 'forward':
            result_df = result_df.fillna(method='ffill')
        elif fill_method == 'backward':
            result_df = result_df.fillna(method='bfill')
        elif fill_method == 'interpolate':
            numeric_cols = result_df.select_dtypes(include=[np.number]).columns
            result_df[numeric_cols] = result_df[numeric_cols].interpolate()
        
        # Remove any remaining NaN rows
        result_df = result_df.dropna()
        
        return result_df

def test_utils():
    """
    Test function for utility classes.
    """
    print("Testing Utils...")
    
    # Test OptionsDataManager
    print("\n=== Testing Options Data Manager (Parquet) ===")
    options_manager = OptionsDataManager('data/options_data_2023.parquet')
    
    # Test with a sample timestamp and price
    test_timestamp = '2023-01-02 09:30:00+05:30'
    test_spot_price = 18150.0
    
    print(f"Finding ATM PUT option for spot price {test_spot_price} at {test_timestamp}")
    atm_put = options_manager.find_atm_options(test_timestamp, test_spot_price, 'PE')
    print(f"ATM PUT: Strike {atm_put['strike_price']}, Premium {atm_put['close']}")
    
    print(f"\nFinding ATM CALL option for spot price {test_spot_price} at {test_timestamp}")
    atm_call = options_manager.find_atm_options(test_timestamp, test_spot_price, 'CE')
    print(f"ATM CALL: Strike {atm_call['strike_price']}, Premium {atm_call['close']}")
    
    # Test Performance Calculator
    print("\n=== Testing Performance Calculator ===")
    
    # Create sample returns
    np.random.seed(42)
    sample_returns = pd.Series(np.random.normal(0.001, 0.02, 252))  # Daily returns for 1 year
    sample_equity = (1 + sample_returns).cumprod() * 100000  # Starting with 100k
    
    sharpe = PerformanceCalculator.calculate_sharpe_ratio(sample_returns)
    max_dd, dd_start, dd_end = PerformanceCalculator.calculate_max_drawdown(sample_equity)
    
    print(f"Sharpe Ratio: {sharpe:.3f}")
    print(f"Max Drawdown: {max_dd:.3%}")
    
    # Test Data Processor
    print("\n=== Testing Data Processor ===")
    
    # Load sample data
    df = pd.read_csv('data/spot_with_composite_signals_2023.csv')
    print(f"Original data shape: {df.shape}")
    
    # Prepare ML features
    df_with_features = DataProcessor.prepare_ml_features(df.head(1000))  # Test with subset
    print(f"Data with ML features shape: {df_with_features.shape}")
    
    # Create targets
    df_with_targets = DataProcessor.create_target_variables(df_with_features)
    print(f"Data with targets shape: {df_with_targets.shape}")
    
    # Clean data
    df_clean = DataProcessor.clean_data(df_with_targets)
    print(f"Cleaned data shape: {df_clean.shape}")
    
    print("\nUtils testing completed successfully!")

if __name__ == "__main__":
    test_utils()