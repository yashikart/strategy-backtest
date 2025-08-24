import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta, time
from typing import Dict, List, Tuple, Optional
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')

from signal_engine import SignalEngine
from utils import OptionsDataManager, PerformanceCalculator, DataProcessor
from model import MLModelManager

class OptionsBacktester:
    """
    Comprehensive options backtesting system with ML-enhanced signals.
    
    Strategy:
    - Buy signal: Sell ATM PUT option (collect premium)
    - Sell signal: Sell ATM CALL option (collect premium)
    - Risk management: 1.5% stop-loss, 3% take-profit, EOD exit at 15:15
    """
    
    def __init__(self, 
                 initial_capital: float = 200000,
                 stop_loss_pct: float = 0.015,
                 take_profit_pct: float = 0.03,
                 exit_time: str = '15:15',
                 options_file: str = 'data/options_data_2023.csv'):
        """
        Initialize the backtester.
        
        Args:
            initial_capital: Starting capital in rupees
            stop_loss_pct: Stop loss percentage (1.5%)
            take_profit_pct: Take profit percentage (3%)
            exit_time: Force exit time (15:15)
            options_file: Path to options data file
        """
        self.initial_capital = initial_capital
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.exit_time = time.fromisoformat(exit_time)
        
        # Initialize components
        self.signal_engine = SignalEngine()
        self.options_manager = OptionsDataManager(options_file)
        self.ml_manager = MLModelManager()
        
        # Tracking variables
        self.trades = []
        self.equity_curve = []
        self.positions = []  # Current open positions
        self.capital = initial_capital
        
    def prepare_data_with_ml(self, spot_df: pd.DataFrame, skip_ml: bool = False) -> pd.DataFrame:
        """
        Prepare data with ML-enhanced signals.
        
        Args:
            spot_df: Spot data with basic signals
            skip_ml: Skip ML training for faster execution
            
        Returns:
            DataFrame with ML-enhanced signals
        """
        print("Preparing data with signal enhancement...")
        
        # Generate composite signals
        df_with_signals = self.signal_engine.generate_composite_signal(spot_df)
        
        if skip_ml:
            print("Skipping ML training for faster execution...")
            df_with_signals['ml_signal'] = 'Hold'
            df_with_signals['final_signal'] = df_with_signals['composite_signal']
            return df_with_signals
        
        # Prepare simplified features for ML training
        print("Preparing features for ML training...")
        
        # Create a clean feature set with available indicators (only numeric)
        feature_columns = [
            'close', 'rsi_14', 'macd', 'macd_hist',  # Removed macd_signal (contains strings)
            'ema_12', 'ema_26', 'bb_middle', 'bb_high', 'bb_low',
            'stoch_k', 'stoch_d', 'adx_14', 'atr_14', 'supertrend',
            'momentum_10', 'momentum_20', 'volatility_20', 'signal_strength'
        ]
        
        # Filter to only existing columns
        available_features = [col for col in feature_columns if col in df_with_signals.columns]
        print(f"Using {len(available_features)} features: {available_features}")
        
        if len(available_features) < 5:
            print("⚠️ Insufficient features for ML training. Using composite signal only.")
            df_with_signals['ml_signal'] = 'Hold'
            df_with_signals['final_signal'] = df_with_signals['composite_signal']
            return df_with_signals
        
        # Create simplified ML training data
        try:
            # Use a subset of data for faster training
            subset_size = min(2000, len(df_with_signals))
            start_idx = max(100, len(df_with_signals) - subset_size)  # Avoid early NaN values
            df_subset = df_with_signals.iloc[start_idx:start_idx + subset_size].copy()
            
            # Prepare features
            X = df_subset[available_features].copy()
            
            # Forward fill any missing values
            X = X.fillna(method='ffill').fillna(method='bfill')
            
            # Create target based on future price movement
            future_price = df_subset['close'].shift(-5)  # 5 periods ahead
            price_change = (future_price - df_subset['close']) / df_subset['close']
            
            # Create classification target: -1 (Sell), 0 (Hold), 1 (Buy)
            y = pd.Series(0, index=df_subset.index)  # Default Hold
            y[price_change > 0.01] = 1   # Buy if price goes up > 1%
            y[price_change < -0.01] = -1  # Sell if price goes down > 1%
            
            # Remove rows with missing targets
            valid_mask = ~pd.isna(price_change)
            X_clean = X[valid_mask]
            y_clean = y[valid_mask]
            
            print(f"Clean dataset: {len(X_clean)} samples")
            print(f"Target distribution: {y_clean.value_counts().to_dict()}")
            
            if len(X_clean) < 100:
                print("⚠️ Insufficient clean data for ML training. Using composite signal only.")
                df_with_signals['ml_signal'] = 'Hold'
                df_with_signals['final_signal'] = df_with_signals['composite_signal']
                return df_with_signals
            
            # Train ML model
            from sklearn.model_selection import train_test_split
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import accuracy_score
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_clean, y_clean, test_size=0.2, random_state=42
            )
            
            print(f"🤖 Training ML model on {len(X_train)} samples...")
            
            # Train Random Forest model
            ml_model = RandomForestClassifier(
                n_estimators=50,  # Smaller for speed
                max_depth=8, 
                random_state=42,
                class_weight='balanced'
            )
            
            ml_model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = ml_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"✅ ML Model trained successfully!")
            print(f"   Training samples: {len(X_train)}")
            print(f"   Test accuracy: {accuracy:.3f}")
            
            # Generate predictions for the full dataset
            X_full = df_with_signals[available_features].copy()
            
            # Clean the prediction data the same way as training data
            X_full = X_full.fillna(method='ffill').fillna(method='bfill')
            
            # Ensure all data is numeric
            for col in X_full.columns:
                X_full[col] = pd.to_numeric(X_full[col], errors='coerce')
            
            # Fill any remaining NaN with 0
            X_full = X_full.fillna(0)
            
            ml_predictions = ml_model.predict(X_full)
            
            # Map predictions to signals
            signal_mapping = {-1: 'Sell', 0: 'Hold', 1: 'Buy'}
            ml_signals = [signal_mapping[pred] for pred in ml_predictions]
            df_with_signals['ml_signal'] = ml_signals
            
            # Create enhanced final signal combining composite and ML
            def combine_signals(row):
                composite = row['composite_signal']
                ml = row['ml_signal']
                
                # Give ML signal more weight for non-Hold predictions
                if ml == 'Buy' and composite != 'Sell':
                    return 'Buy'
                elif ml == 'Sell' and composite != 'Buy':
                    return 'Sell'
                elif composite in ['Buy', 'Sell'] and ml == 'Hold':
                    return composite
                else:
                    return 'Hold'
            
            df_with_signals['final_signal'] = df_with_signals.apply(combine_signals, axis=1)
            
            # Check signal quality
            signal_counts = df_with_signals['final_signal'].value_counts()
            total_signals = signal_counts.get('Buy', 0) + signal_counts.get('Sell', 0)
            
            print(f"✅ ML-enhanced signals generated:")
            print(f"   Buy signals: {signal_counts.get('Buy', 0)}")
            print(f"   Sell signals: {signal_counts.get('Sell', 0)}")
            print(f"   Hold signals: {signal_counts.get('Hold', 0)}")
            print(f"   Total trading signals: {total_signals}")
            
            if total_signals < 10:  # Minimum viable signals
                print("⚠️ Too few ML signals generated. Using composite signals.")
                df_with_signals['final_signal'] = df_with_signals['composite_signal']
            
        except Exception as e:
            print(f"❌ ML enhancement failed: {e}")
            print("   Using composite signal as final signal")
            df_with_signals['ml_signal'] = 'Hold'
            df_with_signals['final_signal'] = df_with_signals['composite_signal']
        
        return df_with_signals
    
    def calculate_option_pnl(self, entry_price: float, current_price: float, 
                            option_type: str, position_type: str = 'short') -> float:
        """
        Calculate P&L for option position.
        
        Args:
            entry_price: Price at which option was sold/bought
            current_price: Current option price
            option_type: 'CE' or 'PE'
            position_type: 'short' (selling) or 'long' (buying)
            
        Returns:
            P&L amount
        """
        if position_type == 'short':
            # We sold the option, so we profit when price decreases
            return entry_price - current_price
        else:
            # We bought the option, so we profit when price increases
            return current_price - entry_price
    
    def get_option_data(self, spot_price: float, current_time: datetime, option_type: str) -> Dict:
        """
        Get realistic option data based on spot price and time.
        
        Args:
            spot_price: Current spot price
            current_time: Current timestamp
            option_type: 'CE' or 'PE'
            
        Returns:
            Dictionary with option details
        """
        # Find ATM strike (round to nearest 50)
        atm_strike = round(spot_price / 50) * 50
        
        # Estimate option premium based on market conditions
        premium = self.estimate_option_premium(spot_price, atm_strike, option_type)
        
        # Generate realistic instrument name
        expiry_date = current_time + timedelta(days=7)  # Assume weekly options
        instrument_name = f"NIFTY{expiry_date.strftime('%d%b%Y').upper()}{int(atm_strike)}{option_type}"
        
        return {
            'strike': atm_strike,
            'premium': premium,
            'instrument': instrument_name,
            'expiry': expiry_date
        }
    
    def check_risk_management(self, position: Dict, current_option_price: float, 
                             current_time: datetime) -> Tuple[bool, str]:
        """
        Check if position should be closed due to risk management rules.
        
        Args:
            position: Position dictionary
            current_option_price: Current option price
            current_time: Current timestamp
            
        Returns:
            Tuple of (should_close, reason)
        """
        entry_price = position['entry_price']
        option_type = position['option_type']
        
        # Calculate current P&L
        pnl = self.calculate_option_pnl(entry_price, current_option_price, option_type, 'short')
        pnl_pct = pnl / entry_price
        
        # Check stop loss
        if pnl_pct <= -self.stop_loss_pct:
            return True, 'stop_loss'
        
        # Check take profit
        if pnl_pct >= self.take_profit_pct:
            return True, 'take_profit'
        
        # Check time-based exit (15:15)
        current_time_only = current_time.time()
        if current_time_only >= self.exit_time:
            return True, 'eod_exit'
        
        return False, 'hold'
    
    def estimate_option_premium(self, spot_price: float, strike_price: float, option_type: str) -> float:
        """
        Estimate option premium using simplified Black-Scholes approximation.
        
        Args:
            spot_price: Current spot price
            strike_price: Option strike price
            option_type: 'CE' or 'PE'
            
        Returns:
            Estimated option premium
        """
        # Simplified option pricing for demo
        if option_type == 'PE':
            # PUT option
            if spot_price < strike_price:
                # ITM
                intrinsic = strike_price - spot_price
                time_value = spot_price * 0.02  # 2% time value
            else:
                # OTM
                intrinsic = 0
                time_value = spot_price * 0.01  # 1% time value
        else:
            # CALL option
            if spot_price > strike_price:
                # ITM
                intrinsic = spot_price - strike_price
                time_value = spot_price * 0.02  # 2% time value
            else:
                # OTM
                intrinsic = 0
                time_value = spot_price * 0.01  # 1% time value
        
        premium = intrinsic + time_value
        return max(premium, 10)  # Minimum premium of 10
    
    def execute_options_strategy(self, signal: str, spot_price: float, current_time: datetime) -> Dict:
        """
        Execute the options strategy based on signal.
        
        Strategy:
        - Buy signal: Sell ATM PUT option (collect premium, profit if market goes up/sideways)
        - Sell signal: Sell ATM CALL option (collect premium, profit if market goes down/sideways)
        
        Args:
            signal: Trading signal ('Buy' or 'Sell')
            spot_price: Current spot price
            current_time: Current timestamp
            
        Returns:
            Dictionary with trade details
        """
        if signal == 'Buy':
            option_type = 'PE'  # Sell ATM PUT
            expectation = 'bullish'  # Expecting market to go up or stay flat
        else:  # signal == 'Sell'
            option_type = 'CE'  # Sell ATM CALL
            expectation = 'bearish'  # Expecting market to go down or stay flat
        
        # Get option details
        option_data = self.get_option_data(spot_price, current_time, option_type)
        
        # Calculate position size (number of lots)
        # Using conservative position sizing - 5% of capital per trade
        position_value = self.capital * 0.05
        lot_size = 50  # NIFTY lot size
        max_lots = max(1, int(position_value / (option_data['premium'] * lot_size)))
        
        # Apply maximum position limit (risk management)
        max_lots = min(max_lots, 5)  # Maximum 5 lots per trade
        
        trade_details = {
            'signal': signal,
            'option_type': option_type,
            'expectation': expectation,
            'strike': option_data['strike'],
            'premium': option_data['premium'],
            'instrument': option_data['instrument'],
            'lots': max_lots,
            'total_premium': option_data['premium'] * lot_size * max_lots,
            'entry_time': current_time,
            'expiry': option_data['expiry']
        }
        
        return trade_details
    
    def simulate_realistic_trade_outcome(self, trade_details: Dict, market_movement: float = None) -> Dict:
        """
        Simulate realistic trade outcome based on market conditions and option behavior.
        
        Args:
            trade_details: Trade details from execute_options_strategy
            market_movement: Optional market movement (for testing)
            
        Returns:
            Dictionary with trade outcome
        """
        import random
        np.random.seed(42)  # For reproducibility
        
        option_type = trade_details['option_type']
        strike = trade_details['strike']
        entry_premium = trade_details['premium']
        total_premium_collected = trade_details['total_premium']
        
        # Simulate market movement if not provided
        if market_movement is None:
            # Realistic daily market movement (normal distribution)
            market_movement = np.random.normal(0, 0.015)  # 1.5% daily volatility
        
        # Simulate time decay (theta)
        time_decay_factor = random.uniform(0.85, 0.95)  # Options lose 5-15% value due to time decay
        
        # Calculate exit premium based on market movement
        if option_type == 'PE':  # We sold PUT
            # PUT gains value when market goes down
            if market_movement < -0.01:  # Market down > 1%
                # PUT becomes more valuable (bad for seller)
                exit_premium = entry_premium * (1 + abs(market_movement) * 2) * time_decay_factor
            else:
                # Market up or flat (good for PUT seller)
                exit_premium = entry_premium * time_decay_factor * (1 - market_movement)
        else:  # option_type == 'CE', we sold CALL
            # CALL gains value when market goes up
            if market_movement > 0.01:  # Market up > 1%
                # CALL becomes more valuable (bad for seller)
                exit_premium = entry_premium * (1 + market_movement * 2) * time_decay_factor
            else:
                # Market down or flat (good for CALL seller)
                exit_premium = entry_premium * time_decay_factor * (1 + abs(market_movement))
        
        # Ensure realistic minimum premium
        exit_premium = max(exit_premium, entry_premium * 0.1)  # Min 10% of entry premium
        
        # Calculate P&L (we are option sellers)
        pnl_per_unit = entry_premium - exit_premium
        total_pnl = pnl_per_unit * 50 * trade_details['lots']  # lot_size * lots
        
        # Apply risk management
        max_loss = total_premium_collected * self.stop_loss_pct
        max_profit = total_premium_collected * self.take_profit_pct
        
        # Determine exit reason
        if total_pnl <= -max_loss:
            exit_reason = 'stop_loss'
            total_pnl = -max_loss
        elif total_pnl >= max_profit:
            exit_reason = 'take_profit'
            total_pnl = max_profit
        else:
            exit_reason = 'time_decay'
        
        outcome = {
            'entry_premium': entry_premium,
            'exit_premium': exit_premium,
            'pnl_per_unit': pnl_per_unit,
            'total_pnl': total_pnl,
            'exit_reason': exit_reason,
            'market_movement': market_movement,
            'return_pct': total_pnl / total_premium_collected if total_premium_collected > 0 else 0
        }
        
        return outcome
    
    def run_backtest(self, spot_df: pd.DataFrame, max_signals: int = 50, sample_pct: float = 0.1, skip_ml: bool = False) -> Dict:
        """
        Run the complete backtest with performance optimizations.
        
        Args:
            spot_df: Spot data with signals
            max_signals: Maximum number of signals to process (for speed)
            sample_pct: Percentage of data to sample for faster processing
            
        Returns:
            Dictionary with backtest results
        """
        print("Starting optimized backtest...")
        
        # Sample data for faster processing if dataset is large
        if len(spot_df) > 10000:
            print(f"Large dataset detected ({len(spot_df)} rows). Sampling {sample_pct:.1%} for faster processing...")
            sample_size = int(len(spot_df) * sample_pct)
            spot_df = spot_df.sample(n=sample_size, random_state=42).sort_values('datetime').reset_index(drop=True)
            print(f"Using {len(spot_df)} sampled data points")
        
        # Prepare data with ML enhancement (using smaller subset)
        df = self.prepare_data_with_ml(spot_df, skip_ml=skip_ml)
        
        # Filter for trading signals only and limit count
        signal_mask = df['final_signal'].isin(['Buy', 'Sell'])
        trading_signals = df[signal_mask].copy()
        
        if len(trading_signals) > max_signals:
            print(f"Limiting to first {max_signals} signals for demo (out of {len(trading_signals)} total)")
            trading_signals = trading_signals.head(max_signals)
        
        print(f"Processing {len(trading_signals)} trading signals...")
        
        # Initialize tracking
        self.trades = []
        self.equity_curve = []
        self.positions = []
        self.capital = self.initial_capital
        
        # Process each trading signal with progress bar
        print(f"\n📊 Processing {len(trading_signals)} trading signals...")
        
        with tqdm(total=len(trading_signals), desc="🔄 Executing trades", unit="signal") as pbar:
            for i, (idx, row) in enumerate(trading_signals.iterrows()):
                try:
                    current_time = pd.to_datetime(row['datetime'])
                    spot_price = row['close']
                    signal = row['final_signal']
                    
                    # Execute options strategy
                    trade_details = self.execute_options_strategy(signal, spot_price, current_time)
                    
                    # Simulate realistic trade outcome
                    outcome = self.simulate_realistic_trade_outcome(trade_details)
                    
                    # Update capital
                    self.capital += outcome['total_pnl']
                    
                    # Record detailed trade
                    trade = {
                        'entry_time': trade_details['entry_time'].strftime('%Y-%m-%d %H:%M:%S'),
                        'exit_time': trade_details['entry_time'].strftime('%Y-%m-%d %H:%M:%S'),  # Simplified for demo
                        'instrument': trade_details['instrument'],
                        'signal': signal,
                        'option_type': trade_details['option_type'],
                        'strike': trade_details['strike'],
                        'lots': trade_details['lots'],
                        'entry_premium': trade_details['premium'],
                        'exit_premium': outcome['exit_premium'],
                        'premium_collected': trade_details['total_premium'],
                        'pnl': outcome['total_pnl'],
                        'return_pct': outcome['return_pct'] * 100,
                        'exit_reason': outcome['exit_reason'],
                        'market_movement': outcome['market_movement'] * 100,
                        'spot_price': spot_price
                    }
                    self.trades.append(trade)
                    
                    # Record equity curve
                    self.equity_curve.append({
                        'datetime': row['datetime'],
                        'equity': self.capital,
                        'total_return_pct': ((self.capital - self.initial_capital) / self.initial_capital) * 100,
                        'positions': len(self.positions),
                        'trade_count': len(self.trades)
                    })
                    
                    # Update progress bar with detailed info
                    current_return = ((self.capital - self.initial_capital) / self.initial_capital) * 100
                    pbar.update(1)
                    pbar.set_postfix({
                        'Capital': f'₹{self.capital:,.0f}',
                        'Return': f'{current_return:+.1f}%',
                        'Last P&L': f'{outcome["total_pnl"]:+.0f}'
                    })
                    
                except Exception as e:
                    print(f"\n⚠️ Error processing signal {i}: {e}")
                    pbar.update(1)
                    continue
        
        final_return = ((self.capital - self.initial_capital) / self.initial_capital) * 100
        print(f"\n✅ Backtest completed! Final capital: ₹{self.capital:,.2f} ({final_return:+.2f}% return)")
        print(f"📊 Total trades executed: {len(self.trades)}")
        
        return self.analyze_results()
    
    def analyze_results(self) -> Dict:
        """
        Analyze backtest results and calculate performance metrics.
        
        Returns:
            Dictionary with performance metrics
        """
        if not self.trades:
            return {'error': 'No trades executed'}
        
        # Convert to DataFrames
        trades_df = pd.DataFrame(self.trades)
        equity_df = pd.DataFrame(self.equity_curve)
        
        # Calculate basic metrics
        total_return = (self.capital - self.initial_capital) / self.initial_capital
        
        # Calculate daily returns
        equity_df['equity_pct_change'] = equity_df['equity'].pct_change()
        returns_series = equity_df['equity_pct_change'].dropna()
        
        # Performance metrics
        sharpe_ratio = PerformanceCalculator.calculate_sharpe_ratio(returns_series)
        max_drawdown, dd_start, dd_end = PerformanceCalculator.calculate_max_drawdown(equity_df['equity'])
        
        # Trade statistics
        trade_stats = PerformanceCalculator.calculate_win_rate(trades_df)
        
        results = {
            'initial_capital': self.initial_capital,
            'final_capital': self.capital,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown * 100,
            'trades_df': trades_df,
            'equity_df': equity_df,
            **trade_stats
        }
        
        return results
    
    def save_results(self, results: Dict, output_dir: str = 'results') -> None:
        """
        Save backtest results to files.
        
        Args:
            results: Results dictionary from analyze_results
            output_dir: Directory to save files
        """
        print("Saving results...")
        
        # Save trades CSV
        if 'trades_df' in results:
            trades_df = results['trades_df']
            trades_df.to_csv(f'{output_dir}/trades.csv', index=False)
            print(f"Trades saved to {output_dir}/trades.csv")
        
        # Save metrics CSV
        metrics_data = {
            'metric': ['Total Return %', 'Sharpe Ratio', 'Max Drawdown %', 'Win Rate %', 
                      'Total Trades', 'Profit Factor', 'Avg Win', 'Avg Loss'],
            'value': [
                results.get('total_return_pct', 0),
                results.get('sharpe_ratio', 0),
                results.get('max_drawdown_pct', 0),
                results.get('win_rate', 0) * 100,
                results.get('total_trades', 0),
                results.get('profit_factor', 0),
                results.get('avg_win', 0),
                results.get('avg_loss', 0)
            ]
        }
        metrics_df = pd.DataFrame(metrics_data)
        metrics_df.to_csv(f'{output_dir}/metrics.csv', index=False)
        print(f"Metrics saved to {output_dir}/metrics.csv")
        
        # Generate charts
        self.generate_charts(results, output_dir)
    
    def generate_charts(self, results: Dict, output_dir: str = 'results') -> None:
        """
        Generate comprehensive charts and visualizations.
        
        Args:
            results: Results dictionary
            output_dir: Directory to save charts
        """
        if 'equity_df' not in results or 'trades_df' not in results:
            return
        
        equity_df = results['equity_df']
        trades_df = results['trades_df']
        equity_df['datetime'] = pd.to_datetime(equity_df['datetime'])
        
        # Set up the plotting style
        plt.style.use('default')
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = 'white'
        
        # Create comprehensive dashboard
        fig = plt.figure(figsize=(20, 15))
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
        
        # 1. Equity Curve (main chart)
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(equity_df['datetime'], equity_df['equity'], linewidth=3, color='#2E86AB', label='Portfolio Value')
        ax1.axhline(y=self.initial_capital, color='#A23B72', linestyle='--', linewidth=2, alpha=0.8, label='Initial Capital')
        ax1.set_title('Portfolio Equity Curve', fontsize=18, fontweight='bold', pad=20)
        ax1.set_ylabel('Portfolio Value (₹)', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=12)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'₹{x:,.0f}'))
        
        # 2. Drawdown Chart
        ax2 = fig.add_subplot(gs[1, :])
        running_max = equity_df['equity'].expanding().max()
        drawdown = (equity_df['equity'] - running_max) / running_max * 100
        ax2.fill_between(equity_df['datetime'], drawdown, 0, alpha=0.8, color='#F18F01')
        ax2.plot(equity_df['datetime'], drawdown, linewidth=2, color='#C73E1D')
        ax2.set_title('Portfolio Drawdown', fontsize=16, fontweight='bold')
        ax2.set_ylabel('Drawdown (%)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=results['max_drawdown_pct'], color='red', linestyle=':', label=f'Max DD: {results["max_drawdown_pct"]:.1f}%')
        ax2.legend()
        
        # 3. Monthly Returns Heatmap
        ax3 = fig.add_subplot(gs[2, 0])
        monthly_returns = equity_df.set_index('datetime')['total_return_pct'].resample('M').last().pct_change().dropna()
        if len(monthly_returns) > 1:
            monthly_data = monthly_returns.values.reshape(-1, 1)
            im = ax3.imshow(monthly_data, cmap='RdYlGn', aspect='auto')
            ax3.set_title('Monthly Returns (%)', fontsize=14, fontweight='bold')
            ax3.set_ylabel('Month', fontsize=12)
            ax3.set_xticks([])
            plt.colorbar(im, ax=ax3, shrink=0.8)
        
        # 4. Trade Distribution
        ax4 = fig.add_subplot(gs[2, 1])
        if len(trades_df) > 0:
            winning_trades = trades_df[trades_df['pnl'] > 0]['pnl']
            losing_trades = trades_df[trades_df['pnl'] < 0]['pnl']
            
            ax4.hist([winning_trades, losing_trades], bins=20, alpha=0.7, 
                    label=['Winning Trades', 'Losing Trades'], color=['green', 'red'])
            ax4.set_title('P&L Distribution', fontsize=14, fontweight='bold')
            ax4.set_xlabel('P&L (₹)', fontsize=12)
            ax4.set_ylabel('Frequency', fontsize=12)
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        # 5. Performance Metrics Table
        ax5 = fig.add_subplot(gs[2, 2])
        ax5.axis('off')
        
        best_trade = trades_df['pnl'].max() if len(trades_df) > 0 else 0
        worst_trade = trades_df['pnl'].min() if len(trades_df) > 0 else 0
        
        metrics_text = f"""
        PERFORMANCE METRICS
        
        Total Return: {results['total_return_pct']:.2f}%
        Sharpe Ratio: {results.get('sharpe_ratio', 0):.3f}
        Max Drawdown: {results['max_drawdown_pct']:.2f}%
        
        Total Trades: {results.get('total_trades', 0)}
        Win Rate: {results.get('win_rate', 0):.1%}
        Profit Factor: {results.get('profit_factor', 0):.2f}
        
        Avg Win: ₹{results.get('avg_win', 0):.0f}
        Avg Loss: ₹{results.get('avg_loss', 0):.0f}
        
        Best Trade: ₹{best_trade:.0f}
        Worst Trade: ₹{worst_trade:.0f}
        """
        ax5.text(0.1, 0.9, metrics_text, transform=ax5.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        # 6. Trade Timeline
        ax6 = fig.add_subplot(gs[3, :])
        if len(trades_df) > 0:
            trade_dates = pd.to_datetime(trades_df['entry_time'])
            cumulative_pnl = trades_df['pnl'].cumsum()
            
            colors = ['green' if pnl > 0 else 'red' for pnl in trades_df['pnl']]
            ax6.scatter(trade_dates, cumulative_pnl, c=colors, alpha=0.7, s=50)
            ax6.plot(trade_dates, cumulative_pnl, linewidth=2, color='blue', alpha=0.6)
            ax6.set_title('Cumulative P&L by Trade', fontsize=16, fontweight='bold')
            ax6.set_xlabel('Date', fontsize=12)
            ax6.set_ylabel('Cumulative P&L (₹)', fontsize=12)
            ax6.grid(True, alpha=0.3)
        
        # Format x-axis for time-based charts
        for ax in [ax1, ax2, ax6]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.suptitle('Options Strategy Backtest Dashboard', fontsize=24, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        plt.savefig(f'{output_dir}/backtest_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create individual equity curve chart
        plt.figure(figsize=(15, 8))
        plt.plot(equity_df['datetime'], equity_df['equity'], linewidth=3, color='#2E86AB')
        plt.axhline(y=self.initial_capital, color='#A23B72', linestyle='--', linewidth=2, alpha=0.8, label='Initial Capital')
        plt.title('Portfolio Equity Curve', fontsize=18, fontweight='bold')
        plt.ylabel('Portfolio Value (₹)', fontsize=14)
        plt.xlabel('Date', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'₹{x:,.0f}'))
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/equity_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create individual drawdown chart
        plt.figure(figsize=(15, 6))
        plt.fill_between(equity_df['datetime'], drawdown, 0, alpha=0.8, color='#F18F01')
        plt.plot(equity_df['datetime'], drawdown, linewidth=2, color='#C73E1D')
        plt.title('Portfolio Drawdown', fontsize=18, fontweight='bold')
        plt.ylabel('Drawdown (%)', fontsize=14)
        plt.xlabel('Date', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/drawdown.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Charts saved to {output_dir}/:")
        print(f"   - backtest_dashboard.png (comprehensive overview)")
        print(f"   - equity_curve.png")
        print(f"   - drawdown.png")

def run_full_backtest():
    """
    Run the complete backtesting system.
    """
    print("=" * 60)
    print("OPTIONS STRATEGY BACKTEST SYSTEM")
    print("=" * 60)
    
    # Load spot data
    print("Loading spot data...")
    spot_df = pd.read_csv('data/spot_with_signals_2023.csv')
    print(f"Loaded {len(spot_df)} rows of spot data")
    
    # Initialize backtester
    backtester = OptionsBacktester(
        initial_capital=200000,
        stop_loss_pct=0.015,
        take_profit_pct=0.03,
        exit_time='15:15'
    )
    
    # Run backtest with ML enhancement enabled
    results = backtester.run_backtest(spot_df, max_signals=50, sample_pct=0.05, skip_ml=False)  # Enable ML training
    
    # Print summary
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS SUMMARY")
    print("=" * 60)
    
    if 'error' in results:
        print(f"Error: {results['error']}")
        return
    
    print(f"Initial Capital: ₹{results['initial_capital']:,.2f}")
    print(f"Final Capital: ₹{results['final_capital']:,.2f}")
    print(f"Total Return: {results['total_return_pct']:.2f}%")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.3f}")
    print(f"Max Drawdown: {results['max_drawdown_pct']:.2f}%")
    print(f"Total Trades: {results['total_trades']}")
    print(f"Win Rate: {results['win_rate']:.1%}")
    print(f"Profit Factor: {results['profit_factor']:.2f}")
    print(f"Average Win: ₹{results['avg_win']:.2f}")
    print(f"Average Loss: ₹{results['avg_loss']:.2f}")
    
    # Save results
    backtester.save_results(results)
    
    print("\nBacktest completed successfully!")
    print("Files generated:")
    print("- trades.csv")
    print("- metrics.csv")
    print("- equity_curve.png")
    print("- drawdown.png")
    
    return results

if __name__ == "__main__":
    run_full_backtest()