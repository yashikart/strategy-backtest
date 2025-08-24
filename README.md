# Options Trading Strategy Backtest System

A comprehensive Python-based backtesting framework for options trading strategies, combining technical indicators, composite signal generation, and machine learning models for enhanced trading signal validation.

![Python](https://img.shields.io/badge/python-v3.10+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-green.svg)

## 🚀 Key Features

- **📊 Technical Analysis**: 8+ comprehensive technical indicators (MACD, RSI, SuperTrend, ADX, Bollinger Bands, EMA crossover, Stochastic, ATR)
- **🤖 ML Enhancement**: Multiple machine learning models (Logistic Regression, XGBoost, Random Forest, LSTM, Ensemble)
- **🎯 Options Strategy**: ATM PUT/CALL selling strategy with realistic options pricing
- **⚖️ Risk Management**: Stop-loss (1.5%), take-profit (3%), and time-based exits (15:15)
- **📈 Performance Analytics**: Comprehensive performance metrics and visualizations
- **🔄 Signal Combination**: Weighted voting system for robust signal generation

## 📁 Project Structure

```
strategy-backtest/
├── data/
│   ├── spot_with_signals_2023.csv      # Input spot data with basic signals
│   ├── options_data_2023.csv           # Options data for realistic pricing
│   └── spot_with_composite_signals_2023.csv  # Generated composite signals
├── results/
│   ├── equity_curve.png                # Portfolio performance chart
│   ├── drawdown.png                    # Drawdown analysis
│   ├── backtest_dashboard.png          # Comprehensive dashboard
│   ├── metrics.csv                     # Performance metrics
│   ├── trades.csv                      # Detailed trade log
│   └── model_performance_detailed.csv  # ML model comparisons
├── indicators.py                       # Technical indicators implementation
├── signal_engine.py                    # Composite signal generation
├── model.py                           # ML models and optimization
├── backtest.py                        # Main backtesting engine
├── utils.py                           # Utility functions and helpers
├── requirements.txt                   # Python dependencies
└── README.md                          # This documentation
```

## 🛠️ Environment Setup & Dependencies

### Prerequisites

- Python 3.10 or higher
- pip package manager

### Installation

1. **Clone or download the project:**
   ```bash
   git clone <repository-url>
   cd strategy-backtest
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation:**
   ```bash
   python -c "import pandas, numpy, sklearn, xgboost, matplotlib; print('All dependencies installed successfully!')"
   ```

### Dependencies Overview

```
pandas>=2.0.0          # Data manipulation and analysis
numpy>=1.24.0          # Numerical computing
ta>=0.11.0             # Technical analysis library
matplotlib>=3.10.0     # Plotting and visualization
seaborn>=0.13.0        # Statistical visualization
scikit-learn>=1.7.0    # Machine learning framework
xgboost>=3.0.0         # Gradient boosting framework
tqdm>=4.65.0           # Progress bars
pyarrow>=10.0.0        # Fast data serialization
tensorflow>=2.10.0     # Deep learning (optional for LSTM)
```

## 🔧 Usage Examples

### Quick Start

Run the complete backtesting system:

```bash
python backtest.py
```

### Step-by-Step Execution

1. **Generate Technical Indicators:**
   ```python
   from indicators import add_indicators
   import pandas as pd
   
   # Load your data
   df = pd.read_csv('data/spot_with_signals_2023.csv')
   
   # Add comprehensive technical indicators
   df_with_indicators = add_indicators(df)
   print(f"Added {len(df_with_indicators.columns) - len(df.columns)} indicators")
   ```

2. **Create Composite Signals:**
   ```python
   from signal_engine import SignalEngine
   
   # Initialize signal engine with custom weights
   engine = SignalEngine(
       inhouse_weight=0.30,    # 30% weight to original signal
       macd_weight=0.15,       # 15% weight to MACD
       supertrend_weight=0.15, # 15% weight to SuperTrend
       rsi_weight=0.12,        # 12% weight to RSI
       # ... other weights
   )
   
   # Generate composite signals
   df_signals = engine.generate_composite_signal(df_with_indicators)
   
   # Get signal summary
   summary = engine.get_signal_summary(df_signals)
   print("Signal Distribution:", summary)
   ```

3. **Train ML Models:**
   ```python
   from model import test_ml_models
   
   # Run comprehensive ML optimization
   ml_manager, model_summary = test_ml_models()
   
   # View model performance
   print(model_summary)
   
   # Get best model
   best_model_name, best_model = ml_manager.find_best_model('test_accuracy')
   print(f"Best model: {best_model_name}")
   ```

4. **Execute Backtest:**
   ```python
   from backtest import OptionsBacktester
   
   # Initialize backtester
   backtester = OptionsBacktester(
       initial_capital=200000,  # ₹2,00,000
       stop_loss_pct=0.015,     # 1.5% stop loss
       take_profit_pct=0.03,    # 3% take profit
       exit_time='15:15'        # Force exit time
   )
   
   # Run backtest
   results = backtester.run_backtest(
       spot_df, 
       max_signals=100,         # Limit signals for demo
       sample_pct=0.1,          # Use 10% of data
       skip_ml=False            # Include ML enhancement
   )
   
   # Save results
   backtester.save_results(results)
   ```

### Custom Configuration Examples

**1. Conservative Strategy (Lower Risk):**
```python
backtester = OptionsBacktester(
    initial_capital=200000,
    stop_loss_pct=0.01,      # Tighter stop loss (1%)
    take_profit_pct=0.02,    # Lower take profit (2%)
    exit_time='14:30'        # Earlier exit
)
```

**2. Aggressive Strategy (Higher Risk):**
```python
backtester = OptionsBacktester(
    initial_capital=500000,
    stop_loss_pct=0.025,     # Wider stop loss (2.5%)
    take_profit_pct=0.05,    # Higher take profit (5%)
    exit_time='15:15'        # Standard exit
)
```

## 📊 Technical Indicators

### Implemented Indicators

| Indicator | Description | Purpose |
|-----------|-------------|---------|
| **MACD** | Moving Average Convergence Divergence | Trend following momentum |
| **RSI** | Relative Strength Index | Overbought/oversold conditions |
| **SuperTrend** | Trend-following indicator | Strong trend identification |
| **ADX** | Average Directional Index | Trend strength measurement |
| **Bollinger Bands** | Volatility bands | Mean reversion signals |
| **EMA Crossover** | Exponential Moving Average | Trend direction changes |
| **Stochastic** | Momentum oscillator | Entry/exit timing |
| **ATR** | Average True Range | Volatility measurement |

### Signal Generation Logic

The system uses a **weighted voting mechanism** to combine signals:

```python
Composite Score = (
    30% × In-house Signal +
    15% × MACD Signal +
    15% × SuperTrend Signal +
    12% × RSI Signal +
    10% × EMA Crossover +
    10% × Bollinger Bands +
    8% × ADX Signal +
    5% × Stochastic Signal +
    5% × ATR Signal
)
```

**Signal Confidence**: Calculated based on agreement between indicators.

## 🤖 Machine Learning Models

### Available Models

1. **Logistic Regression**: Baseline linear classifier
2. **Random Forest**: Ensemble tree-based model
3. **XGBoost**: Gradient boosting with hyperparameter optimization
4. **LSTM**: Deep learning for time series (optional)
5. **Ensemble**: Voting classifier combining multiple models
6. **Clustering-Enhanced**: K-means clustering with classification

### Model Selection Process

```python
# Comprehensive model comparison
models_tested = [
    'logistic_baseline',
    'rf_baseline', 
    'xgb_baseline',
    'xgb_optimized',     # Hyperparameter tuned
    'ensemble_class',    # Voting ensemble
    'cluster_enhanced'   # With clustering features
]

# Evaluation metrics
metrics = ['accuracy', 'precision', 'recall', 'f1_score']

# Time series aware cross-validation
cv_strategy = TimeSeriesSplit(n_splits=5)
```

### Feature Engineering

- **Technical indicators** (14 features)
- **Price momentum** (multiple timeframes)
- **Volatility measures** (rolling std, ATR percentiles)
- **Volume analysis** (if available)
- **Cluster features** (market regime identification)

## 💼 Options Strategy

### Strategy Description

**Objective**: Generate income through option premium collection

**Core Strategy**:
- **Buy Signal** → Sell ATM PUT option (bullish/neutral outlook)
- **Sell Signal** → Sell ATM CALL option (bearish/neutral outlook)

### Trade Execution Logic

```python
def execute_strategy(signal, spot_price, timestamp):
    if signal == 'Buy':
        # Sell ATM PUT - profit if market goes up or stays flat
        option_type = 'PE'
        strike = round_to_atm(spot_price)
        expectation = 'bullish'
    else:  # signal == 'Sell'
        # Sell ATM CALL - profit if market goes down or stays flat
        option_type = 'CE' 
        strike = round_to_atm(spot_price)
        expectation = 'bearish'
    
    return execute_option_trade(option_type, strike, timestamp)
```

### Risk Management

| Parameter | Value | Description |
|-----------|--------|-------------|
| **Stop Loss** | 1.5% | Maximum loss per trade |
| **Take Profit** | 3% | Target profit per trade |
| **Force Exit** | 15:15 | Daily exit time |
| **Position Size** | 5% capital | Risk per trade |
| **Max Positions** | 5 lots | Position limit |

## 📈 Backtest Parameters

### Configuration

```python
BACKTEST_CONFIG = {
    'initial_capital': 200000,     # Starting capital (₹2,00,000)
    'stop_loss_pct': 0.015,        # 1.5% stop loss
    'take_profit_pct': 0.03,       # 3% take profit
    'exit_time': '15:15',          # Force exit time
    'risk_per_trade': 0.05,        # 5% of capital per trade
    'max_positions': 5,            # Maximum open positions
    'commission': 0,               # Ignore transaction costs
    'slippage': 0,                 # Ignore slippage
    'margin_interest': 0           # Ignore margin costs
}
```

### Performance Metrics

The system calculates comprehensive performance metrics:

#### Returns Metrics
- **Total Return %**: Overall portfolio performance
- **Annualized Return**: Risk-adjusted annual return
- **Sharpe Ratio**: Risk-adjusted return measure
- **Maximum Drawdown**: Worst peak-to-trough decline

#### Trade Statistics
- **Total Trades**: Number of completed trades
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profit / Gross loss
- **Average Win/Loss**: Mean profit and loss per trade

#### Risk Metrics
- **Volatility**: Standard deviation of returns
- **Value at Risk (VaR)**: Potential loss at confidence level
- **Maximum Consecutive Losses**: Longest losing streak

## 📊 Output Files & Interpretation

### Generated Files

1. **`trades.csv`** - Detailed trade log
   ```csv
   entry_time,exit_time,instrument,signal,strike,lots,entry_premium,exit_premium,pnl,return_pct
   2023-01-03 09:30:00,2023-01-03 15:15:00,NIFTY03JAN2023PEX,Buy,18000,2,150.5,89.2,6156,20.4
   ```

2. **`metrics.csv`** - Performance summary
   ```csv
   metric,value
   Total Return %,15.67
   Sharpe Ratio,1.23
   Max Drawdown %,-8.45
   Win Rate %,67.5
   ```

3. **`equity_curve.png`** - Portfolio value over time
4. **`drawdown.png`** - Drawdown analysis chart
5. **`backtest_dashboard.png`** - Comprehensive performance dashboard

### Interpreting Results

**📈 Good Performance Indicators:**
- Total Return > 10% annually
- Sharpe Ratio > 1.0
- Max Drawdown < 15%
- Win Rate > 60%
- Profit Factor > 1.5

**⚠️ Warning Signs:**
- Consecutive losses > 5
- Drawdown > 20%
- Sharpe Ratio < 0.5
- Win Rate < 50%

**🔍 Analysis Tips:**
1. **Equity Curve**: Look for smooth upward trend
2. **Drawdown**: Check recovery speed and frequency
3. **Trade Distribution**: Ensure consistent performance
4. **Monthly Returns**: Verify consistency across periods

## 🚨 Important Notes & Limitations

### Assumptions
- **No Transaction Costs**: Commission and fees ignored
- **Perfect Execution**: No slippage or partial fills
- **Simplified Options Pricing**: Using approximation models
- **No Margin Requirements**: Unlimited capital assumption
- **Market Hours**: Standard trading hours only

### Limitations
- **Historical Data Only**: Past performance doesn't guarantee future results
- **Simplified Risk Model**: Real options trading involves complex risks
- **No Market Impact**: Assumes trades don't affect market prices
- **Option Chain Limitations**: Limited to ATM options only

### Risk Disclaimers
⚠️ **This is for educational purposes only. Not financial advice.**
- Options trading involves significant risk
- Past performance doesn't indicate future results
- Always conduct due diligence before live trading
- Consider consulting financial professionals

## 🛠️ Troubleshooting

### Common Issues

**1. Import Errors:**
```bash
# Solution: Reinstall dependencies
pip install --upgrade -r requirements.txt
```

**2. Memory Issues with Large Data:**
```python
# Solution: Use sampling
results = backtester.run_backtest(
    spot_df, 
    max_signals=50,      # Limit signals
    sample_pct=0.05      # Use 5% of data
)
```

**3. No Trading Signals Generated:**
```python
# Check signal distribution
signal_counts = df['final_signal'].value_counts()
print("Signal distribution:", signal_counts)

# Verify data quality
print("Data shape:", df.shape)
print("Missing values:", df.isnull().sum())
```

**4. Poor Model Performance:**
```python
# Check feature importance
for model_name, importance in ml_manager.feature_importance.items():
    print(f"{model_name} top features:")
    # ... display top features
```

### Performance Optimization

**For Large Datasets:**
```python
# Use data sampling
sample_pct = 0.1  # Use 10% of data

# Limit signals processed
max_signals = 100

# Skip ML for faster testing
skip_ml = True
```

**For Production Use:**
```python
# Full dataset with ML
sample_pct = 1.0
max_signals = None
skip_ml = False
```

## 📞 Support & Contributing

### Getting Help
- Check documentation thoroughly
- Review error messages and logs
- Test with smaller datasets first
- Verify data format and quality

### Contributing
1. Fork the repository
2. Create feature branch
3. Test thoroughly
4. Submit pull request with documentation

### Future Enhancements
- Live data integration
- More sophisticated options pricing models
- Portfolio optimization
- Real-time signal generation
- Web-based dashboard

---

**📧 Contact**: For questions or support, please refer to the project documentation or create an issue in the repository.

**🔗 License**: MIT License - see LICENSE file for details.

**⭐ Star this repo** if you find it useful for your quantitative trading research!