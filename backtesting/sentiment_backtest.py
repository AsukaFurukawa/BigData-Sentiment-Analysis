import os
import sys
import json
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Any, Optional, Tuple, Callable

# Configure logging
base_dir = os.path.dirname(os.path.dirname(__file__))  # Get project root directory
logs_dir = os.path.join(base_dir, 'logs')
os.makedirs(logs_dir, exist_ok=True)  # Create logs directory if it doesn't exist

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(logs_dir, 'backtest.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("sentiment_backtest")

class SentimentBacktester:
    """Backtesting framework for sentiment-based trading strategies"""
    
    def __init__(self):
        """Initialize the sentiment backtesting framework"""
        # Paths
        self.base_path = os.path.dirname(__file__)
        self.project_path = os.path.dirname(self.base_path)
        self.sentiment_db_path = os.path.join(self.project_path, 'data-storage', 'data', 'sentiment_data.db')
        self.market_db_path = os.path.join(self.project_path, 'market-data', 'data', 'market_data.db')
        self.results_path = os.path.join(self.base_path, 'results')
        
        # Create results directory if it doesn't exist
        os.makedirs(self.results_path, exist_ok=True)
        
        # Default trading parameters
        self.initial_capital = 10000.0
        self.transaction_cost = 0.001  # 0.1% per trade
        self.slippage = 0.001  # 0.1% slippage
        
        # Results storage
        self.backtest_results = {}
    
    def load_market_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Load market data for a specific symbol and date range"""
        try:
            if not os.path.isfile(self.market_db_path):
                logger.error(f"Market database not found at {self.market_db_path}")
                return pd.DataFrame()
            
            # Connect to the database
            conn = sqlite3.connect(self.market_db_path)
            
            # Query data
            query = f"""
            SELECT timestamp, open, high, low, close, volume
            FROM market_data
            WHERE symbol = '{symbol}'
            AND timestamp BETWEEN '{start_date}' AND '{end_date}'
            ORDER BY timestamp
            """
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            if df.empty:
                logger.warning(f"No market data found for {symbol} between {start_date} and {end_date}")
                return pd.DataFrame()
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            logger.info(f"Loaded {len(df)} market data points for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading market data: {e}")
            return pd.DataFrame()
    
    def load_sentiment_data(self, domain: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Load sentiment data for a specific domain and date range"""
        try:
            if not os.path.isfile(self.sentiment_db_path):
                logger.error(f"Sentiment database not found at {self.sentiment_db_path}")
                return pd.DataFrame()
            
            # Connect to the database
            conn = sqlite3.connect(self.sentiment_db_path)
            
            # Query data
            query = f"""
            SELECT created_at, overall_sentiment, score
            FROM sentiment_results
            WHERE domain = '{domain}'
            AND created_at BETWEEN '{start_date}' AND '{end_date}'
            ORDER BY created_at
            """
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            if df.empty:
                logger.warning(f"No sentiment data found for {domain} between {start_date} and {end_date}")
                return pd.DataFrame()
            
            # Convert created_at to datetime
            df['created_at'] = pd.to_datetime(df['created_at'])
            
            # Convert sentiment to numeric
            sentiment_map = {'positive': 1.0, 'neutral': 0.0, 'negative': -1.0}
            df['sentiment_value'] = df['overall_sentiment'].map(sentiment_map)
            
            # Set index
            df.set_index('created_at', inplace=True)
            
            logger.info(f"Loaded {len(df)} sentiment data points for {domain}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading sentiment data: {e}")
            return pd.DataFrame()
    
    def align_data(self, market_df: pd.DataFrame, sentiment_df: pd.DataFrame, freq: str = 'D') -> pd.DataFrame:
        """Align market and sentiment data to the same timestamps"""
        try:
            if market_df.empty or sentiment_df.empty:
                logger.warning("Cannot align data: one or both datasets are empty")
                return pd.DataFrame()
            
            # Resample sentiment data to the specified frequency
            sentiment_resampled = sentiment_df['sentiment_value'].resample(freq).mean()
            
            # Resample market data to the same frequency
            market_resampled = market_df.resample(freq).last()
            
            # Calculate returns
            market_resampled['returns'] = market_resampled['close'].pct_change()
            
            # Merge sentiment and market data
            merged = pd.merge(
                market_resampled,
                sentiment_resampled.to_frame('sentiment'),
                left_index=True,
                right_index=True,
                how='inner'
            )
            
            # Fill any missing values
            merged.fillna(method='ffill', inplace=True)
            merged.dropna(inplace=True)
            
            logger.info(f"Aligned data with {len(merged)} points")
            return merged
            
        except Exception as e:
            logger.error(f"Error aligning data: {e}")
            return pd.DataFrame()
    
    def simple_threshold_strategy(self, data: pd.DataFrame, sentiment_threshold: float = 0.2) -> pd.DataFrame:
        """Simple threshold-based trading strategy"""
        try:
            if data.empty:
                logger.warning("Cannot run strategy: data is empty")
                return pd.DataFrame()
            
            # Create a copy of the data
            result = data.copy()
            
            # Generate signals based on sentiment threshold
            result['signal'] = 0
            result.loc[result['sentiment'] > sentiment_threshold, 'signal'] = 1  # Buy signal
            result.loc[result['sentiment'] < -sentiment_threshold, 'signal'] = -1  # Sell signal
            
            # Generate positions (1 = long, 0 = neutral, -1 = short)
            result['position'] = result['signal'].shift(1).fillna(0)
            
            # Calculate strategy returns
            result['strategy_returns'] = result['position'] * result['returns']
            
            # Calculate cumulative returns
            result['cumulative_market_returns'] = (1 + result['returns']).cumprod() - 1
            result['cumulative_strategy_returns'] = (1 + result['strategy_returns']).cumprod() - 1
            
            logger.info(f"Completed simple threshold strategy with threshold {sentiment_threshold}")
            return result
            
        except Exception as e:
            logger.error(f"Error running simple threshold strategy: {e}")
            return pd.DataFrame()
    
    def moving_average_strategy(self, data: pd.DataFrame, ma_window: int = 5, sentiment_weight: float = 0.5) -> pd.DataFrame:
        """Strategy based on sentiment moving average crossover"""
        try:
            if data.empty:
                logger.warning("Cannot run strategy: data is empty")
                return pd.DataFrame()
            
            # Create a copy of the data
            result = data.copy()
            
            # Calculate sentiment moving average
            result['sentiment_ma'] = result['sentiment'].rolling(window=ma_window).mean()
            
            # Generate signals
            result['signal'] = 0
            result.loc[result['sentiment'] > result['sentiment_ma'], 'signal'] = 1  # Buy signal
            result.loc[result['sentiment'] < result['sentiment_ma'], 'signal'] = -1  # Sell signal
            
            # Apply sentiment weight
            result['weighted_signal'] = result['signal'] * sentiment_weight
            
            # Generate positions
            result['position'] = result['weighted_signal'].shift(1).fillna(0)
            
            # Calculate strategy returns
            result['strategy_returns'] = result['position'] * result['returns']
            
            # Calculate cumulative returns
            result['cumulative_market_returns'] = (1 + result['returns']).cumprod() - 1
            result['cumulative_strategy_returns'] = (1 + result['strategy_returns']).cumprod() - 1
            
            logger.info(f"Completed moving average strategy with window {ma_window} and weight {sentiment_weight}")
            return result
            
        except Exception as e:
            logger.error(f"Error running moving average strategy: {e}")
            return pd.DataFrame()
    
    def calculate_performance_metrics(self, result: pd.DataFrame) -> Dict[str, float]:
        """Calculate performance metrics for a strategy"""
        try:
            if result.empty or 'strategy_returns' not in result.columns:
                logger.warning("Cannot calculate metrics: invalid strategy results")
                return {}
            
            # Calculate metrics
            total_days = len(result)
            trading_days_per_year = 252
            
            # Returns
            total_return = result['cumulative_strategy_returns'].iloc[-1]
            market_return = result['cumulative_market_returns'].iloc[-1]
            
            # Annualized returns
            years = total_days / trading_days_per_year
            annual_return = (1 + total_return) ** (1 / years) - 1
            market_annual_return = (1 + market_return) ** (1 / years) - 1
            
            # Risk metrics
            daily_returns = result['strategy_returns']
            volatility = daily_returns.std() * np.sqrt(trading_days_per_year)
            market_volatility = result['returns'].std() * np.sqrt(trading_days_per_year)
            
            # Sharpe ratio (assuming risk-free rate of 0)
            sharpe_ratio = annual_return / volatility if volatility != 0 else 0
            market_sharpe = market_annual_return / market_volatility if market_volatility != 0 else 0
            
            # Maximum drawdown
            cumulative_returns = result['cumulative_strategy_returns']
            running_max = cumulative_returns.cummax()
            drawdown = (cumulative_returns - running_max) / (1 + running_max)
            max_drawdown = drawdown.min()
            
            # Win rate
            trades = result['position'].diff().fillna(0) != 0
            trade_returns = result.loc[trades, 'strategy_returns']
            win_rate = (trade_returns > 0).sum() / len(trade_returns) if len(trade_returns) > 0 else 0
            
            # Compile metrics
            metrics = {
                'total_return': total_return,
                'annual_return': annual_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'market_return': market_return,
                'market_annual_return': market_annual_return,
                'market_volatility': market_volatility,
                'market_sharpe': market_sharpe,
                'alpha': annual_return - market_annual_return,
                'beta': np.cov(daily_returns, result['returns'])[0, 1] / np.var(result['returns']) if np.var(result['returns']) != 0 else 1,
                'total_trades': len(trade_returns)
            }
            
            logger.info(f"Calculated performance metrics: Sharpe={sharpe_ratio:.2f}, Return={total_return:.2%}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {}
    
    def visualize_results(self, result: pd.DataFrame, title: str, filename: Optional[str] = None):
        """Visualize strategy results"""
        try:
            if result.empty:
                logger.warning("Cannot visualize: results are empty")
                return
            
            # Create figure
            fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 15), gridspec_kw={'height_ratios': [3, 1, 1]})
            
            # Plot cumulative returns
            result[['cumulative_market_returns', 'cumulative_strategy_returns']].plot(
                ax=axes[0],
                title=f'{title} - Cumulative Returns',
                colormap='viridis'
            )
            axes[0].set_ylabel('Returns')
            axes[0].legend(['Market', 'Strategy'])
            axes[0].grid(True)
            
            # Plot sentiment
            result['sentiment'].plot(
                ax=axes[1],
                title='Sentiment Value',
                color='purple'
            )
            axes[1].set_ylabel('Sentiment')
            axes[1].grid(True)
            
            # Plot positions
            result['position'].plot(
                ax=axes[2],
                title='Strategy Positions',
                color='orange'
            )
            axes[2].set_ylabel('Position')
            axes[2].set_yticks([-1, 0, 1])
            axes[2].set_yticklabels(['Short', 'Neutral', 'Long'])
            axes[2].grid(True)
            
            # Add metrics as text
            metrics = self.calculate_performance_metrics(result)
            if metrics:
                metrics_text = (
                    f"Total Return: {metrics['total_return']:.2%}\n"
                    f"Annual Return: {metrics['annual_return']:.2%}\n"
                    f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}\n"
                    f"Max Drawdown: {metrics['max_drawdown']:.2%}\n"
                    f"Win Rate: {metrics['win_rate']:.2%}\n"
                    f"Alpha: {metrics['alpha']:.2%}\n"
                    f"Beta: {metrics['beta']:.2f}\n"
                    f"Total Trades: {metrics['total_trades']}"
                )
                axes[0].annotate(
                    metrics_text,
                    xy=(0.02, 0.05),
                    xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8)
                )
            
            plt.tight_layout()
            
            # Save or display
            if filename:
                filepath = os.path.join(self.results_path, filename)
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                plt.close()
                logger.info(f"Visualization saved to {filepath}")
            else:
                plt.show()
            
        except Exception as e:
            logger.error(f"Error visualizing results: {e}")
    
    def run_backtest(self, 
                     symbol: str, 
                     domain: str, 
                     start_date: str, 
                     end_date: str,
                     strategy: str = 'threshold',
                     strategy_params: Dict[str, Any] = None,
                     freq: str = 'D') -> Dict[str, Any]:
        """Run a complete backtest of a sentiment-based strategy"""
        try:
            logger.info(f"Starting backtest for {symbol} using {domain} sentiment from {start_date} to {end_date}")
            
            # Load data
            market_data = self.load_market_data(symbol, start_date, end_date)
            sentiment_data = self.load_sentiment_data(domain, start_date, end_date)
            
            if market_data.empty or sentiment_data.empty:
                logger.error("Cannot run backtest: missing data")
                return {}
            
            # Align data
            aligned_data = self.align_data(market_data, sentiment_data, freq)
            
            if aligned_data.empty:
                logger.error("Cannot run backtest: failed to align data")
                return {}
            
            # Set default strategy parameters
            if strategy_params is None:
                strategy_params = {}
            
            # Run strategy
            if strategy == 'threshold':
                sentiment_threshold = strategy_params.get('sentiment_threshold', 0.2)
                result = self.simple_threshold_strategy(aligned_data, sentiment_threshold)
            elif strategy == 'moving_average':
                ma_window = strategy_params.get('ma_window', 5)
                sentiment_weight = strategy_params.get('sentiment_weight', 0.5)
                result = self.moving_average_strategy(aligned_data, ma_window, sentiment_weight)
            else:
                logger.error(f"Unknown strategy: {strategy}")
                return {}
            
            # Calculate performance metrics
            metrics = self.calculate_performance_metrics(result)
            
            # Visualize results
            title = f"{symbol} {strategy.title()} Strategy"
            timestamp = datetime.now().strftime('%Y%m%d_%H%M')
            filename = f"{symbol}_{strategy}_{timestamp}.png"
            self.visualize_results(result, title, filename)
            
            # Save results
            backtest_id = f"{symbol}_{strategy}_{timestamp}"
            self.backtest_results[backtest_id] = {
                'symbol': symbol,
                'domain': domain,
                'strategy': strategy,
                'strategy_params': strategy_params,
                'start_date': start_date,
                'end_date': end_date,
                'metrics': metrics,
                'aligned_data_points': len(aligned_data),
                'timestamp': timestamp
            }
            
            # Save to JSON
            results_file = os.path.join(self.results_path, f"backtest_{backtest_id}.json")
            with open(results_file, 'w', encoding='utf-8') as f:
                backtest_output = self.backtest_results[backtest_id].copy()
                # Convert any numpy/pandas types to Python native types
                for key, value in backtest_output['metrics'].items():
                    if isinstance(value, (np.integer, np.floating)):
                        backtest_output['metrics'][key] = float(value)
                json.dump(backtest_output, f, indent=2)
            
            logger.info(f"Backtest complete. Results saved to {results_file}")
            return self.backtest_results[backtest_id]
            
        except Exception as e:
            logger.error(f"Error running backtest: {e}")
            return {}
    
    def run_optimization(self, 
                         symbol: str, 
                         domain: str, 
                         start_date: str, 
                         end_date: str,
                         strategy: str = 'threshold',
                         param_grid: Dict[str, List[Any]] = None) -> Dict[str, Any]:
        """Run a parameter optimization for a strategy"""
        try:
            logger.info(f"Starting parameter optimization for {strategy} strategy")
            
            # Set default parameter grid
            if param_grid is None:
                if strategy == 'threshold':
                    param_grid = {
                        'sentiment_threshold': [0.1, 0.2, 0.3, 0.4, 0.5]
                    }
                elif strategy == 'moving_average':
                    param_grid = {
                        'ma_window': [3, 5, 10, 15, 20],
                        'sentiment_weight': [0.3, 0.5, 0.7, 1.0]
                    }
                else:
                    logger.error(f"Unknown strategy: {strategy}")
                    return {}
            
            # Load and align data (do this once to save time)
            market_data = self.load_market_data(symbol, start_date, end_date)
            sentiment_data = self.load_sentiment_data(domain, start_date, end_date)
            
            if market_data.empty or sentiment_data.empty:
                logger.error("Cannot run optimization: missing data")
                return {}
            
            # Align data
            aligned_data = self.align_data(market_data, sentiment_data)
            
            if aligned_data.empty:
                logger.error("Cannot run optimization: failed to align data")
                return {}
            
            # Generate all parameter combinations
            param_combinations = self._generate_param_combinations(param_grid)
            
            # Track best parameters and results
            best_sharpe = -float('inf')
            best_params = None
            best_metrics = None
            all_results = []
            
            # Test each parameter combination
            for params in param_combinations:
                # Run strategy with these parameters
                if strategy == 'threshold':
                    result = self.simple_threshold_strategy(aligned_data, params['sentiment_threshold'])
                elif strategy == 'moving_average':
                    result = self.moving_average_strategy(aligned_data, params['ma_window'], params['sentiment_weight'])
                
                # Calculate metrics
                metrics = self.calculate_performance_metrics(result)
                
                # Track result
                result_entry = {
                    'params': params,
                    'metrics': metrics
                }
                all_results.append(result_entry)
                
                # Check if this is the best so far
                if metrics['sharpe_ratio'] > best_sharpe:
                    best_sharpe = metrics['sharpe_ratio']
                    best_params = params
                    best_metrics = metrics
                    
                    logger.info(f"New best parameters: {best_params} with Sharpe {best_sharpe:.2f}")
            
            # Run backtest with the best parameters
            if best_params is not None:
                logger.info(f"Running final backtest with best parameters: {best_params}")
                final_result = self.run_backtest(
                    symbol=symbol,
                    domain=domain,
                    start_date=start_date,
                    end_date=end_date,
                    strategy=strategy,
                    strategy_params=best_params
                )
                
                # Save optimization results
                timestamp = datetime.now().strftime('%Y%m%d_%H%M')
                results_file = os.path.join(self.results_path, f"optimization_{symbol}_{strategy}_{timestamp}.json")
                
                with open(results_file, 'w', encoding='utf-8') as f:
                    optimization_results = {
                        'symbol': symbol,
                        'domain': domain,
                        'strategy': strategy,
                        'start_date': start_date,
                        'end_date': end_date,
                        'best_params': best_params,
                        'best_metrics': {k: float(v) if isinstance(v, (np.integer, np.floating)) else v for k, v in best_metrics.items()},
                        'all_results': [{
                            'params': r['params'],
                            'metrics': {k: float(v) if isinstance(v, (np.integer, np.floating)) else v for k, v in r['metrics'].items()}
                        } for r in all_results],
                        'timestamp': timestamp
                    }
                    json.dump(optimization_results, f, indent=2)
                
                logger.info(f"Optimization complete. Results saved to {results_file}")
                return optimization_results
            
            logger.warning("Optimization did not find any valid parameters")
            return {}
            
        except Exception as e:
            logger.error(f"Error running optimization: {e}")
            return {}
    
    def _generate_param_combinations(self, param_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """Generate all possible combinations of parameters from a grid"""
        if not param_grid:
            return [{}]
        
        # Get all parameter names and possible values
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        # Generate combinations recursively
        def generate_recursive(param_idx: int, current_params: Dict[str, Any]):
            if param_idx >= len(param_names):
                return [current_params.copy()]
            
            name = param_names[param_idx]
            values = param_values[param_idx]
            
            combinations = []
            for value in values:
                current_params[name] = value
                combinations.extend(generate_recursive(param_idx + 1, current_params))
            
            return combinations
        
        return generate_recursive(0, {})

# Example usage
if __name__ == "__main__":
    backtester = SentimentBacktester()
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Sentiment-based Trading Strategy Backtester')
    parser.add_argument('--symbol', type=str, default='AAPL', help='Symbol to backtest (e.g., AAPL, MSFT)')
    parser.add_argument('--domain', type=str, default='finance', help='Sentiment domain (e.g., finance, technology)')
    parser.add_argument('--start-date', type=str, default='2022-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default='2022-12-31', help='End date (YYYY-MM-DD)')
    parser.add_argument('--strategy', type=str, default='threshold', choices=['threshold', 'moving_average'],
                        help='Trading strategy to use')
    parser.add_argument('--optimize', action='store_true', help='Run parameter optimization')
    
    args = parser.parse_args()
    
    if args.optimize:
        # Run parameter optimization
        backtester.run_optimization(
            symbol=args.symbol,
            domain=args.domain,
            start_date=args.start_date,
            end_date=args.end_date,
            strategy=args.strategy
        )
    else:
        # Run backtest with default parameters
        backtester.run_backtest(
            symbol=args.symbol,
            domain=args.domain,
            start_date=args.start_date,
            end_date=args.end_date,
            strategy=args.strategy
        ) 