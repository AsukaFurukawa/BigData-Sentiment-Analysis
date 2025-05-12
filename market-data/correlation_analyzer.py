import os
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.api import VAR
import logging
from scipy.stats import pearsonr, spearmanr
import pickle

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("correlation_analysis.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("correlation_analyzer")

class SentimentMarketCorrelation:
    """Analyzer for correlations between sentiment and market data"""
    
    def __init__(self):
        """Initialize the correlation analyzer"""
        # Paths
        self.base_path = os.path.dirname(__file__)
        self.project_path = os.path.dirname(self.base_path)
        self.market_data_path = os.path.join(self.base_path, 'data', 'market_data.db')
        self.sentiment_data_path = os.path.join(self.project_path, 'data-storage', 'data', 'sentiment_data.db')
        self.results_path = os.path.join(self.base_path, 'analysis_results')
        
        # Create results directory if it doesn't exist
        os.makedirs(self.results_path, exist_ok=True)
        
        # Load configuration
        self.load_config()
        
        # Check if databases exist
        self.market_db_exists = os.path.isfile(self.market_data_path)
        self.sentiment_db_exists = os.path.isfile(self.sentiment_data_path)
        
        if not self.market_db_exists:
            logger.warning(f"Market database not found at {self.market_data_path}")
        
        if not self.sentiment_db_exists:
            logger.warning(f"Sentiment database not found at {self.sentiment_data_path}")
    
    def load_config(self):
        """Load configuration from the config file"""
        try:
            config_path = os.path.join(self.project_path, 'data-ingestion', 'config', 'keys.json')
            
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # Get market data config
            market_config = config.get('market_data', {})
            self.stocks = market_config.get('stocks', [])
            self.indices = market_config.get('indices', [])
            self.crypto = market_config.get('crypto', [])
            self.all_symbols = self.stocks + self.indices + self.crypto
            
            # Analysis settings
            analysis_config = config.get('analysis', {
                'default_lookback_days': 30,
                'correlation_methods': ['pearson', 'spearman'],
                'granger_max_lag': 5,
                'min_data_points': 20
            })
            
            self.lookback_days = analysis_config.get('default_lookback_days', 30)
            self.correlation_methods = analysis_config.get('correlation_methods', ['pearson', 'spearman'])
            self.granger_max_lag = analysis_config.get('granger_max_lag', 5)
            self.min_data_points = analysis_config.get('min_data_points', 20)
            
            logger.info("Configuration loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            # Default values
            self.stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
            self.indices = ["^GSPC", "^DJI", "^IXIC"]
            self.crypto = ["BTC-USD", "ETH-USD"]
            self.all_symbols = self.stocks + self.indices + self.crypto
            self.lookback_days = 30
            self.correlation_methods = ['pearson', 'spearman']
            self.granger_max_lag = 5
            self.min_data_points = 20
    
    def load_market_data(self, symbols=None, days=None):
        """Load market data from the database"""
        if not self.market_db_exists:
            logger.error("Market database not found")
            return pd.DataFrame()
        
        try:
            # Connect to the database
            conn = sqlite3.connect(self.market_data_path)
            
            # Set parameters
            days = days if days is not None else self.lookback_days
            symbols = symbols if symbols is not None else self.all_symbols
            
            # Create the query
            cutoff_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            symbols_str = "', '".join(symbols)
            
            query = f"""
            SELECT symbol, timestamp, close
            FROM market_data
            WHERE symbol IN ('{symbols_str}')
            AND timestamp > '{cutoff_date}'
            ORDER BY symbol, timestamp
            """
            
            # Load data
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            if df.empty:
                logger.warning("No market data found for the specified parameters")
                return pd.DataFrame()
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Pivot the dataframe to have symbols as columns
            pivot_df = df.pivot(index='timestamp', columns='symbol', values='close')
            
            logger.info(f"Loaded market data with shape {pivot_df.shape}")
            return pivot_df
            
        except Exception as e:
            logger.error(f"Error loading market data: {e}")
            return pd.DataFrame()
    
    def load_sentiment_data(self, domains=None, days=None):
        """Load sentiment data from the database"""
        if not self.sentiment_db_exists:
            logger.error("Sentiment database not found")
            return pd.DataFrame()
        
        try:
            # Connect to the database
            conn = sqlite3.connect(self.sentiment_data_path)
            
            # Set parameters
            days = days if days is not None else self.lookback_days
            domains = domains if domains is not None else ['finance', 'technology', 'finance-tech', 'tech-finance']
            
            # Create the query
            cutoff_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            domains_str = "', '".join(domains)
            
            query = f"""
            SELECT created_at, domain, overall_sentiment
            FROM sentiment_results
            WHERE domain IN ('{domains_str}')
            AND created_at > '{cutoff_date}'
            ORDER BY created_at
            """
            
            # Load data
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            if df.empty:
                logger.warning("No sentiment data found for the specified parameters")
                return pd.DataFrame()
            
            # Convert timestamp to datetime
            df['created_at'] = pd.to_datetime(df['created_at'])
            
            # Convert sentiment to numeric
            sentiment_map = {'positive': 1.0, 'neutral': 0.0, 'negative': -1.0}
            df['sentiment_score'] = df['overall_sentiment'].map(sentiment_map)
            
            # Resample to hourly data with mean sentiment
            df.set_index('created_at', inplace=True)
            
            # Create separate columns for each domain
            sentiment_data = pd.DataFrame(index=pd.date_range(
                start=df.index.min(),
                end=df.index.max(),
                freq='H'
            ))
            
            for domain in domains:
                domain_df = df[df['domain'] == domain]
                if not domain_df.empty:
                    # Resample to hourly with mean sentiment
                    domain_hourly = domain_df['sentiment_score'].resample('H').mean()
                    sentiment_data[f'{domain}_sentiment'] = domain_hourly
            
            # Forward fill missing values (carry last sentiment forward)
            sentiment_data.fillna(method='ffill', inplace=True)
            
            # Add a combined finance sentiment column
            finance_cols = [col for col in sentiment_data.columns if 'finance' in col]
            if finance_cols:
                sentiment_data['finance_combined'] = sentiment_data[finance_cols].mean(axis=1)
            
            # Add a combined tech sentiment column
            tech_cols = [col for col in sentiment_data.columns if 'tech' in col]
            if tech_cols:
                sentiment_data['tech_combined'] = sentiment_data[tech_cols].mean(axis=1)
            
            # Add overall sentiment
            sentiment_data['overall'] = sentiment_data.mean(axis=1)
            
            logger.info(f"Loaded sentiment data with shape {sentiment_data.shape}")
            return sentiment_data
            
        except Exception as e:
            logger.error(f"Error loading sentiment data: {e}")
            return pd.DataFrame()
    
    def align_data(self, market_data, sentiment_data):
        """Align market and sentiment data to the same timestamps"""
        try:
            if market_data.empty or sentiment_data.empty:
                logger.warning("Cannot align data: one or both datasets are empty")
                return pd.DataFrame()
            
            # Resample market data to hourly if needed
            if len(market_data.index) > len(market_data.index.unique()):
                # This means we have multiple entries per timestamp
                market_hourly = market_data.groupby(level=0).last()
            else:
                market_hourly = market_data
            
            # Ensure both have a datetime index
            market_hourly.index = pd.to_datetime(market_hourly.index)
            sentiment_data.index = pd.to_datetime(sentiment_data.index)
            
            # Find the common date range
            start_date = max(market_hourly.index.min(), sentiment_data.index.min())
            end_date = min(market_hourly.index.max(), sentiment_data.index.max())
            
            # Filter both dataframes to the common range
            market_filtered = market_hourly.loc[start_date:end_date]
            sentiment_filtered = sentiment_data.loc[start_date:end_date]
            
            # Combine the data
            combined = pd.concat([market_filtered, sentiment_filtered], axis=1)
            
            # Drop rows with any NaN values
            combined.dropna(inplace=True)
            
            logger.info(f"Aligned data with shape {combined.shape}")
            return combined
            
        except Exception as e:
            logger.error(f"Error aligning data: {e}")
            return pd.DataFrame()
    
    def calculate_correlations(self, combined_data):
        """Calculate correlations between sentiment and market data"""
        try:
            if combined_data.empty:
                logger.warning("Cannot calculate correlations: dataset is empty")
                return None
            
            results = {}
            
            # Separate market and sentiment columns
            market_cols = [col for col in combined_data.columns if col in self.all_symbols]
            sentiment_cols = [col for col in combined_data.columns if 'sentiment' in col or col in ['overall', 'finance_combined', 'tech_combined']]
            
            if not market_cols or not sentiment_cols:
                logger.warning("Cannot find market or sentiment columns in the combined data")
                return None
            
            # Calculate correlations using different methods
            for method in self.correlation_methods:
                corr_matrix = combined_data[market_cols + sentiment_cols].corr(method=method)
                
                # Extract the correlations between market and sentiment
                market_sentiment_corr = corr_matrix.loc[market_cols, sentiment_cols]
                
                results[method] = market_sentiment_corr
            
            # Save correlation plots
            for method, corr_df in results.items():
                plt.figure(figsize=(12, 10))
                sns.heatmap(corr_df, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
                plt.title(f"{method.capitalize()} Correlation between Market and Sentiment")
                plt.tight_layout()
                
                # Save the plot
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                plot_path = os.path.join(self.results_path, f'correlation_{method}_{timestamp}.png')
                plt.savefig(plot_path)
                plt.close()
                
                logger.info(f"Saved correlation plot to {plot_path}")
            
            # Save raw correlation data
            for method, corr_df in results.items():
                csv_path = os.path.join(self.results_path, f'correlation_{method}_{timestamp}.csv')
                corr_df.to_csv(csv_path)
                logger.info(f"Saved correlation data to {csv_path}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error calculating correlations: {e}")
            return None
    
    def calculate_granger_causality(self, combined_data):
        """Calculate Granger causality between sentiment and market movements"""
        try:
            if combined_data.empty or len(combined_data) < self.min_data_points:
                logger.warning(f"Not enough data points for Granger causality test (need {self.min_data_points})")
                return None
            
            results = {}
            
            # Get market and sentiment columns
            market_cols = [col for col in combined_data.columns if col in self.all_symbols]
            sentiment_cols = [col for col in combined_data.columns if 'sentiment' in col or col in ['overall', 'finance_combined', 'tech_combined']]
            
            # Convert to returns for market data (stationary series)
            market_returns = combined_data[market_cols].pct_change().dropna()
            
            # Merge with sentiment data
            analysis_data = pd.concat([market_returns, combined_data[sentiment_cols]], axis=1).dropna()
            
            # Run Granger causality tests
            for sentiment_col in sentiment_cols:
                sentiment_results = {}
                
                for market_col in market_cols:
                    # Test if sentiment causes market
                    try:
                        # Create test data
                        test_data = analysis_data[[sentiment_col, market_col]].dropna()
                        
                        if len(test_data) < self.min_data_points:
                            continue
                        
                        # Run test
                        granger_result = grangercausalitytests(
                            test_data, 
                            maxlag=self.granger_max_lag,
                            verbose=False
                        )
                        
                        # Extract p-values for each lag
                        p_values = {}
                        for lag, result in granger_result.items():
                            p_values[lag] = result[0]['ssr_chi2test'][1]
                        
                        sentiment_results[market_col] = p_values
                        
                    except Exception as inner_e:
                        logger.warning(f"Error in Granger test for {sentiment_col} -> {market_col}: {inner_e}")
                
                results[sentiment_col] = sentiment_results
            
            # Save results
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            results_path = os.path.join(self.results_path, f'granger_causality_{timestamp}.pkl')
            with open(results_path, 'wb') as f:
                pickle.dump(results, f)
            
            # Create summary of significant results
            summary = []
            for sentiment_col, market_results in results.items():
                for market_col, lag_results in market_results.items():
                    for lag, p_value in lag_results.items():
                        if p_value < 0.05:  # Significant at 5% level
                            summary.append({
                                'sentiment': sentiment_col,
                                'market': market_col,
                                'lag': lag,
                                'p_value': p_value,
                                'significant': 'Yes'
                            })
            
            if summary:
                summary_df = pd.DataFrame(summary)
                summary_path = os.path.join(self.results_path, f'granger_summary_{timestamp}.csv')
                summary_df.to_csv(summary_path, index=False)
                logger.info(f"Saved Granger causality summary to {summary_path}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error calculating Granger causality: {e}")
            return None
    
    def run_correlation_analysis(self, days=None, symbols=None, domains=None):
        """Run a complete correlation analysis between sentiment and market data"""
        try:
            logger.info("Starting correlation analysis")
            
            # Load data
            market_data = self.load_market_data(symbols, days)
            sentiment_data = self.load_sentiment_data(domains, days)
            
            if market_data.empty or sentiment_data.empty:
                logger.warning("Cannot run analysis: market or sentiment data is empty")
                return None
            
            # Align data
            combined_data = self.align_data(market_data, sentiment_data)
            
            if combined_data.empty:
                logger.warning("Cannot run analysis: combined data is empty")
                return None
            
            # Calculate correlations
            correlation_results = self.calculate_correlations(combined_data)
            
            # Calculate Granger causality
            granger_results = self.calculate_granger_causality(combined_data)
            
            # Return combined results
            results = {
                'correlations': correlation_results,
                'granger_causality': granger_results,
                'timestamp': datetime.now().isoformat(),
                'data_points': len(combined_data),
                'market_symbols': [col for col in combined_data.columns if col in self.all_symbols],
                'sentiment_types': [col for col in combined_data.columns if 'sentiment' in col or col in ['overall', 'finance_combined', 'tech_combined']]
            }
            
            # Save the complete results
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            results_path = os.path.join(self.results_path, f'analysis_results_{timestamp}.pkl')
            with open(results_path, 'wb') as f:
                pickle.dump(results, f)
            
            logger.info(f"Analysis complete. Results saved to {results_path}")
            return results
            
        except Exception as e:
            logger.error(f"Error running correlation analysis: {e}")
            return None

if __name__ == "__main__":
    analyzer = SentimentMarketCorrelation()
    
    # Run analysis for different time periods
    for days in [7, 30]:
        logger.info(f"Running analysis for past {days} days")
        results = analyzer.run_correlation_analysis(days=days)
        
        if results:
            logger.info(f"Analysis for {days} days complete with {results['data_points']} data points") 