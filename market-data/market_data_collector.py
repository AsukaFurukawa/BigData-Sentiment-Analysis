import os
import json
import time
import pandas as pd
import numpy as np
import yfinance as yf
from alpha_vantage.timeseries import TimeSeries
from datetime import datetime, timedelta
import sqlite3
import threading
import logging

# Set up logging
base_dir = os.path.dirname(os.path.dirname(__file__))  # Get project root directory
logs_dir = os.path.join(base_dir, 'logs')
os.makedirs(logs_dir, exist_ok=True)  # Create logs directory if it doesn't exist

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(logs_dir, 'market_data.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("market_data")

# Path to current script
base_path = os.path.dirname(__file__)
config_path = os.path.join(os.path.dirname(base_path), 'data-ingestion', 'config', 'keys.json')
data_dir = os.path.join(base_path, 'data')

# Create data directory if it doesn't exist
os.makedirs(data_dir, exist_ok=True)

try:
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
        # Add market data configuration if not exists
        if 'market_data' not in config:
            config['market_data'] = {
                "alpha_vantage_key": "YOUR_ALPHA_VANTAGE_API_KEY",
                "update_interval_minutes": 60,
                "stocks": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "AMD", "INTC", "IBM"],
                "indices": ["^GSPC", "^DJI", "^IXIC", "^RUT"],
                "crypto": ["BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD"],
                "forex": ["EURUSD=X", "GBPUSD=X", "USDJPY=X"]
            }
            with open(config_path, 'w', encoding='utf-8') as f_write:
                json.dump(config, f_write, indent=2)
except Exception as e:
    logger.error(f"Error loading config: {e}")
    # Create default config if one doesn't exist
    config = {
        "market_data": {
            "alpha_vantage_key": "YOUR_ALPHA_VANTAGE_API_KEY",
            "update_interval_minutes": 60,
            "stocks": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "AMD", "INTC", "IBM"],
            "indices": ["^GSPC", "^DJI", "^IXIC", "^RUT"],
            "crypto": ["BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD"],
            "forex": ["EURUSD=X", "GBPUSD=X", "USDJPY=X"]
        }
    }

# Access market data configuration
market_config = config.get('market_data', {})
ALPHA_VANTAGE_KEY = market_config.get('alpha_vantage_key')
UPDATE_INTERVAL = market_config.get('update_interval_minutes', 60)

# Define assets to track
STOCKS = market_config.get('stocks', [])
INDICES = market_config.get('indices', [])
CRYPTO = market_config.get('crypto', [])
FOREX = market_config.get('forex', [])

# All symbols
ALL_SYMBOLS = STOCKS + INDICES + CRYPTO + FOREX

# SQLite DB setup
db_path = os.path.join(data_dir, 'market_data.db')
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

def setup_database():
    """Create SQLite tables for market data if they don't exist"""
    # Market data table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS market_data (
        symbol TEXT,
        timestamp TEXT,
        open REAL,
        high REAL,
        low REAL,
        close REAL,
        volume REAL,
        data_type TEXT,
        source TEXT,
        PRIMARY KEY (symbol, timestamp)
    )
    ''')
    
    # Market metadata table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS market_metadata (
        symbol TEXT PRIMARY KEY,
        name TEXT,
        sector TEXT,
        industry TEXT,
        asset_type TEXT,
        last_updated TEXT
    )
    ''')
    
    conn.commit()
    logger.info("Database tables created")

def fetch_yfinance_data(symbols, period="1d", interval="1m"):
    """Fetch data from Yahoo Finance for multiple symbols"""
    try:
        logger.info(f"Fetching data for {len(symbols)} symbols from Yahoo Finance")
        data = yf.download(
            tickers=symbols,
            period=period,
            interval=interval,
            group_by="ticker",
            auto_adjust=True,
            prepost=False,
            threads=True
        )
        
        # Process and store the data
        for symbol in symbols:
            try:
                # Get symbol-specific data
                if len(symbols) > 1:
                    if symbol not in data.columns.levels[0]:
                        logger.warning(f"No data found for {symbol}")
                        continue
                    symbol_data = data[symbol].copy()
                else:
                    symbol_data = data.copy()
                
                # Reset index to make datetime a column
                symbol_data = symbol_data.reset_index()
                
                # Determine asset type
                if symbol in STOCKS:
                    asset_type = "stock"
                elif symbol in INDICES:
                    asset_type = "index"
                elif symbol in CRYPTO:
                    asset_type = "crypto"
                elif symbol in FOREX:
                    asset_type = "forex"
                else:
                    asset_type = "other"
                
                # Store in database
                for _, row in symbol_data.iterrows():
                    cursor.execute('''
                    INSERT OR REPLACE INTO market_data
                    (symbol, timestamp, open, high, low, close, volume, data_type, source)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        symbol,
                        row['Datetime'].strftime('%Y-%m-%d %H:%M:%S'),
                        row.get('Open', 0),
                        row.get('High', 0),
                        row.get('Low', 0),
                        row.get('Close', 0),
                        row.get('Volume', 0),
                        interval,
                        'yfinance'
                    ))
                
                conn.commit()
                
                # Update metadata
                ticker = yf.Ticker(symbol)
                info = ticker.info
                
                cursor.execute('''
                INSERT OR REPLACE INTO market_metadata
                (symbol, name, sector, industry, asset_type, last_updated)
                VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    symbol,
                    info.get('shortName', symbol),
                    info.get('sector', ''),
                    info.get('industry', ''),
                    asset_type,
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                ))
                
                conn.commit()
                logger.info(f"Stored data for {symbol}")
                
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                conn.rollback()
        
        return True
        
    except Exception as e:
        logger.error(f"Error fetching data from Yahoo Finance: {e}")
        return False

def fetch_alpha_vantage_data(symbols):
    """Fetch data from Alpha Vantage for specified symbols"""
    if not ALPHA_VANTAGE_KEY or ALPHA_VANTAGE_KEY == "YOUR_ALPHA_VANTAGE_API_KEY":
        logger.warning("Alpha Vantage API key not configured. Skipping.")
        return False
    
    try:
        logger.info(f"Fetching data for {len(symbols)} symbols from Alpha Vantage")
        ts = TimeSeries(key=ALPHA_VANTAGE_KEY, output_format='pandas')
        
        for symbol in symbols[:5]:  # Limit to 5 due to API constraints
            try:
                # Get intraday data
                data, meta_data = ts.get_intraday(symbol=symbol, interval='1min', outputsize='compact')
                
                # Reset index to make datetime a column
                data = data.reset_index()
                
                # Determine asset type
                if symbol in STOCKS:
                    asset_type = "stock"
                else:
                    asset_type = "other"
                
                # Store in database
                for _, row in data.iterrows():
                    cursor.execute('''
                    INSERT OR REPLACE INTO market_data
                    (symbol, timestamp, open, high, low, close, volume, data_type, source)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        symbol,
                        row['date'].strftime('%Y-%m-%d %H:%M:%S'),
                        row['1. open'],
                        row['2. high'],
                        row['3. low'],
                        row['4. close'],
                        row['5. volume'],
                        '1min',
                        'alpha_vantage'
                    ))
                
                # Update metadata
                cursor.execute('''
                INSERT OR REPLACE INTO market_metadata
                (symbol, name, sector, industry, asset_type, last_updated)
                VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    symbol,
                    meta_data.get('2. Symbol', symbol),
                    '',
                    '',
                    asset_type,
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                ))
                
                conn.commit()
                logger.info(f"Stored Alpha Vantage data for {symbol}")
                
                # Respect API rate limits
                time.sleep(15)
                
            except Exception as e:
                logger.error(f"Error processing Alpha Vantage data for {symbol}: {e}")
                conn.rollback()
        
        return True
        
    except Exception as e:
        logger.error(f"Error fetching data from Alpha Vantage: {e}")
        return False

def export_data_to_csv():
    """Export market data to CSV files for analysis"""
    try:
        # Create today's date folder
        date_str = datetime.now().strftime('%Y-%m-%d')
        export_dir = os.path.join(data_dir, date_str)
        os.makedirs(export_dir, exist_ok=True)
        
        # Query all data for the last 7 days
        week_ago = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        
        # Export market data
        query = f"""
        SELECT * FROM market_data 
        WHERE timestamp > '{week_ago}'
        ORDER BY symbol, timestamp
        """
        
        df = pd.read_sql_query(query, conn)
        if not df.empty:
            for symbol in df['symbol'].unique():
                symbol_df = df[df['symbol'] == symbol]
                symbol_df.to_csv(os.path.join(export_dir, f"{symbol}.csv"), index=False)
            
            # Export all data combined
            df.to_csv(os.path.join(export_dir, "all_market_data.csv"), index=False)
        
        # Export metadata
        metadata_df = pd.read_sql_query("SELECT * FROM market_metadata", conn)
        if not metadata_df.empty:
            metadata_df.to_csv(os.path.join(export_dir, "market_metadata.csv"), index=False)
        
        logger.info(f"Exported market data to {export_dir}")
        return True
        
    except Exception as e:
        logger.error(f"Error exporting data to CSV: {e}")
        return False

def update_market_data():
    """Main function to update market data periodically"""
    try:
        # Fetch data from Yahoo Finance
        fetch_yfinance_data(ALL_SYMBOLS)
        
        # Fetch data from Alpha Vantage (if configured)
        if ALPHA_VANTAGE_KEY and ALPHA_VANTAGE_KEY != "YOUR_ALPHA_VANTAGE_API_KEY":
            # Only use Alpha Vantage for stocks, as it's better for those
            fetch_alpha_vantage_data(STOCKS[:5])  # Limit due to API constraints
        
        # Export to CSV
        export_data_to_csv()
        
        logger.info("Market data update completed")
        
    except Exception as e:
        logger.error(f"Error in update_market_data: {e}")

def start_market_data_service():
    """Start the market data collection service"""
    logger.info("Starting market data collection service")
    
    # Set up database
    setup_database()
    
    # Update immediately on start
    update_market_data()
    
    # Schedule updates
    while True:
        # Sleep until next update
        logger.info(f"Next update in {UPDATE_INTERVAL} minutes")
        time.sleep(UPDATE_INTERVAL * 60)
        
        # Update data
        update_market_data()

if __name__ == "__main__":
    start_market_data_service() 