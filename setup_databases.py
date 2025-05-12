#!/usr/bin/env python3
import os
import argparse
import sqlite3
import logging
from datetime import datetime

# Configure logging
logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
os.makedirs(logs_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(logs_dir, 'database_setup.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("database_setup")

def setup_sentiment_db(db_path):
    """Set up the sentiment analysis database"""
    logger.info(f"Setting up sentiment database at {db_path}")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create tables if they don't exist
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS sentiment_results (
            id TEXT PRIMARY KEY,
            created_at TEXT,
            source TEXT,
            text TEXT,
            title TEXT,
            url TEXT,
            domain TEXT,
            overall_sentiment TEXT,
            score REAL,
            confidence TEXT,
            model TEXT
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS finance_analysis (
            id TEXT PRIMARY KEY,
            entities TEXT,
            positive_terms TEXT,
            negative_terms TEXT,
            FOREIGN KEY (id) REFERENCES sentiment_results(id)
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS tech_analysis (
            id TEXT PRIMARY KEY,
            entities TEXT,
            categories TEXT,
            FOREIGN KEY (id) REFERENCES sentiment_results(id)
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS entity_mentions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity TEXT,
            sentiment_id TEXT,
            domain TEXT,
            sentiment_score REAL,
            created_at TEXT,
            FOREIGN KEY (sentiment_id) REFERENCES sentiment_results(id)
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS entity_relationships (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity1 TEXT,
            entity2 TEXT,
            weight INTEGER,
            sentiment_avg REAL,
            domain TEXT
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS source_metadata (
            source TEXT PRIMARY KEY,
            source_type TEXT,
            reliability_score REAL,
            last_updated TEXT
        )
        ''')
        
        # Create indexes for faster querying
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_sentiment_created_at ON sentiment_results(created_at)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_sentiment_domain ON sentiment_results(domain)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_sentiment_source ON sentiment_results(source)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_entity_mentions_entity ON entity_mentions(entity)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_entity_mentions_created_at ON entity_mentions(created_at)')
        
        conn.commit()
        logger.info("Sentiment database setup completed successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"Error setting up sentiment database: {e}")
        return False
    finally:
        if 'conn' in locals() and conn:
            conn.close()

def setup_market_db(db_path):
    """Set up the market data database"""
    logger.info(f"Setting up market database at {db_path}")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create tables if they don't exist
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS market_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            symbol TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS symbols (
            symbol TEXT PRIMARY KEY,
            name TEXT,
            sector TEXT,
            industry TEXT,
            exchange TEXT,
            is_index BOOLEAN
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS market_sentiment (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            symbol TEXT,
            sentiment_score REAL,
            mentions INTEGER,
            source TEXT
        )
        ''')
        
        # Create indexes for faster querying
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_market_data_timestamp ON market_data(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_market_data_symbol ON market_data(symbol)')
        
        conn.commit()
        logger.info("Market database setup completed successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"Error setting up market database: {e}")
        return False
    finally:
        if 'conn' in locals() and conn:
            conn.close()

def setup_analytics_db(db_path):
    """Set up the analytics database for tracking insights"""
    logger.info(f"Setting up analytics database at {db_path}")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create tables if they don't exist
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS correlation_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT,
            sentiment_domain TEXT,
            correlation_value REAL,
            p_value REAL,
            time_period TEXT,
            lag_days INTEGER,
            date_calculated TEXT
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS entity_network (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity TEXT,
            mentions INTEGER,
            centrality_score REAL,
            sentiment_score REAL,
            domains TEXT,
            date_calculated TEXT
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS entity_temporal (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity TEXT,
            date TEXT,
            mentions INTEGER,
            sentiment_score REAL
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS topic_models (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            topic_id INTEGER,
            topic_terms TEXT,
            domain TEXT,
            importance_score REAL,
            date_calculated TEXT
        )
        ''')
        
        # Create indexes for faster querying
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_correlation_symbol ON correlation_results(symbol)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_entity_network_entity ON entity_network(entity)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_entity_temporal_entity ON entity_temporal(entity)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_entity_temporal_date ON entity_temporal(date)')
        
        conn.commit()
        logger.info("Analytics database setup completed successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"Error setting up analytics database: {e}")
        return False
    finally:
        if 'conn' in locals() and conn:
            conn.close()

def setup_data_sources_db(db_path):
    """Set up the data sources database for open data sources"""
    logger.info(f"Setting up data sources database at {db_path}")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create tables for tracking RSS feeds, APIs and other data sources
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS data_sources (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            url TEXT,
            source_type TEXT,
            domain TEXT,
            is_active BOOLEAN,
            last_fetched TEXT,
            fetch_frequency INTEGER,
            reliability_score REAL
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS fetch_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_id INTEGER,
            fetch_time TEXT,
            success BOOLEAN,
            items_fetched INTEGER,
            error_message TEXT,
            FOREIGN KEY (source_id) REFERENCES data_sources(id)
        )
        ''')
        
        # Create index for faster querying
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_data_sources_domain ON data_sources(domain)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_data_sources_type ON data_sources(source_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_fetch_history_source ON fetch_history(source_id)')
        
        # Insert example data sources
        cursor.execute('''
        INSERT OR IGNORE INTO data_sources 
        (name, url, source_type, domain, is_active, fetch_frequency, reliability_score)
        VALUES 
        ('Yahoo Finance', 'https://finance.yahoo.com/news/rssindex', 'rss', 'finance', 1, 60, 0.8),
        ('CNBC', 'https://www.cnbc.com/id/10000664/device/rss/rss.html', 'rss', 'finance', 1, 60, 0.8),
        ('TechCrunch', 'https://techcrunch.com/feed/', 'rss', 'technology', 1, 60, 0.9),
        ('The Verge', 'https://www.theverge.com/rss/index.xml', 'rss', 'technology', 1, 60, 0.85),
        ('Reddit r/investing', 'https://www.reddit.com/r/investing/hot.json', 'api', 'finance', 1, 120, 0.7),
        ('Reddit r/tech', 'https://www.reddit.com/r/tech/hot.json', 'api', 'technology', 1, 120, 0.7),
        ('Hacker News', 'https://hacker-news.firebaseio.com/v0/topstories.json', 'api', 'technology', 1, 120, 0.9)
        ''')
        
        conn.commit()
        logger.info("Data sources database setup completed successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"Error setting up data sources database: {e}")
        return False
    finally:
        if 'conn' in locals() and conn:
            conn.close()

def main():
    """Main function to set up all databases"""
    parser = argparse.ArgumentParser(description="Set up sentiment analysis databases")
    parser.add_argument('--overwrite', action='store_true', help="Overwrite existing databases")
    args = parser.parse_args()
    
    # Define database paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    sentiment_db = os.path.join(base_dir, 'data-storage', 'data', 'sentiment_data.db')
    market_db = os.path.join(base_dir, 'market-data', 'data', 'market_data.db')
    analytics_db = os.path.join(base_dir, 'data-storage', 'data', 'analytics.db')
    data_sources_db = os.path.join(base_dir, 'data-storage', 'data', 'data_sources.db')
    
    # Create the directories if they don't exist
    os.makedirs(os.path.dirname(sentiment_db), exist_ok=True)
    os.makedirs(os.path.dirname(market_db), exist_ok=True)
    
    # If overwrite flag is set, delete existing databases
    if args.overwrite:
        for db_path in [sentiment_db, market_db, analytics_db, data_sources_db]:
            if os.path.exists(db_path):
                try:
                    os.remove(db_path)
                    logger.info(f"Deleted existing database: {db_path}")
                except Exception as e:
                    logger.error(f"Error deleting database {db_path}: {e}")
    
    # Set up the databases
    if setup_sentiment_db(sentiment_db):
        logger.info(f"Sentiment database setup complete: {sentiment_db}")
    else:
        logger.error("Failed to set up sentiment database")
    
    if setup_market_db(market_db):
        logger.info(f"Market database setup complete: {market_db}")
    else:
        logger.error("Failed to set up market database")
    
    if setup_analytics_db(analytics_db):
        logger.info(f"Analytics database setup complete: {analytics_db}")
    else:
        logger.error("Failed to set up analytics database")
    
    if setup_data_sources_db(data_sources_db):
        logger.info(f"Data sources database setup complete: {data_sources_db}")
    else:
        logger.error("Failed to set up data sources database")
    
    logger.info("Database setup process completed")

if __name__ == "__main__":
    main() 