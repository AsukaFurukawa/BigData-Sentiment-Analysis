#!/usr/bin/env python3
import os
import json
import sqlite3
import random
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', 'sample_data_generator.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("sample_data_generator")

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SENTIMENT_DB = os.path.join(BASE_DIR, 'data-storage', 'data', 'sentiment_data.db')
MARKET_DB = os.path.join(BASE_DIR, 'market-data', 'data', 'market_data.db')

# Ensure directories exist
os.makedirs(os.path.dirname(SENTIMENT_DB), exist_ok=True)
os.makedirs(os.path.dirname(MARKET_DB), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, 'logs'), exist_ok=True)

def generate_sentiment_data(num_records=1000):
    """Generate sample sentiment data"""
    logger.info(f"Creating sentiment database...")
    conn = sqlite3.connect(SENTIMENT_DB)
    cursor = conn.cursor()
    
    # Create tables if they don't exist
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS sentiment_results (
        id TEXT PRIMARY KEY,
        created_at TEXT,
        source TEXT,
        text TEXT,
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
    
    # Sample sources
    sources = ['twitter', 'news', 'reddit', 'blog']
    
    # Sample domains
    domains = ['finance', 'technology', 'finance-tech']
    
    # Sample sentiment categories
    sentiments = ['positive', 'neutral', 'negative']
    sentiment_weights = [0.6, 0.3, 0.1]  # More positive than negative
    
    # Sample finance entities
    finance_entities = ['Apple', 'Microsoft', 'Google', 'Amazon', 'Tesla', 'Bitcoin', 
                       'S&P 500', 'Dow Jones', 'NASDAQ', 'Federal Reserve', 'inflation', 
                       'interest rates', 'GDP', 'earnings', 'stocks', 'bonds']
    
    # Sample tech entities
    tech_entities = ['AI', 'machine learning', 'cloud computing', 'blockchain', 'IoT', 
                    'cybersecurity', '5G', 'quantum computing', 'virtual reality', 
                    'big data', 'ChatGPT', 'neural networks']
    
    # Sample positive terms
    positive_terms = ['growth', 'profit', 'surge', 'exceed expectations', 'breakthrough', 
                     'innovation', 'outperform', 'record high', 'bullish', 'strong performance']
    
    # Sample negative terms
    negative_terms = ['decline', 'loss', 'drop', 'miss expectations', 'setback', 
                     'downgrade', 'bearish', 'weak performance', 'volatility', 'underperform']
    
    # Sample confidence levels
    confidence_levels = ['high', 'medium', 'low']
    
    # Sample models
    models = ['FinBERT', 'TechBERT', 'RoBERTa', 'BERT-base']
    
    # Generate random data
    logger.info(f"Generating {num_records} sample sentiment records...")
    records_added = 0
    
    for i in range(num_records):
        # Generate random datetime within the last 30 days
        days_ago = random.randint(0, 30)
        hours_ago = random.randint(0, 24)
        minutes_ago = random.randint(0, 60)
        created_at = (datetime.now() - timedelta(days=days_ago, hours=hours_ago, minutes=minutes_ago)).isoformat()
        
        # Generate random source, domain, and sentiment
        source = random.choice(sources)
        domain = random.choice(domains)
        sentiment = random.choices(sentiments, weights=sentiment_weights)[0]
        
        # Generate random score based on sentiment
        if sentiment == 'positive':
            score = random.uniform(0.6, 1.0)
        elif sentiment == 'negative':
            score = random.uniform(0.0, 0.4)
        else:
            score = random.uniform(0.4, 0.6)
        
        # Generate random confidence and model
        confidence = random.choice(confidence_levels)
        model = random.choice(models)
        
        # Generate random sample text
        if domain == 'finance' or domain == 'finance-tech':
            entities = random.sample(finance_entities, random.randint(1, 3))
            text_entities = ' and '.join(entities)
            
            if sentiment == 'positive':
                term = random.choice(positive_terms)
                text = f"{text_entities} showing {term} in recent market activity."
            elif sentiment == 'negative':
                term = random.choice(negative_terms)
                text = f"{text_entities} experiencing {term} due to market conditions."
            else:
                text = f"{text_entities} showing mixed signals in the current market environment."
        else:
            entities = random.sample(tech_entities, random.randint(1, 3))
            text_entities = ' and '.join(entities)
            
            if sentiment == 'positive':
                term = random.choice(positive_terms)
                text = f"New developments in {text_entities} demonstrating {term} in the tech sector."
            elif sentiment == 'negative':
                term = random.choice(negative_terms)
                text = f"Recent {text_entities} projects facing {term} as industry evolves."
            else:
                text = f"The impact of {text_entities} remains uncertain in today's technological landscape."
        
        # Generate unique ID
        record_id = f"sample_{i}_{int(datetime.now().timestamp())}"
        
        # Insert into sentiment_results
        cursor.execute('''
        INSERT OR REPLACE INTO sentiment_results 
        (id, created_at, source, text, domain, overall_sentiment, score, confidence, model)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (record_id, created_at, source, text, domain, sentiment, score, confidence, model))
        
        # Insert into domain-specific tables
        if domain == 'finance' or domain == 'finance-tech':
            # Get random entities, positive and negative terms
            fin_entities = random.sample(finance_entities, random.randint(2, 5))
            pos_terms = random.sample(positive_terms, random.randint(1, 3))
            neg_terms = random.sample(negative_terms, random.randint(1, 3))
            
            cursor.execute('''
            INSERT OR REPLACE INTO finance_analysis 
            (id, entities, positive_terms, negative_terms)
            VALUES (?, ?, ?, ?)
            ''', (record_id, json.dumps(fin_entities), json.dumps(pos_terms), json.dumps(neg_terms)))
        
        if domain == 'technology' or domain == 'finance-tech':
            # Get random tech entities and categories
            tech_ents = random.sample(tech_entities, random.randint(2, 5))
            categories = random.sample(['AI', 'cloud', 'security', 'data', 'mobile', 'hardware'], random.randint(1, 3))
            
            cursor.execute('''
            INSERT OR REPLACE INTO tech_analysis 
            (id, entities, categories)
            VALUES (?, ?, ?)
            ''', (record_id, json.dumps(tech_ents), json.dumps(categories)))
        
        records_added += 1
    
    conn.commit()
    conn.close()
    
    logger.info(f"Sample sentiment data generation complete. Added {records_added} records.")
    return records_added

def generate_market_data(days=90):
    """Generate sample market data"""
    logger.info(f"Creating market database...")
    conn = sqlite3.connect(MARKET_DB)
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
    
    # Sample symbols
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', '^GSPC', '^DJI']
    
    # Starting prices for each symbol
    base_prices = {
        'AAPL': 150.0,
        'MSFT': 280.0,
        'GOOGL': 120.0,
        'AMZN': 130.0,
        'META': 320.0,
        'TSLA': 250.0,
        '^GSPC': 4500.0,
        '^DJI': 35000.0
    }
    
    # Volatility for each symbol (standard deviation of daily returns)
    volatility = {
        'AAPL': 0.015,
        'MSFT': 0.012,
        'GOOGL': 0.018,
        'AMZN': 0.020,
        'META': 0.025,
        'TSLA': 0.030,
        '^GSPC': 0.008,
        '^DJI': 0.007
    }
    
    logger.info(f"Generating market data for {len(symbols)} symbols over {days} days...")
    records_added = 0
    
    for symbol in symbols:
        price = base_prices[symbol]
        vol = volatility[symbol]
        
        # Generate data for each day
        for day in range(days):
            date = (datetime.now() - timedelta(days=days-day)).strftime('%Y-%m-%d')
            
            # Random daily return with mild trend (slightly upward biased)
            daily_return = np.random.normal(0.0005, vol)
            price = price * (1 + daily_return)
            
            # Generate OHLC data
            daily_vol = vol * price
            open_price = price
            high_price = price + abs(np.random.normal(0, daily_vol))
            low_price = price - abs(np.random.normal(0, daily_vol))
            close_price = price * (1 + np.random.normal(0, vol/5))  # End of day price
            
            # Update the current price for next iteration
            price = close_price
            
            # Generate volume (in thousands)
            volume = int(np.random.normal(5000, 2000)) * 1000
            
            # Insert into database
            cursor.execute('''
            INSERT INTO market_data 
            (timestamp, symbol, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (date, symbol, open_price, high_price, low_price, close_price, volume))
            
            records_added += 1
    
    conn.commit()
    conn.close()
    
    logger.info(f"Sample market data generation complete. Added {records_added} records.")
    return records_added

def main():
    """Main function to generate all sample data"""
    logger.info("Starting sample data generation...")
    
    sentiment_records = generate_sentiment_data(1000)
    market_records = generate_market_data(90)
    
    logger.info("Sample data generation complete:")
    logger.info(f"- Sentiment records: {sentiment_records}")
    logger.info(f"- Market data points: {market_records}")
    
    print("Generated sample data:")
    print(f"- {sentiment_records} sentiment records in {SENTIMENT_DB}")
    print(f"- {market_records} market data points in {MARKET_DB}")
    print("You can now run the dashboard or analysis components.")

if __name__ == "__main__":
    main() 