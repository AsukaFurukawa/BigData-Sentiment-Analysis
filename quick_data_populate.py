#!/usr/bin/env python3
import sqlite3
import requests
import json
import pandas as pd
from datetime import datetime, timedelta
import random
import time
from textblob import TextBlob
import feedparser
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database setup
DB_PATH = 'data-storage/data/sentiment_data.db'

def setup_database():
    """Set up the database with required tables"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create sentiment_results table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS sentiment_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        source TEXT NOT NULL,
        created_at TEXT NOT NULL,
        text TEXT NOT NULL,
        domain TEXT NOT NULL,
        overall_sentiment TEXT NOT NULL,
        confidence REAL DEFAULT 0.0
    )
    ''')
    
    # Create finance_analysis table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS finance_analysis (
        id INTEGER PRIMARY KEY,
        sentiment TEXT,
        finance_score REAL,
        entities TEXT,
        FOREIGN KEY (id) REFERENCES sentiment_results (id)
    )
    ''')
    
    # Create tech_analysis table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS tech_analysis (
        id INTEGER PRIMARY KEY,
        sentiment TEXT,
        tech_score REAL,
        entities TEXT,
        categories TEXT,
        FOREIGN KEY (id) REFERENCES sentiment_results (id)
    )
    ''')
    
    conn.commit()
    conn.close()
    logger.info("Database setup complete")

def analyze_sentiment(text):
    """Analyze sentiment using TextBlob"""
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    
    if polarity > 0.1:
        return 'positive', abs(polarity)
    elif polarity < -0.1:
        return 'negative', abs(polarity)
    else:
        return 'neutral', abs(polarity)

def extract_entities(text, domain):
    """Extract entities based on domain"""
    entities = []
    
    if domain == 'finance':
        # Look for stock symbols, financial terms
        finance_terms = ['stock', 'market', 'trading', 'investment', 'crypto', 'bitcoin', 'ethereum', 'nasdaq', 'sp500', 'dow']
        entities = [term for term in finance_terms if term.lower() in text.lower()]
    elif domain == 'technology':
        # Look for tech terms
        tech_terms = ['AI', 'machine learning', 'blockchain', 'cloud', 'software', 'hardware', 'startup', 'tech', 'innovation']
        entities = [term for term in tech_terms if term.lower() in text.lower()]
    
    return json.dumps(entities)

def collect_news_data():
    """Collect data from news RSS feeds"""
    feeds = [
        {'url': 'https://feeds.finance.yahoo.com/rss/2.0/headline', 'domain': 'finance'},
        {'url': 'https://www.cnbc.com/id/100003114/device/rss/rss.html', 'domain': 'finance'},
        {'url': 'https://feeds.feedburner.com/TechCrunch/', 'domain': 'technology'},
        {'url': 'https://www.wired.com/feed/rss', 'domain': 'technology'},
        {'url': 'https://feeds.arstechnica.com/arstechnica/technology-lab', 'domain': 'technology'}
    ]
    
    all_articles = []
    
    for feed_info in feeds:
        try:
            logger.info(f"Collecting from {feed_info['url']}")
            feed = feedparser.parse(feed_info['url'])
            
            for entry in feed.entries[:20]:  # Limit to 20 articles per feed
                title = entry.title if hasattr(entry, 'title') else ''
                summary = entry.summary if hasattr(entry, 'summary') else ''
                text = f"{title}. {summary}"
                
                if len(text) > 50:  # Only process substantial text
                    sentiment, confidence = analyze_sentiment(text)
                    
                    article = {
                        'source': 'news',
                        'created_at': datetime.now().isoformat(),
                        'text': text[:500],  # Limit text length
                        'domain': feed_info['domain'],
                        'overall_sentiment': sentiment,
                        'confidence': confidence
                    }
                    all_articles.append(article)
            
            time.sleep(1)  # Be respectful to servers
            
        except Exception as e:
            logger.error(f"Error collecting from {feed_info['url']}: {e}")
            continue
    
    return all_articles

def generate_sample_social_data():
    """Generate realistic sample social media data"""
    finance_posts = [
        "Just bought more $AAPL stock! This company is going to the moon 🚀",
        "Market crash incoming? Time to sell everything and go cash",
        "Bitcoin hitting new highs! Cryptocurrency is the future",
        "Fed rate decision tomorrow. Expecting volatility in the markets",
        "Tech stocks are overvalued. Time for a correction",
        "Love this bull market! My portfolio is up 20% this year",
        "Inflation data worse than expected. Markets are tanking",
        "Tesla earnings beat expectations! $TSLA to $500",
        "Oil prices surging. Energy stocks looking good",
        "Banking sector showing strength. $JPM is a buy"
    ]
    
    tech_posts = [
        "ChatGPT is revolutionizing how we work. AI is incredible!",
        "New iPhone features are amazing. Apple keeps innovating",
        "Cloud computing growth is unstoppable. AWS dominating",
        "Blockchain technology will change everything",
        "5G rollout is slower than expected. Disappointing",
        "Cybersecurity threats are increasing. Need better protection",
        "Machine learning models are getting scary good",
        "Quantum computing breakthrough announced today!",
        "Social media platforms need better content moderation",
        "Electric vehicle adoption accelerating rapidly"
    ]
    
    all_posts = []
    
    # Generate finance posts
    for i, post in enumerate(finance_posts):
        sentiment, confidence = analyze_sentiment(post)
        
        article = {
            'source': random.choice(['twitter', 'reddit']),
            'created_at': (datetime.now() - timedelta(hours=random.randint(1, 168))).isoformat(),
            'text': post,
            'domain': 'finance',
            'overall_sentiment': sentiment,
            'confidence': confidence
        }
        all_posts.append(article)
    
    # Generate tech posts
    for i, post in enumerate(tech_posts):
        sentiment, confidence = analyze_sentiment(post)
        
        article = {
            'source': random.choice(['twitter', 'reddit']),
            'created_at': (datetime.now() - timedelta(hours=random.randint(1, 168))).isoformat(),
            'text': post,
            'domain': 'technology',
            'overall_sentiment': sentiment,
            'confidence': confidence
        }
        all_posts.append(article)
    
    return all_posts

def insert_data(articles):
    """Insert articles into the database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    for article in articles:
        # Generate unique ID
        article_id = f"{article['source']}_{int(time.time() * 1000)}_{random.randint(1000, 9999)}"
        
        # Insert into sentiment_results
        cursor.execute('''
        INSERT INTO sentiment_results (id, source, created_at, text, domain, overall_sentiment, score, confidence, model)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            article_id,
            article['source'],
            article['created_at'],
            article['text'],
            article['domain'],
            article['overall_sentiment'],
            article['confidence'],
            str(article['confidence']),
            'textblob'
        ))
        
        # Insert domain-specific analysis
        if article['domain'] == 'finance':
            entities = extract_entities(article['text'], 'finance')
            positive_terms = json.dumps(['bullish', 'buy', 'growth', 'profit', 'gains'])
            negative_terms = json.dumps(['bearish', 'sell', 'loss', 'decline', 'crash'])
            
            cursor.execute('''
            INSERT INTO finance_analysis (id, entities, positive_terms, negative_terms)
            VALUES (?, ?, ?, ?)
            ''', (article_id, entities, positive_terms, negative_terms))
        
        elif article['domain'] == 'technology':
            entities = extract_entities(article['text'], 'technology')
            categories = json.dumps(['general', 'innovation'])
            
            cursor.execute('''
            INSERT INTO tech_analysis (id, entities, categories)
            VALUES (?, ?, ?)
            ''', (article_id, entities, categories))
    
    conn.commit()
    conn.close()
    logger.info(f"Inserted {len(articles)} articles into database")

def main():
    """Main function to collect and store data"""
    logger.info("Starting quick data population...")
    
    # Setup database
    setup_database()
    
    # Collect news data
    logger.info("Collecting news data...")
    news_articles = collect_news_data()
    
    # Generate sample social data
    logger.info("Generating sample social media data...")
    social_articles = generate_sample_social_data()
    
    # Combine all articles
    all_articles = news_articles + social_articles
    
    # Insert into database
    logger.info(f"Inserting {len(all_articles)} articles...")
    insert_data(all_articles)
    
    logger.info("Data population complete!")
    logger.info(f"Total articles inserted: {len(all_articles)}")
    
    # Print summary
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM sentiment_results')
    total_count = cursor.fetchone()[0]
    conn.close()
    
    logger.info(f"Total records in database: {total_count}")

if __name__ == "__main__":
    main() 