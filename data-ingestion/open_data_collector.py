#!/usr/bin/env python3
import os
import sys
import json
import time
import logging
import random
import sqlite3
import datetime
import feedparser
import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from newspaper import Article
from concurrent.futures import ThreadPoolExecutor
from kafka import KafkaProducer

# Set up paths for imports
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

# Configure logging
logs_dir = os.path.join(base_dir, 'logs')
os.makedirs(logs_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(logs_dir, 'open_data_collector.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("open_data_collector")

# Path configurations
config_path = os.path.join(os.path.dirname(__file__), 'config', 'data_sources.json')
data_dir = os.path.join(base_dir, 'data-storage', 'data')
os.makedirs(os.path.join(os.path.dirname(__file__), 'config'), exist_ok=True)
os.makedirs(data_dir, exist_ok=True)

class OpenDataCollector:
    """Collector for open data sources like RSS feeds, news APIs and Reddit"""
    
    def __init__(self):
        self.config = self.load_config()
        self.db_path = os.path.join(data_dir, 'sentiment_data.db')
        
        # Initialize Kafka if available
        self.producer = None
        self.use_kafka = self.config.get('use_kafka', False)
        
        if self.use_kafka:
            try:
                kafka_config = self.config.get('kafka', {})
                bootstrap_servers = kafka_config.get('bootstrap_servers', 'localhost:29092')
                self.topic = kafka_config.get('input_topics', ['raw_posts'])[0]
                
                self.producer = KafkaProducer(
                    bootstrap_servers=bootstrap_servers,
                    value_serializer=lambda x: json.dumps(x).encode('utf-8')
                )
                logger.info(f"Connected to Kafka at {bootstrap_servers}")
            except Exception as e:
                logger.error(f"Failed to connect to Kafka: {e}")
                self.producer = None
        
        # Ensure the database is set up
        self.setup_database()
    
    def load_config(self):
        """Load configuration from JSON file or create default if not exists"""
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                return config
            except Exception as e:
                logger.error(f"Error loading configuration: {e}")
        
        # Create default config
        config = {
            "use_kafka": False,
            "kafka": {
                "bootstrap_servers": "localhost:29092",
                "input_topics": ["raw_posts"],
                "output_topic": "processed_sentiment"
            },
            "sources": {
                "rss_feeds": {
                    "finance": [
                        {"name": "Yahoo Finance", "url": "https://finance.yahoo.com/news/rssindex"},
                        {"name": "CNBC", "url": "https://www.cnbc.com/id/10000664/device/rss/rss.html"},
                        {"name": "MarketWatch", "url": "http://feeds.marketwatch.com/marketwatch/topstories/"},
                        {"name": "Financial Times", "url": "https://www.ft.com/?format=rss"}
                    ],
                    "technology": [
                        {"name": "TechCrunch", "url": "https://techcrunch.com/feed/"},
                        {"name": "Wired", "url": "https://www.wired.com/feed/rss"},
                        {"name": "The Verge", "url": "https://www.theverge.com/rss/index.xml"},
                        {"name": "ArsTechnica", "url": "https://feeds.arstechnica.com/arstechnica/technology-lab"}
                    ]
                },
                "news_api": {
                    "enabled": False,
                    "endpoint": "https://newsapi.org/v2/everything",
                    "api_key": "YOUR_API_KEY_HERE",
                    "queries": {
                        "finance": ["stock market", "investing", "financial news", "economy"],
                        "technology": ["technology", "AI", "blockchain", "tech industry"]
                    }
                },
                "reddit": {
                    "enabled": True,
                    "subreddits": {
                        "finance": ["investing", "stocks", "wallstreetbets", "finance", "economy"],
                        "technology": ["technology", "tech", "programming", "MachineLearning", "artificial"]
                    }
                },
                "hackernews": {
                    "enabled": True
                }
            },
            "collection": {
                "interval_minutes": 60,
                "max_articles_per_source": 50,
                "lookback_days": 1
            }
        }
        
        # Save the default config
        try:
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2)
            logger.info(f"Created default configuration at {config_path}")
        except Exception as e:
            logger.error(f"Error saving default configuration: {e}")
        
        return config
    
    def setup_database(self):
        """Set up the SQLite database for storing articles"""
        try:
            conn = sqlite3.connect(self.db_path)
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
            
            conn.commit()
            conn.close()
            logger.info("Database setup completed")
            
        except Exception as e:
            logger.error(f"Error setting up database: {e}")
    
    def fetch_rss_feed(self, feed_info, domain):
        """Fetch and parse an RSS feed"""
        feed_name = feed_info.get('name', 'Unknown')
        feed_url = feed_info.get('url', '')
        
        if not feed_url:
            logger.error(f"Missing URL for feed {feed_name}")
            return []
        
        try:
            logger.info(f"Fetching RSS feed: {feed_name} ({feed_url})")
            feed = feedparser.parse(feed_url)
            
            articles = []
            max_articles = self.config.get('collection', {}).get('max_articles_per_source', 50)
            
            for entry in feed.entries[:max_articles]:
                article = {
                    'id': f"rss_{hash(entry.get('link', '') + entry.get('title', ''))}",
                    'title': entry.get('title', ''),
                    'link': entry.get('link', ''),
                    'published': entry.get('published', entry.get('updated', datetime.now().isoformat())),
                    'summary': entry.get('summary', ''),
                    'source': feed_name,
                    'domain': domain,
                    'type': 'rss'
                }
                articles.append(article)
            
            logger.info(f"Fetched {len(articles)} articles from {feed_name}")
            return articles
        
        except Exception as e:
            logger.error(f"Error fetching RSS feed {feed_name}: {e}")
            return []
    
    def fetch_article_content(self, article):
        """Fetch full article content using newspaper3k"""
        try:
            if not article.get('link'):
                return article
            
            news_article = Article(article['link'])
            news_article.download()
            news_article.parse()
            
            # Update with full content
            article['text'] = news_article.text
            
            # Try to get better date
            if news_article.publish_date:
                article['published'] = news_article.publish_date.isoformat()
            
            return article
        except Exception as e:
            logger.warning(f"Error fetching article content: {e}")
            # Fall back to summary if available
            article['text'] = article.get('summary', '')
            return article
    
    def fetch_reddit_posts(self, subreddit, domain, limit=25):
        """Fetch posts from Reddit (without authentication)"""
        url = f"https://www.reddit.com/r/{subreddit}/hot.json?limit={limit}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        try:
            logger.info(f"Fetching Reddit posts from r/{subreddit}")
            response = requests.get(url, headers=headers)
            
            if response.status_code != 200:
                logger.error(f"Error fetching Reddit posts: Status code {response.status_code}")
                return []
            
            data = response.json()
            posts = []
            
            for post in data.get('data', {}).get('children', []):
                post_data = post.get('data', {})
                
                # Skip stickied posts
                if post_data.get('stickied', False):
                    continue
                
                article = {
                    'id': f"reddit_{post_data.get('id', '')}",
                    'title': post_data.get('title', ''),
                    'link': f"https://www.reddit.com{post_data.get('permalink', '')}",
                    'published': datetime.fromtimestamp(post_data.get('created_utc', time.time())).isoformat(),
                    'summary': post_data.get('selftext', '')[:500],
                    'text': post_data.get('selftext', ''),
                    'source': f"Reddit r/{subreddit}",
                    'domain': domain,
                    'type': 'reddit'
                }
                
                posts.append(article)
            
            logger.info(f"Fetched {len(posts)} posts from r/{subreddit}")
            return posts
            
        except Exception as e:
            logger.error(f"Error fetching Reddit posts from r/{subreddit}: {e}")
            return []
    
    def fetch_hackernews_stories(self, limit=30):
        """Fetch top stories from Hacker News"""
        try:
            logger.info("Fetching Hacker News top stories")
            
            # Get top story IDs
            top_stories_url = "https://hacker-news.firebaseio.com/v0/topstories.json"
            response = requests.get(top_stories_url)
            
            if response.status_code != 200:
                logger.error(f"Error fetching Hacker News stories: Status code {response.status_code}")
                return []
            
            story_ids = response.json()[:limit]
            stories = []
            
            # Fetch each story
            for story_id in story_ids:
                story_url = f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json"
                story_response = requests.get(story_url)
                
                if story_response.status_code != 200:
                    continue
                
                story = story_response.json()
                
                # Skip jobs, polls, etc.
                if story.get('type') != 'story':
                    continue
                
                # Determine domain (tech-focused)
                story_domain = 'technology'
                
                article = {
                    'id': f"hackernews_{story.get('id', '')}",
                    'title': story.get('title', ''),
                    'link': story.get('url', f"https://news.ycombinator.com/item?id={story_id}"),
                    'published': datetime.fromtimestamp(story.get('time', time.time())).isoformat(),
                    'summary': story.get('text', '')[:500] if story.get('text') else '',
                    'source': 'Hacker News',
                    'domain': story_domain,
                    'type': 'hackernews'
                }
                
                stories.append(article)
                
                # Avoid hitting rate limits
                time.sleep(0.1)
            
            logger.info(f"Fetched {len(stories)} stories from Hacker News")
            return stories
            
        except Exception as e:
            logger.error(f"Error fetching Hacker News stories: {e}")
            return []
    
    def store_article(self, article):
        """Store an article in the database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if article already exists
            cursor.execute("SELECT id FROM sentiment_results WHERE id = ?", (article['id'],))
            if cursor.fetchone():
                conn.close()
                return False
            
            # Insert into sentiment_results table
            cursor.execute('''
            INSERT INTO sentiment_results 
            (id, created_at, source, text, title, url, domain, overall_sentiment, score, confidence, model)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                article['id'],
                article.get('published', datetime.now().isoformat()),
                article['source'],
                article.get('text', article.get('summary', '')),
                article['title'],
                article['link'],
                article['domain'],
                'neutral',  # Will be updated by sentiment analyzer
                0.5,        # Default neutral score
                'medium',   # Default confidence
                'pending'   # Will be updated by sentiment analyzer
            ))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Error storing article: {e}")
            return False
    
    def send_to_kafka(self, article):
        """Send article to Kafka if producer is available"""
        if not self.producer:
            return False
        
        try:
            self.producer.send(self.topic, article)
            return True
        except Exception as e:
            logger.error(f"Error sending to Kafka: {e}")
            return False
    
    def collect_from_all_sources(self):
        """Collect data from all configured sources"""
        all_articles = []
        
        # 1. Collect from RSS feeds
        rss_feeds = self.config.get('sources', {}).get('rss_feeds', {})
        
        for domain, feeds in rss_feeds.items():
            for feed in feeds:
                articles = self.fetch_rss_feed(feed, domain)
                all_articles.extend(articles)
        
        # 2. Collect from Reddit
        reddit_config = self.config.get('sources', {}).get('reddit', {})
        if reddit_config.get('enabled', False):
            subreddits = reddit_config.get('subreddits', {})
            
            for domain, subs in subreddits.items():
                for sub in subs:
                    posts = self.fetch_reddit_posts(sub, domain)
                    all_articles.extend(posts)
        
        # 3. Collect from Hacker News
        hn_config = self.config.get('sources', {}).get('hackernews', {})
        if hn_config.get('enabled', False):
            stories = self.fetch_hackernews_stories()
            all_articles.extend(stories)
        
        # Fetch full content for articles
        logger.info(f"Fetching full content for {len(all_articles)} articles")
        
        articles_with_content = []
        with ThreadPoolExecutor(max_workers=8) as executor:
            articles_with_content = list(executor.map(self.fetch_article_content, all_articles))
        
        # Store articles and send to Kafka
        new_articles = 0
        for article in articles_with_content:
            # Store in database
            if self.store_article(article):
                new_articles += 1
                
                # Send to Kafka if enabled
                if self.use_kafka:
                    self.send_to_kafka(article)
        
        logger.info(f"Collected and stored {new_articles} new articles")
        
        # Also save to CSV as backup
        try:
            articles_df = pd.DataFrame(articles_with_content)
            if not articles_df.empty:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                csv_path = os.path.join(data_dir, f'articles_{timestamp}.csv')
                articles_df.to_csv(csv_path, index=False)
                logger.info(f"Saved {len(articles_df)} articles to {csv_path}")
        except Exception as e:
            logger.error(f"Error saving to CSV: {e}")
        
        return new_articles
    
    def run_collection_service(self, once=False):
        """Run the collection service"""
        try:
            if once:
                logger.info("Running one-time collection from all sources")
                return self.collect_from_all_sources()
            
            interval_minutes = self.config.get('collection', {}).get('interval_minutes', 60)
            
            logger.info(f"Starting continuous collection service with {interval_minutes} minute interval")
            
            while True:
                try:
                    articles_count = self.collect_from_all_sources()
                    logger.info(f"Waiting {interval_minutes} minutes until next collection")
                    time.sleep(interval_minutes * 60)
                except Exception as e:
                    logger.error(f"Error in collection cycle: {e}")
                    time.sleep(300)  # Wait 5 minutes on error
        
        except KeyboardInterrupt:
            logger.info("Collection service interrupted by user")
        finally:
            if self.producer:
                self.producer.close()
            logger.info("Collection service stopped")

def main():
    """Main function to run the collector"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Open Data Collector for Sentiment Analysis')
    parser.add_argument('--once', action='store_true', help='Run collection once and exit')
    
    args = parser.parse_args()
    
    collector = OpenDataCollector()
    collector.run_collection_service(once=args.once)

if __name__ == "__main__":
    main() 