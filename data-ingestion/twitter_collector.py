#!/usr/bin/env python3
import os
import json
import time
import logging
from datetime import datetime
import tweepy
from kafka import KafkaProducer
import pandas as pd
import threading
import sys

# Set up logging
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
logs_dir = os.path.join(base_dir, 'logs')
os.makedirs(logs_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(logs_dir, 'twitter_collector.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("twitter_collector")

# Path configurations
config_path = os.path.join(os.path.dirname(__file__), 'config', 'keys.json')
data_dir = os.path.join(base_dir, 'data-storage', 'data')
os.makedirs(os.path.join(os.path.dirname(__file__), 'config'), exist_ok=True)

# Load configuration
def load_config():
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            return config
        else:
            logger.warning(f"Config file not found at {config_path}, creating template")
            config = {
                "twitter": {
                    "bearer_token": "YOUR_BEARER_TOKEN",
                    "consumer_key": "YOUR_CONSUMER_KEY",
                    "consumer_secret": "YOUR_CONSUMER_SECRET",
                    "access_token": "YOUR_ACCESS_TOKEN",
                    "access_token_secret": "YOUR_ACCESS_TOKEN_SECRET"
                },
                "kafka": {
                    "bootstrap_servers": "localhost:29092",
                    "input_topics": ["raw_twitter_posts"],
                    "output_topic": "processed_sentiment"
                },
                "streaming": {
                    "keywords": {
                        "finance": ["stock market", "investing", "NASDAQ", "NYSE", "market crash", "bull market", "bear market", 
                                   "earnings", "IPO", "stocks", "bonds", "crypto", "bitcoin", "inflation", "recession"],
                        "technology": ["AI", "artificial intelligence", "machine learning", "deep learning", "big data", "blockchain", 
                                      "cloud computing", "tech stocks", "semiconductor", "quantum computing", "ChatGPT", "LLM"]
                    },
                    "accounts": {
                        "finance": ["FinancialTimes", "WSJ", "TheEconomist", "Bloomberg", "CNBCnow", "YahooFinance"],
                        "technology": ["WIRED", "TechCrunch", "verge", "techreview", "CNET"]
                    },
                    "max_tweets_per_keyword": 100,
                    "max_tweets_per_account": 50,
                    "streaming_buffer_size": 1000,
                    "batch_size": 20
                }
            }
            
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"Created template configuration file at {config_path}")
            logger.info("Please edit this file with your API keys and restart the collector")
            return config
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return None

# Twitter API v2 collector for historical data
class TwitterCollector:
    def __init__(self, config):
        self.config = config
        self.twitter_config = config.get('twitter', {})
        self.streaming_config = config.get('streaming', {})
        self.kafka_config = config.get('kafka', {})
        
        # Set up Kafka producer
        self.bootstrap_servers = self.kafka_config.get('bootstrap_servers', 'localhost:29092')
        self.topic = self.kafka_config.get('input_topics', ['raw_twitter_posts'])[0]
        
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda x: json.dumps(x).encode('utf-8')
            )
            logger.info(f"Connected to Kafka at {self.bootstrap_servers}")
        except Exception as e:
            logger.error(f"Failed to connect to Kafka: {e}")
            self.producer = None
        
        # Twitter API setup
        try:
            self.client = tweepy.Client(
                bearer_token=self.twitter_config.get('bearer_token'),
                consumer_key=self.twitter_config.get('consumer_key'),
                consumer_secret=self.twitter_config.get('consumer_secret'),
                access_token=self.twitter_config.get('access_token'),
                access_token_secret=self.twitter_config.get('access_token_secret')
            )
            logger.info("Twitter API client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Twitter API client: {e}")
            self.client = None
    
    def collect_tweets_by_keyword(self, keyword, domain, max_results=100):
        """Collect tweets containing a specific keyword"""
        if not self.client:
            logger.error("Twitter client not initialized")
            return []
        
        try:
            logger.info(f"Collecting tweets for keyword: {keyword}")
            tweets = self.client.search_recent_tweets(
                query=keyword,
                tweet_fields=['created_at', 'text', 'public_metrics', 'lang'],
                max_results=max_results
            )
            
            if not tweets.data:
                logger.warning(f"No tweets found for keyword: {keyword}")
                return []
            
            processed_tweets = []
            for tweet in tweets.data:
                processed_tweet = {
                    'id': str(tweet.id),
                    'text': tweet.text,
                    'created_at': tweet.created_at.isoformat(),
                    'source': 'twitter',
                    'domain': domain,
                    'search_term': keyword,
                    'lang': tweet.lang,
                    'likes': tweet.public_metrics.get('like_count', 0),
                    'retweets': tweet.public_metrics.get('retweet_count', 0),
                    'collected_at': datetime.now().isoformat()
                }
                processed_tweets.append(processed_tweet)
                
                # Send to Kafka if producer is available
                if self.producer:
                    self.producer.send(self.topic, processed_tweet)
            
            logger.info(f"Collected {len(processed_tweets)} tweets for keyword: {keyword}")
            return processed_tweets
            
        except Exception as e:
            logger.error(f"Error collecting tweets for keyword {keyword}: {e}")
            return []
    
    def collect_tweets_by_account(self, account, domain, max_results=50):
        """Collect tweets from a specific account"""
        if not self.client:
            logger.error("Twitter client not initialized")
            return []
        
        try:
            logger.info(f"Collecting tweets for account: {account}")
            
            # First, get the user ID
            user = self.client.get_user(username=account)
            if not user.data:
                logger.warning(f"User not found: {account}")
                return []
            
            user_id = user.data.id
            
            # Then get their tweets
            tweets = self.client.get_users_tweets(
                id=user_id,
                tweet_fields=['created_at', 'text', 'public_metrics', 'lang'],
                max_results=max_results
            )
            
            if not tweets.data:
                logger.warning(f"No tweets found for account: {account}")
                return []
            
            processed_tweets = []
            for tweet in tweets.data:
                processed_tweet = {
                    'id': str(tweet.id),
                    'text': tweet.text,
                    'created_at': tweet.created_at.isoformat(),
                    'source': 'twitter',
                    'domain': domain,
                    'user': account,
                    'lang': tweet.lang,
                    'likes': tweet.public_metrics.get('like_count', 0),
                    'retweets': tweet.public_metrics.get('retweet_count', 0),
                    'collected_at': datetime.now().isoformat()
                }
                processed_tweets.append(processed_tweet)
                
                # Send to Kafka if producer is available
                if self.producer:
                    self.producer.send(self.topic, processed_tweet)
            
            logger.info(f"Collected {len(processed_tweets)} tweets for account: {account}")
            return processed_tweets
            
        except Exception as e:
            logger.error(f"Error collecting tweets for account {account}: {e}")
            return []
    
    def run_collection_cycle(self):
        """Run a full collection cycle for keywords and accounts"""
        all_tweets = []
        
        # Collect finance keywords
        finance_keywords = self.streaming_config.get('keywords', {}).get('finance', [])
        for keyword in finance_keywords:
            tweets = self.collect_tweets_by_keyword(
                keyword, 
                'finance', 
                self.streaming_config.get('max_tweets_per_keyword', 100)
            )
            all_tweets.extend(tweets)
            time.sleep(2)  # Avoid rate limits
        
        # Collect tech keywords
        tech_keywords = self.streaming_config.get('keywords', {}).get('technology', [])
        for keyword in tech_keywords:
            tweets = self.collect_tweets_by_keyword(
                keyword, 
                'technology', 
                self.streaming_config.get('max_tweets_per_keyword', 100)
            )
            all_tweets.extend(tweets)
            time.sleep(2)  # Avoid rate limits
        
        # Collect finance accounts
        finance_accounts = self.streaming_config.get('accounts', {}).get('finance', [])
        for account in finance_accounts:
            tweets = self.collect_tweets_by_account(
                account, 
                'finance', 
                self.streaming_config.get('max_tweets_per_account', 50)
            )
            all_tweets.extend(tweets)
            time.sleep(2)  # Avoid rate limits
        
        # Collect tech accounts
        tech_accounts = self.streaming_config.get('accounts', {}).get('technology', [])
        for account in tech_accounts:
            tweets = self.collect_tweets_by_account(
                account, 
                'technology', 
                self.streaming_config.get('max_tweets_per_account', 50)
            )
            all_tweets.extend(tweets)
            time.sleep(2)  # Avoid rate limits
        
        # Store locally as backup
        if all_tweets:
            try:
                now = datetime.now().strftime('%Y%m%d_%H%M%S')
                tweets_df = pd.DataFrame(all_tweets)
                os.makedirs(os.path.join(data_dir, 'twitter'), exist_ok=True)
                tweets_df.to_csv(os.path.join(data_dir, 'twitter', f'tweets_{now}.csv'), index=False)
                logger.info(f"Saved {len(all_tweets)} tweets to CSV")
            except Exception as e:
                logger.error(f"Error saving tweets to CSV: {e}")
        
        return len(all_tweets)

# Stream listener for real-time streaming
class TwitterStreamListener(tweepy.StreamingClient):
    def __init__(self, bearer_token, kafka_producer, kafka_topic):
        super().__init__(bearer_token)
        self.producer = kafka_producer
        self.topic = kafka_topic
        self.buffer = []
        self.buffer_size = 100
        self.last_flush = time.time()
        self.flush_interval = 10  # seconds
    
    def on_tweet(self, tweet):
        """Process an incoming tweet"""
        try:
            # Process tweet
            processed_tweet = {
                'id': str(tweet.id),
                'text': tweet.text,
                'created_at': tweet.created_at.isoformat() if hasattr(tweet, 'created_at') else datetime.now().isoformat(),
                'source': 'twitter_stream',
                'collected_at': datetime.now().isoformat()
            }
            
            # Add to buffer
            self.buffer.append(processed_tweet)
            
            # Send to Kafka
            if self.producer:
                self.producer.send(self.topic, processed_tweet)
            
            # Flush buffer if needed
            now = time.time()
            if len(self.buffer) >= self.buffer_size or (now - self.last_flush) > self.flush_interval:
                self.flush_buffer()
                
        except Exception as e:
            logger.error(f"Error processing tweet: {e}")
    
    def flush_buffer(self):
        """Flush the buffer to disk"""
        if not self.buffer:
            return
        
        try:
            # Save to CSV
            now = datetime.now().strftime('%Y%m%d_%H%M%S')
            tweets_df = pd.DataFrame(self.buffer)
            os.makedirs(os.path.join(data_dir, 'twitter_stream'), exist_ok=True)
            tweets_df.to_csv(os.path.join(data_dir, 'twitter_stream', f'stream_{now}.csv'), index=False)
            
            # Clear buffer
            buffer_size = len(self.buffer)
            self.buffer = []
            self.last_flush = time.time()
            
            logger.info(f"Flushed {buffer_size} tweets to CSV")
        except Exception as e:
            logger.error(f"Error flushing buffer: {e}")
    
    def on_error(self, status):
        logger.error(f"Stream error: {status}")

def setup_stream(config):
    """Set up the Twitter stream"""
    twitter_config = config.get('twitter', {})
    kafka_config = config.get('kafka', {})
    streaming_config = config.get('streaming', {})
    
    # Set up Kafka producer
    bootstrap_servers = kafka_config.get('bootstrap_servers', 'localhost:29092')
    topic = kafka_config.get('input_topics', ['raw_twitter_posts'])[0]
    
    try:
        producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda x: json.dumps(x).encode('utf-8')
        )
        logger.info(f"Connected to Kafka at {bootstrap_servers}")
    except Exception as e:
        logger.error(f"Failed to connect to Kafka: {e}")
        producer = None
    
    # Set up stream
    try:
        stream = TwitterStreamListener(
            twitter_config.get('bearer_token'),
            producer,
            topic
        )
        
        # Add rules for keywords
        existing_rules = stream.get_rules()
        if existing_rules.data:
            rule_ids = [rule.id for rule in existing_rules.data]
            stream.delete_rules(rule_ids)
        
        # Add finance keywords
        finance_keywords = " OR ".join(streaming_config.get('keywords', {}).get('finance', []))
        if finance_keywords:
            stream.add_rules(tweepy.StreamRule(finance_keywords, tag='finance'))
        
        # Add tech keywords
        tech_keywords = " OR ".join(streaming_config.get('keywords', {}).get('technology', []))
        if tech_keywords:
            stream.add_rules(tweepy.StreamRule(tech_keywords, tag='technology'))
        
        logger.info("Twitter stream set up with rules")
        return stream
    except Exception as e:
        logger.error(f"Error setting up Twitter stream: {e}")
        return None

def run_collection_service():
    """Run the Twitter collection service"""
    # Load configuration
    config = load_config()
    if not config:
        logger.error("Failed to load configuration, exiting")
        return
    
    # Check if Twitter API keys are set
    twitter_config = config.get('twitter', {})
    if (twitter_config.get('bearer_token') == "YOUR_BEARER_TOKEN" or
        twitter_config.get('consumer_key') == "YOUR_CONSUMER_KEY"):
        logger.error("Twitter API keys not configured. Please edit the config file and restart")
        return
    
    # Set up collector
    collector = TwitterCollector(config)
    
    # Set up stream
    stream = setup_stream(config)
    
    # Start stream in a separate thread
    if stream:
        stream_thread = threading.Thread(target=stream.filter, kwargs={"tweet_fields": ["created_at"]})
        stream_thread.daemon = True
        stream_thread.start()
        logger.info("Twitter stream started")
    
    # Run periodic collection in the main thread
    try:
        while True:
            tweets_collected = collector.run_collection_cycle()
            logger.info(f"Collection cycle completed with {tweets_collected} tweets")
            
            # Wait for next cycle (1 hour)
            logger.info("Waiting for next collection cycle (1 hour)")
            time.sleep(3600)
    except KeyboardInterrupt:
        logger.info("Collection service interrupted by user")
    except Exception as e:
        logger.error(f"Error in collection service: {e}")
    finally:
        # Clean up
        if stream:
            stream.disconnect()
        if collector.producer:
            collector.producer.close()
        logger.info("Collection service stopped")

if __name__ == "__main__":
    run_collection_service() 