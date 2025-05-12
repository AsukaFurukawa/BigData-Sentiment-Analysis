import tweepy
import json
import os
import time
from kafka import KafkaProducer

# Path to current script
base_path = os.path.dirname(__file__)
keys_path = os.path.join(base_path, 'config', 'keys.json')

# Debug prints to verify file path and existence
print(f"Looking for config file at: {keys_path}")
print(f"File exists: {os.path.isfile(keys_path)}")

try:
    # Try reading the raw content of the file before parsing
    with open(keys_path, 'r', encoding='utf-8') as f:
        content = f.read()
        print(f"First 100 chars of file content: {content[:100]}")
        
        # Reset file position to beginning
        f.seek(0)
        config = json.load(f)
except FileNotFoundError:
    print(f"File not found: {keys_path}")
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(keys_path), exist_ok=True)
    
    # Create a default config file
    default_config = {
        "reddit": {
            "client_id": "YOUR_CLIENT_ID",
            "client_secret": "YOUR_CLIENT_SECRET",
            "user_agent": "linux:Sentiment Analysis Dashboard:v1.0 (by /u/YourUsername)"
        },
        "twitter": {
            "bearer_token": "YOUR_BEARER_TOKEN",
            "consumer_key": "YOUR_CONSUMER_KEY",
            "consumer_secret": "YOUR_CONSUMER_SECRET",
            "access_token": "YOUR_ACCESS_TOKEN",
            "access_token_secret": "YOUR_ACCESS_TOKEN_SECRET"
        },
        "kafka": {
            "bootstrap_servers": ["localhost:9092"],
            "reddit_topic": "reddit-raw-data",
            "twitter_topic": "twitter-raw-data"
        },
        "keywords": {
            "finance": ["stock market", "investment", "trading", "nasdaq", "dow jones", "s&p 500", 
                       "fintech", "banking", "cryptocurrency", "bitcoin", "ethereum", "financial report"],
            "technology": ["artificial intelligence", "machine learning", "blockchain", "cloud computing", 
                          "cybersecurity", "big data", "tech startup", "product launch", "technology innovation"]
        }
    }
    
    with open(keys_path, 'w', encoding='utf-8') as f:
        json.dump(default_config, f, indent=2)
    print(f"Created default config file at {keys_path}")
    print("Please update with your actual Twitter API credentials before running again.")
    exit(1)
except json.JSONDecodeError as e:
    print(f"JSON parsing error: {e}")
    exit(1)
except Exception as e:
    print(f"Unknown error: {e}")
    exit(1)

# Twitter API credentials
bearer_token = config['twitter']['bearer_token']
consumer_key = config['twitter']['consumer_key']
consumer_secret = config['twitter']['consumer_secret']
access_token = config['twitter']['access_token']
access_token_secret = config['twitter']['access_token_secret']

# Kafka configuration
kafka_bootstrap_servers = config['kafka']['bootstrap_servers']
kafka_twitter_topic = config['kafka']['twitter_topic']

# Initialize Kafka producer
producer = KafkaProducer(
    bootstrap_servers=kafka_bootstrap_servers,
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# Define keywords to track
finance_keywords = config['keywords']['finance']
technology_keywords = config['keywords']['technology']
all_keywords = finance_keywords + technology_keywords

# Twitter Stream Listener
class TwitterStreamListener(tweepy.StreamingClient):
    def on_tweet(self, tweet):
        # Process the tweet data
        tweet_data = {
            'id': tweet.id,
            'created_at': tweet.created_at.isoformat() if hasattr(tweet, 'created_at') else None,
            'text': tweet.text,
            'user_id': tweet.author_id,
            'domain': self._classify_domain(tweet.text),
            'raw_data': tweet.data
        }
        
        # Send to Kafka
        producer.send(kafka_twitter_topic, tweet_data)
        print(f"Tweet sent to Kafka: {tweet.text[:50]}...")
    
    def on_error(self, status):
        print(f"Error: {status}")
        if status == 420:  # Rate limit
            return False
    
    def _classify_domain(self, text):
        """Classify tweet into finance or technology domain based on keywords"""
        text_lower = text.lower()
        
        # Check for finance keywords
        for keyword in finance_keywords:
            if keyword.lower() in text_lower:
                return "finance"
                
        # Check for technology keywords
        for keyword in technology_keywords:
            if keyword.lower() in text_lower:
                return "technology"
                
        # Default to general if no specific domain found
        return "general"

def start_twitter_stream():
    """Initialize and start the Twitter stream"""
    try:
        # Initialize stream
        stream = TwitterStreamListener(bearer_token)
        
        # Clear existing rules
        rules = stream.get_rules()
        if rules.data:
            rule_ids = [rule.id for rule in rules.data]
            stream.delete_rules(rule_ids)
        
        # Add new filter rules
        for keyword in all_keywords:
            stream.add_rules(tweepy.StreamRule(keyword))
        
        # Start streaming
        print(f"Starting Twitter stream with keywords: {all_keywords}")
        stream.filter(tweet_fields=['created_at', 'text', 'author_id', 'public_metrics'])
        
    except Exception as e:
        print(f"Error in Twitter stream: {e}")
        time.sleep(5)  # Wait before reconnecting
        start_twitter_stream()  # Restart stream

if __name__ == "__main__":
    start_twitter_stream()
