import praw
import json
import os
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
            "user_agent": "linux:Sentiment Analysis Dashboard:v1.0 (by /u/TraditionalFact4215)"
        },
        "kafka": {
            "bootstrap_servers": ["localhost:9092"],
            "reddit_topic": "reddit-raw-data"
        },
        "subreddits": ["wallstreetbets", "stocks", "investing", "cryptocurrency"]
    }
    
    with open(keys_path, 'w', encoding='utf-8') as f:
        json.dump(default_config, f, indent=2)
    print(f"Created default config file at {keys_path}")
    print("Please update with your actual API credentials before running again.")
    exit(1)
except json.JSONDecodeError as e:
    print(f"JSON parsing error: {e}")
    exit(1)
except Exception as e:
    print(f"Unknown error: {e}")
    exit(1)

# Access the Reddit API credentials
reddit_client_id = config['reddit']['client_id']
reddit_client_secret = config['reddit']['client_secret']
reddit_user_agent = config['reddit']['user_agent']

# Set up the Reddit API client
reddit = praw.Reddit(
    client_id=reddit_client_id,
    client_secret=reddit_client_secret,
    user_agent=reddit_user_agent
)

# Access Kafka configuration
kafka_bootstrap_servers = config['kafka']['bootstrap_servers']
kafka_reddit_topic = config['kafka']['reddit_topic']

# Initialize Kafka producer
producer = KafkaProducer(bootstrap_servers=kafka_bootstrap_servers)

# Example of reading posts from subreddits and sending to Kafka
subreddits = config['subreddits']
for subreddit_name in subreddits:
    subreddit = reddit.subreddit(subreddit_name)
    for submission in subreddit.top(limit=5):  # Example: top 5 posts
        print(f"Title: {submission.title}, Score: {submission.score}")
        # Send the post's data to Kafka
        message = f"Title: {submission.title}, Score: {submission.score}"
        producer.send(kafka_reddit_topic, value=message.encode('utf-8'))

# Close Kafka producer connection
producer.close()