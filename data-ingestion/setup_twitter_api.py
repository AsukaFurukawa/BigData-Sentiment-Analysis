#!/usr/bin/env python3
import os
import json
import getpass
import logging
from datetime import datetime

# Set up logging
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
logs_dir = os.path.join(base_dir, 'logs')
os.makedirs(logs_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(logs_dir, 'setup_twitter.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("setup_twitter")

def setup_twitter_api():
    """Interactive setup for Twitter API credentials"""
    print("\n=== Twitter API Setup ===")
    print("This script will help you set up your Twitter API credentials.")
    print("You need a Twitter Developer account to access the Twitter API.")
    print("Follow these steps to get your credentials:")
    print("1. Go to https://developer.twitter.com/ and sign in with your Twitter account")
    print("2. Navigate to the 'Projects & Apps' section")
    print("3. Create a new Project and App (or use an existing one)")
    print("4. Go to the 'Keys and tokens' tab")
    print("5. Copy the API Key, API Secret, Access Token, and Access Token Secret")
    print("\nYou will also need a Bearer Token for v2 API access, which can be found in the same place.")
    print("\nEnter your Twitter API credentials below:")
    
    # Config path
    config_path = os.path.join(os.path.dirname(__file__), 'config', 'keys.json')
    config_dir = os.path.dirname(config_path)
    
    # Create config directory if it doesn't exist
    os.makedirs(config_dir, exist_ok=True)
    
    # Initialize config
    config = {}
    
    # Try to load existing config
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            print("Loaded existing configuration file.")
        except Exception as e:
            logger.error(f"Error loading existing config: {e}")
    
    # Initialize Twitter config
    if 'twitter' not in config:
        config['twitter'] = {}
    
    # Get Twitter credentials
    print("\nPress Enter to keep existing values in [brackets]")
    
    bearer_token = input(f"Bearer Token [{config['twitter'].get('bearer_token', 'YOUR_BEARER_TOKEN')}]: ")
    if bearer_token:
        config['twitter']['bearer_token'] = bearer_token
    elif 'bearer_token' not in config['twitter']:
        config['twitter']['bearer_token'] = "YOUR_BEARER_TOKEN"
    
    consumer_key = input(f"API Key (Consumer Key) [{config['twitter'].get('consumer_key', 'YOUR_CONSUMER_KEY')}]: ")
    if consumer_key:
        config['twitter']['consumer_key'] = consumer_key
    elif 'consumer_key' not in config['twitter']:
        config['twitter']['consumer_key'] = "YOUR_CONSUMER_KEY"
    
    consumer_secret = input(f"API Secret (Consumer Secret) [{config['twitter'].get('consumer_secret', 'YOUR_CONSUMER_SECRET')}]: ")
    if consumer_secret:
        config['twitter']['consumer_secret'] = consumer_secret
    elif 'consumer_secret' not in config['twitter']:
        config['twitter']['consumer_secret'] = "YOUR_CONSUMER_SECRET"
    
    access_token = input(f"Access Token [{config['twitter'].get('access_token', 'YOUR_ACCESS_TOKEN')}]: ")
    if access_token:
        config['twitter']['access_token'] = access_token
    elif 'access_token' not in config['twitter']:
        config['twitter']['access_token'] = "YOUR_ACCESS_TOKEN"
    
    access_token_secret = input(f"Access Token Secret [{config['twitter'].get('access_token_secret', 'YOUR_ACCESS_TOKEN_SECRET')}]: ")
    if access_token_secret:
        config['twitter']['access_token_secret'] = access_token_secret
    elif 'access_token_secret' not in config['twitter']:
        config['twitter']['access_token_secret'] = "YOUR_ACCESS_TOKEN_SECRET"
    
    # Initialize Kafka config if not present
    if 'kafka' not in config:
        config['kafka'] = {
            "bootstrap_servers": "localhost:29092",
            "input_topics": ["raw_twitter_posts"],
            "output_topic": "processed_sentiment"
        }
    
    # Initialize streaming config if not present
    if 'streaming' not in config:
        config['streaming'] = {
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
    
    # Save the configuration
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        print(f"\nConfiguration saved to {config_path}")
        logger.info("Twitter API configuration saved")
        
        # Check if credentials are set
        if (config['twitter']['bearer_token'] != "YOUR_BEARER_TOKEN" and
            config['twitter']['consumer_key'] != "YOUR_CONSUMER_KEY"):
            print("\nTwitter API credentials have been set up successfully!")
            return True
        else:
            print("\nWarning: Some Twitter API credentials are still using placeholder values.")
            print("You need to set all credentials before you can use the Twitter API.")
            return False
            
    except Exception as e:
        logger.error(f"Error saving configuration: {e}")
        print(f"\nError saving configuration: {e}")
        return False

def setup_keywords_and_accounts():
    """Set up keywords and accounts to track"""
    print("\n=== Twitter Tracking Setup ===")
    print("Now, let's configure what to track on Twitter.")
    
    # Config path
    config_path = os.path.join(os.path.dirname(__file__), 'config', 'keys.json')
    
    # Try to load existing config
    if not os.path.exists(config_path):
        print("Please run the Twitter API setup first.")
        return False
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        print(f"Error loading configuration: {e}")
        return False
    
    # Initialize streaming config if not present
    if 'streaming' not in config:
        config['streaming'] = {
            "keywords": {
                "finance": [],
                "technology": []
            },
            "accounts": {
                "finance": [],
                "technology": []
            },
            "max_tweets_per_keyword": 100,
            "max_tweets_per_account": 50
        }
    
    # Edit finance keywords
    print("\n--- Finance Keywords ---")
    print("Enter keywords to track for finance (comma-separated):")
    print(f"Current: {', '.join(config['streaming']['keywords']['finance'])}")
    finance_keywords = input("New (or Enter to keep current): ")
    if finance_keywords:
        config['streaming']['keywords']['finance'] = [k.strip() for k in finance_keywords.split(',')]
    
    # Edit technology keywords
    print("\n--- Technology Keywords ---")
    print("Enter keywords to track for technology (comma-separated):")
    print(f"Current: {', '.join(config['streaming']['keywords']['technology'])}")
    tech_keywords = input("New (or Enter to keep current): ")
    if tech_keywords:
        config['streaming']['keywords']['technology'] = [k.strip() for k in tech_keywords.split(',')]
    
    # Edit finance accounts
    print("\n--- Finance Accounts ---")
    print("Enter Twitter accounts to track for finance (comma-separated, without @):")
    print(f"Current: {', '.join(config['streaming']['accounts']['finance'])}")
    finance_accounts = input("New (or Enter to keep current): ")
    if finance_accounts:
        config['streaming']['accounts']['finance'] = [a.strip() for a in finance_accounts.split(',')]
    
    # Edit technology accounts
    print("\n--- Technology Accounts ---")
    print("Enter Twitter accounts to track for technology (comma-separated, without @):")
    print(f"Current: {', '.join(config['streaming']['accounts']['technology'])}")
    tech_accounts = input("New (or Enter to keep current): ")
    if tech_accounts:
        config['streaming']['accounts']['technology'] = [a.strip() for a in tech_accounts.split(',')]
    
    # Save the configuration
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        print(f"\nTracking configuration saved to {config_path}")
        logger.info("Twitter tracking configuration saved")
        return True
            
    except Exception as e:
        logger.error(f"Error saving tracking configuration: {e}")
        print(f"\nError saving tracking configuration: {e}")
        return False

if __name__ == "__main__":
    if setup_twitter_api():
        setup_keywords_and_accounts()
        
        print("\n=== Setup Complete ===")
        print("You can now run the Twitter collector with:")
        print("python run_advanced.py twitter")
    else:
        print("\nSetup incomplete. Please try again and make sure to enter valid credentials.") 