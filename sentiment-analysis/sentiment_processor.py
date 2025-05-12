import json
import re
import nltk
from textblob import TextBlob
from kafka import KafkaConsumer, KafkaProducer
import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np

# Download required NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# Path to current script
base_path = os.path.dirname(__file__)
config_path = os.path.join(os.path.dirname(base_path), 'data-ingestion', 'config', 'keys.json')

print(f"Looking for config file at: {config_path}")
print(f"File exists: {os.path.isfile(config_path)}")

try:
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
except Exception as e:
    print(f"Error loading config: {e}")
    exit(1)

# Kafka configuration
kafka_bootstrap_servers = config['kafka']['bootstrap_servers']
twitter_raw_topic = config['kafka']['twitter_topic']
reddit_raw_topic = config['kafka']['reddit_topic']
sentiment_topic = "sentiment-analysis-results"

# Initialize Kafka consumer and producer
consumer = KafkaConsumer(
    twitter_raw_topic, 
    reddit_raw_topic,
    bootstrap_servers=kafka_bootstrap_servers,
    value_deserializer=lambda m: json.loads(m.decode('utf-8')),
    auto_offset_reset='latest',
    group_id='sentiment-processor-group'
)

producer = KafkaProducer(
    bootstrap_servers=kafka_bootstrap_servers,
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# Load finance and technology specific terms
finance_terms = config['keywords']['finance']
tech_terms = config['keywords']['technology']

# Custom stopwords - words that don't contribute to sentiment
custom_stopwords = stopwords.words('english')
custom_stopwords.extend(['rt', 'http', 'https', 'amp', 'com', 'www'])

# Function to clean text
def clean_text(text):
    """Clean the text by removing URLs, mentions, hashtags, and non-alphanumeric characters"""
    # Convert to string if not already
    if not isinstance(text, str):
        return ""
        
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove user mentions and hashtags
    text = re.sub(r'@\w+|\#\w+', '', text)
    
    # Remove non-alphanumeric characters
    text = re.sub(r'[^\w\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Function to analyze general sentiment
def analyze_general_sentiment(text):
    """Analyze sentiment using TextBlob"""
    cleaned_text = clean_text(text)
    analysis = TextBlob(cleaned_text)
    
    # Get polarity score (-1.0 to 1.0)
    polarity = analysis.sentiment.polarity
    
    # Get subjectivity score (0.0 to 1.0)
    subjectivity = analysis.sentiment.subjectivity
    
    # Categorize sentiment
    if polarity > 0.1:
        sentiment = "positive"
    elif polarity < -0.1:
        sentiment = "negative"
    else:
        sentiment = "neutral"
        
    return {
        "polarity": polarity,
        "subjectivity": subjectivity,
        "sentiment": sentiment
    }

# Function to identify domain-specific sentiment modifiers
def identify_domain_terms(text, domain):
    """Identify domain-specific terms in the text and their influence on sentiment"""
    cleaned_text = clean_text(text.lower())
    tokens = word_tokenize(cleaned_text)
    
    # Filter out stopwords
    filtered_tokens = [w for w in tokens if w not in custom_stopwords]
    
    # Domain-specific terms
    if domain == "finance":
        domain_terms = finance_terms
    elif domain == "technology":
        domain_terms = tech_terms
    else:
        domain_terms = finance_terms + tech_terms
    
    # Flatten and lowercase domain terms
    flat_domain_terms = [term.lower() for term_list in domain_terms for term in term_list.split()]
    
    # Find matches
    matches = [token for token in filtered_tokens if token in flat_domain_terms]
    
    return {
        "domain": domain,
        "domain_terms_found": list(set(matches)),
        "domain_terms_count": len(list(set(matches)))
    }

# Main function to process messages
def process_message(message):
    """Process a message from Kafka and perform sentiment analysis"""
    try:
        # Determine source (Twitter or Reddit)
        topic = message.topic
        source = "twitter" if topic == twitter_raw_topic else "reddit"
        
        # Extract text based on source
        if source == "twitter":
            text = message.value.get('text', '')
            domain = message.value.get('domain', 'general')
            created_at = message.value.get('created_at', '')
            id = message.value.get('id', '')
        else:  # Reddit
            # Assuming Reddit messages come in a different format
            text = message.value.get('title', '') + ' ' + message.value.get('selftext', '')
            domain = 'general'  # Default, could be determined by subreddit
            created_at = message.value.get('created_utc', '')
            id = message.value.get('id', '')
        
        # Perform sentiment analysis
        sentiment_results = analyze_general_sentiment(text)
        
        # Identify domain-specific terms
        domain_info = identify_domain_terms(text, domain)
        
        # Combine results
        result = {
            "source": source,
            "id": id,
            "created_at": created_at,
            "text": text,
            "sentiment": sentiment_results,
            "domain_analysis": domain_info,
            "processed_at": pd.Timestamp.now().isoformat()
        }
        
        # Send results to Kafka
        producer.send(sentiment_topic, result)
        print(f"Processed {source} message: {id}")
        
        return result
        
    except Exception as e:
        print(f"Error processing message: {e}")
        return None

# Main loop to consume messages
def start_sentiment_processing():
    print(f"Starting sentiment analysis processor. Listening to topics: {twitter_raw_topic}, {reddit_raw_topic}")
    
    for message in consumer:
        process_message(message)

if __name__ == "__main__":
    start_sentiment_processing() 