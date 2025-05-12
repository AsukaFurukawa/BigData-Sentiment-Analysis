import json
import os
import time
import pandas as pd
from kafka import KafkaConsumer, KafkaProducer
from finance_sentiment import FinanceSentimentAnalyzer
from tech_sentiment import TechSentimentAnalyzer

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
twitter_topic = config['kafka']['twitter_topic']
reddit_topic = config['kafka']['reddit_topic']
enhanced_sentiment_topic = "enhanced-sentiment-results"

# Initialize specialized sentiment analyzers
finance_analyzer = FinanceSentimentAnalyzer()
tech_analyzer = TechSentimentAnalyzer()

# Initialize Kafka consumer and producer
consumer = KafkaConsumer(
    twitter_topic,
    reddit_topic,
    bootstrap_servers=kafka_bootstrap_servers,
    value_deserializer=lambda m: json.loads(m.decode('utf-8')),
    auto_offset_reset='latest',
    group_id='enhanced-sentiment-processor-group'
)

producer = KafkaProducer(
    bootstrap_servers=kafka_bootstrap_servers,
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

def determine_domain(text, message_value):
    """Determine if the message is related to finance, technology, or both"""
    # Check if domain is already identified in the message
    if 'domain' in message_value and message_value['domain'] in ['finance', 'technology']:
        return message_value['domain']
    
    # Extract entities from both domains
    finance_entities = finance_analyzer.extract_financial_entities(text)
    tech_entities = tech_analyzer.extract_tech_entities(text)
    
    # Determine primary domain based on entity count
    if len(finance_entities) > 0 and len(tech_entities) > 0:
        if len(finance_entities) >= len(tech_entities):
            return "finance-tech"
        else:
            return "tech-finance"
    elif len(finance_entities) > 0:
        return "finance"
    elif len(tech_entities) > 0:
        return "technology"
    else:
        return "general"

def process_message(message):
    """Process a message from Kafka and perform enhanced sentiment analysis"""
    try:
        # Determine source (Twitter or Reddit)
        topic = message.topic
        source = "twitter" if topic == twitter_topic else "reddit"
        
        # Extract content from message based on source
        if source == "twitter":
            text = message.value.get('text', '')
            created_at = message.value.get('created_at', '')
            id = message.value.get('id', '')
            user_id = message.value.get('user_id', '')
        else:  # Reddit
            title = message.value.get('title', '')
            selftext = message.value.get('selftext', '')
            text = f"{title} {selftext}"
            created_at = message.value.get('created_utc', '')
            id = message.value.get('id', '')
            user_id = message.value.get('author', '')
        
        # Determine primary domain
        domain = determine_domain(text, message.value)
        
        # Perform domain-specific analysis
        if domain in ["finance", "finance-tech"]:
            # Finance is primary or co-primary domain
            finance_result = finance_analyzer.process(text)
            
            if domain == "finance-tech":
                # Also analyze technology aspects
                tech_result = tech_analyzer.process(text)
            else:
                tech_result = None
                
        elif domain in ["technology", "tech-finance"]:
            # Technology is primary or co-primary domain
            tech_result = tech_analyzer.process(text)
            
            if domain == "tech-finance":
                # Also analyze finance aspects
                finance_result = finance_analyzer.process(text)
            else:
                finance_result = None
                
        else:  # general domain
            # Perform both analyses but with lower confidence
            finance_result = finance_analyzer.process(text)
            tech_result = tech_analyzer.process(text)
        
        # Create combined result
        result = {
            "source": source,
            "id": id,
            "user_id": user_id,
            "created_at": created_at,
            "text": text,
            "domain": domain,
            "processed_at": pd.Timestamp.now().isoformat(),
        }
        
        # Add finance analysis if available
        if finance_result:
            result["finance_analysis"] = {
                "sentiment": finance_result["sentiment"],
                "entities": finance_result["entities"],
                "finance_score": finance_result["analysis"]["finance_score"],
                "positive_terms": finance_result["analysis"]["finance_pos_words"],
                "negative_terms": finance_result["analysis"]["finance_neg_words"]
            }
        
        # Add tech analysis if available
        if tech_result:
            result["tech_analysis"] = {
                "sentiment": tech_result["sentiment"],
                "entities": tech_result["entities"],
                "categories": tech_result["categories"],
                "tech_score": tech_result["analysis"]["tech_score"],
                "positive_terms": tech_result["analysis"]["tech_pos_words"],
                "negative_terms": tech_result["analysis"]["tech_neg_words"]
            }
        
        # Add overall sentiment based on domain
        if domain in ["finance", "finance-tech"]:
            result["overall_sentiment"] = finance_result["sentiment"]
        elif domain in ["technology", "tech-finance"]:
            result["overall_sentiment"] = tech_result["sentiment"]
        else:
            # For general domain, use the stronger sentiment or neutral
            finance_score = abs(finance_result["analysis"]["finance_score"]) if finance_result else 0
            tech_score = abs(tech_result["analysis"]["tech_score"]) if tech_result else 0
            
            if finance_score > tech_score:
                result["overall_sentiment"] = finance_result["sentiment"]
            elif tech_score > finance_score:
                result["overall_sentiment"] = tech_result["sentiment"]
            else:
                result["overall_sentiment"] = "neutral"
        
        # Send results to Kafka
        producer.send(enhanced_sentiment_topic, result)
        print(f"Processed {source} message {id} - Domain: {domain}, Sentiment: {result['overall_sentiment']}")
        
        return result
        
    except Exception as e:
        print(f"Error processing message: {e}")
        return None

def start_enhanced_processing():
    """Main loop to consume messages and process them"""
    print(f"Starting enhanced sentiment processor. Listening to topics: {twitter_topic}, {reddit_topic}")
    print(f"Results will be published to topic: {enhanced_sentiment_topic}")
    
    for message in consumer:
        process_message(message)

if __name__ == "__main__":
    start_enhanced_processing() 