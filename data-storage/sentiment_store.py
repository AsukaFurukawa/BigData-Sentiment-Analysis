import json
import os
import time
import pandas as pd
from kafka import KafkaConsumer
from datetime import datetime
import sqlite3
import threading

# Path to current script
base_path = os.path.dirname(__file__)
config_path = os.path.join(os.path.dirname(base_path), 'data-ingestion', 'config', 'keys.json')
data_dir = os.path.join(base_path, 'data')

# Create data directory if it doesn't exist
os.makedirs(data_dir, exist_ok=True)

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
sentiment_topic = "enhanced-sentiment-results"

# SQLite DB setup
db_path = os.path.join(data_dir, 'sentiment_data.db')
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Create SQLite tables if they don't exist
def setup_database():
    # Main sentiment results table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS sentiment_results (
        id TEXT PRIMARY KEY,
        source TEXT,
        user_id TEXT,
        created_at TEXT,
        text TEXT,
        domain TEXT,
        overall_sentiment TEXT,
        processed_at TEXT
    )
    ''')
    
    # Finance analysis table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS finance_analysis (
        id TEXT PRIMARY KEY,
        sentiment TEXT,
        finance_score REAL,
        entities TEXT,
        positive_terms TEXT,
        negative_terms TEXT,
        FOREIGN KEY (id) REFERENCES sentiment_results (id)
    )
    ''')
    
    # Tech analysis table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS tech_analysis (
        id TEXT PRIMARY KEY,
        sentiment TEXT,
        tech_score REAL,
        entities TEXT,
        categories TEXT,
        positive_terms TEXT,
        negative_terms TEXT,
        FOREIGN KEY (id) REFERENCES sentiment_results (id)
    )
    ''')
    
    conn.commit()

# Set up CSV files for batch exports
def get_date_partitioned_path(domain, file_type='csv'):
    today = datetime.now().strftime('%Y-%m-%d')
    filename = f"{domain}_sentiment_{today}.{file_type}"
    return os.path.join(data_dir, filename)

# In-memory dataframes for batch processing
all_results = []
finance_results = []
tech_results = []
batch_size = 100  # Number of records to process before writing to disk

# Process a sentiment analysis result
def process_result(result):
    try:
        # Extract main fields
        result_id = result.get('id', '')
        source = result.get('source', '')
        user_id = result.get('user_id', '')
        created_at = result.get('created_at', '')
        text = result.get('text', '')
        domain = result.get('domain', 'general')
        overall_sentiment = result.get('overall_sentiment', 'neutral')
        processed_at = result.get('processed_at', datetime.now().isoformat())
        
        # Insert into main sentiment results table
        cursor.execute('''
        INSERT OR REPLACE INTO sentiment_results 
        (id, source, user_id, created_at, text, domain, overall_sentiment, processed_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (result_id, source, user_id, created_at, text, domain, overall_sentiment, processed_at))
        
        # Process finance analysis if available
        if 'finance_analysis' in result:
            finance = result['finance_analysis']
            finance_sentiment = finance.get('sentiment', 'neutral')
            finance_score = finance.get('finance_score', 0.0)
            finance_entities = json.dumps(finance.get('entities', []))
            finance_pos = json.dumps(finance.get('positive_terms', []))
            finance_neg = json.dumps(finance.get('negative_terms', []))
            
            cursor.execute('''
            INSERT OR REPLACE INTO finance_analysis 
            (id, sentiment, finance_score, entities, positive_terms, negative_terms)
            VALUES (?, ?, ?, ?, ?, ?)
            ''', (result_id, finance_sentiment, finance_score, finance_entities, finance_pos, finance_neg))
            
            # Add to finance results for CSV export
            finance_results.append({
                'id': result_id,
                'source': source,
                'created_at': created_at,
                'domain': domain,
                'sentiment': finance_sentiment,
                'score': finance_score,
                'entities': finance_entities
            })
        
        # Process tech analysis if available
        if 'tech_analysis' in result:
            tech = result['tech_analysis']
            tech_sentiment = tech.get('sentiment', 'neutral')
            tech_score = tech.get('tech_score', 0.0)
            tech_entities = json.dumps(tech.get('entities', []))
            tech_categories = json.dumps(tech.get('categories', {}))
            tech_pos = json.dumps(tech.get('positive_terms', []))
            tech_neg = json.dumps(tech.get('negative_terms', []))
            
            cursor.execute('''
            INSERT OR REPLACE INTO tech_analysis 
            (id, sentiment, tech_score, entities, categories, positive_terms, negative_terms)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (result_id, tech_sentiment, tech_score, tech_entities, tech_categories, tech_pos, tech_neg))
            
            # Add to tech results for CSV export
            tech_results.append({
                'id': result_id,
                'source': source,
                'created_at': created_at,
                'domain': domain,
                'sentiment': tech_sentiment,
                'score': tech_score,
                'entities': tech_entities,
                'categories': tech_categories
            })
        
        # Add to all results for CSV export
        all_results.append({
            'id': result_id,
            'source': source,
            'created_at': created_at,
            'text': text,
            'domain': domain,
            'sentiment': overall_sentiment,
            'processed_at': processed_at
        })
        
        # Commit the database transaction
        conn.commit()
        
    except Exception as e:
        print(f"Error processing result for storage: {e}")
        conn.rollback()

# Batch write results to CSV files
def batch_write_to_csv():
    if all_results:
        all_df = pd.DataFrame(all_results)
        all_df.to_csv(get_date_partitioned_path('all'), index=False)
        
    if finance_results:
        finance_df = pd.DataFrame(finance_results)
        finance_df.to_csv(get_date_partitioned_path('finance'), index=False)
        
    if tech_results:
        tech_df = pd.DataFrame(tech_results)
        tech_df.to_csv(get_date_partitioned_path('tech'), index=False)
    
    print(f"Batch data written to CSV: {len(all_results)} records")

# Periodic batch write thread
def periodic_batch_writer():
    while True:
        time.sleep(300)  # Every 5 minutes
        batch_write_to_csv()

# Initialize Kafka consumer
consumer = KafkaConsumer(
    sentiment_topic,
    bootstrap_servers=kafka_bootstrap_servers,
    value_deserializer=lambda m: json.loads(m.decode('utf-8')),
    auto_offset_reset='latest',
    group_id='sentiment-storage-group'
)

def start_storage_service():
    """Start consuming enhanced sentiment results and storing them"""
    print(f"Starting sentiment storage service. Listening to topic: {sentiment_topic}")
    print(f"Data will be stored in: {data_dir}")
    
    # Set up database
    setup_database()
    
    # Start periodic batch writer thread
    batch_thread = threading.Thread(target=periodic_batch_writer, daemon=True)
    batch_thread.start()
    
    # Process messages
    records_processed = 0
    for message in consumer:
        result = message.value
        process_result(result)
        records_processed += 1
        
        # Check if we need to do a batch write
        if records_processed % batch_size == 0:
            batch_write_to_csv()
            
            # Clear the lists after writing to avoid memory issues
            all_results.clear()
            finance_results.clear()
            tech_results.clear()
            
            print(f"Processed and stored {records_processed} records")

if __name__ == "__main__":
    start_storage_service() 