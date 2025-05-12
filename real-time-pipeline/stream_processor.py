import os
import sys
import json
import time
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from kafka import KafkaConsumer, KafkaProducer
from threading import Thread
import sqlite3
import requests

# Configure logging
base_dir = os.path.dirname(os.path.dirname(__file__))  # Get project root directory
logs_dir = os.path.join(base_dir, 'logs')
os.makedirs(logs_dir, exist_ok=True)  # Create logs directory if it doesn't exist

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(logs_dir, 'stream_processor.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("stream_processor")

# Import local modules - adjust paths as needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from advanced_models.transformer_sentiment import AdvancedSentimentAnalyzer
    from market_data.market_data_collector import fetch_yfinance_data
except ImportError:
    logger.error("Error importing required modules. Check module paths.")
    sys.exit(1)

class RealTimeSentimentProcessor:
    """Process sentiment data in real-time using Kafka streams"""
    
    def __init__(self):
        """Initialize the real-time processor"""
        # Paths
        self.base_path = os.path.dirname(__file__)
        self.project_path = os.path.dirname(self.base_path)
        self.config_path = os.path.join(self.project_path, 'data-ingestion', 'config', 'keys.json')
        self.alert_history_path = os.path.join(self.base_path, 'alert_history.json')
        
        # Create directories
        os.makedirs(self.base_path, exist_ok=True)
        
        # Load configuration
        self.load_config()
        
        # Initialize sentiment analyzer
        self.sentiment_analyzer = AdvancedSentimentAnalyzer()
        
        # Initialize Kafka
        self.setup_kafka()
        
        # Alert settings
        self.alert_cooldown = 15 * 60  # 15 minutes in seconds
        self.last_alert_time = {}
        self.load_alert_history()
        
        # Tracking sentiment data
        self.sentiment_buffer = {
            "finance": [],
            "technology": [],
            "general": []
        }
        self.buffer_max_size = 1000
        self.buffer_window = 30 * 60  # 30 minutes in seconds
        
        # Database connection for persistent storage
        self.db_path = os.path.join(self.project_path, 'data-storage', 'data', 'sentiment_data.db')
        
        # Market data cache
        self.market_data = {}
        self.last_market_update = datetime.now() - timedelta(minutes=10)  # Force initial update
        self.market_update_interval = timedelta(minutes=5)
    
    def load_config(self):
        """Load configuration from JSON file"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
            
            # Kafka settings
            kafka_config = self.config.get('kafka', {})
            self.kafka_bootstrap_servers = kafka_config.get('bootstrap_servers', 'localhost:9092')
            self.input_topics = kafka_config.get('input_topics', ['raw_social_posts', 'raw_news'])
            self.output_topic = kafka_config.get('output_topic', 'processed_sentiment')
            self.alert_topic = kafka_config.get('alert_topic', 'sentiment_alerts')
            
            # Alert settings
            alert_config = self.config.get('alerts', {})
            self.alert_threshold = alert_config.get('threshold', 0.2)
            self.webhooks = alert_config.get('webhooks', [])
            self.alert_enabled = alert_config.get('enabled', True)
            
            # Market data settings
            market_config = self.config.get('market_data', {})
            self.tracked_symbols = market_config.get('stocks', []) + market_config.get('indices', [])
            self.correlation_lookback = market_config.get('correlation_lookback_hours', 24)
            
            logger.info("Configuration loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            # Default values
            self.kafka_bootstrap_servers = 'localhost:9092'
            self.input_topics = ['raw_social_posts', 'raw_news']
            self.output_topic = 'processed_sentiment'
            self.alert_topic = 'sentiment_alerts'
            self.alert_threshold = 0.2
            self.webhooks = []
            self.alert_enabled = True
            self.tracked_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "^GSPC", "^DJI"]
            self.correlation_lookback = 24
            self.config = {}
    
    def setup_kafka(self):
        """Initialize Kafka consumer and producer"""
        try:
            # Create Kafka consumer for input topics
            self.consumer = KafkaConsumer(
                *self.input_topics,
                bootstrap_servers=self.kafka_bootstrap_servers,
                auto_offset_reset='latest',
                enable_auto_commit=True,
                group_id='sentiment-processor',
                value_deserializer=lambda x: json.loads(x.decode('utf-8'))
            )
            
            # Create Kafka producer for output topic
            self.producer = KafkaProducer(
                bootstrap_servers=self.kafka_bootstrap_servers,
                value_serializer=lambda x: json.dumps(x).encode('utf-8')
            )
            
            logger.info("Kafka consumer and producer initialized")
            
        except Exception as e:
            logger.error(f"Error setting up Kafka: {e}")
            raise
    
    def load_alert_history(self):
        """Load alert history to prevent duplicate alerts"""
        try:
            if os.path.exists(self.alert_history_path):
                with open(self.alert_history_path, 'r', encoding='utf-8') as f:
                    self.last_alert_time = json.load(f)
            else:
                self.last_alert_time = {}
                
        except Exception as e:
            logger.error(f"Error loading alert history: {e}")
            self.last_alert_time = {}
    
    def save_alert_history(self):
        """Save alert history to file"""
        try:
            with open(self.alert_history_path, 'w', encoding='utf-8') as f:
                json.dump(self.last_alert_time, f)
                
        except Exception as e:
            logger.error(f"Error saving alert history: {e}")
    
    def process_message(self, message):
        """Process a message from Kafka"""
        try:
            # Extract message content
            content = message.value
            topic = message.topic
            
            # Prepare a base record
            record = {
                "id": content.get('id', str(time.time())),
                "source": content.get('source', topic),
                "created_at": content.get('created_at', datetime.now().isoformat()),
                "user_id": content.get('user_id', ''),
                "text": content.get('text', ''),
                "processing_time": datetime.now().isoformat()
            }
            
            # Determine domain based on message metadata or content
            domain = content.get('domain', 'general')
            if 'finance' in content.get('categories', []) or 'finance' in topic:
                domain = 'finance'
            elif 'tech' in content.get('categories', []) or 'technology' in topic:
                domain = 'technology'
            
            record["domain"] = domain
            
            # Run sentiment analysis
            sentiment_result = self.sentiment_analyzer.process(record["text"], domain)
            record.update(sentiment_result)
            
            # Store in buffer for trend analysis
            self.update_sentiment_buffer(record)
            
            # Save to database
            self.store_sentiment(record)
            
            # Send processed result to output topic
            self.producer.send(self.output_topic, record)
            
            # Check for significant sentiment changes
            self.check_for_alerts(domain)
            
            logger.debug(f"Processed message: {record['id']}")
            return record
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return None
    
    def update_sentiment_buffer(self, record):
        """Update the sentiment buffer with a new record"""
        try:
            domain = record.get('domain', 'general')
            
            # Add timestamp for buffer management
            record['timestamp'] = datetime.now()
            
            # Add to appropriate buffer
            if domain in self.sentiment_buffer:
                self.sentiment_buffer[domain].append(record)
            else:
                self.sentiment_buffer['general'].append(record)
            
            # Clean up old entries
            self.clean_sentiment_buffer()
            
        except Exception as e:
            logger.error(f"Error updating sentiment buffer: {e}")
    
    def clean_sentiment_buffer(self):
        """Remove old entries from the sentiment buffer"""
        try:
            now = datetime.now()
            cutoff_time = now - timedelta(seconds=self.buffer_window)
            
            for domain in self.sentiment_buffer:
                # Remove old entries
                self.sentiment_buffer[domain] = [
                    r for r in self.sentiment_buffer[domain] 
                    if r['timestamp'] > cutoff_time
                ]
                
                # Trim to max size
                if len(self.sentiment_buffer[domain]) > self.buffer_max_size:
                    self.sentiment_buffer[domain] = self.sentiment_buffer[domain][-self.buffer_max_size:]
        
        except Exception as e:
            logger.error(f"Error cleaning sentiment buffer: {e}")
    
    def get_domain_sentiment_trend(self, domain, minutes=15):
        """Calculate sentiment trend for a domain over specified minutes"""
        try:
            if domain not in self.sentiment_buffer:
                return 0, 0, 0
            
            buffer = self.sentiment_buffer[domain]
            if not buffer:
                return 0, 0, 0
            
            now = datetime.now()
            cutoff_time = now - timedelta(minutes=minutes)
            
            # Get recent and older records
            recent_records = [r for r in buffer if r['timestamp'] > cutoff_time]
            older_records = [r for r in buffer if r['timestamp'] <= cutoff_time and r['timestamp'] > cutoff_time - timedelta(minutes=minutes)]
            
            # If no data in one period, return no change
            if not recent_records or not older_records:
                current_sentiment = np.mean([r.get('score', 0) for r in recent_records]) if recent_records else 0
                return current_sentiment, 0, len(recent_records)
            
            # Calculate average sentiment for each period
            recent_sentiment = np.mean([r.get('score', 0) for r in recent_records])
            older_sentiment = np.mean([r.get('score', 0) for r in older_records])
            
            # Calculate change
            sentiment_change = recent_sentiment - older_sentiment
            
            return recent_sentiment, sentiment_change, len(recent_records)
            
        except Exception as e:
            logger.error(f"Error calculating sentiment trend: {e}")
            return 0, 0, 0
    
    def check_for_alerts(self, domain):
        """Check if sentiment changes warrant an alert"""
        if not self.alert_enabled:
            return
        
        try:
            # Get sentiment change
            current_sentiment, sentiment_change, sample_size = self.get_domain_sentiment_trend(domain)
            
            # Check if change exceeds threshold with sufficient sample size
            if abs(sentiment_change) >= self.alert_threshold and sample_size >= 5:
                alert_key = f"{domain}_{sentiment_change > 0}"
                
                # Check cooldown
                now = time.time()
                if alert_key in self.last_alert_time and now - self.last_alert_time[alert_key] < self.alert_cooldown:
                    return
                
                # Create alert
                alert = {
                    "timestamp": datetime.now().isoformat(),
                    "domain": domain,
                    "sentiment_change": sentiment_change,
                    "current_sentiment": current_sentiment,
                    "sample_size": sample_size,
                    "direction": "positive" if sentiment_change > 0 else "negative",
                    "alert_level": "high" if abs(sentiment_change) >= self.alert_threshold * 2 else "medium"
                }
                
                # Update market correlations
                self.update_market_data()
                market_correlations = self.get_market_correlations(domain)
                alert["market_correlations"] = market_correlations
                
                # Send alert to Kafka
                self.producer.send(self.alert_topic, alert)
                
                # Send to webhooks
                for webhook in self.webhooks:
                    try:
                        requests.post(webhook, json=alert, timeout=5)
                    except Exception as webhook_err:
                        logger.error(f"Error sending webhook: {webhook_err}")
                
                # Update alert history
                self.last_alert_time[alert_key] = now
                self.save_alert_history()
                
                logger.info(f"Sent sentiment alert: {domain} sentiment change of {sentiment_change:.3f}")
                
        except Exception as e:
            logger.error(f"Error checking for alerts: {e}")
    
    def update_market_data(self):
        """Update cached market data if needed"""
        now = datetime.now()
        
        # Check if update is needed
        if now - self.last_market_update < self.market_update_interval:
            return
        
        try:
            # Connect to the market data database
            market_db_path = os.path.join(self.project_path, 'market-data', 'data', 'market_data.db')
            
            if os.path.exists(market_db_path):
                conn = sqlite3.connect(market_db_path)
                
                # Fetch recent market data
                hours = self.correlation_lookback + 1
                cutoff_time = (now - timedelta(hours=hours)).strftime('%Y-%m-%d %H:%M:%S')
                symbols_str = "', '".join(self.tracked_symbols)
                
                query = f"""
                SELECT symbol, timestamp, close
                FROM market_data
                WHERE symbol IN ('{symbols_str}')
                AND timestamp > '{cutoff_time}'
                ORDER BY symbol, timestamp
                """
                
                # Load data
                self.market_data = pd.read_sql_query(query, conn)
                conn.close()
                
                # Update timestamp
                self.last_market_update = now
                
                logger.info(f"Updated market data cache with {len(self.market_data)} records")
                
            else:
                # Fallback to direct API call
                logger.warning("Market database not found, fetching data directly")
                fetch_yfinance_data(self.tracked_symbols, period="2d", interval="1h")
                
        except Exception as e:
            logger.error(f"Error updating market data: {e}")
    
    def get_market_correlations(self, domain):
        """Calculate correlations between sentiment and market movements"""
        try:
            if not isinstance(self.market_data, pd.DataFrame) or self.market_data.empty:
                return {}
            
            # Get hourly sentiment for the domain
            sentiment_data = pd.DataFrame([
                {
                    'timestamp': r['timestamp'].strftime('%Y-%m-%d %H:00:00'),
                    'sentiment': r.get('score', 0)
                }
                for r in self.sentiment_buffer[domain]
            ])
            
            if sentiment_data.empty:
                return {}
            
            # Group by hour and average
            sentiment_data['timestamp'] = pd.to_datetime(sentiment_data['timestamp'])
            hourly_sentiment = sentiment_data.groupby('timestamp')['sentiment'].mean().reset_index()
            
            # Format market data
            market_df = self.market_data.copy()
            market_df['timestamp'] = pd.to_datetime(market_df['timestamp'])
            
            # Calculate hourly returns
            market_df['hour'] = market_df['timestamp'].dt.floor('H')
            hourly_market = market_df.groupby(['symbol', 'hour'])['close'].last().reset_index()
            hourly_market = hourly_market.pivot(index='hour', columns='symbol', values='close')
            
            # Calculate returns
            market_returns = hourly_market.pct_change().dropna()
            
            # Merge sentiment and market data on timestamp
            merged_data = pd.merge(
                hourly_sentiment, 
                market_returns.reset_index(), 
                left_on='timestamp', 
                right_on='hour', 
                how='inner'
            )
            
            if len(merged_data) < 3:
                return {}
            
            # Calculate correlations
            correlations = {}
            for symbol in self.tracked_symbols:
                if symbol in merged_data.columns:
                    correlation = merged_data['sentiment'].corr(merged_data[symbol])
                    if not pd.isna(correlation):
                        correlations[symbol] = round(correlation, 3)
            
            return correlations
            
        except Exception as e:
            logger.error(f"Error calculating market correlations: {e}")
            return {}
    
    def store_sentiment(self, record):
        """Store sentiment record in the database"""
        try:
            # Check if database exists
            if not os.path.exists(self.db_path):
                logger.warning(f"Database not found at {self.db_path}, creating")
                
                # Ensure directory exists
                os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
                
                # Create database and tables
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Create sentiment_results table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS sentiment_results (
                    id TEXT PRIMARY KEY,
                    source TEXT,
                    user_id TEXT,
                    created_at TEXT,
                    text TEXT,
                    domain TEXT,
                    overall_sentiment TEXT,
                    score REAL,
                    confidence TEXT,
                    model TEXT,
                    processing_time TEXT
                )
                ''')
                
                conn.commit()
                conn.close()
            
            # Connect to database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Insert record
            cursor.execute('''
            INSERT OR REPLACE INTO sentiment_results
            (id, source, user_id, created_at, text, domain, overall_sentiment, 
            score, confidence, model, processing_time)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                record.get('id', ''),
                record.get('source', ''),
                record.get('user_id', ''),
                record.get('created_at', ''),
                record.get('text', ''),
                record.get('domain', 'general'),
                record.get('sentiment', 'neutral'),
                record.get('score', 0.5),
                record.get('confidence', 'medium'),
                record.get('model', 'default'),
                record.get('processing_time', datetime.now().isoformat())
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing sentiment record: {e}")
    
    def run(self):
        """Main method to process messages from Kafka"""
        logger.info("Starting real-time sentiment processor")
        
        try:
            # Start processing loop
            for message in self.consumer:
                self.process_message(message)
                
        except KeyboardInterrupt:
            logger.info("Processor stopped by user")
        except Exception as e:
            logger.error(f"Error in main processing loop: {e}")
        finally:
            # Close Kafka connections
            try:
                self.consumer.close()
                self.producer.close()
            except:
                pass
            
            logger.info("Real-time sentiment processor stopped")

if __name__ == "__main__":
    processor = RealTimeSentimentProcessor()
    processor.run() 