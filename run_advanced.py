#!/usr/bin/env python3
import os
import sys
import argparse
import subprocess
import logging
import json
from datetime import datetime, timedelta
import time
import threading

# Configure logging - create logs directory and store log file there
os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs'), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs', 'advanced_runner.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("advanced_runner")

# Get the base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_config():
    """Load configuration from config file"""
    try:
        config_path = os.path.join(BASE_DIR, 'data-ingestion', 'config', 'data_sources.json')
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return {}

def run_script(script_path, args=None):
    """Run a Python script with arguments"""
    try:
        cmd = [sys.executable, script_path]
        if args:
            cmd.extend(args)
        
        logger.info(f"Running: {' '.join(cmd)}")
        
        # Run the script
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Capture output
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            logger.error(f"Script failed with return code {process.returncode}")
            logger.error(f"Stderr: {stderr}")
            return False
        
        logger.info(f"Script completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error running script: {e}")
        return False

def run_market_data_collection():
    """Run market data collection"""
    script_path = os.path.join(BASE_DIR, 'market-data', 'market_data_collector.py')
    return run_script(script_path)

def run_correlation_analysis(days=30):
    """Run correlation analysis between sentiment and market data"""
    script_path = os.path.join(BASE_DIR, 'market-data', 'correlation_analyzer.py')
    return run_script(script_path, [f"--days={days}"])

def run_entity_analysis(days=30, domain=None):
    """Run entity relationship analysis"""
    script_path = os.path.join(BASE_DIR, 'entity-analysis', 'entity_graph.py')
    
    args = [f"--days={days}"]
    if domain:
        args.append(f"--domain={domain}")
    
    return run_script(script_path, args)

def run_spark_processing(days=30):
    """Run Spark big data processing"""
    script_path = os.path.join(BASE_DIR, 'spark-processing', 'spark_processor.py')
    return run_script(script_path, [str(days)])

def run_sentiment_backtest(symbol, domain, start_date, end_date, strategy="threshold", optimize=False):
    """Run backtesting for sentiment-based trading strategies"""
    script_path = os.path.join(BASE_DIR, 'backtesting', 'sentiment_backtest.py')
    
    args = [
        f"--symbol={symbol}",
        f"--domain={domain}",
        f"--start-date={start_date}",
        f"--end-date={end_date}",
        f"--strategy={strategy}"
    ]
    
    if optimize:
        args.append("--optimize")
    
    return run_script(script_path, args)

def run_streamlit_dashboard():
    """Run the Streamlit dashboard"""
    script_path = os.path.join(BASE_DIR, 'dashboard', 'streamlit_app.py')
    return run_script(script_path)

def run_kafka_stream_processor():
    """Run the Kafka stream processor"""
    script_path = os.path.join(BASE_DIR, 'real-time-pipeline', 'stream_processor.py')
    return run_script(script_path)

def run_kafka_setup():
    """Run the Kafka and Zookeeper via Docker"""
    logger.info("Starting Kafka and Zookeeper via Docker")
    
    try:
        script_path = os.path.join(BASE_DIR, 'kafka-setup')
        compose_path = os.path.join(script_path, 'docker-compose.yml')
        
        if not os.path.exists(compose_path):
            logger.error(f"Docker Compose file not found at {compose_path}")
            return False
        
        # Start Docker containers
        cmd = ["docker-compose", "-f", compose_path, "up", "-d"]
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            logger.error(f"Docker Compose failed: {stderr}")
            return False
        
        logger.info(f"Kafka and Zookeeper started via Docker")
        
        # Wait for Kafka to be ready
        time.sleep(10)
        
        # Create topics
        create_topics_script = os.path.join(script_path, 'create_topics.sh')
        if os.path.exists(create_topics_script):
            # Make the script executable
            os.chmod(create_topics_script, 0o755)
            
            # Run in Docker
            cmd = ["docker", "exec", "kafka", "bash", "-c", f"kafka-topics --create --if-not-exists --bootstrap-server kafka:9092 --replication-factor 1 --partitions 3 --topic raw_posts"]
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            stdout, stderr = process.communicate()
            
            if process.returncode != 0:
                logger.error(f"Failed to create Kafka topics: {stderr}")
                return False
            
            logger.info("Kafka topics created")
        
        return True
        
    except Exception as e:
        logger.error(f"Error starting Kafka: {e}")
        return False

def run_open_data_collector(once=False):
    """Run the open data collector"""
    logger.info("Starting open data collector")
    
    script_path = os.path.join(BASE_DIR, 'data-ingestion', 'open_data_collector.py')
    
    if not os.path.exists(script_path):
        logger.error(f"Open data collector script not found at {script_path}")
        return False
    
    args = ["--once"] if once else []
    return run_script(script_path, args)

def run_advanced_entity_analysis(days=30, domain=None):
    """Run advanced entity analytics"""
    logger.info("Starting advanced entity analytics")
    
    script_path = os.path.join(BASE_DIR, 'entity-analysis', 'advanced_entity_analytics.py')
    
    if not os.path.exists(script_path):
        logger.error(f"Advanced entity analytics script not found at {script_path}")
        return False
    
    args = [f"--days={days}"]
    if domain:
        args.append(f"--domain={domain}")
    
    return run_script(script_path, args)

def run_all_analysis(days=30):
    """Run all analysis components"""
    logger.info("Starting full analysis pipeline")
    
    # First, start Kafka and Zookeeper
    logger.info("Step 1: Setting up Kafka")
    if not run_kafka_setup():
        logger.error("Kafka setup failed, continuing with other steps")
    
    # Run market data collection
    logger.info("Step 2: Market data collection")
    if not run_market_data_collection():
        logger.error("Market data collection failed, continuing with other steps")
    
    # Start open data collector in background
    logger.info("Step 3: Starting open data collector")
    data_collector_thread = threading.Thread(target=run_open_data_collector)
    data_collector_thread.daemon = True
    data_collector_thread.start()
    
    # Run Spark processing
    logger.info("Step 4: Spark big data processing")
    if not run_spark_processing(days):
        logger.error("Spark processing failed, continuing with other steps")
    
    # Run correlation analysis
    logger.info("Step 5: Correlation analysis")
    if not run_correlation_analysis(days):
        logger.error("Correlation analysis failed, continuing with other steps")
    
    # Run entity analysis for finance
    logger.info("Step 6: Entity analysis for finance domain")
    if not run_entity_analysis(days, "finance"):
        logger.error("Entity analysis for finance failed, continuing with other steps")
    
    # Run entity analysis for technology
    logger.info("Step 7: Entity analysis for technology domain")
    if not run_entity_analysis(days, "technology"):
        logger.error("Entity analysis for technology failed, continuing with other steps")
    
    # Run advanced entity analysis
    logger.info("Step 8: Advanced entity analytics")
    if not run_advanced_entity_analysis(days, None):  # Run for all domains
        logger.error("Advanced entity analytics failed, continuing with other steps")
    
    # Run the Kafka stream processor
    logger.info("Step 9: Starting real-time stream processor")
    if not run_kafka_stream_processor():
        logger.error("Stream processor failed, continuing with other steps")
    
    # Start the dashboard
    logger.info("Step 10: Starting dashboard")
    if not run_streamlit_dashboard():
        logger.error("Dashboard failed to start")
    
    # All done
    logger.info("Analysis pipeline completed")
    return True

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Advanced Sentiment Analysis Runner")
    
    # Define command groups
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Market data collection
    market_parser = subparsers.add_parser("market", help="Collect market data")
    
    # Correlation analysis
    correlation_parser = subparsers.add_parser("correlation", help="Run correlation analysis")
    correlation_parser.add_argument("--days", type=int, default=30, help="Number of days to analyze")
    
    # Entity analysis
    entity_parser = subparsers.add_parser("entity", help="Run entity relationship analysis")
    entity_parser.add_argument("--days", type=int, default=30, help="Number of days to analyze")
    entity_parser.add_argument("--domain", type=str, choices=["finance", "technology"], help="Domain to analyze")
    
    # Advanced entity analysis
    advanced_entity_parser = subparsers.add_parser("advanced-entity", help="Run advanced entity analytics")
    advanced_entity_parser.add_argument("--days", type=int, default=30, help="Number of days to analyze")
    advanced_entity_parser.add_argument("--domain", type=str, choices=["finance", "technology"], help="Domain to analyze")
    
    # Spark processing
    spark_parser = subparsers.add_parser("spark", help="Run Spark big data processing")
    spark_parser.add_argument("--days", type=int, default=30, help="Number of days to process")
    
    # Backtesting
    backtest_parser = subparsers.add_parser("backtest", help="Run sentiment trading strategy backtesting")
    backtest_parser.add_argument("--symbol", type=str, required=True, help="Symbol to backtest (e.g., AAPL)")
    backtest_parser.add_argument("--domain", type=str, default="finance", help="Sentiment domain")
    backtest_parser.add_argument("--start-date", type=str, default=(datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'), help="Start date (YYYY-MM-DD)")
    backtest_parser.add_argument("--end-date", type=str, default=datetime.now().strftime('%Y-%m-%d'), help="End date (YYYY-MM-DD)")
    backtest_parser.add_argument("--strategy", type=str, default="threshold", choices=["threshold", "moving_average"], help="Trading strategy")
    backtest_parser.add_argument("--optimize", action="store_true", help="Run parameter optimization")
    
    # Dashboard
    dashboard_parser = subparsers.add_parser("dashboard", help="Run the interactive dashboard")
    
    # Stream processor
    stream_parser = subparsers.add_parser("stream", help="Run the Kafka stream processor")
    
    # Open data collector
    data_parser = subparsers.add_parser("collect", help="Run the open data collector")
    data_parser.add_argument("--once", action="store_true", help="Run collection once and exit")
    
    # Kafka setup
    kafka_parser = subparsers.add_parser("kafka", help="Set up Kafka and Zookeeper with Docker")
    
    # Run all analysis
    all_parser = subparsers.add_parser("all", help="Run all analysis components")
    all_parser.add_argument("--days", type=int, default=30, help="Number of days to analyze")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run the appropriate command
    if args.command == "market":
        run_market_data_collection()
    elif args.command == "correlation":
        run_correlation_analysis(args.days)
    elif args.command == "entity":
        run_entity_analysis(args.days, args.domain)
    elif args.command == "advanced-entity":
        run_advanced_entity_analysis(args.days, args.domain)
    elif args.command == "spark":
        run_spark_processing(args.days)
    elif args.command == "backtest":
        run_sentiment_backtest(args.symbol, args.domain, args.start_date, args.end_date, args.strategy, args.optimize)
    elif args.command == "dashboard":
        run_streamlit_dashboard()
    elif args.command == "stream":
        run_kafka_stream_processor()
    elif args.command == "collect":
        run_open_data_collector(args.once)
    elif args.command == "kafka":
        run_kafka_setup()
    elif args.command == "all":
        run_all_analysis(args.days)
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 