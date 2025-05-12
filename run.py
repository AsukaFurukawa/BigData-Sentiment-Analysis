import os
import sys
import subprocess
import time
import signal
import argparse
from datetime import datetime

# Get the absolute path of the current script
base_dir = os.path.dirname(os.path.abspath(__file__))

# Define component paths
kafka_setup_path = os.path.join(base_dir, "kafka-setup")
data_ingestion_path = os.path.join(base_dir, "data-ingestion")
sentiment_analysis_path = os.path.join(base_dir, "sentiment-analysis")
data_storage_path = os.path.join(base_dir, "data-storage")
dashboard_path = os.path.join(base_dir, "dashboard")

# Define process groups
KAFKA_GROUP = "kafka"
PRODUCERS_GROUP = "producers"
PROCESSORS_GROUP = "processors"
STORAGE_GROUP = "storage"
DASHBOARD_GROUP = "dashboard"

# Global processes dictionary
processes = {
    KAFKA_GROUP: [],
    PRODUCERS_GROUP: [],
    PROCESSORS_GROUP: [],
    STORAGE_GROUP: [],
    DASHBOARD_GROUP: []
}

def log_message(message):
    """Log a message with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

def start_kafka():
    """Start Kafka and Zookeeper using docker-compose"""
    log_message("Starting Kafka and Zookeeper...")
    
    # Change to the kafka-setup directory
    os.chdir(kafka_setup_path)
    
    # Start Kafka and Zookeeper using docker-compose
    process = subprocess.Popen(
        ["docker-compose", "up", "-d"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Wait for Kafka to start
    time.sleep(10)
    
    # Create Kafka topics
    log_message("Creating Kafka topics...")
    topic_process = subprocess.Popen(
        ["sh", "kafka-topic.sh"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    processes[KAFKA_GROUP].append(process)
    processes[KAFKA_GROUP].append(topic_process)
    
    log_message("Kafka and Zookeeper started successfully!")

def start_data_producers():
    """Start the data producers for Reddit and Twitter"""
    log_message("Starting data producers...")
    
    # Change back to the base directory
    os.chdir(base_dir)
    
    # Start the Reddit producer
    reddit_process = subprocess.Popen(
        [sys.executable, os.path.join(data_ingestion_path, "reddit_producer.py")],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    processes[PRODUCERS_GROUP].append(reddit_process)
    log_message("Reddit producer started")
    
    # Start the Twitter producer
    twitter_process = subprocess.Popen(
        [sys.executable, os.path.join(data_ingestion_path, "twitter_producer.py")],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    processes[PRODUCERS_GROUP].append(twitter_process)
    log_message("Twitter producer started")

def start_sentiment_processors():
    """Start the sentiment analysis processors"""
    log_message("Starting sentiment analysis processors...")
    
    # Change back to the base directory
    os.chdir(base_dir)
    
    # Start the basic sentiment processor
    basic_processor = subprocess.Popen(
        [sys.executable, os.path.join(sentiment_analysis_path, "sentiment_processor.py")],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    processes[PROCESSORS_GROUP].append(basic_processor)
    log_message("Basic sentiment processor started")
    
    # Start the enhanced sentiment processor
    enhanced_processor = subprocess.Popen(
        [sys.executable, os.path.join(sentiment_analysis_path, "enhanced_processor.py")],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    processes[PROCESSORS_GROUP].append(enhanced_processor)
    log_message("Enhanced sentiment processor started")

def start_data_storage():
    """Start the data storage service"""
    log_message("Starting data storage service...")
    
    # Change back to the base directory
    os.chdir(base_dir)
    
    # Start the sentiment data storage
    storage_process = subprocess.Popen(
        [sys.executable, os.path.join(data_storage_path, "sentiment_store.py")],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    processes[STORAGE_GROUP].append(storage_process)
    log_message("Data storage service started")

def start_dashboard():
    """Start the Streamlit dashboard"""
    log_message("Starting Streamlit dashboard...")
    
    # Change to the dashboard directory
    os.chdir(dashboard_path)
    
    # Start the Streamlit dashboard
    dashboard_process = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", "8501"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    processes[DASHBOARD_GROUP].append(dashboard_process)
    log_message("Streamlit dashboard started at http://localhost:8501")

def stop_processes(process_group=None):
    """Stop processes by group or all if group is None"""
    if process_group:
        log_message(f"Stopping {process_group} processes...")
        for process in processes[process_group]:
            if process.poll() is None:  # Process is still running
                process.terminate()
        processes[process_group] = []
    else:
        log_message("Stopping all processes...")
        for group in processes:
            for process in processes[group]:
                if process.poll() is None:  # Process is still running
                    process.terminate()
            processes[group] = []
    
    if process_group == KAFKA_GROUP or process_group is None:
        # Specifically stop Kafka using docker-compose
        os.chdir(kafka_setup_path)
        subprocess.run(["docker-compose", "down"])
        log_message("Kafka and Zookeeper stopped")

def signal_handler(sig, frame):
    """Handle Ctrl+C to gracefully shut down"""
    log_message("Shutdown signal received, stopping all processes...")
    stop_processes()
    log_message("All processes stopped. Exiting.")
    sys.exit(0)

def main():
    """Main function to start all components"""
    parser = argparse.ArgumentParser(description="Run the Sentiment Analysis System")
    
    parser.add_argument("--no-kafka", action="store_true", help="Don't start Kafka")
    parser.add_argument("--no-producers", action="store_true", help="Don't start data producers")
    parser.add_argument("--no-processors", action="store_true", help="Don't start sentiment processors")
    parser.add_argument("--no-storage", action="store_true", help="Don't start data storage")
    parser.add_argument("--dashboard-only", action="store_true", help="Only start the dashboard")
    
    args = parser.parse_args()
    
    # Register signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    
    log_message("Starting Sentiment Analysis System...")
    
    try:
        if args.dashboard_only:
            start_dashboard()
        else:
            if not args.no_kafka:
                start_kafka()
            
            if not args.no_producers:
                start_data_producers()
            
            if not args.no_processors:
                start_sentiment_processors()
            
            if not args.no_storage:
                start_data_storage()
            
            start_dashboard()
        
        log_message("All components started successfully!")
        log_message("Press Ctrl+C to stop all processes")
        
        # Keep the main thread alive
        while True:
            time.sleep(1)
    
    except Exception as e:
        log_message(f"Error starting system: {e}")
        stop_processes()
        sys.exit(1)

if __name__ == "__main__":
    main() 