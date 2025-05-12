#!/bin/bash

# Directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Ensure we're in the right directory
cd "$DIR"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Docker is not installed. Please install Docker and try again."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "Docker Compose is not installed. Please install Docker Compose and try again."
    exit 1
fi

echo "=== Big Data Sentiment Analysis Engine ==="
echo "Starting the system with Docker..."

# Create necessary directories
mkdir -p logs data-storage/data market-data/data

# Start Kafka and Zookeeper
echo "Starting Kafka environment..."
cd kafka-setup
docker-compose up -d
cd ..

echo "Waiting for Kafka to be ready..."
sleep 15

# Create Kafka topics
echo "Creating Kafka topics..."
docker exec kafka kafka-topics --create --if-not-exists --bootstrap-server kafka:9092 --replication-factor 1 --partitions 3 --topic raw_twitter_posts
docker exec kafka kafka-topics --create --if-not-exists --bootstrap-server kafka:9092 --replication-factor 1 --partitions 3 --topic raw_news_data
docker exec kafka kafka-topics --create --if-not-exists --bootstrap-server kafka:9092 --replication-factor 1 --partitions 3 --topic processed_sentiment
docker exec kafka kafka-topics --create --if-not-exists --bootstrap-server kafka:9092 --replication-factor 1 --partitions 3 --topic sentiment_alerts

# Create databases if they don't exist
echo "Setting up databases..."
python setup_databases.py

# Start the components in the right order
echo "Starting main components..."

# Start Twitter data collector in the background
echo "Starting Twitter data collector..."
python run_advanced.py twitter &
TWITTER_PID=$!

# Start stream processor in the background
echo "Starting stream processor..."
python run_advanced.py stream &
STREAM_PID=$!

# Run market data collection
echo "Collecting market data..."
python run_advanced.py market

# Run entity analysis
echo "Running entity analysis..."
python run_advanced.py entity --domain finance
python run_advanced.py entity --domain technology

# Run advanced entity analysis
echo "Running advanced entity analytics..."
python run_advanced.py advanced-entity

# Start the dashboard
echo "Starting dashboard..."
python run_advanced.py dashboard

# Clean up when the dashboard is closed
echo "Shutting down components..."
kill $TWITTER_PID
kill $STREAM_PID

# Stop Kafka and Zookeeper
echo "Stopping Kafka environment..."
cd kafka-setup
docker-compose down
cd ..

echo "System shutdown complete." 