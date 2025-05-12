#!/bin/bash

# Directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Ensure we're in the right directory
cd "$DIR"

# Start Kafka and Zookeeper
echo "Starting Kafka and Zookeeper..."
docker-compose up -d

# Wait for Kafka to be ready
echo "Waiting for Kafka to be ready..."
sleep 15

# Check if Kafka is running
KAFKA_RUNNING=$(docker ps | grep kafka | wc -l)
if [ "$KAFKA_RUNNING" -eq 0 ]; then
  echo "Error: Kafka is not running. Check docker logs."
  exit 1
fi

# Create topics
echo "Creating Kafka topics..."
docker exec kafka kafka-topics --create --if-not-exists --bootstrap-server kafka:9092 --replication-factor 1 --partitions 3 --topic raw_twitter_posts
docker exec kafka kafka-topics --create --if-not-exists --bootstrap-server kafka:9092 --replication-factor 1 --partitions 3 --topic raw_news_data  
docker exec kafka kafka-topics --create --if-not-exists --bootstrap-server kafka:9092 --replication-factor 1 --partitions 3 --topic processed_sentiment
docker exec kafka kafka-topics --create --if-not-exists --bootstrap-server kafka:9092 --replication-factor 1 --partitions 3 --topic sentiment_alerts

# List created topics
echo "Listing created topics:"
docker exec kafka kafka-topics --list --bootstrap-server kafka:9092

echo "Kafka environment is ready!"
echo "Kafka UI is available at: http://localhost:8080" 