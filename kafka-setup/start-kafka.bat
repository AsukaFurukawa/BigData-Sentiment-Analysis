@echo off
REM Change to the directory of this script
cd /d "%~dp0"

echo Starting Kafka and Zookeeper...
docker-compose up -d

echo Waiting for Kafka to be ready...
timeout /t 15 /nobreak > nul

REM Check if Kafka is running
for /f %%i in ('docker ps ^| findstr kafka ^| find /c /v ""') do set KAFKA_RUNNING=%%i
if "%KAFKA_RUNNING%"=="0" (
  echo Error: Kafka is not running. Check docker logs.
  exit /b 1
)

echo Creating Kafka topics...
docker exec kafka kafka-topics --create --if-not-exists --bootstrap-server kafka:9092 --replication-factor 1 --partitions 3 --topic raw_twitter_posts
docker exec kafka kafka-topics --create --if-not-exists --bootstrap-server kafka:9092 --replication-factor 1 --partitions 3 --topic raw_news_data  
docker exec kafka kafka-topics --create --if-not-exists --bootstrap-server kafka:9092 --replication-factor 1 --partitions 3 --topic processed_sentiment
docker exec kafka kafka-topics --create --if-not-exists --bootstrap-server kafka:9092 --replication-factor 1 --partitions 3 --topic sentiment_alerts

echo Listing created topics:
docker exec kafka kafka-topics --list --bootstrap-server kafka:9092

echo Kafka environment is ready!
echo Kafka UI is available at: http://localhost:8080 