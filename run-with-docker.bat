@echo off
REM Change to the directory of this script
cd /d "%~dp0"

REM Check if Docker is installed
docker --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Docker is not installed. Please install Docker and try again.
    exit /b 1
)

REM Check if Docker Compose is installed
docker-compose --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Docker Compose is not installed. Please install Docker Compose and try again.
    exit /b 1
)

echo === Big Data Sentiment Analysis Engine ===
echo Starting the system with Docker...

REM Create necessary directories
mkdir logs 2>nul
mkdir data-storage\data 2>nul
mkdir market-data\data 2>nul

REM Start Kafka and Zookeeper
echo Starting Kafka environment...
cd kafka-setup
docker-compose up -d
cd ..

echo Waiting for Kafka to be ready...
timeout /t 15 /nobreak > nul

REM Create Kafka topics
echo Creating Kafka topics...
docker exec kafka kafka-topics --create --if-not-exists --bootstrap-server kafka:9092 --replication-factor 1 --partitions 3 --topic raw_twitter_posts
docker exec kafka kafka-topics --create --if-not-exists --bootstrap-server kafka:9092 --replication-factor 1 --partitions 3 --topic raw_news_data
docker exec kafka kafka-topics --create --if-not-exists --bootstrap-server kafka:9092 --replication-factor 1 --partitions 3 --topic processed_sentiment
docker exec kafka kafka-topics --create --if-not-exists --bootstrap-server kafka:9092 --replication-factor 1 --partitions 3 --topic sentiment_alerts

REM Create databases if they don't exist
echo Setting up databases...
python setup_databases.py

REM Start the components in the right order
echo Starting main components...

REM Start Twitter data collector in the background
echo Starting Twitter data collector...
start /b python run_advanced.py twitter

REM Start stream processor in the background
echo Starting stream processor...
start /b python run_advanced.py stream

REM Run market data collection
echo Collecting market data...
python run_advanced.py market

REM Run entity analysis
echo Running entity analysis...
python run_advanced.py entity --domain finance
python run_advanced.py entity --domain technology

REM Run advanced entity analysis
echo Running advanced entity analytics...
python run_advanced.py advanced-entity

REM Start the dashboard
echo Starting dashboard...
python run_advanced.py dashboard

REM Stop Kafka and Zookeeper when the dashboard is closed
echo Stopping Kafka environment...
cd kafka-setup
docker-compose down
cd ..

echo System shutdown complete. 