import os
import sys
import json
import logging
from datetime import datetime, timedelta
import sqlite3
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("spark_processing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("spark_processor")

# Import PySpark
try:
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import col, udf, explode, split, regexp_replace, when, lit
    from pyspark.sql.types import StringType, FloatType, ArrayType, StructType, StructField
    from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer, IDF
    from pyspark.ml.clustering import LDA
    import pyspark.sql.functions as F
    HAS_SPARK = True
except ImportError:
    logger.warning("PySpark not available. Will run in local mode with limited functionality.")
    HAS_SPARK = False

class SparkSentimentProcessor:
    """Process sentiment data using Spark for big data capabilities"""
    
    def __init__(self):
        """Initialize the Spark processor"""
        # Paths
        self.base_path = os.path.dirname(__file__)
        self.project_path = os.path.dirname(self.base_path)
        self.sentiment_data_path = os.path.join(self.project_path, 'data-storage', 'data', 'sentiment_data.db')
        self.results_path = os.path.join(self.base_path, 'results')
        
        # Create results directory if it doesn't exist
        os.makedirs(self.results_path, exist_ok=True)
        
        # Configure Spark
        if HAS_SPARK:
            self._init_spark()
        else:
            self.spark = None
    
    def _init_spark(self):
        """Initialize Spark session"""
        try:
            self.spark = (SparkSession
                .builder
                .appName("SentimentBigDataProcessor")
                .config("spark.driver.memory", "4g")
                .config("spark.executor.memory", "4g")
                .config("spark.sql.shuffle.partitions", "8")
                .config("spark.default.parallelism", "8")
                .getOrCreate())
            
            logger.info("Spark session initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing Spark: {e}")
            self.spark = None
    
    def load_data(self, days=30):
        """Load sentiment data for processing"""
        if not os.path.isfile(self.sentiment_data_path):
            logger.error(f"Sentiment database not found at {self.sentiment_data_path}")
            return None
        
        try:
            # Connect to the SQLite database
            conn = sqlite3.connect(self.sentiment_data_path)
            
            # Calculate the cutoff date
            cutoff_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            
            # Query to fetch sentiment data with all associated analytics
            query = f"""
            SELECT 
                s.id, s.source, s.user_id, s.created_at, s.text, s.domain, s.overall_sentiment,
                f.sentiment as finance_sentiment, f.finance_score, f.entities as finance_entities, 
                f.positive_terms as finance_positive_terms, f.negative_terms as finance_negative_terms,
                t.sentiment as tech_sentiment, t.tech_score, t.entities as tech_entities, 
                t.categories as tech_categories, t.positive_terms as tech_positive_terms, 
                t.negative_terms as tech_negative_terms
            FROM sentiment_results s
            LEFT JOIN finance_analysis f ON s.id = f.id
            LEFT JOIN tech_analysis t ON s.id = t.id
            WHERE s.created_at > '{cutoff_date}'
            """
            
            # Load data into a pandas DataFrame first
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            if df.empty:
                logger.warning("No sentiment data found")
                return None
            
            logger.info(f"Loaded {len(df)} sentiment records")
            
            # Convert to Spark DataFrame if available
            if HAS_SPARK and self.spark:
                spark_df = self.spark.createDataFrame(df)
                return spark_df
            else:
                return df
            
        except Exception as e:
            logger.error(f"Error loading sentiment data: {e}")
            return None
    
    def clean_and_prepare_data(self, df):
        """Clean and prepare data for analysis"""
        if df is None:
            return None
        
        try:
            if HAS_SPARK and self.spark and isinstance(df, self.spark.sql.dataframe.DataFrame):
                # Spark DataFrame processing
                
                # Clean text
                cleaned_df = df.withColumn(
                    "clean_text", 
                    regexp_replace(regexp_replace(col("text"), "http\\S+", ""), "[^a-zA-Z\\s]", " ")
                )
                
                # Convert timestamps to datetime
                cleaned_df = cleaned_df.withColumn(
                    "created_at", 
                    F.to_timestamp(col("created_at"))
                )
                
                # Parse JSON columns
                schema_entities = ArrayType(StringType())
                
                # Define UDFs for JSON parsing
                def parse_json_array(json_str):
                    if json_str is None:
                        return []
                    try:
                        return json.loads(json_str)
                    except:
                        return []
                
                parse_json_array_udf = udf(parse_json_array, schema_entities)
                
                # Apply UDFs to parse JSON columns
                for col_name in ["finance_entities", "finance_positive_terms", "finance_negative_terms", 
                                "tech_entities", "tech_positive_terms", "tech_negative_terms"]:
                    if col_name in df.columns:
                        cleaned_df = cleaned_df.withColumn(
                            col_name, 
                            parse_json_array_udf(col(col_name))
                        )
                
                # Add numeric sentiment scores
                cleaned_df = cleaned_df.withColumn(
                    "sentiment_score",
                    when(col("overall_sentiment") == "positive", 1.0)
                    .when(col("overall_sentiment") == "negative", -1.0)
                    .otherwise(0.0)
                )
                
                # Add timestamp-based features
                cleaned_df = cleaned_df.withColumn("hour_of_day", F.hour(col("created_at")))
                cleaned_df = cleaned_df.withColumn("day_of_week", F.dayofweek(col("created_at")))
                
                logger.info("Data cleaned and prepared with Spark")
                
            else:
                # Pandas DataFrame processing for local mode
                # Clean text
                cleaned_df = df.copy()
                cleaned_df["clean_text"] = df["text"].str.replace("http\S+", "", regex=True).str.replace("[^a-zA-Z\s]", " ", regex=True)
                
                # Convert timestamps to datetime
                cleaned_df["created_at"] = pd.to_datetime(cleaned_df["created_at"])
                
                # Parse JSON columns
                for col_name in ["finance_entities", "finance_positive_terms", "finance_negative_terms", 
                                "tech_entities", "tech_positive_terms", "tech_negative_terms"]:
                    if col_name in df.columns:
                        cleaned_df[col_name] = df[col_name].apply(
                            lambda x: json.loads(x) if pd.notna(x) and x else []
                        )
                
                # Add numeric sentiment scores
                cleaned_df["sentiment_score"] = df["overall_sentiment"].map({
                    "positive": 1.0,
                    "neutral": 0.0,
                    "negative": -1.0
                })
                
                # Add timestamp-based features
                cleaned_df["hour_of_day"] = cleaned_df["created_at"].dt.hour
                cleaned_df["day_of_week"] = cleaned_df["created_at"].dt.dayofweek
                
                logger.info("Data cleaned and prepared with Pandas")
            
            return cleaned_df
            
        except Exception as e:
            logger.error(f"Error cleaning and preparing data: {e}")
            return df
    
    def run_topic_modeling(self, df, num_topics=5, domain=None):
        """Run topic modeling on text data using LDA"""
        if df is None or not HAS_SPARK or self.spark is None:
            logger.warning("Cannot run topic modeling: Spark not available or data is empty")
            return None
        
        try:
            # Filter data by domain if specified
            if domain:
                domain_df = df.filter(col("domain") == domain)
            else:
                domain_df = df
            
            # Tokenize text
            tokenizer = Tokenizer(inputCol="clean_text", outputCol="tokens")
            tokenized_df = tokenizer.transform(domain_df)
            
            # Remove stop words
            remover = StopWordsRemover(inputCol="tokens", outputCol="filtered_tokens")
            filtered_df = remover.transform(tokenized_df)
            
            # Convert tokens to term frequency vectors
            cv = CountVectorizer(inputCol="filtered_tokens", outputCol="tf", vocabSize=5000, minDF=5)
            cv_model = cv.fit(filtered_df)
            tf_df = cv_model.transform(filtered_df)
            
            # Calculate IDF vectors
            idf = IDF(inputCol="tf", outputCol="features")
            idf_model = idf.fit(tf_df)
            tfidf_df = idf_model.transform(tf_df)
            
            # Run LDA for topic modeling
            lda = LDA(k=num_topics, maxIter=10, featuresCol="features")
            lda_model = lda.fit(tfidf_df)
            
            # Get topics and their terms
            topics = lda_model.describeTopics(maxTermsPerTopic=10)
            
            # Get vocabulary from the CountVectorizer model
            vocab = cv_model.vocabulary
            
            # Convert to readable format
            topic_terms = []
            
            # Convert Spark DataFrame to Pandas for easier processing
            topics_pd = topics.toPandas()
            
            for i, (termIndices, termWeights) in enumerate(zip(topics_pd["termIndices"], topics_pd["termWeights"])):
                terms = [vocab[idx] for idx in termIndices]
                weights = termWeights
                topic_terms.append({
                    "topic_id": i,
                    "terms": [{"term": term, "weight": weight} for term, weight in zip(terms, weights)]
                })
            
            # Save results
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            domain_str = f"_{domain}" if domain else ""
            results_path = os.path.join(self.results_path, f'topics{domain_str}_{timestamp}.json')
            
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(topic_terms, f, indent=2)
            
            logger.info(f"Topic modeling complete. Results saved to {results_path}")
            return topic_terms
            
        except Exception as e:
            logger.error(f"Error running topic modeling: {e}")
            return None
    
    def analyze_sentiment_trends(self, df):
        """Analyze sentiment trends over time"""
        if df is None:
            logger.warning("Cannot analyze trends: data is empty")
            return None
        
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            if HAS_SPARK and self.spark and isinstance(df, self.spark.sql.dataframe.DataFrame):
                # Spark processing
                
                # Calculate hourly sentiment averages
                hourly_sentiment = df.groupBy(
                    F.date_trunc("hour", col("created_at")).alias("hour"),
                    "domain"
                ).agg(
                    F.avg("sentiment_score").alias("avg_sentiment"),
                    F.count("id").alias("post_count")
                ).orderBy("hour")
                
                # Save results as CSV
                hourly_sentiment_pd = hourly_sentiment.toPandas()
                results_path = os.path.join(self.results_path, f'hourly_sentiment_{timestamp}.csv')
                hourly_sentiment_pd.to_csv(results_path, index=False)
                
                # Calculate sentiment by hour of day
                hour_sentiment = df.groupBy(
                    "hour_of_day",
                    "domain"
                ).agg(
                    F.avg("sentiment_score").alias("avg_sentiment"),
                    F.count("id").alias("post_count")
                ).orderBy("hour_of_day")
                
                # Save results
                hour_sentiment_pd = hour_sentiment.toPandas()
                hour_results_path = os.path.join(self.results_path, f'hour_sentiment_{timestamp}.csv')
                hour_sentiment_pd.to_csv(hour_results_path, index=False)
                
                # Calculate sentiment by day of week
                day_sentiment = df.groupBy(
                    "day_of_week",
                    "domain"
                ).agg(
                    F.avg("sentiment_score").alias("avg_sentiment"),
                    F.count("id").alias("post_count")
                ).orderBy("day_of_week")
                
                # Save results
                day_sentiment_pd = day_sentiment.toPandas()
                day_results_path = os.path.join(self.results_path, f'day_sentiment_{timestamp}.csv')
                day_sentiment_pd.to_csv(day_results_path, index=False)
                
                logger.info(f"Sentiment trend analysis complete. Results saved to {self.results_path}")
                
                return {
                    "hourly": hourly_sentiment_pd,
                    "hour_of_day": hour_sentiment_pd,
                    "day_of_week": day_sentiment_pd
                }
                
            else:
                # Pandas processing
                df_copy = df.copy()
                
                # Calculate hourly sentiment averages
                df_copy['hour'] = df_copy['created_at'].dt.floor('H')
                hourly_sentiment = df_copy.groupby(['hour', 'domain']).agg({
                    'sentiment_score': 'mean',
                    'id': 'count'
                }).rename(columns={'sentiment_score': 'avg_sentiment', 'id': 'post_count'}).reset_index()
                
                # Save results
                results_path = os.path.join(self.results_path, f'hourly_sentiment_{timestamp}.csv')
                hourly_sentiment.to_csv(results_path, index=False)
                
                # Calculate sentiment by hour of day
                hour_sentiment = df_copy.groupby(['hour_of_day', 'domain']).agg({
                    'sentiment_score': 'mean',
                    'id': 'count'
                }).rename(columns={'sentiment_score': 'avg_sentiment', 'id': 'post_count'}).reset_index()
                
                # Save results
                hour_results_path = os.path.join(self.results_path, f'hour_sentiment_{timestamp}.csv')
                hour_sentiment.to_csv(hour_results_path, index=False)
                
                # Calculate sentiment by day of week
                day_sentiment = df_copy.groupby(['day_of_week', 'domain']).agg({
                    'sentiment_score': 'mean',
                    'id': 'count'
                }).rename(columns={'sentiment_score': 'avg_sentiment', 'id': 'post_count'}).reset_index()
                
                # Save results
                day_results_path = os.path.join(self.results_path, f'day_sentiment_{timestamp}.csv')
                day_sentiment.to_csv(day_results_path, index=False)
                
                logger.info(f"Sentiment trend analysis complete. Results saved to {self.results_path}")
                
                return {
                    "hourly": hourly_sentiment,
                    "hour_of_day": hour_sentiment,
                    "day_of_week": day_sentiment
                }
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment trends: {e}")
            return None
    
    def extract_trending_entities(self, df):
        """Extract and analyze trending entities from sentiment data"""
        if df is None:
            logger.warning("Cannot extract trending entities: data is empty")
            return None
        
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            if HAS_SPARK and self.spark and isinstance(df, self.spark.sql.dataframe.DataFrame):
                # Spark processing
                
                # Extract finance entities
                finance_entities_df = df.select(
                    "id", 
                    "domain",
                    "overall_sentiment",
                    "created_at",
                    explode(col("finance_entities")).alias("entity")
                ).filter(col("entity").isNotNull())
                
                # Get entity counts and sentiment
                finance_entity_stats = finance_entities_df.groupBy("entity").agg(
                    F.count("id").alias("mentions"),
                    F.avg(when(col("overall_sentiment") == "positive", 1.0)
                         .when(col("overall_sentiment") == "negative", -1.0)
                         .otherwise(0.0)).alias("avg_sentiment"),
                    F.min("created_at").alias("first_seen"),
                    F.max("created_at").alias("last_seen")
                ).orderBy(F.desc("mentions"))
                
                # Save results
                finance_entities_pd = finance_entity_stats.limit(100).toPandas()
                finance_path = os.path.join(self.results_path, f'finance_entities_{timestamp}.csv')
                finance_entities_pd.to_csv(finance_path, index=False)
                
                # Tech entities
                tech_entities_df = df.select(
                    "id", 
                    "domain",
                    "overall_sentiment",
                    "created_at",
                    explode(col("tech_entities")).alias("entity")
                ).filter(col("entity").isNotNull())
                
                # Get entity counts and sentiment
                tech_entity_stats = tech_entities_df.groupBy("entity").agg(
                    F.count("id").alias("mentions"),
                    F.avg(when(col("overall_sentiment") == "positive", 1.0)
                         .when(col("overall_sentiment") == "negative", -1.0)
                         .otherwise(0.0)).alias("avg_sentiment"),
                    F.min("created_at").alias("first_seen"),
                    F.max("created_at").alias("last_seen")
                ).orderBy(F.desc("mentions"))
                
                # Save results
                tech_entities_pd = tech_entity_stats.limit(100).toPandas()
                tech_path = os.path.join(self.results_path, f'tech_entities_{timestamp}.csv')
                tech_entities_pd.to_csv(tech_path, index=False)
                
                logger.info(f"Entity extraction complete. Results saved to {self.results_path}")
                
                return {
                    "finance": finance_entities_pd,
                    "tech": tech_entities_pd
                }
                
            else:
                # Pandas processing
                df_copy = df.copy()
                
                # Process finance entities
                finance_entities = []
                for _, row in df_copy.iterrows():
                    if isinstance(row.get('finance_entities'), list):
                        for entity in row['finance_entities']:
                            finance_entities.append({
                                'id': row['id'],
                                'entity': entity,
                                'sentiment': row['overall_sentiment'],
                                'created_at': row['created_at']
                            })
                
                if finance_entities:
                    finance_entities_df = pd.DataFrame(finance_entities)
                    
                    # Calculate entity stats
                    finance_entity_stats = finance_entities_df.groupby('entity').agg({
                        'id': 'count',
                        'sentiment': lambda x: sum(1 if s == 'positive' else (-1 if s == 'negative' else 0) for s in x) / len(x),
                        'created_at': ['min', 'max']
                    })
                    
                    finance_entity_stats.columns = ['mentions', 'avg_sentiment', 'first_seen', 'last_seen']
                    finance_entity_stats = finance_entity_stats.reset_index().sort_values('mentions', ascending=False)
                    
                    # Save results
                    finance_path = os.path.join(self.results_path, f'finance_entities_{timestamp}.csv')
                    finance_entity_stats.head(100).to_csv(finance_path, index=False)
                else:
                    finance_entity_stats = pd.DataFrame()
                
                # Process tech entities
                tech_entities = []
                for _, row in df_copy.iterrows():
                    if isinstance(row.get('tech_entities'), list):
                        for entity in row['tech_entities']:
                            tech_entities.append({
                                'id': row['id'],
                                'entity': entity,
                                'sentiment': row['overall_sentiment'],
                                'created_at': row['created_at']
                            })
                
                if tech_entities:
                    tech_entities_df = pd.DataFrame(tech_entities)
                    
                    # Calculate entity stats
                    tech_entity_stats = tech_entities_df.groupby('entity').agg({
                        'id': 'count',
                        'sentiment': lambda x: sum(1 if s == 'positive' else (-1 if s == 'negative' else 0) for s in x) / len(x),
                        'created_at': ['min', 'max']
                    })
                    
                    tech_entity_stats.columns = ['mentions', 'avg_sentiment', 'first_seen', 'last_seen']
                    tech_entity_stats = tech_entity_stats.reset_index().sort_values('mentions', ascending=False)
                    
                    # Save results
                    tech_path = os.path.join(self.results_path, f'tech_entities_{timestamp}.csv')
                    tech_entity_stats.head(100).to_csv(tech_path, index=False)
                else:
                    tech_entity_stats = pd.DataFrame()
                
                logger.info(f"Entity extraction complete. Results saved to {self.results_path}")
                
                return {
                    "finance": finance_entity_stats,
                    "tech": tech_entity_stats
                }
            
        except Exception as e:
            logger.error(f"Error extracting trending entities: {e}")
            return None
    
    def run_batch_processing(self, days=30):
        """Run a complete batch processing pipeline"""
        try:
            logger.info(f"Starting batch processing for the past {days} days")
            
            # Load data
            df = self.load_data(days)
            
            if df is None or (isinstance(df, pd.DataFrame) and df.empty):
                logger.warning("No data to process")
                return False
            
            # Clean and prepare data
            cleaned_df = self.clean_and_prepare_data(df)
            
            # Run topic modeling
            if HAS_SPARK and self.spark:
                # Run for all domains combined
                self.run_topic_modeling(cleaned_df, num_topics=10)
                
                # Run separately for finance and tech domains
                self.run_topic_modeling(cleaned_df, num_topics=5, domain="finance")
                self.run_topic_modeling(cleaned_df, num_topics=5, domain="technology")
            
            # Analyze sentiment trends
            self.analyze_sentiment_trends(cleaned_df)
            
            # Extract trending entities
            self.extract_trending_entities(cleaned_df)
            
            # Stop Spark session if active
            if HAS_SPARK and self.spark:
                self.spark.stop()
                logger.info("Spark session stopped")
            
            logger.info("Batch processing complete")
            return True
            
        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
            
            # Stop Spark session if active
            if HAS_SPARK and self.spark:
                self.spark.stop()
                logger.info("Spark session stopped")
                
            return False

if __name__ == "__main__":
    processor = SparkSentimentProcessor()
    
    # Get days from command line args if provided, default to 30
    days = 30
    if len(sys.argv) > 1:
        try:
            days = int(sys.argv[1])
        except ValueError:
            logger.warning(f"Invalid days parameter '{sys.argv[1]}'. Using default of 30 days.")
    
    processor.run_batch_processing(days) 