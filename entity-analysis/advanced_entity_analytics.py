#!/usr/bin/env python3
import os
import sys
import json
import numpy as np
import pandas as pd
import sqlite3
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from typing import Dict, List, Any, Optional, Tuple, Union

# NLP and ML libraries
import nltk
from nltk.corpus import stopwords
import gensim
from gensim import corpora
from gensim.models import LdaModel, CoherenceModel
from gensim.utils import simple_preprocess
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
import networkx as nx

# Set up logging
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
logs_dir = os.path.join(base_dir, 'logs')
os.makedirs(logs_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(logs_dir, 'advanced_entity_analytics.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("advanced_entity_analytics")

# Download NLTK resources
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    stop_words = set(stopwords.words('english'))
except Exception as e:
    logger.warning(f"Could not download NLTK resources: {e}")
    stop_words = set()

class AdvancedEntityAnalytics:
    """Advanced analytics for entity relationship analysis"""
    
    def __init__(self):
        """Initialize the advanced entity analytics module"""
        # Paths
        self.base_path = os.path.dirname(__file__)
        self.project_path = os.path.dirname(self.base_path)
        self.sentiment_db_path = os.path.join(self.project_path, 'data-storage', 'data', 'sentiment_data.db')
        self.results_path = os.path.join(self.base_path, 'results')
        
        # Create results directory if it doesn't exist
        os.makedirs(self.results_path, exist_ok=True)
        
        # Analysis parameters
        self.min_cooccurrence = 2
        self.min_samples = 2
        self.eps = 0.3  # Distance threshold for DBSCAN
        self.n_topics = 10  # Number of topics for topic modeling
        
        # Initialize state
        self.data = None
        self.entities = []
        self.entity_docs = {}
        self.entity_mentions = {}
        self.entity_sentiment = {}
        self.topics = {}
        self.entity_clusters = {}
        self.entity_temporal = {}
        
        # Network graph
        self.graph = nx.Graph()
    
    def load_data(self, days: int = 30, domain: Optional[str] = None):
        """Load sentiment and entity data from the database"""
        if not os.path.isfile(self.sentiment_db_path):
            logger.error(f"Sentiment database not found at {self.sentiment_db_path}")
            return False
        
        try:
            # Connect to the database
            conn = sqlite3.connect(self.sentiment_db_path)
            
            # Calculate cutoff date
            cutoff_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            
            # Build the query
            domain_filter = f"AND s.domain = '{domain}'" if domain else ""
            
            query = f"""
            SELECT 
                s.id, s.source, s.created_at, s.text, s.domain, s.overall_sentiment, s.score,
                f.entities as finance_entities, f.positive_terms as finance_positive, f.negative_terms as finance_negative,
                t.entities as tech_entities, t.categories as tech_categories
            FROM sentiment_results s
            LEFT JOIN finance_analysis f ON s.id = f.id
            LEFT JOIN tech_analysis t ON s.id = t.id
            WHERE s.created_at > '{cutoff_date}'
            {domain_filter}
            """
            
            # Load data
            self.data = pd.read_sql_query(query, conn)
            conn.close()
            
            if self.data.empty:
                logger.warning("No data found in database")
                return False
            
            # Process data
            self.data['created_at'] = pd.to_datetime(self.data['created_at'])
            
            # Extract entities and prepare for analysis
            self._extract_entities()
            logger.info(f"Loaded {len(self.data)} records with {len(self.entities)} unique entities")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False
    
    def _extract_entities(self):
        """Extract entities from the data and prepare for analysis"""
        try:
            # Initialize
            entity_docs = defaultdict(list)
            entity_mentions = Counter()
            entity_sentiment = defaultdict(list)
            entity_temporal = defaultdict(lambda: defaultdict(int))
            
            # Process records
            for _, row in self.data.iterrows():
                # Parse entity fields
                finance_entities = self._parse_json(row.get('finance_entities', '[]'))
                tech_entities = self._parse_json(row.get('tech_entities', '[]'))
                
                # Date for temporal analysis
                date = row['created_at'].date()
                
                # Combine entities
                all_entities = finance_entities + tech_entities
                if not all_entities:
                    continue
                
                # Record sentiment
                sentiment_value = row.get('score', 0.5)
                
                # Update entity data
                for entity in all_entities:
                    # Add document text
                    entity_docs[entity].append(row['text'])
                    
                    # Count mention
                    entity_mentions[entity] += 1
                    
                    # Add sentiment
                    entity_sentiment[entity].append(sentiment_value)
                    
                    # Add to temporal data
                    entity_temporal[entity][date] += 1
                    
                    # Add to graph
                    if entity not in self.graph:
                        self.graph.add_node(entity)
                    
                # Add co-occurrence edges
                for i, entity1 in enumerate(all_entities):
                    for entity2 in all_entities[i+1:]:
                        if entity1 != entity2:
                            # Add edge or increment weight
                            if self.graph.has_edge(entity1, entity2):
                                self.graph[entity1][entity2]['weight'] += 1
                            else:
                                self.graph.add_edge(entity1, entity2, weight=1)
            
            # Store results
            self.entities = list(entity_mentions.keys())
            self.entity_docs = entity_docs
            self.entity_mentions = entity_mentions
            self.entity_sentiment = {e: np.mean(s) for e, s in entity_sentiment.items()}
            self.entity_temporal = entity_temporal
            
            # Remove weak connections
            self._prune_graph()
            
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
    
    def _parse_json(self, json_str: str) -> List[str]:
        """Parse JSON string to list"""
        try:
            if not json_str or json_str == '[]':
                return []
            return json.loads(json_str)
        except:
            return []
    
    def _prune_graph(self):
        """Remove weak connections from the graph"""
        edges_to_remove = []
        
        for u, v, data in self.graph.edges(data=True):
            # Remove if weight is below threshold
            if data.get('weight', 0) < self.min_cooccurrence:
                edges_to_remove.append((u, v))
        
        # Remove edges
        for u, v in edges_to_remove:
            self.graph.remove_edge(u, v)
        
        # Remove isolated nodes
        isolated_nodes = list(nx.isolates(self.graph))
        self.graph.remove_nodes_from(isolated_nodes)
        
        logger.info(f"Pruned graph: removed {len(edges_to_remove)} weak edges and {len(isolated_nodes)} isolated entities")
    
    def run_topic_modeling(self, num_topics: int = 10, words_per_topic: int = 10):
        """Run topic modeling on documents containing entities"""
        try:
            # Prepare corpus for topic modeling
            all_texts = self.data['text'].tolist()
            if not all_texts:
                logger.warning("No text data available for topic modeling")
                return {}
            
            # Preprocess texts
            processed_texts = []
            for text in all_texts:
                tokens = simple_preprocess(text)
                # Remove stopwords
                tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
                processed_texts.append(tokens)
            
            # Create dictionary
            dictionary = corpora.Dictionary(processed_texts)
            
            # Create corpus
            corpus = [dictionary.doc2bow(text) for text in processed_texts]
            
            # Train LDA model
            lda_model = LdaModel(
                corpus=corpus,
                id2word=dictionary,
                num_topics=num_topics,
                random_state=42,
                passes=10,
                alpha='auto',
                eta='auto'
            )
            
            # Extract topics
            topics = {}
            for i, topic in lda_model.print_topics():
                topics[i] = {
                    'id': i,
                    'words': [(w.split('*')[1].strip().replace('"', ''), float(w.split('*')[0])) 
                              for w in topic.split('+')][:words_per_topic],
                    'coherence': 0.0  # Will be updated later
                }
            
            # Calculate coherence
            coherence_model = CoherenceModel(model=lda_model, texts=processed_texts, dictionary=dictionary, coherence='c_v')
            coherence = coherence_model.get_coherence()
            
            # Update overall coherence
            for topic_id in topics:
                topics[topic_id]['coherence'] = coherence / num_topics
            
            # Assign entities to topics
            entity_topics = {}
            
            for entity in self.entities:
                if entity in self.entity_docs:
                    # Get all documents containing this entity
                    docs = self.entity_docs[entity]
                    if not docs:
                        continue
                    
                    # Process docs
                    entity_processed_docs = []
                    for doc in docs:
                        tokens = simple_preprocess(doc)
                        tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
                        entity_processed_docs.append(tokens)
                    
                    # Convert to corpus format
                    entity_corpus = [dictionary.doc2bow(doc) for doc in entity_processed_docs]
                    
                    # Get topic distributions
                    topic_dists = []
                    for bow in entity_corpus:
                        topic_dist = lda_model.get_document_topics(bow)
                        topic_dists.append({tid: prob for tid, prob in topic_dist})
                    
                    # Aggregate topic probabilities across all docs
                    agg_topics = defaultdict(float)
                    for dist in topic_dists:
                        for tid, prob in dist.items():
                            agg_topics[tid] += prob
                    
                    # Normalize
                    total = sum(agg_topics.values())
                    if total > 0:
                        agg_topics = {tid: prob/total for tid, prob in agg_topics.items()}
                    
                    # Get dominant topic
                    if agg_topics:
                        dominant_topic = max(agg_topics.items(), key=lambda x: x[1])
                        entity_topics[entity] = {
                            'dominant_topic': dominant_topic[0],
                            'probability': dominant_topic[1],
                            'topic_distribution': agg_topics
                        }
            
            # Store results
            self.topics = {
                'topics': topics,
                'coherence': coherence,
                'entity_topics': entity_topics
            }
            
            logger.info(f"Topic modeling completed with {num_topics} topics, coherence: {coherence:.4f}")
            return self.topics
            
        except Exception as e:
            logger.error(f"Error in topic modeling: {e}")
            return {}
    
    def cluster_entities(self):
        """Cluster entities based on co-occurrence and similarities"""
        try:
            if not self.entities:
                logger.warning("No entities available for clustering")
                return {}
            
            # Create a similarity matrix based on graph
            n_entities = len(self.entities)
            entity_to_idx = {entity: i for i, entity in enumerate(self.entities)}
            
            # Initialize similarity matrix
            similarity_matrix = np.zeros((n_entities, n_entities))
            
            # Fill similarity based on co-occurrences
            for u, v, data in self.graph.edges(data=True):
                if u in entity_to_idx and v in entity_to_idx:
                    i, j = entity_to_idx[u], entity_to_idx[v]
                    weight = data.get('weight', 1)
                    similarity_matrix[i, j] = weight
                    similarity_matrix[j, i] = weight  # Matrix is symmetric
            
            # Normalize
            row_sums = similarity_matrix.sum(axis=1)
            non_zero_rows = row_sums > 0
            similarity_matrix[non_zero_rows] = similarity_matrix[non_zero_rows] / row_sums[non_zero_rows, np.newaxis]
            
            # Apply DBSCAN clustering
            clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric='precomputed').fit(1 - similarity_matrix)
            
            # Extract clusters
            labels = clustering.labels_
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            
            # Organize results
            clusters = defaultdict(list)
            for i, label in enumerate(labels):
                if label != -1:  # Skip noise points
                    clusters[int(label)].append(self.entities[i])
            
            # Calculate cluster metrics
            cluster_metrics = {}
            for cluster_id, entities in clusters.items():
                # Calculate average sentiment
                sentiments = [self.entity_sentiment.get(e, 0.5) for e in entities]
                avg_sentiment = np.mean(sentiments)
                
                # Calculate central entities
                if len(entities) > 1:
                    subgraph = self.graph.subgraph(entities)
                    centrality = nx.degree_centrality(subgraph)
                    central_entities = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:3]
                else:
                    central_entities = [(entities[0], 1.0)] if entities else []
                
                # Get topic, if available
                if self.topics and 'entity_topics' in self.topics:
                    topic_counts = Counter()
                    for entity in entities:
                        if entity in self.topics['entity_topics']:
                            topic_counts[self.topics['entity_topics'][entity]['dominant_topic']] += 1
                    
                    dominant_topic = topic_counts.most_common(1)[0][0] if topic_counts else None
                else:
                    dominant_topic = None
                
                # Store metrics
                cluster_metrics[cluster_id] = {
                    'id': cluster_id,
                    'size': len(entities),
                    'avg_sentiment': avg_sentiment,
                    'central_entities': central_entities,
                    'dominant_topic': dominant_topic
                }
            
            # Store results
            self.entity_clusters = {
                'clusters': clusters,
                'metrics': cluster_metrics,
                'n_clusters': n_clusters,
                'noise_points': sum(1 for label in labels if label == -1)
            }
            
            logger.info(f"Entity clustering completed with {n_clusters} clusters")
            return self.entity_clusters
            
        except Exception as e:
            logger.error(f"Error in entity clustering: {e}")
            return {}
    
    def analyze_temporal_patterns(self):
        """Analyze temporal patterns in entity mentions"""
        try:
            if not self.entity_temporal:
                logger.warning("No temporal data available for analysis")
                return {}
            
            # Get top entities by mention count
            top_entities = sorted(self.entity_mentions.items(), key=lambda x: x[1], reverse=True)[:20]
            
            # Get all dates in the data
            all_dates = set()
            for entity, date_counts in self.entity_temporal.items():
                all_dates.update(date_counts.keys())
            
            all_dates = sorted(all_dates)
            if not all_dates:
                return {}
            
            # Create a dataframe for temporal analysis
            temporal_data = []
            
            for entity, count in top_entities:
                entity_data = self.entity_temporal.get(entity, {})
                
                # Get counts for each date
                for date in all_dates:
                    temporal_data.append({
                        'entity': entity,
                        'date': date,
                        'count': entity_data.get(date, 0),
                        'sentiment': self.entity_sentiment.get(entity, 0.5)
                    })
            
            if not temporal_data:
                return {}
                
            # Create dataframe
            temporal_df = pd.DataFrame(temporal_data)
            
            # Calculate metrics
            entity_volatility = {}
            entity_trend = {}
            entity_momentum = {}
            
            for entity, count in top_entities:
                entity_df = temporal_df[temporal_df['entity'] == entity]
                if len(entity_df) < 2:
                    continue
                
                # Calculate volatility (coefficient of variation)
                counts = entity_df['count'].values
                mean = np.mean(counts)
                std = np.std(counts)
                volatility = std / mean if mean > 0 else 0
                
                # Calculate trend (simple linear regression)
                x = np.arange(len(counts))
                if len(x) > 1 and not np.all(counts == counts[0]):
                    trend = np.polyfit(x, counts, 1)[0]
                else:
                    trend = 0
                
                # Calculate momentum (recent vs overall average)
                if len(counts) >= 3:
                    recent_avg = np.mean(counts[-3:])
                    overall_avg = np.mean(counts)
                    momentum = recent_avg / overall_avg if overall_avg > 0 else 1.0
                else:
                    momentum = 1.0
                
                # Store metrics
                entity_volatility[entity] = volatility
                entity_trend[entity] = trend
                entity_momentum[entity] = momentum
            
            # Find emerging entities (high momentum, positive trend)
            emerging_entities = [
                entity for entity, mom in entity_momentum.items()
                if mom > 1.2 and entity_trend.get(entity, 0) > 0
            ]
            
            # Find fading entities (low momentum, negative trend)
            fading_entities = [
                entity for entity, mom in entity_momentum.items()
                if mom < 0.8 and entity_trend.get(entity, 0) < 0
            ]
            
            # Find volatile entities
            volatile_entities = [
                entity for entity, vol in entity_volatility.items()
                if vol > np.median(list(entity_volatility.values())) * 1.5
            ]
            
            # Store results
            temporal_analysis = {
                'entity_volatility': entity_volatility,
                'entity_trend': entity_trend,
                'entity_momentum': entity_momentum,
                'emerging_entities': emerging_entities,
                'fading_entities': fading_entities,
                'volatile_entities': volatile_entities,
                'temporal_data': temporal_df.to_dict(orient='records')
            }
            
            logger.info(f"Temporal analysis completed for {len(top_entities)} entities")
            return temporal_analysis
            
        except Exception as e:
            logger.error(f"Error in temporal analysis: {e}")
            return {}
    
    def calculate_advanced_metrics(self):
        """Calculate advanced metrics for entity analysis"""
        try:
            if not self.entities:
                logger.warning("No entities available for advanced metrics")
                return {}
            
            # Calculate centrality measures
            centrality_metrics = {}
            
            # Check if graph has nodes
            if self.graph.number_of_nodes() > 0:
                # Degree centrality
                degree_centrality = nx.degree_centrality(self.graph)
                
                # Betweenness centrality
                betweenness_centrality = nx.betweenness_centrality(self.graph)
                
                # Eigenvector centrality
                try:
                    eigenvector_centrality = nx.eigenvector_centrality(self.graph, max_iter=1000)
                except:
                    eigenvector_centrality = {node: 0.0 for node in self.graph.nodes()}
                
                # PageRank
                pagerank = nx.pagerank(self.graph)
                
                # Combine metrics
                for entity in self.graph.nodes():
                    centrality_metrics[entity] = {
                        'degree': degree_centrality.get(entity, 0),
                        'betweenness': betweenness_centrality.get(entity, 0),
                        'eigenvector': eigenvector_centrality.get(entity, 0),
                        'pagerank': pagerank.get(entity, 0)
                    }
            
            # Sentiment polarity
            sentiment_polarity = {}
            for entity, sentiment in self.entity_sentiment.items():
                if sentiment > 0.6:
                    polarity = "positive"
                elif sentiment < 0.4:
                    polarity = "negative"
                else:
                    polarity = "neutral"
                
                sentiment_polarity[entity] = polarity
            
            # Calculate entity importance score
            entity_importance = {}
            for entity in self.entities:
                # Factors for importance
                mention_score = np.log1p(self.entity_mentions.get(entity, 1)) / np.log1p(max(self.entity_mentions.values()))
                
                # Centrality (if available)
                if entity in centrality_metrics:
                    centrality_score = (
                        0.4 * centrality_metrics[entity]['pagerank'] / max(max(cm['pagerank'] for cm in centrality_metrics.values()), 0.001) +
                        0.3 * centrality_metrics[entity]['degree'] / max(max(cm['degree'] for cm in centrality_metrics.values()), 0.001) +
                        0.3 * centrality_metrics[entity]['betweenness'] / max(max(cm['betweenness'] for cm in centrality_metrics.values()), 0.001)
                    )
                else:
                    centrality_score = 0
                
                # Sentiment intensity (distance from neutral)
                sentiment = self.entity_sentiment.get(entity, 0.5)
                sentiment_intensity = 2 * abs(sentiment - 0.5)
                
                # Combined score
                importance = 0.5 * mention_score + 0.3 * centrality_score + 0.2 * sentiment_intensity
                entity_importance[entity] = importance
            
            # Combine all metrics
            advanced_metrics = {
                'centrality': centrality_metrics,
                'sentiment_polarity': sentiment_polarity,
                'importance': entity_importance
            }
            
            logger.info(f"Advanced metrics calculated for {len(self.entities)} entities")
            return advanced_metrics
            
        except Exception as e:
            logger.error(f"Error calculating advanced metrics: {e}")
            return {}
    
    def run_advanced_analysis(self, days: int = 30, domain: Optional[str] = None):
        """Run a complete advanced entity analysis"""
        try:
            # Load data
            if not self.load_data(days, domain):
                logger.error("Failed to load data for analysis")
                return None
            
            # Run topic modeling
            topics = self.run_topic_modeling()
            
            # Cluster entities
            clusters = self.cluster_entities()
            
            # Analyze temporal patterns
            temporal = self.analyze_temporal_patterns()
            
            # Calculate advanced metrics
            metrics = self.calculate_advanced_metrics()
            
            # Combine results
            results = {
                'topics': topics,
                'clusters': clusters,
                'temporal': temporal,
                'metrics': metrics,
                'entity_count': len(self.entities),
                'relationship_count': self.graph.number_of_edges()
            }
            
            # Save results
            self.save_results(results, domain)
            
            logger.info(f"Advanced entity analysis completed for {len(self.entities)} entities")
            return results
            
        except Exception as e:
            logger.error(f"Error running advanced entity analysis: {e}")
            return None
    
    def save_results(self, results: Dict[str, Any], domain: Optional[str] = None):
        """Save analysis results to JSON file"""
        try:
            # Create timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M')
            domain_str = f"_{domain}" if domain else ""
            
            # Create output file path
            output_file = os.path.join(self.results_path, f"advanced_entity_analysis{domain_str}_{timestamp}.json")
            
            # Convert to serializable format
            serializable_results = self._make_json_serializable(results)
            
            # Save to file
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_results, f, indent=2)
            
            logger.info(f"Analysis results saved to {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            return None
    
    def _make_json_serializable(self, obj):
        """Make an object JSON serializable"""
        if isinstance(obj, (datetime, np.datetime64)):
            return obj.isoformat()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {self._make_json_serializable(k): self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(i) for i in obj]
        elif isinstance(obj, tuple):
            return tuple(self._make_json_serializable(i) for i in obj)
        elif isinstance(obj, set):
            return list(self._make_json_serializable(i) for i in obj)
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')
        elif pd.isna(obj):
            return None
        else:
            return obj

if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Advanced Entity Analytics')
    parser.add_argument('--days', type=int, default=30, help='Number of days to analyze')
    parser.add_argument('--domain', type=str, choices=['finance', 'technology'], help='Limit analysis to domain')
    
    args = parser.parse_args()
    
    # Run analysis
    analyzer = AdvancedEntityAnalytics()
    results = analyzer.run_advanced_analysis(args.days, args.domain)
    
    if results:
        print(f"\nAdvanced analysis complete:")
        print(f"- Entities analyzed: {results['entity_count']}")
        print(f"- Relationships: {results['relationship_count']}")
        print(f"- Topics identified: {len(results['topics'].get('topics', {}))}")
        print(f"- Entity clusters: {results['clusters'].get('n_clusters', 0)}")
        
        if 'temporal' in results and 'emerging_entities' in results['temporal']:
            print(f"- Emerging entities: {', '.join(results['temporal']['emerging_entities'][:5])}")
            
        print(f"\nResults saved to {analyzer.results_path}") 