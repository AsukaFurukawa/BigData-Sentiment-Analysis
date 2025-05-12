import os
import sys
import json
import sqlite3
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import logging
import pickle
from collections import defaultdict
from typing import Dict, List, Any, Optional, Tuple
import re

# Configure logging
base_dir = os.path.dirname(os.path.dirname(__file__))  # Get project root directory
logs_dir = os.path.join(base_dir, 'logs')
os.makedirs(logs_dir, exist_ok=True)  # Create logs directory if it doesn't exist

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(logs_dir, 'entity_graph.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("entity_graph")

class EntityRelationshipAnalyzer:
    """Analyze relationships between entities in sentiment data using graph analysis"""
    
    def __init__(self):
        """Initialize the entity relationship analyzer"""
        # Paths
        self.base_path = os.path.dirname(__file__)
        self.project_path = os.path.dirname(self.base_path)
        self.sentiment_db_path = os.path.join(self.project_path, 'data-storage', 'data', 'sentiment_data.db')
        self.results_path = os.path.join(self.base_path, 'results')
        
        # Create results directory if it doesn't exist
        os.makedirs(self.results_path, exist_ok=True)
        
        # Initialize graph
        self.graph = nx.Graph()
        
        # Entity metadata
        self.entity_metadata = {}
        self.entity_sentiments = defaultdict(list)
        
        # Settings
        self.min_cooccurrence = 2  # Minimum co-occurrences to establish a relationship
        self.max_entity_distance = 50  # Maximum distance (in characters) between entities to be considered related
    
    def load_entities_from_db(self, days: int = 30, domain: Optional[str] = None):
        """Load entity data from the sentiment database"""
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
                s.id, s.created_at, s.text, s.domain, s.overall_sentiment,
                f.entities as finance_entities, 
                t.entities as tech_entities
            FROM sentiment_results s
            LEFT JOIN finance_analysis f ON s.id = f.id
            LEFT JOIN tech_analysis t ON s.id = t.id
            WHERE s.created_at > '{cutoff_date}'
            {domain_filter}
            """
            
            # Execute query
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            if df.empty:
                logger.warning("No entity data found in database")
                return False
            
            logger.info(f"Loaded {len(df)} records with entity data")
            
            # Process the data
            self._process_entity_data(df)
            return True
            
        except Exception as e:
            logger.error(f"Error loading entity data: {e}")
            return False
    
    def _process_entity_data(self, df: pd.DataFrame):
        """Process entity data from the database"""
        try:
            # Reset graph and entity data
            self.graph = nx.Graph()
            self.entity_metadata = {}
            self.entity_sentiments = defaultdict(list)
            
            # Process each record
            for _, row in df.iterrows():
                # Parse entity JSON strings
                finance_entities = self._parse_json(row.get('finance_entities', '[]'))
                tech_entities = self._parse_json(row.get('tech_entities', '[]'))
                
                # Get sentiment from row
                sentiment_value = self._sentiment_to_value(row.get('overall_sentiment', 'neutral'))
                
                # Combine all entities
                all_entities = finance_entities + tech_entities
                
                if not all_entities:
                    continue
                
                # Add entities to graph
                text = row.get('text', '')
                
                # Extract entity positions in text
                entity_positions = self._find_entity_positions(text, all_entities)
                
                # Update entity metadata and sentiments
                self._update_entity_metadata(all_entities, row, sentiment_value)
                
                # Build relationships between entities
                self._build_entity_relationships(all_entities, entity_positions, text)
            
            # Remove weak connections
            self._prune_graph()
            
            logger.info(f"Processed entity data: {len(self.entity_metadata)} entities, {self.graph.number_of_edges()} relationships")
            
        except Exception as e:
            logger.error(f"Error processing entity data: {e}")
    
    def _parse_json(self, json_str: str) -> List[str]:
        """Parse JSON string to list"""
        try:
            if not json_str or json_str == '[]':
                return []
            return json.loads(json_str)
        except:
            return []
    
    def _sentiment_to_value(self, sentiment: str) -> float:
        """Convert sentiment string to numeric value"""
        sentiment_map = {
            'positive': 1.0,
            'negative': -1.0,
            'neutral': 0.0
        }
        return sentiment_map.get(sentiment.lower(), 0.0)
    
    def _find_entity_positions(self, text: str, entities: List[str]) -> Dict[str, List[int]]:
        """Find positions of entities in text"""
        positions = {}
        for entity in entities:
            # Find all occurrences
            entity_positions = []
            for match in re.finditer(re.escape(entity), text, re.IGNORECASE):
                entity_positions.append(match.start())
            
            if entity_positions:
                positions[entity] = entity_positions
        
        return positions
    
    def _update_entity_metadata(self, entities: List[str], row: pd.Series, sentiment_value: float):
        """Update entity metadata and sentiment"""
        for entity in entities:
            # Initialize entity if not exists
            if entity not in self.entity_metadata:
                self.entity_metadata[entity] = {
                    'first_seen': row.get('created_at'),
                    'mentions': 0,
                    'domains': set(),
                    'sentiment_avg': 0.0,
                }
            
            # Update metadata
            metadata = self.entity_metadata[entity]
            metadata['mentions'] += 1
            metadata['domains'].add(row.get('domain', 'unknown'))
            
            # Add sentiment
            self.entity_sentiments[entity].append(sentiment_value)
            
            # Update average sentiment
            metadata['sentiment_avg'] = np.mean(self.entity_sentiments[entity])
    
    def _build_entity_relationships(self, entities: List[str], positions: Dict[str, List[int]], text: str):
        """Build relationships between entities based on co-occurrence and proximity"""
        # Add all entities as nodes
        for entity in entities:
            if entity not in self.graph:
                self.graph.add_node(entity)
        
        # Connect entities that appear in the same text
        for i, entity1 in enumerate(entities):
            for j, entity2 in enumerate(entities[i+1:], i+1):
                if entity1 == entity2:
                    continue
                
                # Add edge or increment weight if exists
                if self.graph.has_edge(entity1, entity2):
                    self.graph[entity1][entity2]['weight'] += 1
                else:
                    # Check if entities are close to each other in the text
                    is_close = False
                    
                    # If position data is available
                    if entity1 in positions and entity2 in positions:
                        # Check distances between all occurrences
                        for pos1 in positions[entity1]:
                            for pos2 in positions[entity2]:
                                if abs(pos1 - pos2) <= self.max_entity_distance:
                                    is_close = True
                                    break
                            if is_close:
                                break
                    
                    # Add edge with proximity information
                    self.graph.add_edge(entity1, entity2, weight=1, proximity=is_close)
    
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
        
        logger.info(f"Pruned graph: removed {len(edges_to_remove)} weak edges and {len(isolated_nodes)} isolated nodes")
    
    def get_entity_communities(self):
        """Identify communities of related entities"""
        try:
            if self.graph.number_of_nodes() == 0:
                logger.warning("No entities in graph to detect communities")
                return []
            
            # Detect communities using Louvain method
            try:
                import community as community_louvain
                partition = community_louvain.best_partition(self.graph)
                
                # Group entities by community
                communities = defaultdict(list)
                for entity, community_id in partition.items():
                    communities[community_id].append(entity)
                
                # Sort communities by size
                sorted_communities = sorted(communities.values(), key=len, reverse=True)
                
                # Add metadata to each community
                result_communities = []
                for i, community in enumerate(sorted_communities):
                    community_data = {
                        'id': i,
                        'size': len(community),
                        'entities': community,
                        'avg_sentiment': np.mean([self.entity_metadata[e]['sentiment_avg'] for e in community]),
                        'top_entities': sorted(community, 
                                             key=lambda e: self.entity_metadata[e]['mentions'], 
                                             reverse=True)[:5]
                    }
                    result_communities.append(community_data)
                
                logger.info(f"Detected {len(result_communities)} communities of entities")
                return result_communities
            except ImportError:
                logger.warning("Community detection requires python-louvain package. Falling back to connected components.")
                raise ImportError
                
        except (ImportError, Exception) as e:
            # Fallback to connected components
            components = list(nx.connected_components(self.graph))
            
            # Process components
            result_communities = []
            for i, component in enumerate(sorted(components, key=len, reverse=True)):
                component_list = list(component)
                community_data = {
                    'id': i,
                    'size': len(component_list),
                    'entities': component_list,
                    'avg_sentiment': np.mean([self.entity_metadata[e]['sentiment_avg'] for e in component_list]),
                    'top_entities': sorted(component_list, 
                                         key=lambda e: self.entity_metadata[e]['mentions'], 
                                         reverse=True)[:5]
                }
                result_communities.append(community_data)
            
            logger.info(f"Detected {len(result_communities)} connected components of entities")
            return result_communities
    
    def get_central_entities(self, limit: int = 20):
        """Identify the most central entities in the network"""
        try:
            if self.graph.number_of_nodes() == 0:
                logger.warning("No entities in graph to calculate centrality")
                return []
            
            # Calculate different centrality measures
            degree_centrality = nx.degree_centrality(self.graph)
            betweenness_centrality = nx.betweenness_centrality(self.graph)
            closeness_centrality = nx.closeness_centrality(self.graph)
            
            # Combine centrality measures
            combined_centrality = {}
            for entity in self.graph.nodes():
                # Weighted average of different centrality measures
                combined_centrality[entity] = (
                    0.4 * degree_centrality.get(entity, 0) +
                    0.4 * betweenness_centrality.get(entity, 0) +
                    0.2 * closeness_centrality.get(entity, 0)
                )
            
            # Get top entities
            top_entities = sorted(combined_centrality.items(), key=lambda x: x[1], reverse=True)[:limit]
            
            # Format result
            results = []
            for entity, score in top_entities:
                entity_data = {
                    'entity': entity,
                    'centrality_score': score,
                    'mentions': self.entity_metadata[entity]['mentions'],
                    'sentiment': self.entity_metadata[entity]['sentiment_avg'],
                    'domains': list(self.entity_metadata[entity]['domains'])
                }
                results.append(entity_data)
            
            logger.info(f"Identified {len(results)} central entities")
            return results
            
        except Exception as e:
            logger.error(f"Error calculating entity centrality: {e}")
            return []
    
    def visualize_graph(self, filename: str = None, limit: int = 100):
        """Visualize the entity relationship graph"""
        try:
            if self.graph.number_of_nodes() == 0:
                logger.warning("No entities in graph to visualize")
                return False
            
            # Create a subgraph of the most important entities if too many
            if self.graph.number_of_nodes() > limit:
                # Get top entities by degree
                top_entities = sorted(self.graph.degree, key=lambda x: x[1], reverse=True)[:limit]
                top_entity_names = [name for name, _ in top_entities]
                
                # Create subgraph
                subgraph = self.graph.subgraph(top_entity_names)
            else:
                subgraph = self.graph
            
            # Set up the figure
            plt.figure(figsize=(12, 12))
            
            # Create positions for the graph layout
            pos = nx.spring_layout(subgraph, seed=42)
            
            # Get sentiment colors for nodes
            node_colors = []
            for node in subgraph.nodes():
                sentiment = self.entity_metadata[node]['sentiment_avg']
                if sentiment > 0.2:
                    node_colors.append('green')
                elif sentiment < -0.2:
                    node_colors.append('red')
                else:
                    node_colors.append('gray')
            
            # Get edge weights for thickness
            edge_widths = [data['weight'] * 0.5 for _, _, data in subgraph.edges(data=True)]
            
            # Draw the graph
            nx.draw_networkx(
                subgraph,
                pos=pos,
                with_labels=True,
                node_color=node_colors,
                node_size=[self.entity_metadata[node]['mentions'] * 10 + 100 for node in subgraph.nodes()],
                edge_color='lightgray',
                width=edge_widths,
                font_size=8,
                alpha=0.8
            )
            
            # Add title
            plt.title(f"Entity Relationship Network (top {subgraph.number_of_nodes()} entities)")
            plt.axis('off')
            
            # Save or show
            if filename:
                filepath = os.path.join(self.results_path, filename)
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                logger.info(f"Graph visualization saved to {filepath}")
                plt.close()
            else:
                plt.tight_layout()
                plt.show()
            
            return True
            
        except Exception as e:
            logger.error(f"Error visualizing graph: {e}")
            return False
    
    def save_graph(self, filename: str = "entity_graph.pkl"):
        """Save the graph and entity data to file"""
        try:
            # Create data package
            data = {
                'graph': self.graph,
                'entity_metadata': self.entity_metadata,
                'entity_sentiments': dict(self.entity_sentiments),
                'timestamp': datetime.now().isoformat()
            }
            
            # Save to file
            filepath = os.path.join(self.results_path, filename)
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
            
            logger.info(f"Graph data saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving graph data: {e}")
            return False
    
    def load_graph(self, filename: str = "entity_graph.pkl"):
        """Load the graph and entity data from file"""
        try:
            filepath = os.path.join(self.results_path, filename)
            
            if not os.path.isfile(filepath):
                logger.warning(f"Graph file not found at {filepath}")
                return False
            
            # Load data
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            # Restore data
            self.graph = data['graph']
            self.entity_metadata = data['entity_metadata']
            self.entity_sentiments = defaultdict(list, data['entity_sentiments'])
            
            logger.info(f"Graph data loaded from {filepath}")
            logger.info(f"Loaded {len(self.entity_metadata)} entities and {self.graph.number_of_edges()} relationships")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading graph data: {e}")
            return False
    
    def run_analysis(self, days: int = 30, domain: Optional[str] = None, visualize: bool = True):
        """Run a complete entity relationship analysis"""
        try:
            # Load entity data
            if not self.load_entities_from_db(days, domain):
                logger.error("Failed to load entity data for analysis")
                return False
            
            # Get communities
            communities = self.get_entity_communities()
            
            # Get central entities
            central_entities = self.get_central_entities()
            
            # Visualize graph
            if visualize:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M')
                domain_str = f"_{domain}" if domain else ""
                self.visualize_graph(f"entity_graph{domain_str}_{timestamp}.png")
            
            # Save results
            timestamp = datetime.now().strftime('%Y%m%d_%H%M')
            domain_str = f"_{domain}" if domain else ""
            
            # Save communities
            communities_file = os.path.join(self.results_path, f"entity_communities{domain_str}_{timestamp}.json")
            with open(communities_file, 'w', encoding='utf-8') as f:
                json.dump(communities, f, indent=2)
            
            # Save central entities
            central_file = os.path.join(self.results_path, f"central_entities{domain_str}_{timestamp}.json")
            with open(central_file, 'w', encoding='utf-8') as f:
                json.dump(central_entities, f, indent=2)
            
            # Save graph
            self.save_graph(f"entity_graph{domain_str}_{timestamp}.pkl")
            
            logger.info(f"Entity relationship analysis complete. Results saved to {self.results_path}")
            
            # Return analysis results
            return {
                'communities': communities,
                'central_entities': central_entities,
                'entity_count': len(self.entity_metadata),
                'relationship_count': self.graph.number_of_edges()
            }
            
        except Exception as e:
            logger.error(f"Error running entity relationship analysis: {e}")
            return False

# Example usage
if __name__ == "__main__":
    analyzer = EntityRelationshipAnalyzer()
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Entity Relationship Analysis')
    parser.add_argument('--days', type=int, default=30, help='Number of days to analyze')
    parser.add_argument('--domain', type=str, help='Domain to filter by (e.g., finance, technology)')
    parser.add_argument('--no-visualize', action='store_true', help='Disable graph visualization')
    
    args = parser.parse_args()
    
    # Run analysis
    results = analyzer.run_analysis(args.days, args.domain, not args.no_visualize)
    
    if results:
        print(f"\nAnalysis complete:")
        print(f"- Entities: {results['entity_count']}")
        print(f"- Relationships: {results['relationship_count']}")
        print(f"- Communities: {len(results['communities'])}")
        print(f"- Top entities: {', '.join([e['entity'] for e in results['central_entities'][:5]])}")
        print(f"\nResults saved to {analyzer.results_path}") 