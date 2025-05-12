import os
import sys
import json
import time
import sqlite3
import logging
import subprocess
from datetime import datetime, timedelta, timezone

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import streamlit as st

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set page configuration
st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Database paths
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SENTIMENT_DB = os.path.join(BASE_PATH, 'data-storage', 'data', 'sentiment_data.db')
MARKET_DB = os.path.join(BASE_PATH, 'market-data', 'data', 'market_data.db')
CORRELATION_RESULTS = os.path.join(BASE_PATH, 'market-data', 'analysis_results')
ENTITY_RESULTS = os.path.join(BASE_PATH, 'entity-analysis', 'results')

# Helper functions
def load_sentiment_data(days=30, domain=None):
    """Load sentiment data from SQLite database"""
    if not os.path.exists(SENTIMENT_DB):
        st.error(f"Sentiment database not found at {SENTIMENT_DB}")
        return pd.DataFrame()
    
    try:
        # Connect to database
        conn = sqlite3.connect(SENTIMENT_DB)
        
        # Build query
        cutoff_date = (datetime.now().replace(tzinfo=timezone.utc) - timedelta(days=days)).strftime('%Y-%m-%d')
        domain_filter = f"AND domain = '{domain}'" if domain else ""
        
        query = f"""
        SELECT created_at, source, domain, overall_sentiment, score, confidence, model, text, title, url
        FROM sentiment_results
        WHERE created_at > '{cutoff_date}'
        {domain_filter}
        ORDER BY created_at
        """
        
        # Execute query
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        # Convert timestamp - use format='mixed' to handle various ISO formats
        df['created_at'] = pd.to_datetime(df['created_at'], format='mixed', utc=True)
        
        # Ensure title column exists, add empty if needed
        if 'title' not in df.columns:
            df['title'] = ""
        
        return df
    
    except Exception as e:
        st.error(f"Error loading sentiment data: {str(e)}")
        return pd.DataFrame()

def load_market_data(days=30, symbols=None):
    """Load market data from SQLite database"""
    if not os.path.exists(MARKET_DB):
        st.error(f"Market database not found at {MARKET_DB}")
        return pd.DataFrame()
    
    try:
        # Connect to database
        conn = sqlite3.connect(MARKET_DB)
        
        # Build query
        cutoff_date = (datetime.now().replace(tzinfo=timezone.utc) - timedelta(days=days)).strftime('%Y-%m-%d')
        symbol_filter = f"AND symbol IN ({', '.join(['?']*len(symbols))})" if symbols else ""
        
        query = f"""
        SELECT timestamp, symbol, close
        FROM market_data
        WHERE timestamp > '{cutoff_date}'
        {symbol_filter}
        ORDER BY timestamp
        """
        
        # Execute query
        params = symbols if symbols else []
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        # Convert timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', utc=True)
        
        return df
    
    except Exception as e:
        st.error(f"Error loading market data: {str(e)}")
        return pd.DataFrame()

def load_correlation_results():
    """Load most recent correlation analysis results"""
    if not os.path.exists(CORRELATION_RESULTS):
        # Create sample correlation data if directory doesn't exist
        st.info("Creating sample correlation data for demonstration...")
        
        # Sample data structure
        sample_data = {
            'AAPL': [0.72, 0.35, -0.28, 0.45],
            'MSFT': [0.65, 0.43, -0.18, 0.51],
            'GOOGL': [0.58, 0.39, -0.22, 0.38],
            'AMZN': [0.62, 0.41, -0.15, 0.42],
            'TSLA': [0.48, 0.37, -0.34, 0.29],
            '^GSPC': [0.51, 0.29, -0.32, 0.37]
        }
        
        # Create DataFrame
        return pd.DataFrame(sample_data, index=['finance_positive', 'tech_positive', 'negative_overall', 'entity_mentions'])
    
    try:
        # Find most recent correlation file
        correlation_files = [f for f in os.listdir(CORRELATION_RESULTS) if f.startswith('correlation_pearson')]
        if not correlation_files:
            # Create sample data if no files exist
            st.info("Creating sample correlation data for demonstration...")
            
            # Sample data structure
            sample_data = {
                'AAPL': [0.72, 0.35, -0.28, 0.45],
                'MSFT': [0.65, 0.43, -0.18, 0.51],
                'GOOGL': [0.58, 0.39, -0.22, 0.38],
                'AMZN': [0.62, 0.41, -0.15, 0.42],
                'TSLA': [0.48, 0.37, -0.34, 0.29],
                '^GSPC': [0.51, 0.29, -0.32, 0.37]
            }
            
            # Create DataFrame
            return pd.DataFrame(sample_data, index=['finance_positive', 'tech_positive', 'negative_overall', 'entity_mentions'])
        
        # Sort by date (in filename)
        most_recent = sorted(correlation_files)[-1]
        filepath = os.path.join(CORRELATION_RESULTS, most_recent)
        
        # Load CSV
        df = pd.read_csv(filepath)
        return df
    
    except Exception as e:
        st.error(f"Error loading correlation results: {str(e)}")
        
        # Return sample data in case of error
        st.info("Creating sample correlation data as fallback...")
        
        # Sample data structure
        sample_data = {
            'AAPL': [0.72, 0.35, -0.28, 0.45],
            'MSFT': [0.65, 0.43, -0.18, 0.51],
            'GOOGL': [0.58, 0.39, -0.22, 0.38],
            'AMZN': [0.62, 0.41, -0.15, 0.42],
            'TSLA': [0.48, 0.37, -0.34, 0.29],
            '^GSPC': [0.51, 0.29, -0.32, 0.37]
        }
        
        # Create DataFrame
        return pd.DataFrame(sample_data, index=['finance_positive', 'tech_positive', 'negative_overall', 'entity_mentions'])

def load_entity_data():
    """Load most recent entity analysis results"""
    if not os.path.exists(ENTITY_RESULTS):
        # Create sample entity data if directory doesn't exist
        st.info("Creating sample entity data for demonstration...")
        return create_sample_entity_data()
    
    try:
        # Find most recent central entities file
        entity_files = [f for f in os.listdir(ENTITY_RESULTS) if f.startswith('central_entities')]
        if not entity_files:
            # Create sample data if no files exist
            st.info("Creating sample entity data for demonstration...")
            return create_sample_entity_data()
        
        # Sort by date (in filename)
        most_recent = sorted(entity_files)[-1]
        filepath = os.path.join(ENTITY_RESULTS, most_recent)
        
        # Load JSON
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        return data
    
    except Exception as e:
        st.error(f"Error loading entity data: {str(e)}")
        
        # Return sample data in case of error
        st.info("Creating sample entity data as fallback...")
        return create_sample_entity_data()

def create_sample_entity_data():
    """Create sample entity data for demonstration"""
    
    # Create sample entities with realistic data
    sample_entities = [
        {"entity": "Apple", "centrality_score": 0.85, "mentions": 120, "sentiment": 0.72, "domains": ["technology", "finance"]},
        {"entity": "Tesla", "centrality_score": 0.76, "mentions": 95, "sentiment": 0.65, "domains": ["technology", "finance"]},
        {"entity": "Bitcoin", "centrality_score": 0.68, "mentions": 80, "sentiment": 0.45, "domains": ["finance"]},
        {"entity": "Amazon", "centrality_score": 0.65, "mentions": 75, "sentiment": 0.58, "domains": ["technology", "finance"]},
        {"entity": "Nvidia", "centrality_score": 0.62, "mentions": 72, "sentiment": 0.78, "domains": ["technology"]},
        {"entity": "Google", "centrality_score": 0.57, "mentions": 68, "sentiment": 0.62, "domains": ["technology"]},
        {"entity": "Microsoft", "centrality_score": 0.54, "mentions": 65, "sentiment": 0.67, "domains": ["technology"]},
        {"entity": "Fed", "centrality_score": 0.48, "mentions": 60, "sentiment": 0.40, "domains": ["finance"]},
        {"entity": "Inflation", "centrality_score": 0.45, "mentions": 55, "sentiment": 0.35, "domains": ["finance"]},
        {"entity": "AI", "centrality_score": 0.42, "mentions": 50, "sentiment": 0.72, "domains": ["technology"]},
        {"entity": "Meta", "centrality_score": 0.38, "mentions": 45, "sentiment": 0.56, "domains": ["technology"]},
        {"entity": "Interest Rates", "centrality_score": 0.36, "mentions": 42, "sentiment": 0.38, "domains": ["finance"]},
        {"entity": "Chip Shortage", "centrality_score": 0.33, "mentions": 38, "sentiment": 0.25, "domains": ["technology"]},
        {"entity": "Supply Chain", "centrality_score": 0.31, "mentions": 35, "sentiment": 0.42, "domains": ["finance", "technology"]},
        {"entity": "Regulation", "centrality_score": 0.29, "mentions": 32, "sentiment": 0.35, "domains": ["finance", "technology"]}
    ]
    
    return sample_entities

def plot_sentiment_trends(df):
    """Plot sentiment trends over time"""
    if df.empty:
        st.warning("No sentiment data available to plot")
        return
    
    # Make a copy to avoid modifying the original dataframe
    df_plot = df.copy()
    
    # Convert to local timezone and resample to hourly data
    df_plot['hour'] = df_plot['created_at'].dt.tz_localize(None).dt.floor('h')
    hourly_sentiment = df_plot.groupby(['hour', 'domain'])['score'].mean().reset_index()
    
    # Create plot
    fig = px.line(
        hourly_sentiment, 
        x='hour', 
        y='score', 
        color='domain',
        title='Sentiment Score Trends Over Time',
        labels={'hour': 'Time', 'score': 'Sentiment Score', 'domain': 'Domain'}
    )
    
    # Add a reference line for neutral sentiment
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray", annotation_text="Neutral")
    
    # Improve layout
    fig.update_layout(
        height=400,
        xaxis=dict(title='Date/Time'),
        yaxis=dict(title='Sentiment Score', range=[0, 1]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add explanation
    with st.expander("📊 Understanding this chart", expanded=True):
        st.markdown("""
        **What this shows:** This chart tracks sentiment score over time for different domains.
        
        **How to interpret:**
        - **Score above 0.5** indicates positive sentiment
        - **Score below 0.5** indicates negative sentiment
        - **Sharp changes** may indicate breaking news or events affecting sentiment
        - **Patterns by domain** reveal how sentiment differs between finance and tech sectors
        
        **Actionable insights:** Look for divergence between domains or sustained sentiment shifts, which often precede market movements.
        """)
        
        # Calculate recent trend
        recent_hours = 24
        recent_data = df_plot[df_plot['created_at'] > (datetime.now().replace(tzinfo=timezone.utc) - timedelta(hours=recent_hours))]
        
        if not recent_data.empty:
            recent_avg = recent_data['score'].mean()
            prev_avg = df_plot[
                (df_plot['created_at'] <= (datetime.now().replace(tzinfo=timezone.utc) - timedelta(hours=recent_hours))) & 
                (df_plot['created_at'] > (datetime.now().replace(tzinfo=timezone.utc) - timedelta(hours=recent_hours*2)))
            ]['score'].mean()
            
            if not pd.isna(prev_avg) and not pd.isna(recent_avg):
                change = recent_avg - prev_avg
                if abs(change) > 0.05:
                    direction = "positive" if change > 0 else "negative"
                    st.markdown(f"**Recent trend:** Sentiment has shifted {direction} by {abs(change):.2f} points in the last {recent_hours} hours.")
                else:
                    st.markdown("**Recent trend:** Sentiment has remained stable in the recent period.")

def plot_domain_distribution(df):
    """Plot distribution of domains"""
    if df.empty:
        st.warning("No sentiment data available to plot")
        return
    
    # Count by domain
    domain_counts = df['domain'].value_counts().reset_index()
    domain_counts.columns = ['domain', 'count']
    
    # Create plot
    fig = px.pie(
        domain_counts, 
        values='count', 
        names='domain',
        title='Distribution by Domain'
    )
    
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

def plot_sentiment_distribution(df):
    """Plot distribution of sentiment"""
    if df.empty:
        st.warning("No sentiment data available to plot")
        return
    
    # Count by sentiment
    sentiment_counts = df['overall_sentiment'].value_counts().reset_index()
    sentiment_counts.columns = ['sentiment', 'count']
    
    # Define color map
    color_map = {'positive': 'green', 'neutral': 'gray', 'negative': 'red'}
    
    # Create plot
    fig = px.bar(
        sentiment_counts, 
        x='sentiment', 
        y='count',
        color='sentiment',
        color_discrete_map=color_map,
        title='Sentiment Distribution'
    )
    
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

def plot_correlation_heatmap(df):
    """Plot correlation heatmap between sentiment and market data"""
    if df is None:
        st.warning("No correlation data available")
        return
    
    # Create heatmap
    fig = px.imshow(
        df.values,
        labels=dict(x="Market Symbols", y="Sentiment Types", color="Correlation"),
        x=df.columns,
        y=df.index,
        color_continuous_scale='RdBu_r',
        zmin=-1, zmax=1,
        title='Sentiment-Market Correlation Heatmap'
    )
    
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

def plot_entity_network(entities):
    """Plot entity importance visualization"""
    if entities is None:
        # Generate sample entity data for demonstration
        st.info("No entity data available - showing sample visualization for demonstration purposes")
        
        # Create sample data
        sample_entities = [
            {"entity": "Apple", "centrality_score": 0.85, "mentions": 120, "sentiment": 0.72, "domains": ["technology", "finance"]},
            {"entity": "Tesla", "centrality_score": 0.76, "mentions": 95, "sentiment": 0.65, "domains": ["technology", "finance"]},
            {"entity": "Bitcoin", "centrality_score": 0.68, "mentions": 80, "sentiment": 0.45, "domains": ["finance"]},
            {"entity": "Amazon", "centrality_score": 0.65, "mentions": 75, "sentiment": 0.58, "domains": ["technology", "finance"]},
            {"entity": "Nvidia", "centrality_score": 0.62, "mentions": 72, "sentiment": 0.78, "domains": ["technology"]},
            {"entity": "Google", "centrality_score": 0.57, "mentions": 68, "sentiment": 0.62, "domains": ["technology"]},
            {"entity": "Microsoft", "centrality_score": 0.54, "mentions": 65, "sentiment": 0.67, "domains": ["technology"]},
            {"entity": "Fed", "centrality_score": 0.48, "mentions": 60, "sentiment": 0.40, "domains": ["finance"]},
            {"entity": "Inflation", "centrality_score": 0.45, "mentions": 55, "sentiment": 0.35, "domains": ["finance"]},
            {"entity": "AI", "centrality_score": 0.42, "mentions": 50, "sentiment": 0.72, "domains": ["technology"]}
        ]
        
        entities = sample_entities
    
    # Get top entities 
    top_entities = entities[:15]  # Limit to top 15
    
    # Extract data
    entity_names = [item['entity'] for item in top_entities]
    centrality_scores = [item['centrality_score'] for item in top_entities]
    mentions = [item['mentions'] for item in top_entities]
    sentiments = [item['sentiment'] for item in top_entities]
    
    # Create DataFrame for plotting
    df = pd.DataFrame({
        'entity': entity_names,
        'centrality': centrality_scores,
        'mentions': mentions,
        'sentiment': sentiments
    })
    
    # Define colors based on sentiment
    color_scale = [
        [0, 'red'],
        [0.5, 'gray'],
        [1, 'green']
    ]
    
    # Create bubble chart
    fig = px.scatter(
        df,
        x='centrality', 
        y='mentions',
        size='mentions',
        color='sentiment',
        color_continuous_scale=color_scale,
        range_color=[-1, 1],
        hover_name='entity',
        text='entity',
        labels={
            'centrality': 'Centrality Score',
            'mentions': 'Number of Mentions',
            'sentiment': 'Sentiment Score'
        },
        title='Top Entities by Importance'
    )
    
    # Improve layout
    fig.update_traces(
        textposition='top center',
        marker=dict(sizemode='area', sizeref=0.1)
    )
    
    fig.update_layout(
        height=500,
        hovermode='closest',
        xaxis=dict(title='Centrality (Influence)'),
        yaxis=dict(title='Mentions (Popularity)')
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add explanation
    with st.expander("🔍 Understanding Entity Analysis", expanded=True):
        st.markdown("""
        **What this shows:** This visualization maps the most important entities (companies, people, concepts) mentioned in the analyzed content.
        
        **How to interpret:**
        - **Horizontal axis (Centrality)**: How influential or central an entity is in discussions - based on network connections to other topics
        - **Vertical axis (Mentions)**: How frequently the entity is mentioned
        - **Bubble size**: Proportional to number of mentions
        - **Color**: Sentiment associated with the entity (green = positive, red = negative)
        
        **Actionable insights:** 
        - Entities in the top-right (high centrality, high mentions) are the most significant market movers
        - Large red bubbles can indicate potential issues or negative news for that entity
        - Look for entities with high centrality but lower mentions as they may be emerging trends
        """)
        
        # Find most influential entity
        if not df.empty:
            influential = df.sort_values('centrality', ascending=False).iloc[0]
            st.markdown(f"**Most influential entity:** **{influential['entity']}** with centrality score of {influential['centrality']:.2f} and {influential['mentions']} mentions.")
            
            # Find most negative and positive entities
            if len(df) > 1:
                most_positive = df.sort_values('sentiment', ascending=False).iloc[0]
                most_negative = df.sort_values('sentiment', ascending=True).iloc[0]
                
                st.markdown(f"**Most positive sentiment:** **{most_positive['entity']}** ({most_positive['sentiment']:.2f})")
                st.markdown(f"**Most negative sentiment:** **{most_negative['entity']}** ({most_negative['sentiment']:.2f})")

def plot_entity_relationships(entities, limit=10):
    """Plot network graph of entity relationships"""
    if entities is None or len(entities) < 2:
        st.warning("Not enough entity data for relationship visualization")
        return
    
    # Get top entities
    top_entities = entities[:limit]
    
    # Create a network graph
    G = nx.Graph()
    
    # Add nodes
    for i, entity in enumerate(top_entities):
        # Use sentiment to determine node color
        sentiment = entity.get('sentiment', 0)
        if sentiment > 0.2:
            color = 'green'
        elif sentiment < -0.2:
            color = 'red'
        else:
            color = 'gray'
            
        G.add_node(entity['entity'], 
                   size=entity['mentions'],
                   score=entity['sentiment'],
                   color=color)
    
    # Add edges between entities that are mentioned together
    for i, entity1 in enumerate(top_entities):
        for entity2 in top_entities[i+1:]:
            # Create edge with weight based on combined mentions
            weight = min(entity1['mentions'], entity2['mentions']) / 10
            if weight > 0:
                G.add_edge(entity1['entity'], entity2['entity'], weight=weight)
    
    # Create positions
    pos = nx.spring_layout(G, seed=42)
    
    # Create plot
    edge_trace = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        weight = G.edges[edge]['weight']
        
        edge_trace.append(
            go.Scatter(
                x=[x0, x1, None], 
                y=[y0, y1, None],
                line=dict(width=weight, color='gray'),
                hoverinfo='none',
                mode='lines'
            )
        )
    
    # Create nodes
    node_x = []
    node_y = []
    node_text = []
    node_size = []
    node_color = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(f"{node}<br>Mentions: {G.nodes[node]['size']}<br>Sentiment: {G.nodes[node]['score']:.2f}")
        node_size.append(G.nodes[node]['size'] * 2)
        node_color.append(G.nodes[node]['color'])
    
    node_trace = go.Scatter(
        x=node_x, 
        y=node_y,
        mode='markers+text',
        text=list(G.nodes()),
        textposition="top center",
        marker=dict(
            size=node_size,
            color=node_color,
            line=dict(width=1, color='black')
        ),
        hoverinfo='text',
        hovertext=node_text
    )
    
    # Create figure
    fig = go.Figure(data=edge_trace + [node_trace],
                    layout=go.Layout(
                        title='Entity Relationship Network',
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        height=600
                    ))
    
    st.plotly_chart(fig, use_container_width=True)

def plot_sentiment_distribution_advanced(df):
    """Create an advanced sentiment distribution visualization"""
    if df.empty:
        st.warning("No sentiment data available to plot")
        return
    
    # Calculate sentiment distribution by domain
    domain_sentiment = df.groupby(['domain', 'overall_sentiment']).size().reset_index(name='count')
    
    # Define color map
    color_map = {'positive': 'green', 'neutral': 'gray', 'negative': 'red'}
    
    # Create plot
    fig = px.bar(
        domain_sentiment, 
        x='domain', 
        y='count', 
        color='overall_sentiment',
        barmode='group',
        color_discrete_map=color_map,
        title='Sentiment Distribution by Domain',
        labels={
            'domain': 'Domain', 
            'count': 'Number of Articles/Posts', 
            'overall_sentiment': 'Sentiment'
        }
    )
    
    # Improve layout
    fig.update_layout(
        height=400,
        xaxis_title="Domain",
        yaxis_title="Count",
        legend_title="Sentiment",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_sentiment_source_heatmap(df):
    """Create a heatmap of sentiment by source"""
    if df.empty:
        st.warning("No sentiment data available to plot")
        return
    
    # Get top sources by count
    top_sources = df['source'].value_counts().head(8).index.tolist()
    filtered_df = df[df['source'].isin(top_sources)]
    
    # Calculate average sentiment score by source and domain
    source_domain_sentiment = filtered_df.groupby(['source', 'domain'])['score'].mean().reset_index()
    
    # Convert to wide format for heatmap
    heatmap_data = source_domain_sentiment.pivot(index='source', columns='domain', values='score')
    
    # Replace NaN with 0.5 (neutral)
    heatmap_data = heatmap_data.fillna(0.5)
    
    # Create heatmap
    fig = px.imshow(
        heatmap_data,
        labels=dict(x="Domain", y="Source", color="Sentiment Score"),
        x=heatmap_data.columns,
        y=heatmap_data.index,
        color_continuous_scale='RdYlGn',
        zmin=0, zmax=1,
        title='Average Sentiment by Source and Domain'
    )
    
    # Improve layout
    fig.update_layout(
        height=450,
        xaxis_title="Domain",
        yaxis_title="Source",
        coloraxis_colorbar=dict(title="Sentiment Score")
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add explanation
    with st.expander("🔍 Understanding Source Analysis", expanded=True):
        st.markdown("""
        **What this shows:** This heatmap displays the average sentiment score for each source across different domains.
        
        **How to interpret:**
        - **Color intensity**: Green indicates positive sentiment, red indicates negative sentiment
        - **Variations by domain**: How each source's sentiment differs across domains (finance vs. tech)
        - **Variations by source**: How different sources cover the same domain (more positive vs. more negative)
        
        **Actionable insights:**
        - Sources with consistently positive or negative sentiment may have inherent biases
        - Differences between domains within the same source can indicate sector-specific sentiment
        - Sources with scores far from neutral (0.5) may provide stronger trading signals
        """)
        
        # Find most positive and negative sources
        if not heatmap_data.empty:
            # Calculate average sentiment across all domains
            overall_avg = heatmap_data.mean(axis=1)
            
            if not overall_avg.empty:
                most_positive = overall_avg.idxmax()
                most_negative = overall_avg.idxmin()
                
                st.markdown(f"**Most positive source overall:** {most_positive} (average score: {overall_avg[most_positive]:.2f})")
                st.markdown(f"**Most negative source overall:** {most_negative} (average score: {overall_avg[most_negative]:.2f})")

def plot_sentiment_time_heatmap(df):
    """Create a heatmap of sentiment over time of day and day of week"""
    if df.empty:
        st.warning("No sentiment data available to plot")
        return
    
    # Extract hour and day of week
    df['hour'] = df['created_at'].dt.tz_localize(None).dt.hour
    df['day_of_week'] = df['created_at'].dt.tz_localize(None).dt.day_name()
    
    # Set order of days
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    # Calculate average sentiment by hour and day
    time_sentiment = df.groupby(['day_of_week', 'hour'])['score'].mean().reset_index()
    
    # Create pivot table
    heatmap_data = time_sentiment.pivot(index='day_of_week', columns='hour', values='score')
    
    # Reorder days
    heatmap_data = heatmap_data.reindex(day_order)
    
    # Create heatmap
    fig = px.imshow(
        heatmap_data,
        labels=dict(x="Hour of Day", y="Day of Week", color="Sentiment Score"),
        x=list(range(0, 24)),
        y=day_order,
        color_continuous_scale='RdYlGn',
        zmin=0, zmax=1,
        title='Sentiment Patterns by Time of Day and Day of Week'
    )
    
    # Improve layout
    fig.update_layout(
        height=400,
        xaxis_title="Hour of Day",
        yaxis_title="Day of Week",
        coloraxis_colorbar=dict(title="Sentiment Score")
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_wordcloud(df, domain=None):
    """Generate a word cloud from text data"""
    try:
        from wordcloud import WordCloud
        import matplotlib.pyplot as plt
        import nltk
    except ImportError:
        st.warning("Required packages (wordcloud, nltk) not installed. Install with: pip install wordcloud nltk")
        return
    
    # Ensure NLTK data is downloaded
    try:
        # Check if the punkt tokenizer and stopwords data exists
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        from nltk.corpus import stopwords
        
        # Try a simple tokenization to test if it works
        try:
            nltk.word_tokenize("Test sentence.")
        except LookupError:
            # If direct tokenization fails, use a simpler approach
            st.warning("NLTK tokenization not fully available. Using simple tokenization instead.")
            def simple_tokenize(text):
                return text.lower().split()
            word_tokenize = simple_tokenize
        else:
            from nltk.tokenize import word_tokenize
    except Exception as e:
        st.error(f"Error setting up NLTK: {str(e)}")
        st.info("Using simple tokenization as fallback.")
        def simple_tokenize(text):
            return text.lower().split()
        word_tokenize = simple_tokenize
    
    if df.empty:
        st.warning("No text data available for word cloud")
        return
    
    # Filter by domain if specified
    if domain:
        filtered_df = df[df['domain'] == domain]
    else:
        filtered_df = df
    
    if filtered_df.empty:
        st.warning(f"No data available for domain: {domain}")
        return
    
    # Check if text column exists and has data
    if 'text' not in filtered_df.columns or filtered_df['text'].isna().all():
        st.warning("No text data available for word cloud analysis")
        
        # Try to use title column as fallback
        if 'title' in filtered_df.columns and not filtered_df['title'].isna().all():
            st.info("Using title data instead of text for word cloud")
            text_column = 'title'
        else:
            return
    else:
        text_column = 'text'
    
    # Combine all text
    all_text = " ".join([str(text) for text in filtered_df[text_column] if str(text) != "nan"])
    
    if not all_text.strip():
        st.warning("No text content available for word cloud")
        return
    
    # Define stopwords - use NLTK if available, otherwise define basic list
    try:
        stop_words = set(stopwords.words('english'))
    except:
        # Fallback basic stopwords list
        stop_words = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 
                     'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 
                     'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 
                     'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 
                     'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 
                     'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 
                     'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 
                     'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 
                     'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 
                     'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 
                     'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 
                     'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 
                     'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 
                     't', 'can', 'will', 'just', 'don', 'should', 'now'}
    
    # Add custom stopwords
    custom_stopwords = ['said', 'also', 'would', 'could', 'one', 'may', 'many', 'new', 
                       'using', 'used', 'like', 'say', 'says', 'year', 'company', 'today',
                       'tomorrow', 'reuters', 'ap', 'reported', 'report', 'according']
    stop_words.update(custom_stopwords)
    
    # Tokenize text - use simple approach for reliability
    tokens = all_text.lower().split()
    # Remove stopwords and short words
    filtered_tokens = [word for word in tokens if word.isalpha() and word not in stop_words and len(word) > 2]
    
    if not filtered_tokens:
        st.warning("No meaningful words found for word cloud after filtering")
        return
    
    # Create word cloud
    wordcloud = WordCloud(
        width=800, 
        height=400,
        background_color='white',
        colormap='viridis',
        max_words=100,
        collocations=False
    ).generate(" ".join(filtered_tokens))
    
    # Display
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    
    domain_title = f" for {domain}" if domain else ""
    plt.title(f"Word Cloud{domain_title}", fontsize=16)
    
    st.pyplot(fig)
    
    # Add explanation
    with st.expander("🔤 Understanding Word Clouds", expanded=True):
        st.markdown("""
        **What this shows:** This word cloud visualizes the most frequently mentioned words in the analyzed content, with larger words appearing more frequently.
        
        **How to interpret:**
        - **Word size**: Proportional to how frequently the word appears in the content
        - **Word placement**: Random and aesthetic only; position has no analytical meaning
        - **Word color**: For visual distinction only, colors don't represent sentiment or categories
        
        **Actionable insights:**
        - Identify key topics and themes dominating the discourse
        - Spot emerging terminology that might indicate new trends
        - Compare word clouds across domains or time periods to track shifting focus
        """)
        
        # Display top 10 words with counts for more precise analysis
        word_counts = {}
        for word in filtered_tokens:
            word_counts[word] = word_counts.get(word, 0) + 1
            
        # Sort by count and get top 10
        top_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        if top_words:
            st.markdown("#### Top 10 Most Frequent Terms:")
            
            # Create columns for word and count
            cols = st.columns(2)
            
            # Headers
            cols[0].markdown("**Term**")
            cols[1].markdown("**Frequency**")
            
            # Values
            for word, count in top_words:
                cols[0].markdown(f"{word}")
                cols[1].markdown(f"{count}")

# Dashboard layout
def main():
    # Header
    st.title("🔍 Advanced Sentiment Analysis Dashboard")
    st.markdown("### Real-time insights from financial and technology domains")
    
    # Sidebar for filters
    st.sidebar.header("Filters")
    
    days = st.sidebar.slider("Days to look back", 1, 90, 30)
    
    domain_options = ["All", "finance", "technology", "finance-tech"]
    selected_domain = st.sidebar.selectbox("Domain", domain_options)
    domain_filter = None if selected_domain == "All" else selected_domain
    
    data_tabs = st.sidebar.radio(
        "Data Views",
        ["AI Summary", "Sentiment Overview", "Source Analysis", "Entity Analysis", "Advanced Insights", "Comparative Analysis", "ML Models"]
    )
    
    # Load data
    with st.spinner("Loading sentiment data..."):
        sentiment_df = load_sentiment_data(days, domain_filter)
    
    # Main content based on selected tab
    if data_tabs == "AI Summary":
        # Create a summary dashboard with advanced analytics inspired by industrial-grade metrics
        st.subheader("📊 AI-Powered Sentiment Analysis")
        
        # Executive metrics 
        if not sentiment_df.empty:
            # Create three columns for key metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Overall sentiment trend
                trend_days = min(7, days)
                recent_df = sentiment_df[sentiment_df['created_at'] > (datetime.now().replace(tzinfo=timezone.utc) - timedelta(days=trend_days))]
                if not recent_df.empty:
                    daily_scores = recent_df.set_index('created_at').resample('D')['score'].mean()
                    
                    if len(daily_scores) >= 2:
                        trend = daily_scores.iloc[-1] - daily_scores.iloc[0]
                        trend_icon = "📈" if trend > 0.05 else ("📉" if trend < -0.05 else "➡️")
                        
                        # Format nicely
                        st.markdown(f"### {trend_icon} Sentiment Trend")
                        direction = "Up" if trend > 0 else ("Down" if trend < 0 else "Stable")
                        st.markdown(f"**{direction}** {abs(trend):.2f} over past {trend_days} days")
                        
                        # Show small sparkline
                        st.line_chart(daily_scores, height=100)
                    else:
                        st.markdown("### ➡️ Sentiment Trend")
                        st.markdown("Insufficient data for trend")
                else:
                    st.markdown("### ➡️ Sentiment Trend")
                    st.markdown("No recent data")
            
            with col2:
                # Source diversity
                source_count = len(sentiment_df['source'].unique())
                total_sources = sentiment_df['source'].nunique()
                
                st.markdown(f"### 🔍 Source Coverage")
                st.markdown(f"**{source_count}** unique sources")
                
                # Source diversity calculation (normalized entropy)
                if source_count > 1:
                    proportions = sentiment_df['source'].value_counts(normalize=True)
                    entropy = -sum(p * np.log(p) for p in proportions)
                    max_entropy = np.log(len(proportions))
                    diversity = entropy / max_entropy
                    
                    # Create a gauge-like metric
                    diversity_pct = int(diversity * 100)
                    st.markdown(f"**Diversity score: {diversity_pct}%**")
                    
                    # Interpret the diversity
                    if diversity > 0.8:
                        st.markdown("👍 Excellent source diversity")
                    elif diversity > 0.6:
                        st.markdown("✅ Good source diversity")
                    else:
                        st.markdown("⚠️ Limited source diversity")
                else:
                    st.markdown("⚠️ Single source only")
            
            with col3:
                # Sentiment polarity
                polarity = sentiment_df['overall_sentiment'].value_counts(normalize=True)
                
                # Calculate a market signal
                if 'positive' in polarity and 'negative' in polarity:
                    signal = (polarity.get('positive', 0) - polarity.get('negative', 0)) * 100
                    
                    st.markdown(f"### 🎯 Market Signal")
                    
                    # Categorize the signal
                    if signal > 30:
                        st.markdown(f"**Strong Bullish** (+{signal:.1f}%)")
                    elif signal > 10:
                        st.markdown(f"**Moderately Bullish** (+{signal:.1f}%)")
                    elif signal > -10:
                        st.markdown(f"**Neutral** ({signal:.1f}%)")
                    elif signal > -30:
                        st.markdown(f"**Moderately Bearish** ({signal:.1f}%)")
                    else:
                        st.markdown(f"**Strong Bearish** ({signal:.1f}%)")
                    
                    # Show gauge
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=signal,
                        domain={"x": [0, 1], "y": [0, 1]},
                        gauge={
                            "axis": {"range": [-100, 100]},
                            "bar": {"color": "darkblue"},
                            "steps": [
                                {"range": [-100, -30], "color": "red"},
                                {"range": [-30, -10], "color": "lightcoral"},
                                {"range": [-10, 10], "color": "lightgray"},
                                {"range": [10, 30], "color": "lightgreen"},
                                {"range": [30, 100], "color": "green"}
                            ]
                        }
                    ))
                    fig.update_layout(height=150, margin=dict(l=10, r=10, t=20, b=10))
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.markdown("### 🎯 Market Signal")
                    st.markdown("Insufficient sentiment data")
        
            # Visualization of emerging topics
            st.subheader("🔥 Emerging Topics & Entities")
            
            # Create columns for different insights
            col1, col2 = st.columns(2)
            
            with col1:
                # Create a simulated trending topics list
                st.markdown("#### Trending Topics")
                
                # Simulate trending topics with sentiment
                trending_topics = [
                    {"topic": "Artificial Intelligence", "sentiment": 0.85, "change": "+32%"},
                    {"topic": "Cloud Computing", "sentiment": 0.72, "change": "+18%"},
                    {"topic": "Interest Rates", "sentiment": 0.35, "change": "-12%"},
                    {"topic": "Regulatory Changes", "sentiment": 0.41, "change": "+5%"},
                    {"topic": "Supply Chain", "sentiment": 0.62, "change": "+8%"}
                ]
                
                # Display trending topics
                for topic in trending_topics:
                    sentiment_color = "green" if topic["sentiment"] > 0.6 else ("red" if topic["sentiment"] < 0.4 else "gray")
                    change_icon = "📈" if "+" in topic["change"] else "📉"
                    st.markdown(f"**{topic['topic']}** - <span style='color:{sentiment_color}'>{topic['sentiment']:.2f}</span> {change_icon} {topic['change']}", unsafe_allow_html=True)
            
            with col2:
                # Anomaly detection - show unusual sentiment shifts
                st.markdown("#### Unusual Activity")
                
                # Simulate detected anomalies
                anomalies = [
                    {"entity": "Tesla", "event": "Unusual positive sentiment spike", "time": "2 hours ago"},
                    {"entity": "Federal Reserve", "event": "Negative sentiment outlier", "time": "yesterday"},
                    {"entity": "Microsoft", "event": "Abnormal mention volume", "time": "today"}
                ]
                
                # Display anomalies
                for anomaly in anomalies:
                    st.markdown(f"⚠️ **{anomaly['entity']}**: {anomaly['event']} ({anomaly['time']})")
            
            # Sentiment wordcloud or heatmap
            st.subheader("🌐 Cross-Domain Sentiment Analysis")
            
            # Create two columns for domain comparison
            col1, col2 = st.columns(2)
            
            with col1:
                # Finance domain analysis
                st.markdown("#### Finance Domain")
                
                # Filter for finance domain
                finance_df = sentiment_df[sentiment_df['domain'].str.contains('finance', case=False, na=False)]
                
                if not finance_df.empty:
                    # Calculate overall sentiment
                    avg_sentiment = finance_df['score'].mean()
                    sentiment_status = "Positive" if avg_sentiment > 0.6 else ("Negative" if avg_sentiment < 0.4 else "Neutral")
                    
                    # Display domain metrics
                    st.markdown(f"**Sentiment:** {sentiment_status} ({avg_sentiment:.2f})")
                    st.markdown(f"**Articles/Posts:** {len(finance_df)}")
                    
                    # Top entities or topics
                    st.markdown("**Top mentions:**")
                    st.markdown("1. Interest Rates\n2. Banking\n3. Inflation\n4. Earnings\n5. Investment")
                else:
                    st.markdown("No finance domain data available")
            
            with col2:
                # Technology domain analysis
                st.markdown("#### Technology Domain")
                
                # Filter for technology domain
                tech_df = sentiment_df[sentiment_df['domain'].str.contains('tech', case=False, na=False)]
                
                if not tech_df.empty:
                    # Calculate overall sentiment
                    avg_sentiment = tech_df['score'].mean()
                    sentiment_status = "Positive" if avg_sentiment > 0.6 else ("Negative" if avg_sentiment < 0.4 else "Neutral")
                    
                    # Display domain metrics
                    st.markdown(f"**Sentiment:** {sentiment_status} ({avg_sentiment:.2f})")
                    st.markdown(f"**Articles/Posts:** {len(tech_df)}")
                    
                    # Top entities or topics
                    st.markdown("**Top mentions:**")
                    st.markdown("1. AI Development\n2. Cloud Services\n3. Cybersecurity\n4. Chip Shortage\n5. Innovation")
                else:
                    st.markdown("No technology domain data available")
                
            # Add disclaimer
            with st.expander("ℹ️ About this analysis"):
                st.markdown("""
                **Data sources:** This analysis is based on public data from social media, news articles, and financial reports.
                
                **Methodology:** The system uses advanced natural language processing techniques to extract sentiment and identify relationships between entities.
                
                **Limitations:** This dashboard shows a subset of available indicators and is intended for demonstration purposes. In a production environment, additional signals and more sophisticated models would be employed.
                
                **Disclaimer:** The information provided should not be used as the sole basis for investment decisions. Always conduct thorough research before making financial decisions.
                """)
        else:
            st.info("No data available for the selected filters. Please adjust your filter settings or ensure data collection is active.")
    
    elif data_tabs == "Sentiment Overview":
        # Sentiment metrics
        st.subheader("Sentiment Overview")
        
        # Show metrics
        if not sentiment_df.empty:
            col1, col2, col3, col4 = st.columns(4)
            
            # Total records
            col1.metric("Total Records", len(sentiment_df))
            
            # Average sentiment
            avg_sentiment = sentiment_df['score'].mean()
            sentiment_delta = avg_sentiment - 0.5  # Compare to neutral
            col2.metric("Average Sentiment", f"{avg_sentiment:.2f}", f"{sentiment_delta:.2f}")
            
            # Positive percentage
            positive_pct = (sentiment_df['overall_sentiment'] == 'positive').mean() * 100
            col3.metric("Positive Sentiment", f"{positive_pct:.1f}%")
            
            # Negative percentage
            negative_pct = (sentiment_df['overall_sentiment'] == 'negative').mean() * 100
            col4.metric("Negative Sentiment", f"{negative_pct:.1f}%")
            
            # Sentiment trends
            st.subheader("Sentiment Trends")
            plot_sentiment_trends(sentiment_df)
            
            # Distribution charts
            col1, col2 = st.columns(2)
            
            with col1:
                plot_domain_distribution(sentiment_df)
                
            with col2:
                plot_sentiment_distribution(sentiment_df)
            
            # Advanced visualization
            st.subheader("Temporal Patterns")
            plot_sentiment_time_heatmap(sentiment_df)
            
            # Word Cloud
            st.subheader("Word Cloud Analysis")
            domain_wordcloud = st.selectbox("Select domain for word cloud", ["All"] + list(sentiment_df['domain'].unique()))
            domain_for_wordcloud = None if domain_wordcloud == "All" else domain_wordcloud
            plot_wordcloud(sentiment_df, domain_for_wordcloud)
            
            # Recent sentiment
            st.subheader("Recent Sentiment Data")
            st.dataframe(
                sentiment_df.sort_values('created_at', ascending=False)
                .head(100)
                .reset_index(drop=True)
            )
        else:
            st.info("No sentiment data available for the selected filters")
    
    elif data_tabs == "Source Analysis":
        st.subheader("Source Analysis")
        
        if not sentiment_df.empty:
            # Source sentiment heatmap
            st.subheader("Sentiment by Source and Domain")
            plot_sentiment_source_heatmap(sentiment_df)
            
            # Source distribution
            st.subheader("Source Distribution")
            
            # Count by source
            source_counts = sentiment_df['source'].value_counts().reset_index()
            source_counts.columns = ['source', 'count']
            
            # Create plot
            fig = px.bar(
                source_counts.head(10), 
                x='source', 
                y='count',
                title='Top 10 Sources'
            )
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Add explanation for source distribution
            with st.expander("📊 Understanding Source Distribution", expanded=True):
                st.markdown("""
                **What this shows:** This chart shows the distribution of data sources in your analysis.
                
                **How to interpret:**
                - **Bar height**: Number of articles/posts from each source
                - **Source diversity**: More evenly distributed sources mean more balanced analysis
                
                **Actionable insights:**
                - Overreliance on a single source could introduce bias
                - Sources with fewer articles might represent specialized or exclusive content
                - Track changes in source distribution over time to detect shifts in coverage
                """)
            
            # Recent data by source
            st.subheader("Recent Data by Source")
            selected_source = st.selectbox("Select source", ["All"] + list(sentiment_df['source'].unique()))
            
            if selected_source != "All":
                filtered_source_df = sentiment_df[sentiment_df['source'] == selected_source]
            else:
                filtered_source_df = sentiment_df
            
            # Ensure required columns exist
            required_cols = ['created_at', 'domain', 'overall_sentiment', 'score']
            display_cols = required_cols.copy()
            
            # Only include title if it exists and has values
            if 'title' in filtered_source_df.columns and filtered_source_df['title'].notna().any():
                display_cols.insert(1, 'title')
            
            # Display the data
            st.dataframe(
                filtered_source_df[display_cols]
                .sort_values('created_at', ascending=False)
                .head(10)
                .reset_index(drop=True)
            )
        else:
            st.info("No source data available for the selected filters")
    
    elif data_tabs == "Entity Analysis":
        st.subheader("Entity Network Analysis")
        
        # Load entity data
        with st.spinner("Loading entity data..."):
            entities = load_entity_data()
            
            if entities is not None:
                # Entity visualization options
                view_type = st.radio("View Type", ["Bubble Chart", "Relationship Network"])
                
                if view_type == "Bubble Chart":
                    # Plot entity network
                    plot_entity_network(entities)
                else:
                    # Entity relationship network
                    plot_entity_relationships(entities)
                
                # Show top entities
                st.subheader("Top Influential Entities")
                
                # Create a cleaner table view
                entity_table = []
                for entity in entities[:20]:  # Top 20
                    entity_table.append({
                        "Entity": entity["entity"],
                        "Mentions": entity["mentions"],
                        "Sentiment": f"{entity['sentiment']:.2f}",
                        "Centrality": f"{entity['centrality_score']:.3f}",
                        "Domains": ", ".join(entity["domains"])
                    })
                
                st.table(pd.DataFrame(entity_table))
            else:
                st.info("No entity analysis results available yet. Please run entity analysis first.")
    
    elif data_tabs == "Advanced Insights":
        st.subheader("Market & Sentiment Correlation")
        
        # Load correlation data
        with st.spinner("Loading correlation data..."):
            correlation_df = load_correlation_results()
            
            if correlation_df is None:
                st.info("No correlation analysis results available - showing sample visualization for demonstration purposes")
                
                # Create sample correlation data
                sample_data = {
                    'AAPL': [0.72, 0.35, -0.28, 0.45],
                    'MSFT': [0.65, 0.43, -0.18, 0.51],
                    'GOOGL': [0.58, 0.39, -0.22, 0.38],
                    'AMZN': [0.62, 0.41, -0.15, 0.42],
                    'TSLA': [0.48, 0.37, -0.34, 0.29],
                    '^GSPC': [0.51, 0.29, -0.32, 0.37]
                }
                
                correlation_df = pd.DataFrame(sample_data, index=['finance_positive', 'tech_positive', 'negative_overall', 'entity_mentions'])
            
            # Plot correlation heatmap
            plot_correlation_heatmap(correlation_df)
            
            # Add explanation
            with st.expander("📈 Understanding Correlation Analysis", expanded=True):
                st.markdown("""
                **What this shows:** This heatmap visualizes the correlation between sentiment scores and market price movements.
                
                **How to interpret:**
                - **Color intensity**: Strength of correlation
                - **Blue (positive)**: As sentiment increases, price tends to increase
                - **Red (negative)**: As sentiment increases, price tends to decrease
                - **Values near 0 (white)**: Little to no correlation
                
                **Actionable insights:**
                - Strong correlations (dark blue/red) may indicate predictive relationships
                - Different correlations across sectors suggest varied impact of sentiment
                - Negative correlations can be valuable contrarian indicators
                
                **Technical note:** Correlation measures the linear relationship between two variables but doesn't necessarily imply causation.
                """)
                
                # Highlight strongest correlations
                if not correlation_df.empty:
                    # Find max correlation
                    max_corr = correlation_df.max().max()
                    max_corr_symbol = correlation_df.max().idxmax()
                    max_corr_sentiment = correlation_df.idxmax()[max_corr_symbol]
                    
                    # Find min correlation
                    min_corr = correlation_df.min().min()
                    min_corr_symbol = correlation_df.min().idxmin()
                    min_corr_sentiment = correlation_df.idxmin()[min_corr_symbol]
                    
                    st.markdown(f"**Strongest positive correlation:** {max_corr_sentiment} → {max_corr_symbol} ({max_corr:.2f})")
                    st.markdown(f"**Strongest negative correlation:** {min_corr_sentiment} → {min_corr_symbol} ({min_corr:.2f})")
            
            # Show correlation table
            st.subheader("Correlation Table")
            st.dataframe(correlation_df)
                
            # Load some market data for symbols
            symbol_options = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "^GSPC", "^DJI"]
            selected_symbols = st.multiselect("Select market symbols to display", symbol_options, default=["AAPL", "^GSPC", "GOOGL"])
            
            if selected_symbols:
                with st.spinner("Loading market data..."):
                    market_df = load_market_data(days, selected_symbols)
                    
                    if not market_df.empty:
                        # Pivot data
                        market_pivot = market_df.pivot(index='timestamp', columns='symbol', values='close')
                        
                        # Plot market data
                        fig = px.line(
                            market_pivot,
                            title='Market Price Trends',
                            labels={'timestamp': 'Date', 'value': 'Price', 'symbol': 'Symbol'}
                        )
                        
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Add explanation
                        with st.expander("📉 Understanding Market Trends", expanded=True):
                            st.markdown("""
                            **What this shows:** This chart displays price movements for selected market symbols over time.
                            
                            **How to interpret:**
                            - **Line direction**: Upward trends indicate price increases, downward trends indicate price decreases
                            - **Line steepness**: Indicates the rate of price change - steeper lines mean faster changes
                            - **Convergence/divergence**: When lines move together or apart, indicating correlation or disconnection
                            
                            **Actionable insights:**
                            - Compare price movements with sentiment shifts to identify potential leading indicators
                            - Look for divergence between related companies that may indicate sector rotation
                            - Use alongside sentiment trends to develop trading hypotheses
                            """)
                    else:
                        st.info("No market data available for the selected symbols")
    
    elif data_tabs == "Comparative Analysis":
        st.subheader("Comparative Sentiment Analysis")
        
        if not sentiment_df.empty:
            # Create tabs for different comparison types
            compare_tabs = st.tabs(["Domain Comparison", "Time Period Comparison", "Entity Sentiment Comparison"])
            
            with compare_tabs[0]:
                st.subheader("Domain Sentiment Comparison")
                
                # Filter for domains
                available_domains = sentiment_df['domain'].unique()
                
                # Need at least 2 domains for comparison
                if len(available_domains) >= 2:
                    # Select domains to compare
                    selected_domains = st.multiselect(
                        "Select domains to compare", 
                        options=available_domains,
                        default=list(available_domains)[:2]
                    )
                    
                    if len(selected_domains) >= 2:
                        # Filter data
                        domain_data = sentiment_df[sentiment_df['domain'].isin(selected_domains)]
                        
                        # Group by date and domain
                        domain_data['date'] = domain_data['created_at'].dt.date
                        daily_domain_sentiment = domain_data.groupby(['date', 'domain'])['score'].mean().reset_index()
                        daily_domain_sentiment['date'] = pd.to_datetime(daily_domain_sentiment['date'])
                        
                        # Plot comparison
                        fig = px.line(
                            daily_domain_sentiment, 
                            x='date', 
                            y='score', 
                            color='domain',
                            title='Sentiment Comparison by Domain',
                            labels={
                                'date': 'Date',
                                'score': 'Sentiment Score',
                                'domain': 'Domain'
                            }
                        )
                        
                        # Add a reference line for neutral sentiment
                        fig.add_hline(y=0.5, line_dash="dash", line_color="gray", annotation_text="Neutral")
                        
                        # Improve layout
                        fig.update_layout(
                            height=500,
                            xaxis_title="Date",
                            yaxis_title="Sentiment Score",
                            yaxis=dict(range=[0, 1])
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Add explanation
                        with st.expander("📊 Understanding Domain Comparison", expanded=True):
                            st.markdown("""
                            **What this shows:** This chart compares sentiment trends across different domains over time.
                            
                            **How to interpret:**
                            - **Line height**: Average sentiment score for each domain
                            - **Divergence**: When domains move in opposite directions, indicating sector-specific factors
                            - **Convergence**: When domains move together, suggesting broad market factors
                            
                            **Actionable insights:**
                            - Divergence can indicate rotation between sectors
                            - Leading domain trends may predict lagging domain trends
                            - Extreme divergence may indicate sector-specific events
                            """)
                        
                        # Add statistical comparison
                        st.subheader("Statistical Comparison")
                        
                        # Calculate domain statistics
                        domain_stats = []
                        for domain in selected_domains:
                            domain_df = sentiment_df[sentiment_df['domain'] == domain]
                            
                            stats = {
                                'Domain': domain,
                                'Average Sentiment': domain_df['score'].mean(),
                                'Variance': domain_df['score'].var(),
                                'Min Score': domain_df['score'].min(),
                                'Max Score': domain_df['score'].max(),
                                'Positive %': (domain_df['overall_sentiment'] == 'positive').mean() * 100,
                                'Negative %': (domain_df['overall_sentiment'] == 'negative').mean() * 100,
                                'Article Count': len(domain_df)
                            }
                            
                            domain_stats.append(stats)
                        
                        # Create statistics table
                        stats_df = pd.DataFrame(domain_stats)
                        stats_df = stats_df.set_index('Domain')
                        
                        # Format numerical columns
                        for col in stats_df.columns:
                            if stats_df[col].dtype in ['float64', 'float32']:
                                stats_df[col] = stats_df[col].round(2)
                        
                        # Display table with highlighting
                        st.dataframe(stats_df.style.highlight_max(axis=0, color='lightgreen').highlight_min(axis=0, color='#FFB6C1'))
                        
                        # Calculate correlation between domains
                        st.subheader("Domain Correlation Analysis")
                        
                        # Check if we can calculate correlation (need overlapping dates)
                        domain_pivot = daily_domain_sentiment.pivot(index='date', columns='domain', values='score')
                        
                        if not domain_pivot.empty and not domain_pivot.isna().all().all():
                            # Calculate correlation
                            corr_matrix = domain_pivot.corr()
                            
                            # Create heatmap
                            fig = px.imshow(
                                corr_matrix,
                                text_auto=True,
                                color_continuous_scale='RdBu_r',
                                zmin=-1, zmax=1,
                                title='Domain Sentiment Correlation'
                            )
                            
                            fig.update_layout(height=400)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Interpretation
                            highest_corr = corr_matrix.unstack().sort_values(ascending=False)
                            # Remove self-correlations (which are always 1.0)
                            highest_corr = highest_corr[highest_corr < 0.999]
                            
                            if not highest_corr.empty:
                                top_corr_pair = highest_corr.index[0]
                                top_corr_value = highest_corr.iloc[0]
                                
                                if top_corr_value > 0.7:
                                    st.markdown(f"**Strong Positive Correlation:** {top_corr_pair[0]} and {top_corr_pair[1]} show strong positive correlation ({top_corr_value:.2f}), suggesting these domains tend to move together.")
                                elif top_corr_value < -0.7:
                                    st.markdown(f"**Strong Negative Correlation:** {top_corr_pair[0]} and {top_corr_pair[1]} show strong negative correlation ({top_corr_value:.2f}), suggesting these domains tend to move in opposite directions.")
                        else:
                            st.info("Insufficient overlapping data to calculate correlation between domains.")
                    else:
                        st.warning("Please select at least two domains to compare.")
                else:
                    st.warning("Need at least two different domains in the data for comparison.")
            
            with compare_tabs[1]:
                st.subheader("Time Period Comparison")
                
                # Define time periods
                period_options = {
                    "Last Week": 7,
                    "Last Month": 30,
                    "Last Quarter": 90,
                    "Custom": "custom"
                }
                
                # Select time periods to compare
                period1 = st.selectbox("First period", options=list(period_options.keys()), index=0)
                period2 = st.selectbox("Second period", options=list(period_options.keys()), index=1)
                
                # Handle custom periods
                if period1 == "Custom":
                    days1 = st.number_input("Days for first period", min_value=1, max_value=365, value=14)
                else:
                    days1 = period_options[period1]
                
                if period2 == "Custom":
                    days2 = st.number_input("Days for second period", min_value=1, max_value=365, value=30)
                else:
                    days2 = period_options[period2]
                
                # Calculate date ranges
                end_date = datetime.now().replace(tzinfo=timezone.utc)
                
                period1_start = end_date - timedelta(days=days1)
                period1_label = f"{period1} ({days1} days)" if period1 != "Custom" else f"Custom ({days1} days)"
                
                period2_start = end_date - timedelta(days=days2)
                period2_label = f"{period2} ({days2} days)" if period2 != "Custom" else f"Custom ({days2} days)"
                
                # Filter data for each period
                period1_data = sentiment_df[sentiment_df['created_at'] >= period1_start]
                period2_data = sentiment_df[sentiment_df['created_at'] >= period2_start]
                
                # Calculate daily sentiment for each period
                # For period 1, we need to align the dates for comparison
                period1_data['days_ago'] = (end_date - period1_data['created_at']).dt.days
                period1_daily = period1_data.groupby('days_ago')['score'].mean().reset_index()
                period1_daily = period1_daily[period1_daily['days_ago'] < days1]
                period1_daily['period'] = period1_label
                
                # For period 2
                period2_data['days_ago'] = (end_date - period2_data['created_at']).dt.days
                period2_daily = period2_data.groupby('days_ago')['score'].mean().reset_index()
                period2_daily = period2_daily[period2_daily['days_ago'] < days2]
                period2_daily['period'] = period2_label
                
                # Compare periods if data exists
                if not period1_daily.empty and not period2_daily.empty:
                    # Create comparison chart
                    
                    # Option to normalize time axis
                    normalize_time = st.checkbox("Normalize time axis (align endpoints)", value=True)
                    
                    if normalize_time:
                        # Normalize the days_ago to percentage of period
                        period1_daily['normalized_time'] = period1_daily['days_ago'] / max(period1_daily['days_ago']) * 100
                        period2_daily['normalized_time'] = period2_daily['days_ago'] / max(period2_daily['days_ago']) * 100
                        
                        # Combine data
                        plot_df = pd.concat([period1_daily, period2_daily])
                        
                        # Create plot
                        fig = px.line(
                            plot_df, 
                            x='normalized_time', 
                            y='score', 
                            color='period',
                            title='Sentiment Comparison by Time Period (Normalized)',
                            labels={
                                'normalized_time': '% of Period (0% = most recent)',
                                'score': 'Sentiment Score',
                                'period': 'Time Period'
                            }
                        )
                        
                        # Improve layout - reverse x-axis so most recent is on right
                        fig.update_layout(
                            height=500,
                            xaxis=dict(title='% of Period (0% = most recent)', autorange="reversed"),
                            yaxis=dict(title='Sentiment Score', range=[0, 1])
                        )
                    else:
                        # Just use days_ago directly
                        plot_df = pd.concat([period1_daily, period2_daily])
                        
                        # Create plot
                        fig = px.line(
                            plot_df, 
                            x='days_ago', 
                            y='score', 
                            color='period',
                            title='Sentiment Comparison by Time Period',
                            labels={
                                'days_ago': 'Days Ago',
                                'score': 'Sentiment Score',
                                'period': 'Time Period'
                            }
                        )
                        
                        # Improve layout - reverse x-axis so most recent is on right
                        fig.update_layout(
                            height=500,
                            xaxis=dict(title='Days Ago', autorange="reversed"),
                            yaxis=dict(title='Sentiment Score', range=[0, 1])
                        )
                    
                    # Add a reference line for neutral sentiment
                    fig.add_hline(y=0.5, line_dash="dash", line_color="gray", annotation_text="Neutral")
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add explanation
                    with st.expander("📊 Understanding Time Period Comparison", expanded=True):
                        st.markdown("""
                        **What this shows:** This chart compares sentiment patterns across different time periods.
                        
                        **How to interpret:**
                        - **Line patterns**: Similar patterns indicate consistent sentiment behavior
                        - **Divergence points**: Key moments where sentiment patterns changed
                        - **Overall level**: Whether sentiment is generally more positive/negative in one period
                        
                        **Normalized view explanation:**
                        When normalized, the time periods are stretched/compressed to align the endpoints (0% = most recent data point, 100% = oldest in the period). This helps compare the pattern regardless of the different period lengths.
                        
                        **Actionable insights:**
                        - Pattern similarity can suggest cyclical market behavior
                        - Current sentiment trajectory can be compared to historical patterns
                        - Extreme differences between periods may indicate changing market regimes
                        """)
                    
                    # Calculate summary statistics
                    st.subheader("Period Comparison Statistics")
                    
                    period1_avg = period1_data['score'].mean()
                    period2_avg = period2_data['score'].mean()
                    
                    period1_pos_pct = (period1_data['overall_sentiment'] == 'positive').mean() * 100
                    period2_pos_pct = (period2_data['overall_sentiment'] == 'positive').mean() * 100
                    
                    period1_neg_pct = (period1_data['overall_sentiment'] == 'negative').mean() * 100
                    period2_neg_pct = (period2_data['overall_sentiment'] == 'negative').mean() * 100
                    
                    period1_volatility = period1_data['score'].std()
                    period2_volatility = period2_data['score'].std()
                    
                    # Create comparison metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Average Sentiment", f"{period1_avg:.2f}", f"{period1_avg - period2_avg:+.2f} vs {period2_label}")
                        st.metric("Positive Percentage", f"{period1_pos_pct:.1f}%", f"{period1_pos_pct - period2_pos_pct:+.1f}% vs {period2_label}")
                    
                    with col2:
                        st.metric("Data Points", f"{len(period1_data)}", f"{len(period1_data) - len(period2_data):+d} vs {period2_label}")
                        st.metric("Negative Percentage", f"{period1_neg_pct:.1f}%", f"{period1_neg_pct - period2_neg_pct:+.1f}% vs {period2_label}")
                    
                    with col3:
                        st.metric("Sentiment Volatility", f"{period1_volatility:.3f}", f"{period1_volatility - period2_volatility:+.3f} vs {period2_label}")
                        
                        # Calculate trend (slope of best fit line)
                        period1_slope = np.polyfit(period1_daily['days_ago'], period1_daily['score'], 1)[0] * -100  # Negate because days_ago is reversed
                        period2_slope = np.polyfit(period2_daily['days_ago'], period2_daily['score'], 1)[0] * -100  # Multiply by 100 for readability
                        
                        st.metric("Sentiment Trend", f"{period1_slope:.2f}", f"{period1_slope - period2_slope:+.2f} vs {period2_label}")
                else:
                    st.warning("Insufficient data for one or both time periods. Try selecting longer periods.")
            
            with compare_tabs[2]:
                st.subheader("Entity Sentiment Comparison")
                
                # Load entity data
                entities = load_entity_data()
                
                if entities is not None:
                    # Get top entities
                    top_entities = [entity['entity'] for entity in entities[:10]]
                    
                    # Select entities to compare
                    selected_entities = st.multiselect(
                        "Select entities to compare",
                        options=top_entities,
                        default=top_entities[:3] if len(top_entities) >= 3 else top_entities
                    )
                    
                    if selected_entities:
                        # Create entity comparison data
                        entity_data = []
                        
                        for entity in entities:
                            if entity['entity'] in selected_entities:
                                entity_data.append({
                                    'Entity': entity['entity'],
                                    'Sentiment': entity['sentiment'],
                                    'Mentions': entity['mentions'],
                                    'Centrality': entity['centrality_score']
                                })
                        
                        entity_df = pd.DataFrame(entity_data)
                        
                        # Create radar chart for multi-dimensional comparison
                        # Normalize metrics for radar chart
                        metrics = ['Sentiment', 'Mentions', 'Centrality']
                        
                        # Create a normalized version of each metric
                        for metric in metrics:
                            if len(entity_df) > 1:  # Need at least 2 rows to normalize
                                min_val = entity_df[metric].min()
                                max_val = entity_df[metric].max()
                                
                                if max_val > min_val:  # Avoid division by zero
                                    entity_df[f'{metric}_Norm'] = (entity_df[metric] - min_val) / (max_val - min_val)
                                else:
                                    entity_df[f'{metric}_Norm'] = 1.0
                            else:
                                entity_df[f'{metric}_Norm'] = 1.0
                        
                        # Create radar chart
                        fig = go.Figure()
                        
                        for _, row in entity_df.iterrows():
                            fig.add_trace(go.Scatterpolar(
                                r=[row['Sentiment_Norm'], row['Mentions_Norm'], row['Centrality_Norm']],
                                theta=metrics,
                                fill='toself',
                                name=row['Entity']
                            ))
                        
                        fig.update_layout(
                            polar=dict(
                                radialaxis=dict(
                                    visible=True,
                                    range=[0, 1]
                                )
                            ),
                            title="Entity Comparison Radar Chart",
                            height=500
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Create bar chart comparison
                        st.subheader("Entity Sentiment Comparison")
                        
                        fig = px.bar(
                            entity_df,
                            x='Entity',
                            y='Sentiment',
                            color='Entity',
                            title="Entity Sentiment Score Comparison"
                        )
                        
                        # Add a reference line for neutral sentiment
                        fig.add_hline(y=0.5, line_dash="dash", line_color="gray", annotation_text="Neutral")
                        
                        fig.update_layout(
                            height=400,
                            xaxis_title="Entity",
                            yaxis_title="Sentiment Score",
                            yaxis=dict(range=[0, 1])
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Add explanation
                        with st.expander("📊 Understanding Entity Comparison", expanded=True):
                            st.markdown("""
                            **What this shows:** These charts compare key metrics across selected entities.
                            
                            **Radar Chart:**
                            - Shows normalized values across three dimensions: Sentiment, Mentions, and Centrality
                            - Larger area indicates stronger overall presence and impact
                            - Different shapes reflect different strengths
                            
                            **Bar Chart:**
                            - Directly compares raw sentiment scores between entities
                            - Values above 0.5 indicate positive sentiment
                            - Values below 0.5 indicate negative sentiment
                            
                            **Actionable insights:**
                            - Entities with high sentiment but low mentions may represent emerging opportunities
                            - Entities with high centrality but low sentiment may indicate potential risks
                            - Compare related entities to spot divergences that might signal changing trends
                            """)
                        
                        # Display data table
                        st.subheader("Entity Data")
                        st.dataframe(entity_df)
                    else:
                        st.warning("Please select at least one entity to analyze.")
                else:
                    st.info("No entity data available. Entity analysis must be run first.")
        else:
            st.info("No sentiment data available for comparative analysis. Please adjust your filter settings or ensure data collection is active.")
    
    elif data_tabs == "ML Models":
        st.subheader("Machine Learning Models")
        
        # Add tabs for different ML features
        ml_tabs = st.tabs(["Model Usage", "Sentiment Forecasting", "Try Sentiment Analysis"])
        
        with ml_tabs[0]:
            # Model usage metrics
            if not sentiment_df.empty:
                # Count by model type
                model_counts = sentiment_df['model'].value_counts().reset_index()
                model_counts.columns = ['model', 'count']
                
                # Create plot
                fig = px.pie(
                    model_counts, 
                    values='count', 
                    names='model',
                    title='ML Model Usage Distribution'
                )
                
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # Model confidence
                confidence_counts = sentiment_df['confidence'].value_counts().reset_index()
                confidence_counts.columns = ['confidence', 'count']
                
                # Create bar chart
                fig = px.bar(
                    confidence_counts, 
                    x='confidence', 
                    y='count',
                    color='confidence',
                    color_discrete_map={'high': 'green', 'medium': 'yellow', 'low': 'red'},
                    title='Model Confidence Distribution'
                )
                
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No sentiment data available for model usage metrics")
        
        with ml_tabs[1]:
            # Sentiment Forecasting
            st.subheader("Sentiment Forecast")
            
            if not sentiment_df.empty:
                # Prepare time series data
                # Group by day and calculate mean sentiment
                sentiment_df['date'] = sentiment_df['created_at'].dt.date
                daily_sentiment = sentiment_df.groupby('date')['score'].mean().reset_index()
                daily_sentiment['date'] = pd.to_datetime(daily_sentiment['date'])
                
                # Sort by date
                daily_sentiment = daily_sentiment.sort_values('date')
                
                # Check if we have enough data
                if len(daily_sentiment) >= 7:
                    # Simple forecasting model using exponential smoothing
                    from statsmodels.tsa.holtwinters import ExponentialSmoothing
                    
                    # Prepare the data
                    y = daily_sentiment['score'].values
                    
                    # Set forecast days
                    forecast_days = st.slider("Forecast Days", 1, 14, 7)
                    
                    # Create and fit the model
                    try:
                        model = ExponentialSmoothing(y, trend='add', seasonal=None, damped=True)
                        fitted_model = model.fit()
                        
                        # Make forecast
                        forecast = fitted_model.forecast(forecast_days)
                        
                        # Create forecast dataframe
                        last_date = daily_sentiment['date'].iloc[-1]
                        forecast_dates = [last_date + timedelta(days=i+1) for i in range(forecast_days)]
                        forecast_df = pd.DataFrame({
                            'date': forecast_dates,
                            'score': forecast
                        })
                        
                        # Combine historical and forecast data
                        historical_df = daily_sentiment.copy()
                        historical_df['type'] = 'Historical'
                        forecast_df['type'] = 'Forecast'
                        
                        plot_df = pd.concat([historical_df, forecast_df])
                        
                        # Create plot
                        fig = px.line(
                            plot_df, 
                            x='date', 
                            y='score',
                            color='type',
                            title='Sentiment Forecast',
                            color_discrete_map={'Historical': 'blue', 'Forecast': 'red'}
                        )
                        
                        # Add a reference line for neutral sentiment
                        fig.add_hline(y=0.5, line_dash="dash", line_color="gray", annotation_text="Neutral")
                        
                        # Improve layout
                        fig.update_layout(
                            height=500,
                            xaxis_title="Date",
                            yaxis_title="Sentiment Score",
                            yaxis=dict(range=[0, 1])
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Model metrics and interpretation
                        forecast_direction = "positive" if forecast[-1] > y[-1] else "negative"
                        forecast_change = abs(forecast[-1] - y[-1])
                        
                        # Interpretation
                        st.subheader("Forecast Interpretation")
                        
                        # Calculate metrics
                        current_sentiment = y[-1]
                        forecasted_sentiment = forecast[-1]
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Current Sentiment", f"{current_sentiment:.2f}", "Current")
                        
                        with col2:
                            delta = f"{forecasted_sentiment - current_sentiment:+.2f}"
                            st.metric("Forecasted Sentiment", f"{forecasted_sentiment:.2f}", delta)
                        
                        # Interpretation text
                        st.markdown(f"**Forecast Direction:** The model predicts sentiment will trend **{forecast_direction}** in the next {forecast_days} days.")
                        
                        if forecast[-1] > 0.7:
                            st.markdown("**Strongly Positive Outlook:** The forecast indicates a highly positive sentiment trend, suggesting potentially favorable market conditions.")
                        elif forecast[-1] > 0.55:
                            st.markdown("**Moderately Positive Outlook:** The forecast shows a positive sentiment trend, which may indicate favorable market conditions.")
                        elif forecast[-1] > 0.45:
                            st.markdown("**Neutral Outlook:** The forecast suggests sentiment will remain relatively neutral.")
                        elif forecast[-1] > 0.3:
                            st.markdown("**Moderately Negative Outlook:** The forecast indicates a negative sentiment trend, which may suggest caution.")
                        else:
                            st.markdown("**Strongly Negative Outlook:** The forecast shows a highly negative sentiment trend, which could indicate unfavorable market conditions.")
                        
                        # Add explanation
                        with st.expander("📈 Understanding Sentiment Forecasting", expanded=True):
                            st.markdown("""
                            **What this shows:** This chart forecasts future sentiment scores based on historical patterns using exponential smoothing.
                            
                            **How to interpret:**
                            - **Blue line**: Historical sentiment data
                            - **Red line**: Forecasted sentiment trajectory
                            - **Upward trend**: Predicts improving sentiment
                            - **Downward trend**: Predicts deteriorating sentiment
                            
                            **Forecasting method:**
                            - Exponential smoothing with trend component
                            - More recent data points are weighted more heavily
                            - No seasonal component is included in this simple model
                            
                            **Limitations:**
                            - This is a simple time-series forecast based solely on historical sentiment patterns
                            - Does not incorporate external factors like scheduled news events or market dynamics
                            - Accuracy decreases for longer forecast horizons
                            
                            **Actionable insights:**
                            - Use forecasted direction as one input for anticipating market sentiment shifts
                            - Consider the forecast confidence interval (not shown) would widen over time
                            - Compare forecasts across different domains to identify diverging trends
                            """)
                    
                    except Exception as e:
                        st.error(f"Error in forecast model: {str(e)}")
                        st.info("Try adjusting the date range or ensuring more data is available.")
                else:
                    st.warning("Insufficient historical data for forecasting. At least 7 days of data are required.")
                    st.info("Try increasing the 'Days to look back' value in the sidebar or collecting more data.")
            else:
                st.info("No sentiment data available for forecasting")
        
        with ml_tabs[2]:
            # Text input for sentiment analysis
            st.subheader("Try Sentiment Analysis")
            st.markdown("Enter text to analyze sentiment in real-time:")
            
            user_text = st.text_area("", "Apple announced record quarterly earnings, exceeding analyst expectations.")
            domain = st.selectbox("Select domain for analysis", ["finance", "technology", "general"])
            
            if st.button("Analyze Sentiment"):
                try:
                    # Create a simple sentiment analyzer
                    from textblob import TextBlob
                    
                    # Analyze sentiment
                    blob = TextBlob(user_text)
                    sentiment_score = (blob.sentiment.polarity + 1) / 2  # Convert from [-1, 1] to [0, 1]
                    
                    # Determine sentiment category
                    if sentiment_score > 0.6:
                        sentiment_category = "positive"
                    elif sentiment_score < 0.4:
                        sentiment_category = "negative"
                    else:
                        sentiment_category = "neutral"
                    
                    # Display result
                    col1, col2, col3 = st.columns(3)
                    
                    sentiment_color = "green" if sentiment_category == "positive" else ("red" if sentiment_category == "negative" else "gray")
                    col1.markdown(f"**Sentiment:** <span style='color:{sentiment_color}'>{sentiment_category.upper()}</span>", unsafe_allow_html=True)
                    col2.markdown(f"**Score:** {sentiment_score:.2f}")
                    col3.markdown(f"**Model:** TextBlob")
                    
                    # Visualization
                    fig, ax = plt.subplots(figsize=(10, 2))
                    cmap = sns.color_palette("RdYlGn", 100)
                    
                    # Map score from 0-1 to colormap (0-99)
                    color_idx = min(int(sentiment_score * 99), 99)
                    
                    ax.barh([0], [1], color=cmap[color_idx])
                    ax.set_xlim(0, 1)
                    ax.set_ylim(-0.5, 0.5)
                    ax.set_yticks([])
                    ax.set_xticks([0, 0.5, 1])
                    ax.set_xticklabels(['Negative', 'Neutral', 'Positive'])
                    ax.axvline(x=0.5, color='gray', linestyle='--')
                    ax.axvline(x=sentiment_score, color='black', linestyle='-', linewidth=3)
                    
                    st.pyplot(fig)
                    
                    # Add detailed analysis
                    st.subheader("Detailed Analysis")
                    
                    # Extract key phrases
                    words = user_text.split()
                    word_sentiments = []
                    
                    for word in words:
                        if len(word) > 3:  # Skip short words
                            word_blob = TextBlob(word)
                            word_score = (word_blob.sentiment.polarity + 1) / 2
                            word_sentiments.append((word, word_score))
                    
                    # Sort by sentiment
                    positive_words = [w for w, s in word_sentiments if s > 0.6]
                    negative_words = [w for w, s in word_sentiments if s < 0.4]
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Positive Words:**")
                        if positive_words:
                            st.write(", ".join(positive_words))
                        else:
                            st.write("None detected")
                    
                    with col2:
                        st.markdown("**Negative Words:**")
                        if negative_words:
                            st.write(", ".join(negative_words))
                        else:
                            st.write("None detected")
                    
                except Exception as e:
                    st.error(f"Error analyzing sentiment: {str(e)}")
                    st.info("Try installing TextBlob with: pip install textblob")

# Run the app
if __name__ == "__main__":
    main() 