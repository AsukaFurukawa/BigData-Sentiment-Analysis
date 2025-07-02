import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import json
import os
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import altair as alt

# Set page configuration
st.set_page_config(
    page_title="Finance & Tech Sentiment Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Path to data directory
base_path = os.path.dirname(__file__)
data_dir = os.path.join(os.path.dirname(base_path), 'data-storage', 'data')
db_path = os.path.join(data_dir, 'sentiment_data.db')

# Check if database exists
db_exists = os.path.isfile(db_path)

# Connect to SQLite database
@st.cache_resource
def get_connection():
    return sqlite3.connect(db_path)

# Load data from database
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_sentiment_data(days=7):
    if not db_exists:
        return pd.DataFrame()
    
    conn = get_connection()
    
    cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
    
    query = f"""
    SELECT 
        s.id, s.source, s.created_at, s.text, s.domain, s.overall_sentiment, s.score,
        f.entities as finance_entities,
        t.entities as tech_entities, t.categories as tech_categories
    FROM sentiment_results s
    LEFT JOIN finance_analysis f ON s.id = f.id
    LEFT JOIN tech_analysis t ON s.id = t.id
    WHERE s.created_at > ?
    ORDER BY s.created_at DESC
    LIMIT 10000
    """
    
    df = pd.read_sql_query(query, conn, params=(cutoff_date,))
    
    # Convert timestamp strings to datetime
    df['created_at'] = pd.to_datetime(df['created_at'])
    
    # Add date and hour columns for aggregation
    df['date'] = df['created_at'].dt.date
    df['hour'] = df['created_at'].dt.hour
    
    return df

# Parse JSON columns
def parse_json_column(df, column):
    if column not in df.columns or df.empty:
        return df
    
    # Create a new dataframe with the parsed JSON
    parsed = df[column].apply(lambda x: json.loads(x) if pd.notna(x) else [])
    
    return parsed

# Sidebar for filters
st.sidebar.title("Sentiment Analysis Dashboard")
st.sidebar.image("https://www.pngall.com/wp-content/uploads/8/Analysis-PNG-Free-Image.png", width=100)

# Time filter
time_filter = st.sidebar.selectbox(
    "Time Range",
    ["Last 24 Hours", "Last 7 Days", "Last 30 Days", "All Time"],
    index=1
)

if time_filter == "Last 24 Hours":
    days = 1
elif time_filter == "Last 7 Days":
    days = 7
elif time_filter == "Last 30 Days":
    days = 30
else:
    days = 365  # All time (limit to a year)

# Domain filter
domain_filter = st.sidebar.multiselect(
    "Domain",
    ["finance", "technology", "finance-tech", "tech-finance", "general"],
    default=["finance", "technology"]
)

# Source filter
source_filter = st.sidebar.multiselect(
    "Source",
    ["twitter", "reddit", "news"],
    default=["twitter", "reddit", "news"]
)

# Sentiment filter
sentiment_filter = st.sidebar.multiselect(
    "Sentiment",
    ["positive", "negative", "neutral"],
    default=["positive", "negative", "neutral"]
)

# Load data
try:
    df = load_sentiment_data(days)
    
    # If no data, try loading all data without date filter
    if df.empty:
        st.warning("No recent data found, loading all available data...")
        conn = get_connection()
        query = "SELECT id, source, created_at, text, domain, overall_sentiment, score FROM sentiment_results ORDER BY created_at DESC LIMIT 1000"
        df = pd.read_sql_query(query, conn)
        df['created_at'] = pd.to_datetime(df['created_at'])
        df['date'] = df['created_at'].dt.date
        df['hour'] = df['created_at'].dt.hour
        conn.close()
    
    # Apply filters
    if domain_filter and not df.empty:
        df = df[df['domain'].isin(domain_filter)]
    
    if source_filter and not df.empty:
        df = df[df['source'].isin(source_filter)]
    
    if sentiment_filter and not df.empty:
        df = df[df['overall_sentiment'].isin(sentiment_filter)]
    
    # Check if we have data after filtering
    if df.empty:
        st.error("No data available. Database might be empty.")
        st.info("Try running: python quick_data_populate.py")
        st.stop()
    
    # Main dashboard
    st.title("Finance & Technology Sentiment Analysis Dashboard")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Posts", len(df))
    
    with col2:
        sentiment_counts = df['overall_sentiment'].value_counts(normalize=True) * 100
        positive_pct = sentiment_counts.get('positive', 0)
        st.metric("Positive Sentiment", f"{positive_pct:.1f}%")
    
    with col3:
        negative_pct = sentiment_counts.get('negative', 0)
        st.metric("Negative Sentiment", f"{negative_pct:.1f}%")
    
    with col4:
        source_counts = df['source'].value_counts(normalize=True) * 100
        twitter_pct = source_counts.get('twitter', 0)
        reddit_pct = source_counts.get('reddit', 0)
        st.metric("Twitter / Reddit", f"{twitter_pct:.1f}% / {reddit_pct:.1f}%")
    
    # Sentiment trends over time
    st.subheader("Sentiment Trends Over Time")
    
    # Aggregate sentiment by date
    sentiment_by_date = df.groupby(['date', 'overall_sentiment']).size().reset_index(name='count')
    
    # Create pivot table for sentiment by date
    pivot_sentiment = sentiment_by_date.pivot(index='date', columns='overall_sentiment', values='count').reset_index()
    pivot_sentiment = pivot_sentiment.fillna(0)
    
    # Make sure all sentiment columns exist
    for sentiment in ['positive', 'negative', 'neutral']:
        if sentiment not in pivot_sentiment.columns:
            pivot_sentiment[sentiment] = 0
    
    # Create a stacked area chart
    fig = px.area(
        pivot_sentiment,
        x="date",
        y=["positive", "negative", "neutral"],
        labels={"value": "Count", "date": "Date"},
        title="Sentiment Distribution Over Time",
        color_discrete_map={"positive": "green", "negative": "red", "neutral": "gray"}
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Domain comparison
    st.subheader("Finance vs Technology Sentiment")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Finance sentiment
        finance_df = df[df['domain'].isin(['finance', 'finance-tech'])]
        if not finance_df.empty:
            finance_sentiment_counts = finance_df['overall_sentiment'].value_counts()
            
            fig = px.pie(
                values=finance_sentiment_counts.values,
                names=finance_sentiment_counts.index,
                title="Finance Sentiment Distribution",
                color=finance_sentiment_counts.index,
                color_discrete_map={"positive": "green", "negative": "red", "neutral": "gray"}
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No finance data available for the selected filters.")
    
    with col2:
        # Technology sentiment
        tech_df = df[df['domain'].isin(['technology', 'tech-finance'])]
        if not tech_df.empty:
            tech_sentiment_counts = tech_df['overall_sentiment'].value_counts()
            
            fig = px.pie(
                values=tech_sentiment_counts.values,
                names=tech_sentiment_counts.index,
                title="Technology Sentiment Distribution",
                color=tech_sentiment_counts.index,
                color_discrete_map={"positive": "green", "negative": "red", "neutral": "gray"}
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No technology data available for the selected filters.")
    
    # Entity extraction
    st.subheader("Most Discussed Entities")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Finance entities
        finance_entities_series = parse_json_column(df, 'finance_entities')
        if not finance_entities_series.empty:
            # Flatten the list of entities
            finance_entities = [entity for sublist in finance_entities_series if sublist for entity in sublist]
            
            if finance_entities:
                # Count entity occurrences
                entity_counts = pd.Series(finance_entities).value_counts().head(15)
                
                fig = px.bar(
                    x=entity_counts.values,
                    y=entity_counts.index,
                    orientation='h',
                    title="Top 15 Finance Entities",
                    labels={"x": "Occurrences", "y": "Entity"}
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No finance entities found in the selected data.")
        else:
            st.info("No finance entity data available.")
    
    with col2:
        # Tech entities
        tech_entities_series = parse_json_column(df, 'tech_entities')
        if not tech_entities_series.empty:
            # Flatten the list of entities
            tech_entities = [entity for sublist in tech_entities_series if sublist for entity in sublist]
            
            if tech_entities:
                # Count entity occurrences
                entity_counts = pd.Series(tech_entities).value_counts().head(15)
                
                fig = px.bar(
                    x=entity_counts.values,
                    y=entity_counts.index,
                    orientation='h',
                    title="Top 15 Technology Entities",
                    labels={"x": "Occurrences", "y": "Entity"}
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No technology entities found in the selected data.")
        else:
            st.info("No technology entity data available.")
    
    # Tech category distribution
    if 'tech_categories' in df.columns:
        st.subheader("Technology Category Distribution")
        
        tech_categories_series = parse_json_column(df, 'tech_categories')
        
        if not tech_categories_series.empty:
            # Extract keys from all dictionaries
            category_keys = []
            for cat_dict in tech_categories_series:
                if isinstance(cat_dict, dict):
                    category_keys.extend(cat_dict.keys())
            
            if category_keys:
                # Count category occurrences
                category_counts = pd.Series(category_keys).value_counts()
                
                fig = px.pie(
                    values=category_counts.values,
                    names=category_counts.index,
                    title="Technology Categories",
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No technology categories found in the selected data.")
        else:
            st.info("No technology category data available.")
    
    # Recent posts with sentiment
    st.subheader("Recent Posts with Sentiment Analysis")
    
    # Display a sample of recent posts
    recent_posts = df.head(10)[['source', 'created_at', 'text', 'domain', 'overall_sentiment']]
    
    # Format the dataframe for display
    recent_posts['created_at'] = recent_posts['created_at'].dt.strftime('%Y-%m-%d %H:%M')
    
    # Add emoji based on sentiment
    recent_posts['sentiment'] = recent_posts['overall_sentiment'].map({
        'positive': '😀 Positive',
        'negative': '😞 Negative',
        'neutral': '😐 Neutral'
    })
    
    # Display in table
    st.dataframe(
        recent_posts[['source', 'created_at', 'text', 'domain', 'sentiment']],
        column_config={
            "text": st.column_config.TextColumn("Content", width="large"),
            "sentiment": st.column_config.TextColumn("Sentiment", width="medium"),
            "source": st.column_config.TextColumn("Source", width="small"),
        },
        use_container_width=True,
        hide_index=True
    )
    
    # Download data
    st.subheader("Download Analyzed Data")
    
    # Prepare download data
    download_df = df[['id', 'source', 'created_at', 'text', 'domain', 'overall_sentiment', 
                     'finance_sentiment', 'finance_score', 'tech_sentiment', 'tech_score']]
    
    # Convert to CSV
    csv = download_df.to_csv(index=False)
    
    # Add download button
    st.download_button(
        label="Download as CSV",
        data=csv,
        file_name=f"sentiment_data_{datetime.now().strftime('%Y-%m-%d')}.csv",
        mime="text/csv"
    )
    
except Exception as e:
    st.error(f"Error loading data: {e}")
    if not db_exists:
        st.info("Database file not found. Please make sure the sentiment analysis pipeline has been run and data has been stored.")
    st.stop()

# Footer
st.markdown("---")
st.markdown("**Social Media Sentiment Analysis for Finance & Technology Research** | Data updated every 5 minutes") 