import streamlit as st
import pandas as pd
import sqlite3
import os
import plotly.express as px
from datetime import datetime, timedelta

# Set page configuration
st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

# Database path - absolute path fix
db_path = r"E:\Sentiment Analysis\Big-Data-Sentiment-Engine\data-storage\data\sentiment_data.db"

# Debug path
print(f"🔍 Dashboard looking for DB at: {db_path}")
print(f"📁 DB exists: {os.path.exists(db_path)}")

@st.cache_data(ttl=60)
def load_data():
    """Load data from database"""
    try:
        conn = sqlite3.connect(db_path)
        
        # Simple query to get all data
        query = """
        SELECT id, source, created_at, text, domain, overall_sentiment, score
        FROM sentiment_results 
        ORDER BY created_at DESC 
        LIMIT 5000
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if not df.empty:
            # Handle different date formats
            try:
                df['created_at'] = pd.to_datetime(df['created_at'], format='mixed', errors='coerce')
            except:
                try:
                    df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
                except:
                    # If all else fails, use current time
                    df['created_at'] = pd.Timestamp.now()
            
            df['date'] = df['created_at'].dt.date
            
        return df
        
    except Exception as e:
        st.error(f"Database error: {e}")
        return pd.DataFrame()

# Main app
st.title("🎯 Sentiment Analysis Dashboard")
st.markdown("### Real-time Finance & Technology Sentiment Analysis")

# Load data
df = load_data()

if df.empty:
    st.error("❌ No data found in database!")
    st.info("💡 Run `python mega_data_generator.py` to add data")
    st.stop()

# Drop rows with invalid dates if any
df = df.dropna(subset=['created_at'])

# Show data summary
st.success(f"✅ Loaded {len(df)} sentiment records")

# Sidebar filters
st.sidebar.header("🔍 Filters")

# Time filter
time_options = ["All Time", "Last 7 Days", "Last 30 Days"]
time_filter = st.sidebar.selectbox("Time Range", time_options)

if time_filter == "Last 7 Days":
    cutoff = datetime.now() - timedelta(days=7)
    df = df[df['created_at'] >= cutoff]
elif time_filter == "Last 30 Days":
    cutoff = datetime.now() - timedelta(days=30)
    df = df[df['created_at'] >= cutoff]

# Domain filter
domains = df['domain'].unique().tolist()
selected_domains = st.sidebar.multiselect("Domain", domains, default=domains)
df = df[df['domain'].isin(selected_domains)]

# Source filter
sources = df['source'].unique().tolist()
selected_sources = st.sidebar.multiselect("Source", sources, default=sources)
df = df[df['source'].isin(selected_sources)]

# Main metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("📊 Total Records", len(df))

with col2:
    positive_pct = (df['overall_sentiment'] == 'positive').mean() * 100
    st.metric("😊 Positive", f"{positive_pct:.1f}%")

with col3:
    negative_pct = (df['overall_sentiment'] == 'negative').mean() * 100
    st.metric("😞 Negative", f"{negative_pct:.1f}%")

with col4:
    neutral_pct = (df['overall_sentiment'] == 'neutral').mean() * 100
    st.metric("😐 Neutral", f"{neutral_pct:.1f}%")

# Charts
st.subheader("📈 Sentiment Distribution")

col1, col2 = st.columns(2)

with col1:
    # Sentiment pie chart
    sentiment_counts = df['overall_sentiment'].value_counts()
    fig = px.pie(
        values=sentiment_counts.values,
        names=sentiment_counts.index,
        title="Overall Sentiment Distribution",
        color_discrete_map={
            "positive": "#2E8B57",
            "negative": "#DC143C", 
            "neutral": "#808080"
        }
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Domain distribution
    domain_counts = df['domain'].value_counts()
    fig = px.bar(
        x=domain_counts.index,
        y=domain_counts.values,
        title="Posts by Domain",
        labels={'x': 'Domain', 'y': 'Count'}
    )
    st.plotly_chart(fig, use_container_width=True)

# Sentiment over time
st.subheader("📅 Sentiment Trends Over Time")

if len(df) > 0:
    # Group by date and sentiment
    daily_sentiment = df.groupby(['date', 'overall_sentiment']).size().reset_index(name='count')
    
    # Create pivot table
    pivot_df = daily_sentiment.pivot(index='date', columns='overall_sentiment', values='count').fillna(0)
    pivot_df = pivot_df.reset_index()
    
    # Ensure all sentiment columns exist
    for sentiment in ['positive', 'negative', 'neutral']:
        if sentiment not in pivot_df.columns:
            pivot_df[sentiment] = 0
    
    # Create stacked area chart
    fig = px.area(
        pivot_df,
        x='date',
        y=['positive', 'negative', 'neutral'],
        title="Daily Sentiment Trends",
        color_discrete_map={
            "positive": "#2E8B57",
            "negative": "#DC143C",
            "neutral": "#808080"
        }
    )
    st.plotly_chart(fig, use_container_width=True)

# Source comparison
st.subheader("📱 Source Analysis")

col1, col2 = st.columns(2)

with col1:
    # Source distribution
    source_counts = df['source'].value_counts()
    fig = px.pie(
        values=source_counts.values,
        names=source_counts.index,
        title="Posts by Source"
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Sentiment by source
    source_sentiment = df.groupby(['source', 'overall_sentiment']).size().reset_index(name='count')
    fig = px.bar(
        source_sentiment,
        x='source',
        y='count',
        color='overall_sentiment',
        title="Sentiment by Source",
        color_discrete_map={
            "positive": "#2E8B57",
            "negative": "#DC143C",
            "neutral": "#808080"
        }
    )
    st.plotly_chart(fig, use_container_width=True)

# Sample data
st.subheader("📝 Recent Posts")
st.dataframe(
    df[['created_at', 'source', 'domain', 'overall_sentiment', 'text']].head(10),
    use_container_width=True
)

# Footer
st.markdown("---")
st.markdown("🚀 **Advanced Sentiment Analysis Engine** | Built with Streamlit & Python") 