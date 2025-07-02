import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px

# Set page configuration
st.set_page_config(
    page_title="🎯 Advanced Sentiment Analysis",
    page_icon="📊",
    layout="wide"
)

# Database path
db_path = r"E:\Sentiment Analysis\Big-Data-Sentiment-Engine\data-storage\data\sentiment_data.db"

@st.cache_data(ttl=30)
def load_simple_data():
    """Load data with minimal processing"""
    try:
        conn = sqlite3.connect(db_path)
        query = "SELECT * FROM sentiment_results LIMIT 5000"
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Database error: {e}")
        return pd.DataFrame()

# Header
st.title("🎯 Advanced Sentiment Analysis Dashboard")
st.markdown("### 🚀 Real-time Finance & Technology Sentiment Analysis")

# Load data
df = load_simple_data()

if df.empty:
    st.error("❌ No data found!")
    st.stop()

# Show success
st.success(f"✅ Successfully loaded {len(df)} sentiment records!")

# Basic stats
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("📊 Total Records", len(df))

with col2:
    if 'overall_sentiment' in df.columns:
        positive_count = len(df[df['overall_sentiment'] == 'positive'])
        st.metric("😊 Positive", positive_count)

with col3:
    if 'overall_sentiment' in df.columns:
        negative_count = len(df[df['overall_sentiment'] == 'negative'])
        st.metric("😞 Negative", negative_count)

with col4:
    if 'domain' in df.columns:
        finance_count = len(df[df['domain'] == 'finance'])
        st.metric("💰 Finance", finance_count)

# Charts section
st.subheader("📈 Data Analysis")

col1, col2 = st.columns(2)

with col1:
    if 'overall_sentiment' in df.columns:
        st.subheader("Sentiment Distribution")
        sentiment_counts = df['overall_sentiment'].value_counts()
        
        fig = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            title="Overall Sentiment",
            color_discrete_map={
                "positive": "#00CC96",
                "negative": "#EF553B",
                "neutral": "#636EFA"
            }
        )
        st.plotly_chart(fig, use_container_width=True)

with col2:
    if 'domain' in df.columns:
        st.subheader("Domain Distribution")
        domain_counts = df['domain'].value_counts()
        
        fig = px.bar(
            x=domain_counts.index,
            y=domain_counts.values,
            title="Posts by Domain",
            labels={'x': 'Domain', 'y': 'Count'},
            color=domain_counts.values,
            color_continuous_scale='viridis'
        )
        st.plotly_chart(fig, use_container_width=True)

# Source analysis
if 'source' in df.columns:
    st.subheader("📱 Source Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        source_counts = df['source'].value_counts()
        fig = px.pie(
            values=source_counts.values,
            names=source_counts.index,
            title="Data Sources"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'overall_sentiment' in df.columns:
            # Cross-tabulation
            cross_tab = pd.crosstab(df['source'], df['overall_sentiment'])
            
            fig = px.bar(
                cross_tab,
                title="Sentiment by Source",
                labels={'value': 'Count', 'index': 'Source'}
            )
            st.plotly_chart(fig, use_container_width=True)

# Sample data
st.subheader("📝 Sample Data")

# Show relevant columns only
display_columns = []
for col in ['created_at', 'source', 'domain', 'overall_sentiment', 'text']:
    if col in df.columns:
        display_columns.append(col)

if display_columns:
    sample_df = df[display_columns].head(10)
    st.dataframe(sample_df, use_container_width=True)

# Advanced stats
st.subheader("📊 Advanced Statistics")

col1, col2, col3 = st.columns(3)

with col1:
    if 'score' in df.columns:
        avg_score = df['score'].mean()
        st.metric("📈 Average Score", f"{avg_score:.3f}")

with col2:
    if 'domain' in df.columns:
        unique_domains = df['domain'].nunique()
        st.metric("🏢 Unique Domains", unique_domains)

with col3:
    if 'source' in df.columns:
        unique_sources = df['source'].nunique()
        st.metric("📱 Data Sources", unique_sources)

# Footer
st.markdown("---")
st.markdown("🚀 **Advanced Big Data Sentiment Analysis Engine** | Research-Grade Analytics")

# Debug info
with st.expander("🔧 Debug Information"):
    st.write("Database path:", db_path)
    st.write("Database exists:", sqlite3.connect(db_path) is not None)
    st.write("DataFrame shape:", df.shape)
    st.write("DataFrame columns:", list(df.columns))
    if not df.empty:
        st.write("Sample record:", df.iloc[0].to_dict()) 