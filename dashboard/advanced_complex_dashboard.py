import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import json
from datetime import datetime, timedelta
import seaborn as sns
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(
    page_title="🎯 Advanced Big Data Sentiment Analysis Research Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem;
    }
    .explanation-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Database path
db_path = r"E:\Sentiment Analysis\Big-Data-Sentiment-Engine\data-storage\data\sentiment_data.db"

@st.cache_data(ttl=60)
def load_comprehensive_data():
    """Load comprehensive data with advanced processing"""
    try:
        conn = sqlite3.connect(db_path)
        
        # Main sentiment data
        query = """
        SELECT s.*, f.entities as finance_entities, f.positive_terms, f.negative_terms,
               t.entities as tech_entities, t.categories as tech_categories
        FROM sentiment_results s
        LEFT JOIN finance_analysis f ON s.id = f.id
        LEFT JOIN tech_analysis t ON s.id = t.id
        LIMIT 8000
        """
        df = pd.read_sql_query(query, conn)
        
        # Entity mentions
        entity_query = "SELECT * FROM entity_mentions LIMIT 5000"
        try:
            entity_df = pd.read_sql_query(entity_query, conn)
        except:
            entity_df = pd.DataFrame()
        
        conn.close()
        
        # Process timestamps with timezone handling
        if not df.empty:
            try:
                df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce', utc=True)
                df['created_at'] = df['created_at'].dt.tz_localize(None)  # Remove timezone
                df['date'] = df['created_at'].dt.date
                df['hour'] = df['created_at'].dt.hour
                df['day_of_week'] = df['created_at'].dt.day_name()
            except:
                # Fallback to current time
                current_time = pd.Timestamp.now()
                df['created_at'] = current_time
                df['date'] = current_time.date()
                df['hour'] = current_time.hour
                df['day_of_week'] = current_time.day_name()
        
        return df, entity_df
    except Exception as e:
        st.error(f"Database error: {e}")
        return pd.DataFrame(), pd.DataFrame()

def create_advanced_metrics(df):
    """Create advanced statistical metrics"""
    metrics = {}
    
    if not df.empty:
        # Basic metrics
        metrics['total_records'] = len(df)
        metrics['avg_sentiment_score'] = df['score'].mean() if 'score' in df.columns else 0
        metrics['sentiment_volatility'] = df['score'].std() if 'score' in df.columns else 0
        
        # Domain metrics
        if 'domain' in df.columns:
            metrics['finance_ratio'] = len(df[df['domain'] == 'finance']) / len(df)
            metrics['tech_ratio'] = len(df[df['domain'] == 'technology']) / len(df)
        
        # Sentiment distribution
        if 'overall_sentiment' in df.columns:
            sentiment_dist = df['overall_sentiment'].value_counts(normalize=True)
            metrics['positive_ratio'] = sentiment_dist.get('positive', 0)
            metrics['negative_ratio'] = sentiment_dist.get('negative', 0)
            metrics['neutral_ratio'] = sentiment_dist.get('neutral', 0)
        
        # Advanced metrics
        metrics['data_quality_score'] = (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        metrics['source_diversity'] = df['source'].nunique() if 'source' in df.columns else 0
        
    return metrics

def create_correlation_heatmap(df):
    """Create correlation heatmap for numerical features"""
    if df.empty:
        return None
    
    # Select numerical columns
    numerical_cols = []
    for col in ['score', 'confidence']:
        if col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                numerical_cols.append(col)
            except:
                pass
    
    # Add engineered features
    if 'overall_sentiment' in df.columns:
        df['sentiment_numeric'] = df['overall_sentiment'].map({'positive': 1, 'neutral': 0, 'negative': -1})
        numerical_cols.append('sentiment_numeric')
    
    if 'hour' in df.columns:
        numerical_cols.append('hour')
    
    if len(numerical_cols) >= 2:
        corr_matrix = df[numerical_cols].corr()
        
        fig = px.imshow(
            corr_matrix,
            title="📊 Advanced Feature Correlation Matrix",
            color_continuous_scale='RdBu_r',
            aspect="auto"
        )
        fig.update_layout(height=500)
        return fig
    
    return None

def create_time_series_analysis(df):
    """Create advanced time series analysis"""
    if df.empty or 'created_at' not in df.columns:
        return None
    
    # Aggregate by hour
    df['datetime_hour'] = df['created_at'].dt.floor('h')  # Use 'h' instead of 'H'
    hourly_sentiment = df.groupby(['datetime_hour', 'overall_sentiment']).size().reset_index(name='count')
    
    # Pivot for stacked area chart
    pivot_hourly = hourly_sentiment.pivot(index='datetime_hour', columns='overall_sentiment', values='count').fillna(0)
    
    fig = go.Figure()
    
    colors = {'positive': '#00CC96', 'negative': '#EF553B', 'neutral': '#636EFA'}
    
    for sentiment in ['positive', 'negative', 'neutral']:
        if sentiment in pivot_hourly.columns:
            fig.add_trace(go.Scatter(
                x=pivot_hourly.index,
                y=pivot_hourly[sentiment],
                mode='lines',
                stackgroup='one',
                name=sentiment.title(),
                line=dict(color=colors.get(sentiment, '#636EFA')),
                fillcolor=colors.get(sentiment, '#636EFA')
            ))
    
    fig.update_layout(
        title="📈 Temporal Sentiment Flow Analysis",
        xaxis_title="Time",
        yaxis_title="Sentiment Volume",
        hovermode='x unified',
        height=500
    )
    
    return fig

def create_domain_sentiment_matrix(df):
    """Create domain vs sentiment matrix"""
    if df.empty or 'domain' not in df.columns or 'overall_sentiment' not in df.columns:
        return None
    
    # Create cross-tabulation
    cross_tab = pd.crosstab(df['domain'], df['overall_sentiment'], normalize='index') * 100
    
    fig = px.imshow(
        cross_tab,
        title="🎯 Domain-Sentiment Distribution Matrix (%)",
        labels=dict(x="Sentiment", y="Domain", color="Percentage"),
        color_continuous_scale='Viridis',
        aspect="auto"
    )
    
    # Add text annotations
    for i, domain in enumerate(cross_tab.index):
        for j, sentiment in enumerate(cross_tab.columns):
            fig.add_annotation(
                x=j, y=i,
                text=f"{cross_tab.iloc[i, j]:.1f}%",
                showarrow=False,
                font=dict(color="white", size=12)
            )
    
    fig.update_layout(height=400)
    return fig

def create_source_performance_radar(df):
    """Create radar chart for source performance"""
    if df.empty or 'source' not in df.columns:
        return None
    
    source_metrics = []
    
    for source in df['source'].unique():
        source_data = df[df['source'] == source]
        
        metrics = {
            'source': source,
            'volume': len(source_data),
            'positive_ratio': len(source_data[source_data['overall_sentiment'] == 'positive']) / len(source_data) * 100,
            'avg_score': source_data['score'].mean() if 'score' in df.columns else 50,
            'recency': (pd.Timestamp.now() - source_data['created_at'].max()).days if 'created_at' in df.columns and not source_data['created_at'].empty else 0
        }
        source_metrics.append(metrics)
    
    metrics_df = pd.DataFrame(source_metrics)
    
    fig = go.Figure()
    
    categories = ['Volume', 'Positive %', 'Avg Score', 'Recency Score']
    
    for _, row in metrics_df.iterrows():
        values = [
            row['volume'] / metrics_df['volume'].max() * 100,
            row['positive_ratio'],
            row['avg_score'] * 100,
            100 - (row['recency'] / max(metrics_df['recency'].max(), 1) * 100)
        ]
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name=row['source'].title()
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        title="🎯 Multi-Source Performance Radar Analysis",
        height=500
    )
    
    return fig

def create_sentiment_distribution_violin(df):
    """Create violin plot for sentiment score distribution"""
    if df.empty or 'score' not in df.columns or 'overall_sentiment' not in df.columns:
        return None
    
    fig = px.violin(
        df, 
        y='score', 
        x='overall_sentiment',
        box=True,
        title="🎻 Sentiment Score Distribution Analysis",
        color='overall_sentiment',
        color_discrete_map={'positive': '#00CC96', 'negative': '#EF553B', 'neutral': '#636EFA'}
    )
    
    fig.update_layout(height=500)
    return fig

def create_entity_network_analysis(df):
    """Create entity co-occurrence analysis"""
    if df.empty:
        return None
    
    # Extract entities from finance and tech columns
    entities = []
    
    for col in ['finance_entities', 'tech_entities']:
        if col in df.columns:
            for entity_list in df[col].dropna():
                try:
                    parsed = json.loads(entity_list) if isinstance(entity_list, str) else entity_list
                    if isinstance(parsed, list):
                        entities.extend(parsed)
                except:
                    pass
    
    if not entities:
        return None
    
    # Count entity mentions
    entity_counts = pd.Series(entities).value_counts().head(20)
    
    fig = px.treemap(
        names=entity_counts.index,
        values=entity_counts.values,
        title="🌐 Entity Mention Hierarchy Map"
    )
    
    fig.update_layout(height=500)
    return fig

# Main Dashboard
def main():
    # Header
    st.markdown('<h1 class="main-header">🎯 Advanced Big Data Sentiment Analysis Research Platform</h1>', unsafe_allow_html=True)
    st.markdown("### 🚀 Multi-Modal Financial & Technology Intelligence Engine")
    
    # Load data
    with st.spinner("🔄 Loading comprehensive dataset..."):
        df, entity_df = load_comprehensive_data()
    
    if df.empty:
        st.error("❌ No data available for analysis")
        return
    
    # Success message
    st.success(f"✅ Successfully loaded {len(df)} sentiment records for advanced analysis")
    
    # Sidebar controls
    st.sidebar.header("🎛️ Analysis Controls")
    
    # Filters
    if 'domain' in df.columns:
        domains = st.sidebar.multiselect("📊 Select Domains", df['domain'].unique(), default=df['domain'].unique())
        df = df[df['domain'].isin(domains)]
    
    if 'source' in df.columns:
        sources = st.sidebar.multiselect("📱 Select Sources", df['source'].unique(), default=df['source'].unique())
        df = df[df['source'].isin(sources)]
    
    # Advanced metrics
    metrics = create_advanced_metrics(df)
    
    # KPI Dashboard
    st.markdown('<h2 class="sub-header">📊 Executive KPI Dashboard</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("📈 Total Records", f"{metrics.get('total_records', 0):,}")
        st.metric("🎯 Data Quality", f"{metrics.get('data_quality_score', 0):.1f}%")
    
    with col2:
        st.metric("😊 Positive Ratio", f"{metrics.get('positive_ratio', 0):.2%}")
        st.metric("😞 Negative Ratio", f"{metrics.get('negative_ratio', 0):.2%}")
    
    with col3:
        st.metric("⚖️ Sentiment Score", f"{metrics.get('avg_sentiment_score', 0):.3f}")
        st.metric("📊 Volatility Index", f"{metrics.get('sentiment_volatility', 0):.3f}")
    
    with col4:
        st.metric("💰 Finance Ratio", f"{metrics.get('finance_ratio', 0):.2%}")
        st.metric("🤖 Tech Ratio", f"{metrics.get('tech_ratio', 0):.2%}")
    
    with col5:
        st.metric("🔄 Source Diversity", metrics.get('source_diversity', 0))
        st.metric("😐 Neutral Ratio", f"{metrics.get('neutral_ratio', 0):.2%}")
    
    # Advanced Visualizations
    st.markdown('<h2 class="sub-header">📈 Advanced Multi-Dimensional Analysis</h2>', unsafe_allow_html=True)
    
    # Row 1: Time series and correlation
    col1, col2 = st.columns(2)
    
    with col1:
        time_fig = create_time_series_analysis(df)
        if time_fig:
            st.plotly_chart(time_fig, use_container_width=True)
    
    with col2:
        corr_fig = create_correlation_heatmap(df)
        if corr_fig:
            st.plotly_chart(corr_fig, use_container_width=True)
    
    # Row 2: Domain matrix and radar
    col1, col2 = st.columns(2)
    
    with col1:
        matrix_fig = create_domain_sentiment_matrix(df)
        if matrix_fig:
            st.plotly_chart(matrix_fig, use_container_width=True)
    
    with col2:
        radar_fig = create_source_performance_radar(df)
        if radar_fig:
            st.plotly_chart(radar_fig, use_container_width=True)
    
    # Row 3: Distribution and entity analysis
    col1, col2 = st.columns(2)
    
    with col1:
        violin_fig = create_sentiment_distribution_violin(df)
        if violin_fig:
            st.plotly_chart(violin_fig, use_container_width=True)
    
    with col2:
        entity_fig = create_entity_network_analysis(df)
        if entity_fig:
            st.plotly_chart(entity_fig, use_container_width=True)
    
    # Detailed Data Tables
    st.markdown('<h2 class="sub-header">📋 Comprehensive Data Analysis Tables</h2>', unsafe_allow_html=True)
    
    # Statistical Summary Table
    if not df.empty:
        st.subheader("📊 Statistical Summary by Domain & Sentiment")
        
        if 'domain' in df.columns and 'overall_sentiment' in df.columns:
            summary_table = df.groupby(['domain', 'overall_sentiment']).agg({
                'score': ['count', 'mean', 'std', 'min', 'max'],
                'id': 'count'
            }).round(3)
            
            summary_table.columns = ['Count', 'Mean Score', 'Std Dev', 'Min Score', 'Max Score', 'Records']
            st.dataframe(summary_table, use_container_width=True)
    
    # Top Entities Table
    st.subheader("🏆 Top Mentioned Entities Analysis")
    
    entities_data = []
    for col in ['finance_entities', 'tech_entities']:
        if col in df.columns:
            for _, row in df.iterrows():
                if pd.notna(row[col]):
                    try:
                        entities = json.loads(row[col]) if isinstance(row[col], str) else row[col]
                        if isinstance(entities, list):
                            for entity in entities:
                                entities_data.append({
                                    'Entity': entity,
                                    'Domain': row.get('domain', 'Unknown'),
                                    'Sentiment': row.get('overall_sentiment', 'Unknown'),
                                    'Score': row.get('score', 0)
                                })
                    except:
                        pass
    
    if entities_data:
        entities_df = pd.DataFrame(entities_data)
        entity_summary = entities_df.groupby('Entity').agg({
            'Score': ['count', 'mean'],
            'Sentiment': lambda x: x.value_counts().index[0] if len(x) > 0 else 'Unknown'
        }).round(3)
        
        entity_summary.columns = ['Mentions', 'Avg Score', 'Dominant Sentiment']
        entity_summary = entity_summary.sort_values('Mentions', ascending=False).head(20)
        st.dataframe(entity_summary, use_container_width=True)
    
    # Raw Data Sample
    st.subheader("📝 Raw Data Sample (Latest Records)")
    display_columns = ['created_at', 'source', 'domain', 'overall_sentiment', 'score', 'text']
    available_columns = [col for col in display_columns if col in df.columns]
    
    if available_columns:
        sample_df = df[available_columns].head(20)
        st.dataframe(sample_df, use_container_width=True)
    
    # Explanations Section
    st.markdown('<h2 class="sub-header">📚 Methodology & Insights Explanation</h2>', unsafe_allow_html=True)
    
    with st.expander("🔬 **Advanced Analytics Methodology**", expanded=True):
        st.markdown("""
        <div class="explanation-box">
        <h4>🎯 Temporal Sentiment Flow Analysis</h4>
        <p>This time-series visualization reveals sentiment patterns across temporal dimensions. The stacked area chart demonstrates:</p>
        <ul>
            <li><strong>Volume Dynamics:</strong> Peak sentiment activity periods indicating market attention cycles</li>
            <li><strong>Sentiment Momentum:</strong> Positive/negative sentiment waves correlating with market events</li>
            <li><strong>Temporal Clustering:</strong> Identification of sentiment convergence periods</li>
        </ul>
        
        <h4>📊 Feature Correlation Matrix</h4>
        <p>The correlation heatmap reveals interdependencies between:</p>
        <ul>
            <li><strong>Sentiment-Score Alignment:</strong> Validation of sentiment classification accuracy</li>
            <li><strong>Temporal Correlations:</strong> Time-based sentiment pattern recognition</li>
            <li><strong>Cross-Feature Dependencies:</strong> Hidden relationships in multi-dimensional data</li>
        </ul>
        
        <h4>🎯 Domain-Sentiment Distribution Matrix</h4>
        <p>This normalized percentage matrix demonstrates:</p>
        <ul>
            <li><strong>Domain Bias Analysis:</strong> Inherent sentiment tendencies per domain</li>
            <li><strong>Sector Performance:</strong> Comparative sentiment health across industries</li>
            <li><strong>Risk Assessment:</strong> Sentiment volatility patterns by domain</li>
        </ul>
        
        <h4>🎻 Sentiment Score Distribution (Violin Plot)</h4>
        <p>Advanced distribution analysis revealing:</p>
        <ul>
            <li><strong>Score Density:</strong> Concentration patterns within sentiment categories</li>
            <li><strong>Outlier Detection:</strong> Extreme sentiment scores requiring investigation</li>
            <li><strong>Distribution Skewness:</strong> Asymmetric patterns indicating bias or trends</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with st.expander("📈 **Business Intelligence Insights**"):
        st.markdown(f"""
        <div class="explanation-box">
        <h4>💼 Executive Summary</h4>
        <p>Based on the analysis of <strong>{metrics.get('total_records', 0):,} sentiment records</strong>, key insights include:</p>
        
        <h5>🎯 Market Sentiment Health</h5>
        <ul>
            <li><strong>Overall Sentiment Score:</strong> {metrics.get('avg_sentiment_score', 0):.3f} (Range: -1 to +1)</li>
            <li><strong>Positive Sentiment Ratio:</strong> {metrics.get('positive_ratio', 0):.1%} indicating market optimism levels</li>
            <li><strong>Volatility Index:</strong> {metrics.get('sentiment_volatility', 0):.3f} measuring sentiment stability</li>
        </ul>
        
        <h5>🏢 Domain Analysis</h5>
        <ul>
            <li><strong>Finance Domain:</strong> {metrics.get('finance_ratio', 0):.1%} of total discussions</li>
            <li><strong>Technology Domain:</strong> {metrics.get('tech_ratio', 0):.1%} of total discussions</li>
            <li><strong>Cross-Domain Correlation:</strong> Identifying sector interdependencies</li>
        </ul>
        
        <h5>📊 Data Quality Metrics</h5>
        <ul>
            <li><strong>Data Completeness:</strong> {metrics.get('data_quality_score', 0):.1f}% ensuring analysis reliability</li>
            <li><strong>Source Diversity:</strong> {metrics.get('source_diversity', 0)} unique sources providing comprehensive coverage</li>
            <li><strong>Temporal Coverage:</strong> Multi-day analysis enabling trend identification</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with st.expander("🔧 **Technical Implementation Details**"):
        st.markdown("""
        <div class="explanation-box">
        <h4>🛠️ Advanced NLP Pipeline</h4>
        <ul>
            <li><strong>Multi-Model Ensemble:</strong> Transformer-based sentiment classification with domain adaptation</li>
            <li><strong>Entity Recognition:</strong> Named Entity Recognition (NER) for financial instruments and technology terms</li>
            <li><strong>Temporal Processing:</strong> Time-series decomposition with trend and seasonality analysis</li>
            <li><strong>Cross-Domain Validation:</strong> Domain-specific sentiment lexicons and validation metrics</li>
        </ul>
        
        <h4>📊 Big Data Processing</h4>
        <ul>
            <li><strong>Real-time Ingestion:</strong> Multi-source data streaming with Apache Kafka integration</li>
            <li><strong>Distributed Processing:</strong> Spark-based parallel processing for scalability</li>
            <li><strong>Advanced Analytics:</strong> Statistical modeling with correlation and regression analysis</li>
            <li><strong>Performance Optimization:</strong> Caching and indexing strategies for sub-second response times</li>
        </ul>
        
        <h4>🎯 Research Applications</h4>
        <ul>
            <li><strong>Market Research:</strong> Consumer sentiment tracking and brand perception analysis</li>
            <li><strong>Risk Management:</strong> Early warning systems for sentiment-driven market volatility</li>
            <li><strong>Academic Research:</strong> Longitudinal studies on social media sentiment patterns</li>
            <li><strong>Policy Analysis:</strong> Public opinion tracking for regulatory and policy decisions</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 2rem;">
        <h3>🚀 Advanced Big Data Sentiment Analysis Research Platform</h3>
        <p><strong>Enterprise-Grade Analytics | Research-Quality Insights | Real-Time Intelligence</strong></p>
        <p>Powered by Advanced NLP, Machine Learning, and Big Data Technologies</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 