# Big Data Sentiment Analysis Engine

Advanced sentiment analysis platform for finance and technology domains leveraging open-source data collection, advanced analytics, and interactive visualization.

## Overview

This system provides a complete sentiment analysis pipeline for financial and technology sectors without relying on Twitter's paid API. It collects data from various free sources, processes it using natural language processing techniques, and visualizes the results in an interactive dashboard.

## Key Features

### Data Collection
- RSS feeds from major financial and tech news sites
- Reddit public JSON endpoints for financial and tech subreddits
- Hacker News API integration
- Sample data generation for testing and demonstration

### Sentiment Analysis
- Multi-model sentiment analysis with confidence scoring
- Entity extraction and relationship mapping
- Domain-specific sentiment contextualization
- Aspect-based sentiment analysis

### Advanced Visualizations
- Interactive sentiment trends
- Source analysis heatmaps
- Entity relationship networks
- Word clouds with real-time explanations
- Domain comparisons with statistical analysis
- Sentiment forecasting using time-series analysis
- Interactive sentiment analyzer

### Database Structure
- Sentiment data storage
- Market data integration
- Analytics results
- Data source management

## Installation

### Prerequisites
- Python 3.9+ (3.10 recommended)
- Windows, macOS, or Linux

### Quick Installation
1. Clone the repository
2. Run the automated installer:
```
python install_dependencies.py
```

### Manual Installation
If the automated installer fails, you can manually install dependencies:
```
pip install -r requirements.txt
pip install nltk
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

## Getting Started

### Using the start.bat Script (Windows)
The easiest way to get started on Windows is to use the provided start.bat script:
1. Open a command prompt in the project directory
2. Run `start.bat`
3. Select an option from the menu

### Manual Execution
You can also run specific components directly:

1. Set up the databases:
```
python setup_databases.py
```

2. Collect data:
```
python run_advanced.py collect
```

3. Launch the dashboard:
```
python run_advanced.py dashboard
```

## Dashboard Sections

### AI Summary
Provides an executive overview of sentiment trends, key entities, and market impact predictions.

### Sentiment Overview
Visualizes sentiment across time, domains, and sources with detailed explanations.

### Source Analysis
Analyzes the sentiment distribution across different news sources with source-specific metrics.

### Entity Analysis
Identifies and analyzes key entities mentioned in the data, with relationship graphs and centrality metrics.

### Advanced Insights
Market correlation analysis and other advanced metrics linking sentiment to potential market impacts.

### Comparative Analysis
- **Domain Comparison**: Compare sentiment across different domains with correlation analysis
- **Time Period Comparison**: Compare sentiment patterns between different time periods
- **Entity Comparison**: Compare metrics across different entities using radar charts

### ML Models
- **Model Usage**: Analytics on model usage and confidence
- **Sentiment Forecasting**: Predicts future sentiment trends using time-series modeling
- **Sentiment Analyzer**: Interactive tool for analyzing custom text

## Troubleshooting

### Common Issues

#### Missing NLTK Data
If you see errors about NLTK data not being found:
```
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

#### Word Cloud Errors
If the word cloud visualization shows errors:
```
pip install wordcloud --prefer-binary
```

#### Database Issues
If you encounter database-related errors:
```
python setup_databases.py --force-recreate
```

## Advanced Usage

### Adding New Data Sources
Edit `data-ingestion/open_data_collector.py` to add new data sources.

### Custom Visualization
Create custom visualizations by adding new functions to `dashboard/streamlit_app.py`.

### Market Data Integration
The system can integrate with market data when available in the market_data database.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
