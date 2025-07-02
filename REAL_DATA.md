# Using Real-Time Data Instead of Samples

The sentiment analysis dashboard is designed to use real-time data from multiple sources, but you may see errors or sample data being displayed if the actual data collection hasn't been run. This guide explains how to collect real data and resolve common errors.

## Current Error Explanation

If you see this error:
```
Error loading sentiment data: '>' not supported between instances of 'str' and 'int'
```

This is happening because the date comparison in the SQL query is trying to compare a string with an integer. The system will automatically fall back to sample data when this occurs.

## Collecting Real-Time Data

We've created a dedicated script to collect real-time data from various sources:

1. **Setup the Collection Environment**:
   ```bash
   pip install feedparser requests textblob
   ```

2. **Setup the Database**:
   ```bash
   python collect_real_data.py --setup
   ```

3. **Collect Data Once**:
   ```bash
   python collect_real_data.py
   ```

4. **Collect Data Continuously** (refreshes every hour):
   ```bash
   python collect_real_data.py --continuous
   ```

5. **Customize Collection Interval** (e.g., every 15 minutes):
   ```bash
   python collect_real_data.py --continuous --interval 15
   ```

## Data Sources

The collection script gathers data from:

1. **Financial News RSS Feeds**:
   - CNBC Finance
   - WSJ Markets
   - CNBC Business

2. **Technology News RSS Feeds**:
   - TechCrunch
   - Wired
   - Ars Technica

3. **Reddit**:
   - r/investing, r/stocks (Finance)
   - r/technology, r/programming (Technology)

4. **Hacker News**:
   - Top stories with comments

## Verifying Data Collection

After running the data collection, you can verify that real data is being used:

1. Check the data collection log:
   ```bash
   cat data_collection.log
   ```

2. Count the records in the database:
   ```bash
   sqlite3 data-storage/data/sentiment_data.db "SELECT COUNT(*) FROM sentiment_results"
   ```

3. View the most recent entries:
   ```bash
   sqlite3 data-storage/data/sentiment_data.db "SELECT created_at, source, domain, overall_sentiment FROM sentiment_results ORDER BY created_at DESC LIMIT 5"
   ```

## Running the Dashboard with Real Data

Once you've collected real data, restart the dashboard:

```bash
python run_advanced.py dashboard
```

You should now see the real-time data instead of sample data in all views.

## Troubleshooting

### No Data Showing Even After Collection

If you've collected data but still don't see it in the dashboard:

1. **Check Database Path**:
   Ensure the database path in `collect_real_data.py` matches the one in `dashboard/streamlit_app.py`.

2. **Date Format Issues**:
   If you're seeing date format errors, you can refresh the database:
   ```bash
   python setup_databases.py --force-recreate
   ```
   Then run the collection again.

3. **Low-Level SQL Issues**:
   If you're comfortable with SQL, you can inspect and fix the database directly:
   ```bash
   sqlite3 data-storage/data/sentiment_data.db
   ```

### Sample Data Still Showing

If sample data is still being shown despite having real data:

1. Check if you're receiving a specific error in the dashboard UI
2. Look for warnings about "creating sample data"
3. Verify the data exists and is in the correct format

## Adding Custom Data Sources

You can extend the script to include additional sources by modifying `collect_real_data.py`:

1. Add new RSS feeds in the `sources` dictionary under the "rss" key
2. Add new Reddit subreddits under the "reddit" key
3. Implement custom collectors for other sources by creating new functions 

## Using the Dashboard with Mixed Data

The dashboard is designed to gracefully handle mixed real and sample data. If some views are showing real data and others are showing sample data, this is expected behavior when:

1. You're just starting to collect data (not enough for all visualizations)
2. Some data sources are unavailable or rate-limited
3. Certain data points are missing required fields

The system will progressively replace sample data with real data as more is collected. 