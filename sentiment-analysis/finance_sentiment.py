import re
import nltk
import pandas as pd
import numpy as np
from textblob import TextBlob
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import json
import os

# Download required NLTK resources if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# Financial dictionaries - positive and negative words in financial context
FINANCE_POSITIVE = [
    "growth", "profit", "increase", "gain", "rising", "recovery", "outperform", 
    "successful", "positive", "upward", "strong", "opportunity", "exceed",
    "beat", "returns", "momentum", "rally", "bullish", "upgrade", "improved",
    "surged", "boosted", "expanded", "upside", "climb"
]

FINANCE_NEGATIVE = [
    "loss", "decline", "decrease", "drop", "falling", "recession", "underperform",
    "bankruptcy", "debt", "inflation", "crisis", "crash", "downgrade", "default",
    "deficit", "risk", "volatile", "plunge", "negative", "bearish", "worry",
    "concern", "disappointing", "downward", "tumble", "slump", "weak"
]

# Financial entities
FINANCIAL_ENTITIES = [
    "stock", "share", "bond", "market", "index", "nasdaq", "nyse", "djia", 
    "s&p", "trading", "trader", "investor", "investment", "fund", "etf",
    "portfolio", "asset", "equity", "security", "dividend", "yield", "growth",
    "earnings", "revenue", "profit", "income", "quarter", "fiscal", "economic",
    "economy", "fed", "federal reserve", "bank", "treasury", "interest rate",
    "inflation", "deflation", "gdp", "recession", "bull", "bear", "volatility"
]

class FinanceSentimentAnalyzer:
    """Specialized sentiment analyzer for financial texts"""
    
    def __init__(self):
        """Initialize the finance sentiment analyzer"""
        # Load stopwords
        self.stopwords = set(stopwords.words('english'))
        
        # Add custom financial stopwords that don't contribute to sentiment
        self.stopwords.update(['the', 'and', 'a', 'in', 'to', 'of', 'for', 'on'])
        
        # Set up dictionaries
        self.fin_pos = set(FINANCE_POSITIVE)
        self.fin_neg = set(FINANCE_NEGATIVE)
        self.fin_entities = set(FINANCIAL_ENTITIES)
        
        # Load additional financial terms from config if available
        try:
            base_path = os.path.dirname(__file__)
            config_path = os.path.join(os.path.dirname(base_path), 'data-ingestion', 'config', 'keys.json')
            
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                
            finance_keywords = config.get('keywords', {}).get('finance', [])
            for keyword in finance_keywords:
                for term in keyword.lower().split():
                    if len(term) > 3:  # Avoid short terms
                        self.fin_entities.add(term)
        except Exception as e:
            print(f"Warning: Could not load additional financial terms: {e}")
    
    def clean_text(self, text):
        """Clean the financial text for analysis"""
        if not isinstance(text, str):
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove mentions and hashtags
        text = re.sub(r'@\w+|\#\w+', '', text)
        
        # Remove special characters but keep $cashtags intact
        text = re.sub(r'[^\w\s\$]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_financial_entities(self, text):
        """Extract financial entities from text"""
        cleaned_text = self.clean_text(text)
        tokens = word_tokenize(cleaned_text)
        
        # Extract $cashtags (stock symbols)
        cashtags = [token for token in tokens if token.startswith('$') and len(token) > 1]
        
        # Extract known financial entities
        entities = [token for token in tokens if token in self.fin_entities]
        
        # Combine and deduplicate
        all_entities = list(set(cashtags + entities))
        
        return all_entities
    
    def calculate_finance_sentiment(self, text):
        """Calculate sentiment specifically for financial text"""
        cleaned_text = self.clean_text(text)
        tokens = word_tokenize(cleaned_text)
        
        # Remove stopwords
        filtered_tokens = [token for token in tokens if token not in self.stopwords]
        
        # Count financial positive and negative words
        pos_matches = [word for word in filtered_tokens if word in self.fin_pos]
        neg_matches = [word for word in filtered_tokens if word in self.fin_neg]
        
        # Calculate finance-specific sentiment score
        pos_count = len(pos_matches)
        neg_count = len(neg_matches)
        
        if pos_count + neg_count == 0:
            fin_score = 0  # Neutral if no financial sentiment words
        else:
            fin_score = (pos_count - neg_count) / (pos_count + neg_count)
        
        # Get general sentiment from TextBlob for comparison
        general_sentiment = TextBlob(cleaned_text).sentiment.polarity
        
        # Determine final sentiment classification with bias toward financial terms
        if fin_score > 0.1:
            sentiment = "positive"
        elif fin_score < -0.1:
            sentiment = "negative"
        else:
            # If financial sentiment is neutral, use general sentiment as backup
            if general_sentiment > 0.2:
                sentiment = "positive"
            elif general_sentiment < -0.2:
                sentiment = "negative"
            else:
                sentiment = "neutral"
        
        return {
            "finance_score": fin_score,
            "general_score": general_sentiment,
            "finance_pos_words": pos_matches,
            "finance_neg_words": neg_matches,
            "sentiment": sentiment
        }
    
    def process(self, text):
        """Process financial text and return comprehensive analysis"""
        if not text or len(text) < 5:
            return {
                "entities": [],
                "sentiment": "neutral",
                "analysis": {
                    "finance_score": 0,
                    "general_score": 0,
                    "finance_pos_words": [],
                    "finance_neg_words": []
                }
            }
        
        # Extract financial entities
        entities = self.extract_financial_entities(text)
        
        # Calculate sentiment
        sentiment_analysis = self.calculate_finance_sentiment(text)
        
        # Return combined analysis
        return {
            "entities": entities,
            "sentiment": sentiment_analysis["sentiment"],
            "analysis": sentiment_analysis
        }

# Example usage
if __name__ == "__main__":
    analyzer = FinanceSentimentAnalyzer()
    
    test_texts = [
        "Apple's stock surged 5% after beating earnings expectations for Q3.",
        "Tesla shares plummeted following disappointing delivery numbers and rising competition.",
        "The market remains uncertain amid inflation concerns and potential rate hikes.",
        "Bitcoin reached an all-time high as institutional investors continue to adopt cryptocurrencies."
    ]
    
    for text in test_texts:
        result = analyzer.process(text)
        print(f"\nText: {text}")
        print(f"Sentiment: {result['sentiment']}")
        print(f"Financial entities: {result['entities']}")
        print(f"Finance score: {result['analysis']['finance_score']:.2f}")
        print(f"Positive words: {result['analysis']['finance_pos_words']}")
        print(f"Negative words: {result['analysis']['finance_neg_words']}") 