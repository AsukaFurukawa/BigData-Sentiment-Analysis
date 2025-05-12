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

# Technology dictionaries - positive and negative words in tech context
TECH_POSITIVE = [
    "innovative", "breakthrough", "advanced", "cutting-edge", "efficient", 
    "powerful", "optimized", "revolutionary", "enhanced", "improved",
    "secure", "scalable", "reliable", "seamless", "intuitive", "user-friendly",
    "robust", "agile", "fast", "responsive", "smart", "intelligent", 
    "compatible", "integrated", "streamlined", "automated"
]

TECH_NEGATIVE = [
    "outdated", "bug", "glitch", "crash", "slow", "vulnerable", "insecure",
    "incompatible", "complex", "complicated", "difficult", "obsolete", "legacy",
    "failure", "error", "issue", "problem", "malware", "virus", "breach", "hack",
    "intrusion", "inefficient", "unstable", "unreliable", "flawed", "defective"
]

# Technology entities
TECH_ENTITIES = [
    "ai", "artificial intelligence", "machine learning", "ml", "nlp", "deep learning",
    "cloud", "aws", "azure", "gcp", "google cloud", "server", "database", "sql", "nosql",
    "blockchain", "crypto", "bitcoin", "ethereum", "nft", "token", "defi", "web3",
    "software", "hardware", "app", "application", "mobile", "ios", "android",
    "api", "sdk", "framework", "library", "code", "programming", "developer",
    "devops", "cicd", "continuous integration", "automation", "pipeline",
    "network", "internet", "wifi", "5g", "bandwidth", "latency", "cybersecurity", 
    "encryption", "firewall", "security", "data", "analytics", "big data", "hadoop",
    "spark", "iot", "internet of things", "sensors", "robot", "robotics", "autonomous",
    "vr", "ar", "virtual reality", "augmented reality", "metaverse", "gpu", "cpu", "processor"
]

class TechSentimentAnalyzer:
    """Specialized sentiment analyzer for technology texts"""
    
    def __init__(self):
        """Initialize the tech sentiment analyzer"""
        # Load stopwords
        self.stopwords = set(stopwords.words('english'))
        
        # Add custom tech stopwords that don't contribute to sentiment
        self.stopwords.update(['the', 'and', 'a', 'in', 'to', 'of', 'for', 'on'])
        
        # Set up dictionaries
        self.tech_pos = set(TECH_POSITIVE)
        self.tech_neg = set(TECH_NEGATIVE)
        self.tech_entities = set(TECH_ENTITIES)
        
        # Load additional tech terms from config if available
        try:
            base_path = os.path.dirname(__file__)
            config_path = os.path.join(os.path.dirname(base_path), 'data-ingestion', 'config', 'keys.json')
            
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                
            tech_keywords = config.get('keywords', {}).get('technology', [])
            for keyword in tech_keywords:
                for term in keyword.lower().split():
                    if len(term) > 3:  # Avoid short terms
                        self.tech_entities.add(term)
        except Exception as e:
            print(f"Warning: Could not load additional technology terms: {e}")
    
    def clean_text(self, text):
        """Clean the technology text for analysis"""
        if not isinstance(text, str):
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove mentions and hashtags
        text = re.sub(r'@\w+|\#\w+', '', text)
        
        # Remove special characters (but keep alphanumerics)
        text = re.sub(r'[^\w\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_tech_entities(self, text):
        """Extract technology entities from text"""
        cleaned_text = self.clean_text(text)
        tokens = word_tokenize(cleaned_text)
        
        # Single word tech terms
        single_terms = [token for token in tokens if token in self.tech_entities]
        
        # Multi-word tech terms
        for entity in self.tech_entities:
            if ' ' in entity and entity in cleaned_text:
                single_terms.append(entity)
        
        # Deduplicate
        all_entities = list(set(single_terms))
        
        return all_entities
    
    def calculate_tech_sentiment(self, text):
        """Calculate sentiment specifically for technology text"""
        cleaned_text = self.clean_text(text)
        tokens = word_tokenize(cleaned_text)
        
        # Remove stopwords
        filtered_tokens = [token for token in tokens if token not in self.stopwords]
        
        # Count technology positive and negative words
        pos_matches = [word for word in filtered_tokens if word in self.tech_pos]
        neg_matches = [word for word in filtered_tokens if word in self.tech_neg]
        
        # Calculate tech-specific sentiment score
        pos_count = len(pos_matches)
        neg_count = len(neg_matches)
        
        if pos_count + neg_count == 0:
            tech_score = 0  # Neutral if no tech sentiment words
        else:
            tech_score = (pos_count - neg_count) / (pos_count + neg_count)
        
        # Get general sentiment from TextBlob for comparison
        general_sentiment = TextBlob(cleaned_text).sentiment.polarity
        
        # Determine final sentiment classification with bias toward tech terms
        if tech_score > 0.1:
            sentiment = "positive"
        elif tech_score < -0.1:
            sentiment = "negative"
        else:
            # If tech sentiment is neutral, use general sentiment as backup
            if general_sentiment > 0.2:
                sentiment = "positive"
            elif general_sentiment < -0.2:
                sentiment = "negative"
            else:
                sentiment = "neutral"
        
        return {
            "tech_score": tech_score,
            "general_score": general_sentiment,
            "tech_pos_words": pos_matches,
            "tech_neg_words": neg_matches,
            "sentiment": sentiment
        }
    
    def identify_tech_categories(self, entities):
        """Categorize tech entities into broader categories"""
        categories = {
            "AI/ML": ["ai", "artificial intelligence", "machine learning", "ml", "nlp", "deep learning"],
            "Cloud": ["cloud", "aws", "azure", "gcp", "google cloud", "server"],
            "Blockchain": ["blockchain", "crypto", "bitcoin", "ethereum", "nft", "token", "defi", "web3"],
            "Mobile": ["app", "mobile", "ios", "android"],
            "Software": ["software", "code", "programming", "developer", "api", "sdk", "framework", "library"],
            "Hardware": ["hardware", "gpu", "cpu", "processor"],
            "Data": ["data", "database", "sql", "nosql", "analytics", "big data", "hadoop", "spark"],
            "Internet": ["network", "internet", "wifi", "5g"],
            "Security": ["cybersecurity", "encryption", "firewall", "security"],
            "Emerging": ["iot", "vr", "ar", "virtual reality", "augmented reality", "metaverse", "robotics"]
        }
        
        result = {}
        for category, terms in categories.items():
            matches = [entity for entity in entities if entity in terms]
            if matches:
                result[category] = matches
                
        return result
    
    def process(self, text):
        """Process technology text and return comprehensive analysis"""
        if not text or len(text) < 5:
            return {
                "entities": [],
                "categories": {},
                "sentiment": "neutral",
                "analysis": {
                    "tech_score": 0,
                    "general_score": 0,
                    "tech_pos_words": [],
                    "tech_neg_words": []
                }
            }
        
        # Extract technology entities
        entities = self.extract_tech_entities(text)
        
        # Categorize entities
        categories = self.identify_tech_categories(entities)
        
        # Calculate sentiment
        sentiment_analysis = self.calculate_tech_sentiment(text)
        
        # Return combined analysis
        return {
            "entities": entities,
            "categories": categories,
            "sentiment": sentiment_analysis["sentiment"],
            "analysis": sentiment_analysis
        }

# Example usage
if __name__ == "__main__":
    analyzer = TechSentimentAnalyzer()
    
    test_texts = [
        "The new AI model achieves breakthrough performance with 20% improved accuracy.",
        "Users reported multiple crashes and security vulnerabilities in the latest app update.",
        "The company is investing heavily in cloud infrastructure and machine learning capabilities.",
        "The blockchain startup failed to deliver on its promises, and its token value has plummeted."
    ]
    
    for text in test_texts:
        result = analyzer.process(text)
        print(f"\nText: {text}")
        print(f"Sentiment: {result['sentiment']}")
        print(f"Tech entities: {result['entities']}")
        print(f"Categories: {result['categories']}")
        print(f"Tech score: {result['analysis']['tech_score']:.2f}")
        print(f"Positive words: {result['analysis']['tech_pos_words']}")
        print(f"Negative words: {result['analysis']['tech_neg_words']}") 