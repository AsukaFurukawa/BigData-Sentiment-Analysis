import os
import torch
import numpy as np
import pandas as pd
from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer, 
    pipeline,
    BertForSequenceClassification,
    BertTokenizer
)
import json
import re
from typing import Dict, List, Tuple, Any, Optional

class AdvancedSentimentAnalyzer:
    """Advanced sentiment analyzer using transformer models"""
    
    def __init__(self):
        """Initialize the advanced sentiment analyzers with transformer models"""
        # Set device - use GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Initialize specialized models
        self.finbert_model = self._init_finbert()
        self.bert_model = self._init_bert()
        
        # Cache for performance
        self.cache = {}
        self.max_cache_size = 1000
        
        # Load configuration if available
        try:
            base_path = os.path.dirname(__file__)
            config_path = os.path.join(os.path.dirname(os.path.dirname(base_path)), 
                                      'data-ingestion', 'config', 'keys.json')
            
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load configuration: {e}")
            self.config = {}
    
    def _init_finbert(self):
        """Initialize FinBERT model for financial sentiment analysis"""
        try:
            # Load FinBERT for financial sentiment
            model_name = "ProsusAI/finbert"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            model.to(self.device)
            
            # Create sentiment analysis pipeline
            fin_sentiment = pipeline(
                "text-classification", 
                model=model, 
                tokenizer=tokenizer,
                device=0 if self.device == "cuda" else -1
            )
            
            print("FinBERT loaded successfully")
            return fin_sentiment
        except Exception as e:
            print(f"Error loading FinBERT: {e}")
            print("Falling back to general BERT")
            return None
    
    def _init_bert(self):
        """Initialize BERT model for general sentiment analysis"""
        try:
            # Load general sentiment BERT
            model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            model.to(self.device)
            
            # Create sentiment analysis pipeline
            general_sentiment = pipeline(
                "sentiment-analysis", 
                model=model, 
                tokenizer=tokenizer,
                device=0 if self.device == "cuda" else -1
            )
            
            print("BERT loaded successfully")
            return general_sentiment
        except Exception as e:
            print(f"Error loading BERT: {e}")
            return None
    
    def clean_text(self, text: str) -> str:
        """Clean the text for analysis"""
        if not isinstance(text, str):
            return ""
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove mentions and hashtags
        text = re.sub(r'@\w+|\#\w+', '', text)
        
        # Keep only necessary punctuation
        text = re.sub(r'[^\w\s.,!?]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def analyze_finance_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze financial text using FinBERT"""
        # Clean the text
        cleaned_text = self.clean_text(text)
        
        # Check cache
        cache_key = f"fin_{cleaned_text}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        if not cleaned_text or len(cleaned_text) < 10:
            result = {
                "label": "neutral",
                "score": 0.5,
                "model": "none"
            }
        elif self.finbert_model:
            try:
                # FinBERT has labels: positive, negative, neutral
                prediction = self.finbert_model(cleaned_text)[0]
                label = prediction["label"].lower()
                score = prediction["score"]
                
                result = {
                    "label": label,
                    "score": score,
                    "model": "finbert"
                }
            except Exception as e:
                print(f"Error in FinBERT prediction: {e}")
                # Fallback to BERT
                result = self.analyze_general_sentiment(cleaned_text)
                result["model"] = "bert_fallback"
        else:
            # Fallback to general BERT
            result = self.analyze_general_sentiment(cleaned_text)
            result["model"] = "bert_fallback"
        
        # Update cache
        if len(self.cache) >= self.max_cache_size:
            # Remove oldest items
            keys = list(self.cache.keys())
            for key in keys[:100]:  # Remove oldest 100 items
                self.cache.pop(key, None)
        
        self.cache[cache_key] = result
        return result
    
    def analyze_tech_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze technology text using BERT"""
        # For now, technology uses the general BERT model
        # This can be replaced with a tech-specialized model in the future
        cleaned_text = self.clean_text(text)
        
        # Check cache
        cache_key = f"tech_{cleaned_text}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        result = self.analyze_general_sentiment(cleaned_text)
        result["model"] = "bert"
        
        # Update cache
        if len(self.cache) >= self.max_cache_size:
            keys = list(self.cache.keys())
            for key in keys[:100]:
                self.cache.pop(key, None)
        
        self.cache[cache_key] = result
        return result
    
    def analyze_general_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze general text using BERT"""
        cleaned_text = self.clean_text(text)
        
        if not cleaned_text or len(cleaned_text) < 10:
            return {
                "label": "neutral",
                "score": 0.5,
                "model": "none"
            }
        
        if self.bert_model:
            try:
                # BERT multi-class sentiment returns a 1-5 star rating
                prediction = self.bert_model(cleaned_text)[0]
                score = float(prediction["label"].split()[0]) / 5.0  # Convert 1-5 stars to 0-1 score
                
                # Map score to sentiment
                if score >= 0.6:
                    label = "positive"
                elif score <= 0.4:
                    label = "negative"
                else:
                    label = "neutral"
                
                return {
                    "label": label,
                    "score": score,
                    "model": "bert"
                }
            except Exception as e:
                print(f"Error in BERT prediction: {e}")
                # Simple fallback
                return {
                    "label": "neutral",
                    "score": 0.5,
                    "model": "fallback"
                }
        else:
            # No model available
            return {
                "label": "neutral",
                "score": 0.5,
                "model": "none"
            }
    
    def process(self, text: str, domain: str = "general") -> Dict[str, Any]:
        """Process text and determine sentiment based on domain"""
        if not text or len(text) < 10:
            return {
                "sentiment": "neutral",
                "score": 0.5,
                "confidence": "low",
                "model": "none"
            }
        
        # Choose the appropriate model based on domain
        if domain in ["finance", "finance-tech"]:
            result = self.analyze_finance_sentiment(text)
        elif domain in ["technology", "tech-finance"]:
            result = self.analyze_tech_sentiment(text)
        else:
            # For general domain, compare results from both models
            fin_result = self.analyze_finance_sentiment(text)
            tech_result = self.analyze_tech_sentiment(text)
            
            # Choose the result with higher confidence
            if abs(fin_result["score"] - 0.5) > abs(tech_result["score"] - 0.5):
                result = fin_result
            else:
                result = tech_result
        
        # Calculate confidence based on distance from neutral (0.5)
        score_distance = abs(result["score"] - 0.5)
        if score_distance > 0.3:
            confidence = "high"
        elif score_distance > 0.15:
            confidence = "medium"
        else:
            confidence = "low"
        
        return {
            "sentiment": result["label"],
            "score": result["score"],
            "confidence": confidence,
            "model": result["model"]
        }

# Example usage
if __name__ == "__main__":
    analyzer = AdvancedSentimentAnalyzer()
    
    test_texts = [
        {
            "text": "The quarterly earnings exceeded analyst expectations, leading to a 15% surge in stock price.",
            "domain": "finance"
        },
        {
            "text": "The new AI chip delivers breakthrough performance with 40% less power consumption.",
            "domain": "technology"
        },
        {
            "text": "Investors grew concerned about rising inflation rates and potential market corrections.",
            "domain": "finance"
        },
        {
            "text": "Security researchers discovered critical vulnerabilities in the latest software update.",
            "domain": "technology"
        }
    ]
    
    for test in test_texts:
        result = analyzer.process(test["text"], test["domain"])
        print(f"\nText: {test['text']}")
        print(f"Domain: {test['domain']}")
        print(f"Sentiment: {result['sentiment']}")
        print(f"Score: {result['score']:.2f}")
        print(f"Confidence: {result['confidence']}")
        print(f"Model: {result['model']}") 