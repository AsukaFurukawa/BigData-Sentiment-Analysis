import os
import torch
import numpy as np
import pandas as pd
from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer,
    pipeline,
    XLMRobertaForSequenceClassification,
    XLMRobertaTokenizer
)
import json
import re
from typing import Dict, List, Tuple, Any, Optional
import langdetect
from langdetect.lang_detect_exception import LangDetectException

class MultilingualSentimentAnalyzer:
    """Multilingual sentiment analyzer using XLM-RoBERTa models"""
    
    def __init__(self):
        """Initialize the multilingual sentiment analyzer"""
        # Set device - use GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Initialize multilingual model
        self.xlm_model = self._init_xlm_roberta()
        
        # Initialize specialized finance model
        self.finbert_model = self._init_finbert()
        
        # Cache for performance
        self.cache = {}
        self.max_cache_size = 1000
        
        # Language detection cache
        self.lang_cache = {}
        
        # Language support
        self.supported_languages = {
            'en': 'English',
            'fr': 'French',
            'de': 'German',
            'es': 'Spanish',
            'it': 'Italian',
            'pt': 'Portuguese',
            'ru': 'Russian',
            'zh': 'Chinese',
            'ja': 'Japanese',
            'ar': 'Arabic'
        }
        
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
    
    def _init_xlm_roberta(self):
        """Initialize XLM-RoBERTa model for multilingual sentiment analysis"""
        try:
            # Load XLM-RoBERTa multilingual model
            model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            model.to(self.device)
            
            # Create sentiment analysis pipeline
            xlm_sentiment = pipeline(
                "sentiment-analysis", 
                model=model, 
                tokenizer=tokenizer,
                device=0 if self.device == "cuda" else -1
            )
            
            print("XLM-RoBERTa loaded successfully")
            return xlm_sentiment
        except Exception as e:
            print(f"Error loading XLM-RoBERTa: {e}")
            return None
    
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
            return None
    
    def detect_language(self, text: str) -> str:
        """Detect the language of the text"""
        if not text or len(text) < 10:
            return 'en'  # Default to English for very short texts
        
        # Check cache
        cache_key = hash(text[:100])  # Use first 100 chars as key
        if cache_key in self.lang_cache:
            return self.lang_cache[cache_key]
        
        try:
            # Detect language
            lang = langdetect.detect(text)
            
            # Cache result
            self.lang_cache[cache_key] = lang
            
            return lang
        except LangDetectException:
            return 'en'  # Default to English on error
    
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
    
    def analyze_multilingual_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment in any language using XLM-RoBERTa"""
        # Clean the text
        cleaned_text = self.clean_text(text)
        
        # Check cache
        cache_key = f"xlm_{cleaned_text}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        if not cleaned_text or len(cleaned_text) < 10:
            result = {
                "label": "neutral",
                "score": 0.5,
                "model": "none",
                "language": "unknown"
            }
        elif self.xlm_model:
            try:
                # Detect language
                lang = self.detect_language(cleaned_text)
                
                # Analyze sentiment with XLM-RoBERTa
                prediction = self.xlm_model(cleaned_text)[0]
                
                # Extract label and score
                label = prediction["label"].lower()
                score = prediction["score"]
                
                # Map label to standardized format
                if "positive" in label:
                    std_label = "positive"
                elif "negative" in label:
                    std_label = "negative"
                else:
                    std_label = "neutral"
                
                # For some models that use different label schemes
                if std_label == "positive" and score < 0.5:
                    score = 1 - score  # Normalize scores
                
                result = {
                    "label": std_label,
                    "score": score,
                    "model": "xlm-roberta",
                    "language": lang
                }
                
            except Exception as e:
                print(f"Error in XLM-RoBERTa prediction: {e}")
                # Fallback to neutral sentiment
                result = {
                    "label": "neutral",
                    "score": 0.5,
                    "model": "fallback",
                    "language": self.detect_language(cleaned_text)
                }
        else:
            # Model not available
            result = {
                "label": "neutral",
                "score": 0.5,
                "model": "none",
                "language": self.detect_language(cleaned_text)
            }
        
        # Update cache
        if len(self.cache) >= self.max_cache_size:
            # Remove oldest items
            keys = list(self.cache.keys())
            for key in keys[:100]:  # Remove oldest 100 items
                self.cache.pop(key, None)
        
        self.cache[cache_key] = result
        return result
    
    def analyze_finance_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze financial text using FinBERT"""
        # Clean the text
        cleaned_text = self.clean_text(text)
        
        # Check cache
        cache_key = f"fin_{cleaned_text}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Detect language
        lang = self.detect_language(cleaned_text)
        
        # Only use FinBERT for English text
        if lang != 'en' or not self.finbert_model:
            # Fallback to multilingual for non-English
            result = self.analyze_multilingual_sentiment(cleaned_text)
            result["model"] = "xlm_fallback"
            return result
        
        if not cleaned_text or len(cleaned_text) < 10:
            result = {
                "label": "neutral",
                "score": 0.5,
                "model": "none",
                "language": lang
            }
        else:
            try:
                # FinBERT has labels: positive, negative, neutral
                prediction = self.finbert_model(cleaned_text)[0]
                label = prediction["label"].lower()
                score = prediction["score"]
                
                result = {
                    "label": label,
                    "score": score,
                    "model": "finbert",
                    "language": lang
                }
            except Exception as e:
                print(f"Error in FinBERT prediction: {e}")
                # Fallback to multilingual
                result = self.analyze_multilingual_sentiment(cleaned_text)
                result["model"] = "xlm_fallback"
        
        # Update cache
        if len(self.cache) >= self.max_cache_size:
            # Remove oldest items
            keys = list(self.cache.keys())
            for key in keys[:100]:
                self.cache.pop(key, None)
        
        self.cache[cache_key] = result
        return result
    
    def translate_label_to_native(self, label: str, language: str) -> str:
        """Translate sentiment label to native language if possible"""
        translations = {
            'fr': {'positive': 'positif', 'negative': 'négatif', 'neutral': 'neutre'},
            'de': {'positive': 'positiv', 'negative': 'negativ', 'neutral': 'neutral'},
            'es': {'positive': 'positivo', 'negative': 'negativo', 'neutral': 'neutro'},
            'it': {'positive': 'positivo', 'negative': 'negativo', 'neutral': 'neutro'},
            'pt': {'positive': 'positivo', 'negative': 'negativo', 'neutral': 'neutro'},
            'ru': {'positive': 'положительный', 'negative': 'отрицательный', 'neutral': 'нейтральный'},
            'zh': {'positive': '积极', 'negative': '消极', 'neutral': '中性'},
            'ja': {'positive': 'ポジティブ', 'negative': 'ネガティブ', 'neutral': '中立'},
            'ar': {'positive': 'إيجابي', 'negative': 'سلبي', 'neutral': 'محايد'}
        }
        
        if language in translations and label in translations[language]:
            return translations[language][label]
        return label
    
    def process(self, text: str, domain: str = "general") -> Dict[str, Any]:
        """Process text and determine sentiment based on domain"""
        if not text or len(text) < 10:
            return {
                "sentiment": "neutral",
                "score": 0.5,
                "confidence": "low",
                "model": "none",
                "language": "unknown",
                "language_name": "Unknown"
            }
        
        # Detect language first
        lang = self.detect_language(text)
        lang_name = self.supported_languages.get(lang, "Other")
        
        # Choose the appropriate model based on domain and language
        if domain == "finance" and lang == 'en':
            result = self.analyze_finance_sentiment(text)
        else:
            # For non-English or non-finance domains, use multilingual model
            result = self.analyze_multilingual_sentiment(text)
        
        # Calculate confidence based on distance from neutral (0.5)
        score_distance = abs(result["score"] - 0.5)
        if score_distance > 0.3:
            confidence = "high"
        elif score_distance > 0.15:
            confidence = "medium"
        else:
            confidence = "low"
        
        # Get native language label if available
        native_label = self.translate_label_to_native(result["label"], lang)
        
        return {
            "sentiment": result["label"],
            "native_sentiment": native_label,
            "score": result["score"],
            "confidence": confidence,
            "model": result["model"],
            "language": lang,
            "language_name": lang_name
        }

# Example usage
if __name__ == "__main__":
    analyzer = MultilingualSentimentAnalyzer()
    
    test_texts = [
        {
            "text": "The quarterly earnings exceeded analyst expectations, leading to a 15% surge in stock price.",
            "domain": "finance"
        },
        {
            "text": "Les résultats trimestriels ont dépassé les attentes des analystes, entraînant une hausse de 15% du cours de l'action.",
            "domain": "finance"
        },
        {
            "text": "Die neue KI-Technologie bietet bahnbrechende Leistung bei 40% geringerem Energieverbrauch.",
            "domain": "technology"
        },
        {
            "text": "Los inversores mostraron preocupación por el aumento de las tasas de inflación y posibles correcciones del mercado.",
            "domain": "finance"
        }
    ]
    
    for test in test_texts:
        result = analyzer.process(test["text"], test["domain"])
        print(f"\nText: {test['text']}")
        print(f"Domain: {test['domain']}")
        print(f"Language: {result['language_name']} ({result['language']})")
        print(f"Sentiment: {result['sentiment']} / {result['native_sentiment']}")
        print(f"Score: {result['score']:.2f}")
        print(f"Confidence: {result['confidence']}")
        print(f"Model: {result['model']}") 