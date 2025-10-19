import re
from typing import Dict, Any, List
import yaml
import os

class KeywordAnalyzer:
    """Keyword-based fallback analyzer when LLMs fail"""
    
    def __init__(self):
        self.keyword_weights = self.load_keyword_weights()
    
    def load_keyword_weights(self) -> Dict[str, Any]:
        """Load keyword weights from config"""
        try:
            with open('config.yaml', 'r') as f:
                config = yaml.safe_load(f)
                return config.get('keyword_weights', {})
        except:
            # Default weights if config fails
            return {
                'emotion': {
                    'Optimistic': ["growth", "success", "positive", "opportunity"],
                    'Fearful': ["risk", "concern", "uncertain", "challenge"],
                    'Angry': ["failure", "problem", "issue", "wrong"],
                    'Sad': ["loss", "decline", "difficult", "struggle"],
                    'Excited': ["innovation", "breakthrough", "amazing", "excellent"],
                    'Cautious': ["monitor", "evaluate", "consider", "potential"],
                    'Neutral': ["report", "data", "analysis", "information"]
                },
                'intent': {
                    'Persuasive': ["should", "must", "need to", "recommend"],
                    'Informative': ["according to", "data shows", "research indicates"],
                    'Neutral': ["the", "is", "are", "was"]
                }
            }
    
    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Analyze text using keyword matching
        
        Returns:
            Dict with emotion, intent, confidence
        """
        text_lower = text.lower()
        
        # Emotion analysis
        emotion_scores = {}
        for emotion, keywords in self.keyword_weights['emotion'].items():
            score = sum(1 for keyword in keywords if keyword.lower() in text_lower)
            emotion_scores[emotion] = score
        
        # Intent analysis  
        intent_scores = {}
        for intent, keywords in self.keyword_weights['intent'].items():
            score = sum(1 for keyword in keywords if keyword.lower() in text_lower)
            intent_scores[intent] = score
        
        # Determine results
        emotion = max(emotion_scores.items(), key=lambda x: x[1])[0]
        intent = max(intent_scores.items(), key=lambda x: x[1])[0]
        
        # Calculate confidence based on keyword matches
        total_possible = max(len(keywords) for keywords in self.keyword_weights['emotion'].values())
        emotion_conf = min(emotion_scores[emotion] / total_possible if total_possible > 0 else 0.5, 0.9)
        
        total_possible_intent = max(len(keywords) for keywords in self.keyword_weights['intent'].values())
        intent_conf = min(intent_scores[intent] / total_possible_intent if total_possible_intent > 0 else 0.5, 0.9)
        
        confidence = (emotion_conf + intent_conf) / 2
        
        return {
            "emotion": emotion,
            "intent": intent,
            "confidence": confidence
        }