from typing import List, Dict, Any
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

class AnalyticsEngine:
    """Compute analytics and generate chart data"""
    
    def __init__(self):
        self.emotion_categories = ["Optimistic", "Fearful", "Angry", "Sad", "Neutral", "Excited", "Cautious"]
        self.intent_categories = ["Persuasive", "Informative", "Neutral"]
    
    def compute_analytics(self, analysis_results: List[Dict]) -> Dict[str, Any]:
        """Compute comprehensive analytics from analysis results"""
        if not analysis_results:
            return self._empty_analytics()
        
        # Basic distributions
        emotion_dist = Counter([r['emotion'] for r in analysis_results])
        intent_dist = Counter([r['intent'] for r in analysis_results])
        
        # Convert to percentages
        total = len(analysis_results)
        emotion_dist_pct = {k: v/total for k, v in emotion_dist.items()}
        intent_dist_pct = {k: v/total for k, v in intent_dist.items()}
        
        # Confidence statistics
        confidences = [r.get('confidence', 0.5) for r in analysis_results]
        avg_confidence = np.mean(confidences) if confidences else 0.5
        
        # Dominant emotion
        dominant_emotion = max(emotion_dist.items(), key=lambda x: x[1])[0] if emotion_dist else "Neutral"
        
        # Page-wise analysis
        page_analysis = self._analyze_by_page(analysis_results)
        
        # Timeline data
        timeline_data = self._prepare_timeline_data(analysis_results)
        
        # Keyword analysis
        keyword_analysis = self._analyze_keywords(analysis_results)
        
        return {
            "emotion_distribution": emotion_dist_pct,
            "intent_distribution": intent_dist_pct,
            "average_confidence": float(avg_confidence),
            "dominant_emotion": dominant_emotion,
            "confidence_stats": {
                "mean": float(np.mean(confidences)),
                "std": float(np.std(confidences)),
                "min": float(np.min(confidences)),
                "max": float(np.max(confidences))
            },
            "page_analysis": page_analysis,
            "timeline_data": timeline_data,
            "keyword_analysis": keyword_analysis,
            "chart_data": self._generate_chart_data(analysis_results)
        }
    
    def _analyze_by_page(self, analysis_results: List[Dict]) -> Dict[int, Dict]:
        """Analyze emotions and intents by page"""
        page_data = {}
        
        for result in analysis_results:
            page = result['page']
            if page not in page_data:
                page_data[page] = {
                    'emotions': Counter(),
                    'intents': Counter(),
                    'confidences': []
                }
            
            page_data[page]['emotions'][result['emotion']] += 1
            page_data[page]['intents'][result['intent']] += 1
            page_data[page]['confidences'].append(result.get('confidence', 0.5))
        
        # Summarize page data
        page_summary = {}
        for page, data in page_data.items():
            dominant_emotion = max(data['emotions'].items(), key=lambda x: x[1])[0]
            dominant_intent = max(data['intents'].items(), key=lambda x: x[1])[0]
            avg_confidence = np.mean(data['confidences'])
            
            page_summary[page] = {
                'dominant_emotion': dominant_emotion,
                'dominant_intent': dominant_intent,
                'emotion_distribution': dict(data['emotions']),
                'intent_distribution': dict(data['intents']),
                'average_confidence': float(avg_confidence)
            }
        
        return page_summary
    
    def _prepare_timeline_data(self, analysis_results: List[Dict]) -> Dict[str, Any]:
        """Prepare data for emotion timeline chart"""
        if not analysis_results:
            return {}
        
        # Sort by block_id for timeline
        sorted_results = sorted(analysis_results, key=lambda x: x['block_id'])
        
        timeline = {
            'blocks': [r['block_id'] for r in sorted_results],
            'emotions': [r['emotion'] for r in sorted_results],
            'intents': [r['intent'] for r in sorted_results],
            'confidences': [r.get('confidence', 0.5) for r in sorted_results]
        }
        
        return timeline
    
    def _analyze_keywords(self, analysis_results: List[Dict]) -> Dict[str, List]:
        """Analyze keywords associated with each emotion"""
        emotion_keywords = {emotion: Counter() for emotion in self.emotion_categories}
        
        for result in analysis_results:
            emotion = result['emotion']
            text = result['text'].lower()
            
            # Simple keyword extraction (improve with NLP in production)
            words = text.split()[:20]  # First 20 words as representative
            for word in words:
                if len(word) > 3:  # Only consider words longer than 3 chars
                    emotion_keywords[emotion][word] += 1
        
        # Get top 5 keywords per emotion
        top_keywords = {}
        for emotion, counter in emotion_keywords.items():
            top_keywords[emotion] = [word for word, count in counter.most_common(5)]
        
        return top_keywords
    
    def _generate_chart_data(self, analysis_results: List[Dict]) -> Dict[str, Any]:
        """Generate data for all charts"""
        if not analysis_results:
            return {}
        
        # Emotion distribution pie chart
        emotion_dist = Counter([r['emotion'] for r in analysis_results])
        
        # Timeline data for line chart
        timeline_data = self._prepare_timeline_data(analysis_results)
        
        # Confidence heatmap data
        confidence_by_page = {}
        for result in analysis_results:
            page = result['page']
            confidence = result.get('confidence', 0.5)
            if page not in confidence_by_page:
                confidence_by_page[page] = []
            confidence_by_page[page].append(confidence)
        
        # Average confidence per page
        page_confidence = {page: np.mean(confs) for page, confs in confidence_by_page.items()}
        
        return {
            "emotion_distribution": {
                "labels": list(emotion_dist.keys()),
                "values": list(emotion_dist.values())
            },
            "intent_distribution": {
                "labels": list(Counter([r['intent'] for r in analysis_results]).keys()),
                "values": list(Counter([r['intent'] for r in analysis_results]).values())
            },
            "timeline": timeline_data,
            "confidence_heatmap": page_confidence
        }
    
    def _empty_analytics(self) -> Dict[str, Any]:
        """Return empty analytics structure"""
        return {
            "emotion_distribution": {},
            "intent_distribution": {},
            "average_confidence": 0.0,
            "dominant_emotion": "Neutral",
            "confidence_stats": {
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0
            },
            "page_analysis": {},
            "timeline_data": {},
            "keyword_analysis": {},
            "chart_data": {}
        }