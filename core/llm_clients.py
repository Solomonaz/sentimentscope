import os
import time
import requests
from typing import Dict, Any, Optional
import json

class LLMClients:
    """Multi-LLM API client with failover support (OpenAI removed)"""
    
    def __init__(self):
        self.setup_clients()
        self.available_models = self.detect_available_models()
    
    def setup_clients(self):
        """Initialize LLM clients (OpenAI removed)"""
        # Gemini - only setup if we have a valid API key
        self.gemini_key = os.environ.get('GEMINI_KEY')
        if self.gemini_key and self.gemini_key.strip() and self.gemini_key != 'test_key':
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.gemini_key)
                self.genai = genai
                print("Gemini client initialized")
            except ImportError:
                print("Google Generative AI not available")
                self.gemini_key = None
            except Exception as e:
                print(f"Gemini setup failed: {e}")
                self.gemini_key = None
        else:
            print("No valid Gemini key found")
            self.gemini_key = None
        
        # Groq - simplified initialization
        self.groq_key = os.environ.get('GROQ_KEY')
        self.groq_client = None
        if self.groq_key and self.groq_key.strip() and self.groq_key != 'test_key':
            try:
                import groq
                # Simple initialization without extra parameters
                self.groq_client = groq.Groq(api_key=self.groq_key)
                print("Groq client initialized")
            except ImportError:
                print("Groq package not available")
            except TypeError as e:
                # Handle version compatibility issues
                print(f"Groq version compatibility issue: {e}")
                try:
                    # Alternative initialization
                    self.groq_client = groq.Groq(api_key=self.groq_key, timeout=30)
                    print("Groq client initialized (alternative method)")
                except:
                    print("Groq setup failed completely")
                    self.groq_client = None
            except Exception as e:
                print(f"Groq setup failed: {e}")
                self.groq_client = None
        else:
            print("No valid Groq key found")
        
        # HuggingFace - always available as fallback
        self.hf_key = os.environ.get('HF_KEY')
        print("LLM clients setup complete")
    
    def detect_available_models(self) -> list:
        """Detect which LLM models are available"""
        available = []
        
        if self.gemini_key:
            available.append('gemini')
            print("Gemini model available")
        
        if self.groq_client:
            available.append('groq')
            print("Groq model available")
        
        # Always available fallbacks
        available.append('huggingface')
        available.append('keyword_fallback')
        
        print(f"Final available models: {available}")
        return available
    
    def analyze_text(self, text: str, model_priority: list = None) -> Dict[str, Any]:
        """
        Analyze text with failover through multiple LLMs (OpenAI removed)
        
        Returns:
            Dict with emotion, intent, confidence, and model_used
        """
        if model_priority is None:
            model_priority = ['gemini', 'groq', 'huggingface', 'keyword_fallback']
        
        print(f"Starting analysis with priority: {model_priority}")
        
        for model in model_priority:
            if model not in self.available_models:
                print(f"Model {model} not available, skipping")
                continue
                
            try:
                print(f"Trying model: {model}")
                
                if model == 'gemini':
                    result = self._analyze_gemini(text)
                elif model == 'groq':
                    result = self._analyze_groq(text)
                elif model == 'huggingface':
                    result = self._analyze_huggingface(text)
                elif model == 'keyword_fallback':
                    from core.fallback_analyzer import KeywordAnalyzer
                    fallback = KeywordAnalyzer()
                    result = fallback.analyze(text)
                else:
                    continue
                
                if result:
                    result['model_used'] = model
                    print(f"Success with model: {model}")
                    return result
                else:
                    print(f"Model {model} returned no result")
                    
            except Exception as e:
                print(f"Model {model} failed: {e}")
                continue
        
        # Ultimate fallback - basic keyword analysis
        print("All models failed, using ultimate fallback")
        from core.fallback_analyzer import KeywordAnalyzer
        fallback = KeywordAnalyzer()
        result = fallback.analyze(text)
        result['model_used'] = 'ultimate_fallback'
        return result
    
    def _analyze_gemini(self, text: str) -> Optional[Dict[str, Any]]:
        """Analyze using Gemini"""
        if not self.gemini_key:
            return None
            
        try:
            model = self.genai.GenerativeModel('gemini-2.5-flash')
            
            prompt = f"""
            Analyze the following text and provide ONLY a JSON response with this exact structure:
            {{
                "emotion": "Optimistic|Fearful|Angry|Sad|Neutral|Excited|Cautious",
                "intent": "Persuasive|Informative|Neutral", 
                "confidence": 0.95
            }}
            
            Text: {text[:1500]}
            
            Be objective and analyze the emotional tone and persuasive intent.
            """
            
            response = model.generate_content(prompt)
            return self._parse_llm_response(response.text)
        except Exception as e:
            print(f"Gemini analysis error: {e}")
            return None
    
    def _analyze_groq(self, text: str) -> Optional[Dict[str, Any]]:
        """Analyze using Groq"""
        if not self.groq_client:
            return None
        
        try:
            prompt = f"""
            Analyze this text and respond with ONLY valid JSON:
            {{
                "emotion": "Optimistic|Fearful|Angry|Sad|Neutral|Excited|Cautious",
                "intent": "Persuasive|Informative|Neutral", 
                "confidence": 0.95
            }}
            
            Text: {text[:1500]}
            """
            
            response = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150
            )
            
            return self._parse_llm_response(response.choices[0].message.content)
        except Exception as e:
            print(f"Groq analysis error: {e}")
            return None
    
    def _analyze_huggingface(self, text: str) -> Optional[Dict[str, Any]]:
        """Analyze using HuggingFace transformers"""
        try:
            from transformers import pipeline
            
            print("Loading HuggingFace emotion classifier...")
            
            # Emotion analysis with error handling for different return formats
            emotion_classifier = pipeline(
                "text-classification",
                model="mistralai/Mistral-7B-Instruct-v0.3",
                return_all_scores=False,
                top_k=1
            )
            
            emotion_results = emotion_classifier(text[:512])
            
            # Handle different return formats
            emotion_label = 'neutral'
            confidence = 0.5
            
            if isinstance(emotion_results, list) and len(emotion_results) > 0:
                emotion_result = emotion_results[0]
                if isinstance(emotion_result, dict):
                    emotion_label = emotion_result.get('label', 'neutral')
                    confidence = emotion_result.get('score', 0.5)
            
            # Map HuggingFace emotions to our categories
            emotion_map = {
                'joy': 'Optimistic',
                'fear': 'Fearful', 
                'anger': 'Angry',
                'sadness': 'Sad',
                'neutral': 'Neutral',
                'surprise': 'Excited',
                'disgust': 'Cautious'
            }
            
            emotion = emotion_map.get(emotion_label, 'Neutral')
            
            # Simple intent detection
            intent = "Informative"  # Default
            persuasive_words = ['should', 'must', 'recommend', 'advise', 'suggest', 'urgent', 'critical']
            informative_words = ['according', 'data', 'research', 'analysis', 'report', 'findings']
            
            text_lower = text.lower()
            persuasive_count = sum(1 for word in persuasive_words if word in text_lower)
            informative_count = sum(1 for word in informative_words if word in text_lower)
            
            if persuasive_count > informative_count:
                intent = "Persuasive"
            elif informative_count > persuasive_count:
                intent = "Informative"
            else:
                intent = "Neutral"
            
            return {
                "emotion": emotion,
                "intent": intent,
                "confidence": float(confidence)
            }
            
        except Exception as e:
            print(f"HuggingFace analysis failed: {e}")
            return None
    
    def _parse_llm_response(self, response_text: str) -> Optional[Dict[str, Any]]:
        """Parse LLM response into structured data"""
        try:
            if not response_text:
                return None
                
            # Extract JSON from response
            json_str = response_text.strip()
            
            # Clean the response
            if '```json' in json_str:
                json_str = json_str.split('```json')[1].split('```')[0]
            elif '```' in json_str:
                json_str = json_str.split('```')[1]
            
            # Remove any non-JSON content
            start_idx = json_str.find('{')
            end_idx = json_str.rfind('}') + 1
            if start_idx != -1 and end_idx != 0:
                json_str = json_str[start_idx:end_idx]
            
            data = json.loads(json_str)
            
            # Validate required fields
            required = ['emotion', 'intent', 'confidence']
            if all(field in data for field in required):
                confidence = min(max(float(data['confidence']), 0.0), 1.0)
                return {
                    'emotion': data['emotion'],
                    'intent': data['intent'],
                    'confidence': confidence
                }
            else:
                print(f"Missing required fields in response: {data}")
        
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"Failed to parse LLM response: {e}")
            print(f"Response was: {response_text}")
        
        return None