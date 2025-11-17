import torch
from transformers import pipeline
import warnings
warnings.filterwarnings("ignore")

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List
from data_models.data_models import Turn


class EmotionAnalyzer:
    """Classify emotion for each conversational turn"""
    
    EMOTION_MAP = {
        "anger": "angry",
        "joy": "happy",
        "sadness": "sad",
        "disgust": "frustrated",
        "fear": "confused",
        "surprise": "confused",
        "neutral": "neutral"
    }
    
    def __init__(self, model_name: str = "j-hartmann/emotion-english-distilroberta-base"):
        device = 0 if torch.cuda.is_available() else -1
        
        try:
            self.classifier = pipeline(
                "text-classification",
                model=model_name,
                device=device,
                top_k=None
            )
            self.model_loaded = True
        except Exception as e:
            self.classifier = None
            self.model_loaded = False
        
        try:
            self.sentiment_classifier = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=device
            )
        except Exception as e:
            self.sentiment_classifier = None
    
    def analyze_turn(self, turn: Turn) -> Turn:
        """
        Analyze emotion for a single turn.
        Updates turn.emotion and turn.emotion_score in place.
        """
        if not self.model_loaded or not turn.text.strip():
            turn.emotion = "neutral"
            turn.emotion_score = 0.5
            return turn
        
        try:
            text = turn.text[:512]
            
            results = self.classifier(text)
            
            if isinstance(results, list) and isinstance(results[0], list):
                results = results[0]

            best_result = max(results, key=lambda x: x['score'])
            raw_label = best_result['label'].lower()
            score = best_result['score']
            
            turn.emotion = self.EMOTION_MAP.get(raw_label, "neutral")
            turn.emotion_score = round(score, 3)
            
        except Exception as e:
            turn.emotion = "neutral"
            turn.emotion_score = 0.5
        
        turn.contains_apology = self._detect_apology_bert(turn.text)
        turn.contains_empathy = self._detect_empathy_bert(turn.text)
        
        sentiment, sentiment_score = self._analyze_sentiment(turn.text)
        turn.sentiment = sentiment
        turn.sentiment_score = sentiment_score
        
        return turn
    
    def _detect_apology_bert(self, text: str) -> bool:
        """
        Detect apology using keyword-based detection with context awareness.
        """
        if not text.strip():
            return False
        
        text_lower = text.lower()
        
        strong_apology_phrases = [
            "i'm sorry",
            "i am sorry",
            "so sorry",
            "very sorry",
            "i apologize",
            "my apologies",
            "apologies for",
            "sorry for",
            "sorry about",
            "i regret"
        ]
        
        for phrase in strong_apology_phrases:
            if phrase in text_lower:
                return True
        
        return False
    
    def _detect_empathy_bert(self, text: str) -> bool:
        """
        Detect empathy using keyword-based detection with context awareness.
        Focuses on genuine empathy expressions, not generic language.
        """
        if not text.strip():
            return False
        
        text_lower = text.lower()
        
        empathy_phrases = [
            "i understand",
            "i can understand",
            "i completely understand",
            "i realize",
            "i can see",
            "i appreciate",
            "thank you for",
            "thanks for",
            "that must be",
            "that sounds",
            "i hear you",
            "you're right",
            "that's frustrating",
            "that's disappointing",
            "i can imagine"
        ]
        
        for phrase in empathy_phrases:
            if phrase in text_lower:
                return True
        
        return False
    
    def _detect_apology_keyword(self, text: str) -> bool:
        """Fallback: keyword-based apology detection"""
        if not hasattr(self, 'APOLOGY_KEYWORDS'):
            return False
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.APOLOGY_KEYWORDS)
    
    def _detect_empathy_keyword(self, text: str) -> bool:
        """Fallback: keyword-based empathy detection"""
        if not hasattr(self, 'EMPATHY_KEYWORDS'):
            return False
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.EMPATHY_KEYWORDS)
    
    def _analyze_sentiment(self, text: str) -> tuple:
        """
        Analyze sentiment with context-aware rules for customer service conversations.
        
        The distilbert sentiment model is trained on movie reviews and often misclassifies
        questions and neutral statements as negative. This method adds intelligent filtering.
        
        Returns: (sentiment: str, score: float) where sentiment is "positive", "negative", or "neutral"
        """
        if not self.sentiment_classifier or not text or not text.strip():
            return None, None

        text_trunc = text[:512].strip()
        text_lower = text_trunc.lower()

        if "call is now being recorded" in text_lower or "now being recorded" in text_lower:
            return "neutral", 0.5

        is_question = text_trunc.endswith('?') or any(text_lower.startswith(w) for w in [
            "did you", "do you", "can you", "could you", "would you", "will you",
            "what", "where", "when", "why", "how", "is that", "are you", "have you"
        ])
        
        if is_question:
            clear_positive = ["great", "excellent", "wonderful", "perfect", "fantastic", "happy"]
            clear_negative = ["problem", "issue", "wrong", "bad", "terrible", "upset", "angry", "hate"]
            
            has_positive = any(word in text_lower for word in clear_positive)
            has_negative = any(word in text_lower for word in clear_negative)
            
            if has_positive and not has_negative:
                return "positive", 0.8
            elif has_negative and not has_positive:
                return "negative", 0.8
            else:
                return "neutral", 0.7

        words = text_lower.split()
        if len(words) <= 5:
            strong_pos = ["thank", "thanks", "excellent", "great", "perfect", "yes", "wonderful", "welcome"]
            strong_neg = ["sorry", "no", "wrong", "upset", "angry", "hate", "disappointed", "frustrated"]
            
            has_strong_pos = any(word in text_lower for word in strong_pos)
            has_strong_neg = any(word in text_lower for word in strong_neg)
            
            if has_strong_pos and not has_strong_neg:
                return "positive", 0.9
            elif has_strong_neg and not has_strong_pos:
                return "negative", 0.9
            else:
                return "neutral", 0.6

        procedural_indicators = [
            "let me", "i will", "i can", "what is", "my name is",
            "the item", "the number", "zip code", "located in",
            "order number", "item number", "customer number"
        ]
        if any(phrase in text_lower for phrase in procedural_indicators):
            return "neutral", 0.7

        try:
            result = self.sentiment_classifier(text_trunc)
            label = result[0]['label'].lower()
            score = float(result[0]['score'])

            sentiment = "positive" if label == "positive" else "negative"

            if score < 0.75:
                return "neutral", round(score, 3)

            return sentiment, round(score, 3)
        except Exception as e:
            return None, None
    
    def analyze_turns(self, turns: List[Turn]) -> List[Turn]:
        """Batch analyze all turns"""
        for turn in turns:
            self.analyze_turn(turn)
        return turns


def emotion_analyze_call(transcript):
    """
    Analyze emotions for a call transcript from WhisperX.
    
    Args:
        transcript: Dict with 'utterances' key containing list of utterance dicts,
                   or JSON string representation of the same
    
    Returns:
        List of Turn objects with emotion analysis, or list of dicts if JSON string input
    """
    import json
    
    if isinstance(transcript, str):
        try:
            transcript = json.loads(transcript)
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON string provided")
    
    if not isinstance(transcript, dict):
        raise ValueError("Transcript must be a dict or JSON string")
    
    utterances = transcript.get("utterances", [])
    if not utterances:
        raise ValueError("Transcript must contain 'utterances' key with list of utterances")
    
    turns = []
    for idx, utt in enumerate(utterances):
        turn = Turn(
            turn_id=idx,
            speaker=utt.get("role", "Unknown"),
            start_sec=utt.get("start", 0.0),
            end_sec=utt.get("end", 0.0),
            text=utt.get("text", ""),
            duration_sec=utt.get("end", 0.0) - utt.get("start", 0.0),
            word_count=len(utt.get("text", "").split())
        )
        turns.append(turn)
    
    analyzer = EmotionAnalyzer()
    analyzed_turns = analyzer.analyze_turns(turns)
    
    return [turn.to_dict() for turn in analyzed_turns]