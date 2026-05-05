import logging
import torch
from transformers import pipeline
import warnings
warnings.filterwarnings("ignore")

from typing import List
from data_models.data_models import Turn
import phrase_loader as _pl

logger = logging.getLogger(__name__)

_emotion_analyzer_instance = None


def get_emotion_analyzer() -> "EmotionAnalyzer":
    global _emotion_analyzer_instance
    if _emotion_analyzer_instance is None:
        _emotion_analyzer_instance = EmotionAnalyzer()
    return _emotion_analyzer_instance


class EmotionAnalyzer:
    """Classify emotion for each conversational turn"""
    
    EMOTION_MAP = {
        "anger": "angry",
        "joy": "happy",
        "sadness": "sad",
        "disgust": "frustrated",
        "fear": "fearful",
        "surprise": "surprised",
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
            logger.error(
                "Emotion model '%s' failed to load — all turns will use neutral/0.5 fallback. Error: %s",
                model_name, e
            )
        
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
        if not self.model_loaded:
            if not getattr(self, '_fallback_warned', False):
                logger.warning("Emotion model not loaded — returning neutral/0.5 fallback for all turns")
                self._fallback_warned = True
            turn.emotion = "neutral"
            turn.emotion_score = 0.5
            return turn

        if not turn.text.strip():
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
            logger.warning("Emotion analysis failed for turn '%.50s...': %s", turn.text, e)
            turn.emotion = "neutral"
            turn.emotion_score = 0.5

        turn.contains_apology = self._detect_apology_bert(turn.text)
        turn.contains_empathy = self._detect_empathy_bert(turn.text)
        
        sentiment, sentiment_score = self._analyze_sentiment(turn.text)
        turn.sentiment = sentiment
        turn.sentiment_score = sentiment_score
        
        return turn
    
    _APOLOGY_PHRASES_DEFAULT = [
        "i'm sorry",
        "i am sorry",
        "so sorry",
        "very sorry",
        "i apologize",
        "my apologies",
        "apologies for",
        "sorry for",
        "sorry about",
        "i regret",
    ]

    def _detect_apology_bert(self, text: str) -> bool:
        """
        Detect apology using keyword-based detection with context awareness.
        """
        if not text.strip():
            return False

        text_lower = text.lower()
        phrases = _pl.get("apology_phrases", self._APOLOGY_PHRASES_DEFAULT)
        return any(phrase in text_lower for phrase in phrases)
    
    _EMPATHY_PHRASES_DEFAULT = [
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
        "i can imagine",
    ]

    _RECORDING_ANNOUNCEMENT_DEFAULT = [
        "call is now being recorded",
        "now being recorded",
    ]

    _QUESTION_STARTERS_DEFAULT = [
        "did you", "do you", "can you", "could you", "would you", "will you",
        "what", "where", "when", "why", "how", "is that", "are you", "have you",
    ]

    _SENTIMENT_CLEAR_POSITIVE_DEFAULT = [
        "great", "excellent", "wonderful", "perfect", "fantastic", "happy",
    ]

    _SENTIMENT_CLEAR_NEGATIVE_DEFAULT = [
        "problem", "issue", "wrong", "bad", "terrible", "upset", "angry", "hate",
    ]

    _SENTIMENT_STRONG_POSITIVE_DEFAULT = [
        "thank", "thanks", "excellent", "great", "perfect", "yes", "wonderful", "welcome",
    ]

    _SENTIMENT_STRONG_NEGATIVE_DEFAULT = [
        "sorry", "no", "wrong", "upset", "angry", "hate", "disappointed", "frustrated",
    ]

    _PROCEDURAL_INDICATORS_DEFAULT = [
        "let me", "i will", "i can", "what is", "my name is",
        "the item", "the number", "zip code", "located in",
        "order number", "item number", "customer number",
    ]

    def _detect_empathy_bert(self, text: str) -> bool:
        """
        Detect empathy using keyword-based detection with context awareness.
        Focuses on genuine empathy expressions, not generic language.
        """
        if not text.strip():
            return False

        text_lower = text.lower()
        phrases = _pl.get("empathy_phrases", self._EMPATHY_PHRASES_DEFAULT)
        return any(phrase in text_lower for phrase in phrases)
    
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

        recording_phrases = _pl.get("recording_announcement_phrases", self._RECORDING_ANNOUNCEMENT_DEFAULT)
        if any(p in text_lower for p in recording_phrases):
            return "neutral", 0.5

        question_starters = _pl.get("question_starters", self._QUESTION_STARTERS_DEFAULT)
        is_question = text_trunc.endswith('?') or any(text_lower.startswith(w) for w in question_starters)

        if is_question:
            clear_positive = _pl.get("sentiment_clear_positive", self._SENTIMENT_CLEAR_POSITIVE_DEFAULT)
            clear_negative = _pl.get("sentiment_clear_negative", self._SENTIMENT_CLEAR_NEGATIVE_DEFAULT)

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
            strong_pos = _pl.get("sentiment_strong_positive", self._SENTIMENT_STRONG_POSITIVE_DEFAULT)
            strong_neg = _pl.get("sentiment_strong_negative", self._SENTIMENT_STRONG_NEGATIVE_DEFAULT)

            has_strong_pos = any(word in text_lower for word in strong_pos)
            has_strong_neg = any(word in text_lower for word in strong_neg)

            if has_strong_pos and not has_strong_neg:
                return "positive", 0.9
            elif has_strong_neg and not has_strong_pos:
                return "negative", 0.9
            else:
                return "neutral", 0.6

        procedural_indicators = _pl.get("procedural_indicators", self._PROCEDURAL_INDICATORS_DEFAULT)
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
    
    analyzer = get_emotion_analyzer()
    analyzed_turns = analyzer.analyze_turns(turns)
    
    return [turn.to_dict() for turn in analyzed_turns]