import json
from typing import List, Dict, Optional
from collections import Counter

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_models.data_models import Turn


class CallSummaryAnalyzer:
    
    def __init__(self):
        pass
    
    def analyze_call(self, turns: List[Turn]) -> Dict:
        if not turns:
            return self._empty_summary()
        
        agent_turns = [t for t in turns if t.speaker == "Agent"]
        customer_turns = [t for t in turns if t.speaker == "Customer"]
        
        emotion_dist = self._get_emotion_distribution(turns)
        sentiment_dist = self._get_sentiment_distribution(turns)
        
        key_moments = self._find_key_moments(turns)
        
        trajectory = self._analyze_trajectory(turns)
        
        resolution = self._determine_resolution(turns, key_moments)
        satisfaction = self._predict_satisfaction(turns, key_moments, trajectory)
        
        narrative = self._generate_narrative(
            turns, trajectory, key_moments, emotion_dist, satisfaction
        )
        
        return {
            "summary": narrative,
            "emotional_journey": trajectory,
            "emotion_distribution": emotion_dist,
            "sentiment_distribution": sentiment_dist,
            "key_moments": key_moments,
            "call_tone": self._determine_tone(sentiment_dist),
            "resolution_status": resolution,
            "customer_satisfaction": satisfaction,
            "agent_empathy_score": self._calculate_agent_empathy(agent_turns),
            "customer_frustration_level": self._calculate_customer_frustration(customer_turns)
        }
    
    def _get_emotion_distribution(self, turns: List[Turn]) -> Dict:
        emotions = [t.emotion for t in turns if t.emotion]
        emotion_counts = dict(Counter(emotions))
        
        total = len(emotions) if emotions else 1
        return {
            emotion: {
                "count": count,
                "percentage": round((count / total) * 100, 1)
            }
            for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True)
        }
    
    def _get_sentiment_distribution(self, turns: List[Turn]) -> Dict:
        sentiments = [t.sentiment for t in turns if t.sentiment]
        sentiment_counts = dict(Counter(sentiments))
        
        total = len(sentiments) if sentiments else 1
        return {
            sentiment: {
                "count": count,
                "percentage": round((count / total) * 100, 1)
            }
            for sentiment, count in sorted(sentiment_counts.items(), key=lambda x: x[1], reverse=True)
        }
    
    def _analyze_trajectory(self, turns: List[Turn]) -> Dict:
        if len(turns) < 2:
            return {
                "start_emotion": turns[0].emotion if turns else "unknown",
                "end_emotion": turns[0].emotion if turns else "unknown",
                "trajectory": "flat",
                "description": "Single or no turns in call"
            }
        
        customer_turns = [t for t in turns if t.speaker == "Customer"]
        
        if not customer_turns:
            customer_turns = turns
        
        start_emotion = customer_turns[0].emotion if customer_turns else turns[0].emotion
        end_emotion = customer_turns[-1].emotion if customer_turns else turns[-1].emotion
        
        start_sentiment = customer_turns[0].sentiment if customer_turns else turns[0].sentiment
        end_sentiment = customer_turns[-1].sentiment if customer_turns else turns[-1].sentiment
        
        trajectory_type = self._categorize_trajectory(start_emotion, end_emotion)
        
        emotions = [t.emotion for t in customer_turns]
        sentiment_changes = self._count_sentiment_changes(customer_turns)
        
        return {
            "start_emotion": start_emotion,
            "end_emotion": end_emotion,
            "start_sentiment": start_sentiment,
            "end_sentiment": end_sentiment,
            "trajectory": trajectory_type,
            "total_customer_turns": len(customer_turns),
            "sentiment_changes": sentiment_changes,
            "description": self._describe_trajectory(start_emotion, end_emotion, trajectory_type)
        }
    
    def _categorize_trajectory(self, start: str, end: str) -> str:
        positive_emotions = ["happy", "neutral"]
        negative_emotions = ["sad", "angry", "frustrated", "confused"]
        
        start_is_negative = start in negative_emotions
        end_is_negative = end in negative_emotions
        
        if not start_is_negative and not end_is_negative:
            return "positive_throughout"
        elif start_is_negative and not end_is_negative:
            return "resolved"
        elif not start_is_negative and end_is_negative:
            return "escalated"
        else:
            return "negative_throughout"
    
    def _describe_trajectory(self, start: str, end: str, trajectory: str) -> str:
        descriptions = {
            "positive_throughout": f"Call maintained positive tone throughout (started {start}, ended {end})",
            "resolved": f"Issue was resolved - customer started {start} but ended {end}",
            "escalated": f"Situation escalated - customer started {start} but ended {end}",
            "negative_throughout": f"Customer remained {end} throughout call (started {start})"
        }
        return descriptions.get(trajectory, "Unknown trajectory")
    
    def _count_sentiment_changes(self, turns: List[Turn]) -> int:
        sentiments = [t.sentiment for t in turns if t.sentiment]
        if len(sentiments) < 2:
            return 0
        
        changes = 0
        for i in range(len(sentiments) - 1):
            if sentiments[i] != sentiments[i + 1]:
                changes += 1
        
        return changes
    
    def _find_key_moments(self, turns: List[Turn]) -> List[Dict]:
        moments = []
        
        for turn in turns:
            if turn.speaker == "Customer" and turn.emotion in ["angry", "frustrated"]:
                moments.append({
                    "type": "escalation",
                    "speaker": turn.speaker,
                    "timestamp_sec": turn.start_sec,
                    "text": turn.text[:100],
                    "emotion": turn.emotion,
                    "turn_id": turn.turn_id
                })
        
        customer_emotions = {t.turn_id: t.emotion for t in turns if t.speaker == "Customer"}
        for turn in turns:
            if turn.speaker == "Agent" and turn.contains_apology:
                prev_customer_turns = [t for t in turns if t.speaker == "Customer" and t.turn_id < turn.turn_id]
                if prev_customer_turns:
                    last_customer_emotion = prev_customer_turns[-1].emotion
                    if last_customer_emotion in ["sad", "angry", "frustrated"]:
                        moments.append({
                            "type": "apology",
                            "speaker": turn.speaker,
                            "timestamp_sec": turn.start_sec,
                            "text": turn.text[:100],
                            "emotion": turn.emotion,
                            "turn_id": turn.turn_id
                        })
        
        for turn in turns:
            if turn.speaker == "Agent" and turn.contains_empathy:
                prev_customer_turns = [t for t in turns if t.speaker == "Customer" and t.turn_id < turn.turn_id]
                if prev_customer_turns:
                    last_customer_emotion = prev_customer_turns[-1].emotion
                    if last_customer_emotion in ["sad", "angry", "frustrated", "confused"]:
                        if not turn.contains_apology:
                            moments.append({
                                "type": "empathy",
                                "speaker": turn.speaker,
                                "timestamp_sec": turn.start_sec,
                                "text": turn.text[:100],
                                "emotion": turn.emotion,
                                "turn_id": turn.turn_id
                            })
        
        if turns:
            customer_turns = [t for t in turns if t.speaker == "Customer"]
            if customer_turns:
                last_customer_turn = customer_turns[-1]
                if last_customer_turn.emotion == "happy":
                    if len(customer_turns) > 1:
                        earlier_negative = any(t.emotion in ["sad", "angry", "frustrated"] for t in customer_turns[:-1])
                        if earlier_negative:
                            moments.append({
                                "type": "resolution",
                                "speaker": last_customer_turn.speaker,
                                "timestamp_sec": last_customer_turn.start_sec,
                                "text": last_customer_turn.text[:100],
                                "emotion": last_customer_turn.emotion,
                                "turn_id": last_customer_turn.turn_id
                            })
        
        seen_turn_ids = set()
        unique_moments = []
        for moment in moments:
            if moment['turn_id'] not in seen_turn_ids:
                unique_moments.append(moment)
                seen_turn_ids.add(moment['turn_id'])
        
        unique_moments.sort(key=lambda x: x["timestamp_sec"])
        
        return unique_moments
    
    def _determine_resolution(self, turns: List[Turn], key_moments: List[Dict]) -> str:
        customer_turns = [t for t in turns if t.speaker == "Customer"]
        
        if not customer_turns:
            return "unknown"
        
        last_customer_turn = customer_turns[-1]
        
        positive_ending = last_customer_turn.emotion in ["happy", "neutral"]
        has_thanks = "thank" in last_customer_turn.text.lower()
        has_apology = any(m["type"] == "apology" and m["speaker"] == "Agent" for m in key_moments)
        
        if positive_ending and has_thanks:
            return "resolved"
        elif last_customer_turn.emotion in ["angry", "frustrated"]:
            return "unresolved"
        elif has_apology and positive_ending:
            return "resolved"
        else:
            return "pending"
    
    def _predict_satisfaction(self, turns: List[Turn], key_moments: List[Dict], trajectory: Dict) -> str:
        customer_turns = [t for t in turns if t.speaker == "Customer"]
        
        if not customer_turns:
            return "unknown"
        
        last_emotion = customer_turns[-1].emotion
        agent_apologies = len([m for m in key_moments if m["type"] == "apology" and m["speaker"] == "Agent"])
        agent_empathy_count = len([m for m in key_moments if m["type"] == "empathy" and m["speaker"] == "Agent"])
        avg_sentiment = self._get_average_sentiment(customer_turns)
        
        score = 0
        
        if last_emotion in ["happy", "neutral"]:
            score += 3
        elif last_emotion in ["sad", "confused"]:
            score += 1
        else:
            score -= 2
        
        score += agent_apologies
        score += agent_empathy_count
        
        if avg_sentiment == "positive":
            score += 2
        elif avg_sentiment == "negative":
            score -= 1
        
        if trajectory["trajectory"] == "resolved":
            score += 2
        elif trajectory["trajectory"] == "escalated":
            score -= 2
        
        if score >= 4:
            return "very_satisfied"
        elif score >= 2:
            return "satisfied"
        elif score >= 0:
            return "neutral"
        elif score >= -2:
            return "dissatisfied"
        else:
            return "very_dissatisfied"
    
    def _calculate_agent_empathy(self, agent_turns: List[Turn]) -> float:
        if not agent_turns:
            return 0.0
        
        empathy_count = len([t for t in agent_turns if t.contains_empathy])
        apology_count = len([t for t in agent_turns if t.contains_apology])
        
        total_empathy_signals = empathy_count + apology_count
        score = min(total_empathy_signals / len(agent_turns), 1.0)
        
        return round(score, 3)
    
    def _calculate_customer_frustration(self, customer_turns: List[Turn]) -> float:
        if not customer_turns:
            return 0.0
        
        frustration_count = len([t for t in customer_turns if t.emotion in ["angry", "frustrated"]])
        negative_sentiment_count = len([t for t in customer_turns if t.sentiment == "negative"])
        
        total_frustration_signals = frustration_count + negative_sentiment_count
        score = min(total_frustration_signals / (len(customer_turns) * 2), 1.0)
        
        return round(score, 3)
    
    def _determine_tone(self, sentiment_dist: Dict) -> str:
        if not sentiment_dist:
            return "neutral"
        
        positive_pct = sentiment_dist.get("positive", {}).get("percentage", 0)
        negative_pct = sentiment_dist.get("negative", {}).get("percentage", 0)
        
        if positive_pct > 60:
            return "positive"
        elif negative_pct > 60:
            return "negative"
        else:
            return "neutral"
    
    def _get_average_sentiment(self, turns: List[Turn]) -> Optional[str]:
        sentiments = [t.sentiment for t in turns if t.sentiment]
        if not sentiments:
            return None
        
        positive = sentiments.count("positive")
        negative = sentiments.count("negative")
        
        if positive > negative:
            return "positive"
        elif negative > positive:
            return "negative"
        else:
            return "neutral"
    
    def _generate_narrative(
        self, 
        turns: List[Turn], 
        trajectory: Dict, 
        key_moments: List[Dict],
        emotion_dist: Dict,
        satisfaction: str
    ) -> str:
        customer_turns = [t for t in turns if t.speaker == "Customer"]
        agent_turns = [t for t in turns if t.speaker == "Agent"]
        
        narrative = []
        
        if customer_turns:
            first_emotion = customer_turns[0].emotion
            narrative.append(f"Call started with customer expressing {first_emotion} sentiment.")
        
        if key_moments:
            apologies = [m for m in key_moments if m["type"] == "apology"]
            escalations = [m for m in key_moments if m["type"] == "escalation"]
            empathy_moments = [m for m in key_moments if m["type"] == "empathy"]
            
            if apologies:
                narrative.append(f"Agent provided {len(apologies)} apology/apologies during the call.")
            
            if escalations:
                narrative.append(f"Customer showed frustration or anger {len(escalations)} time(s).")
            
            if empathy_moments:
                narrative.append(f"Agent demonstrated empathy {len(empathy_moments)} time(s).")
        
        narrative.append(f"Emotional trajectory: {trajectory['description']}")
        
        if customer_turns:
            last_emotion = customer_turns[-1].emotion
            narrative.append(f"Call ended with customer showing {last_emotion} emotion.")
        
        narrative.append(f"Issue resolution: {trajectory.get('trajectory', 'unknown').replace('_', ' ').title()}")
        narrative.append(f"Predicted customer satisfaction: {satisfaction.replace('_', ' ').title()}")
        
        return " ".join(narrative)
    
    def _empty_summary(self) -> Dict:
        return {
            "summary": "No conversation turns found.",
            "emotional_journey": {
                "trajectory": "unknown",
                "description": "Insufficient data"
            },
            "emotion_distribution": {},
            "sentiment_distribution": {},
            "key_moments": [],
            "call_tone": "neutral",
            "resolution_status": "unknown",
            "customer_satisfaction": "unknown",
            "agent_empathy_score": 0.0,
            "customer_frustration_level": 0.0
        }


def summarize_emotion_call(emotion_results):
    turns = []
    for turn_dict in emotion_results:
        turn = Turn(**turn_dict)
        turns.append(turn)
    
    analyzer = CallSummaryAnalyzer()
    return analyzer.analyze_call(turns)
