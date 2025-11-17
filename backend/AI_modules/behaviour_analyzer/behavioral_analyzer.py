import json
from typing import List, Dict, Tuple


class BehavioralAnalyzer:
    
    def __init__(self):
        self.OPTIMAL_WPM = 150
        self.MIN_SILENCE_GAP = 0.5
        self.INTERRUPTION_THRESHOLD = 0.3
        self.OPTIMAL_RESPONSE_TIME = 1.5
    
    def analyze_call(self, turns: List[Dict]) -> Dict:
        if not turns:
            return self._empty_analysis()
        
        agent_turns = [t for t in turns if t.get('speaker', '').lower() == 'agent']
        customer_turns = [t for t in turns if t.get('speaker', '').lower() == 'customer']
        
        wpm_metrics = self._calculate_wpm(agent_turns, customer_turns)
        silence_metrics = self._analyze_silences(turns)
        interruption_metrics = self._detect_interruptions(turns)
        response_metrics = self._analyze_response_times(turns)
        question_metrics = self._analyze_questions(agent_turns, customer_turns)
        listening_metrics = self._analyze_active_listening(agent_turns)
        
        behavioral_score = self._calculate_behavioral_score({
            **wpm_metrics,
            **silence_metrics,
            **interruption_metrics,
            **response_metrics,
            **question_metrics,
            **listening_metrics
        })
        
        return {
            "words_per_minute": wpm_metrics,
            "silence_analysis": silence_metrics,
            "interruption_analysis": interruption_metrics,
            "response_time_analysis": response_metrics,
            "question_analysis": question_metrics,
            "active_listening": listening_metrics,
            "behavioral_score": behavioral_score,
            "overall_assessment": self._generate_assessment(behavioral_score)
        }
    
    def _calculate_wpm(self, agent_turns: List[Dict], customer_turns: List[Dict]) -> Dict:
        def get_wpm(turns: List[Dict]) -> Tuple[float, int, float]:
            if not turns:
                return 0.0, 0, 0.0
            
            total_words = sum(len(t.get('text', '').split()) for t in turns)
            total_duration = sum(t.get('end_time', 0) - t.get('start_time', 0) for t in turns)
            
            if total_duration == 0:
                return 0.0, total_words, 0.0
            
            wpm = (total_words / total_duration) * 60
            return round(wpm, 1), total_words, round(total_duration, 2)
        
        agent_wpm, agent_words, agent_duration = get_wpm(agent_turns)
        customer_wpm, customer_words, customer_duration = get_wpm(customer_turns)
        
        agent_pacing = self._assess_pacing(agent_wpm)
        customer_pacing = self._assess_pacing(customer_wpm)
        
        return {
            "agent_wpm": agent_wpm,
            "agent_total_words": agent_words,
            "agent_talk_time_sec": agent_duration,
            "agent_pacing": agent_pacing,
            "customer_wpm": customer_wpm,
            "customer_total_words": customer_words,
            "customer_talk_time_sec": customer_duration,
            "customer_pacing": customer_pacing,
            "talk_ratio": round(agent_words / max(customer_words, 1), 2)
        }
    
    def _assess_pacing(self, wpm: float) -> str:
        if wpm == 0:
            return "unknown"
        elif wpm < 120:
            return "too_slow"
        elif wpm <= 180:
            return "optimal"
        elif wpm <= 220:
            return "slightly_fast"
        else:
            return "too_fast"
    
    def _analyze_silences(self, turns: List[Dict]) -> Dict:
        if len(turns) < 2:
            return {
                "silence_count": 0,
                "total_silence_sec": 0.0,
                "silence_percentage": 0.0,
                "avg_silence_duration": 0.0,
                "longest_silence": 0.0,
                "silences": []
            }
        
        sorted_turns = sorted(turns, key=lambda t: t.get('start_time', 0))
        
        silences = []
        for i in range(len(sorted_turns) - 1):
            current_end = sorted_turns[i].get('end_time', 0)
            next_start = sorted_turns[i + 1].get('start_time', 0)
            gap = next_start - current_end
            
            if gap >= self.MIN_SILENCE_GAP:
                silences.append({
                    "start": current_end,
                    "end": next_start,
                    "duration": round(gap, 2),
                    "after_speaker": sorted_turns[i].get('speaker', 'Unknown'),
                    "before_speaker": sorted_turns[i + 1].get('speaker', 'Unknown')
                })
        
        total_silence = sum(s['duration'] for s in silences)
        total_call_duration = max(t.get('end_time', 0) for t in turns) if turns else 1
        silence_percentage = (total_silence / total_call_duration) * 100
        
        return {
            "silence_count": len(silences),
            "total_silence_sec": round(total_silence, 2),
            "silence_percentage": round(silence_percentage, 1),
            "avg_silence_duration": round(total_silence / len(silences), 2) if silences else 0.0,
            "longest_silence": round(max((s['duration'] for s in silences), default=0.0), 2),
            "silences": silences[:10]
        }
    
    def _detect_interruptions(self, turns: List[Dict]) -> Dict:
        if len(turns) < 2:
            return {
                "interruption_count": 0,
                "agent_interruptions": 0,
                "customer_interruptions": 0,
                "interruptions": []
            }
        
        sorted_turns = sorted(turns, key=lambda t: t.get('start_time', 0))
        
        interruptions = []
        for i in range(len(sorted_turns) - 1):
            current_speaker = sorted_turns[i].get('speaker', 'Unknown')
            current_end = sorted_turns[i].get('end_time', 0)
            
            next_speaker = sorted_turns[i + 1].get('speaker', 'Unknown')
            next_start = sorted_turns[i + 1].get('start_time', 0)
            
            overlap = current_end - next_start
            
            if overlap > self.INTERRUPTION_THRESHOLD and current_speaker != next_speaker:
                interruptions.append({
                    "timestamp": next_start,
                    "interrupted_speaker": current_speaker,
                    "interrupting_speaker": next_speaker,
                    "overlap_duration": round(overlap, 2),
                    "interrupted_text": sorted_turns[i].get('text', '')[:100]
                })
        
        agent_interruptions = sum(1 for i in interruptions if i['interrupting_speaker'].lower() == 'agent')
        customer_interruptions = sum(1 for i in interruptions if i['interrupting_speaker'].lower() == 'customer')
        
        return {
            "interruption_count": len(interruptions),
            "agent_interruptions": agent_interruptions,
            "customer_interruptions": customer_interruptions,
            "interruptions": interruptions
        }
    
    def _analyze_response_times(self, turns: List[Dict]) -> Dict:
        if len(turns) < 2:
            return {
                "avg_agent_response_time": 0.0,
                "avg_customer_response_time": 0.0,
                "response_times": []
            }
        
        sorted_turns = sorted(turns, key=lambda t: t.get('start_time', 0))
        
        agent_response_times = []
        customer_response_times = []
        response_details = []
        
        for i in range(len(sorted_turns) - 1):
            current_speaker = sorted_turns[i].get('speaker', 'Unknown')
            current_end = sorted_turns[i].get('end_time', 0)
            
            next_speaker = sorted_turns[i + 1].get('speaker', 'Unknown')
            next_start = sorted_turns[i + 1].get('start_time', 0)
            
            if current_speaker != next_speaker:
                response_time = next_start - current_end
                
                if response_time >= 0:
                    response_details.append({
                        "timestamp": next_start,
                        "responding_speaker": next_speaker,
                        "response_time": round(response_time, 2)
                    })
                    
                    if next_speaker.lower() == 'agent':
                        agent_response_times.append(response_time)
                    else:
                        customer_response_times.append(response_time)
        
        return {
            "avg_agent_response_time": round(sum(agent_response_times) / len(agent_response_times), 2) if agent_response_times else 0.0,
            "avg_customer_response_time": round(sum(customer_response_times) / len(customer_response_times), 2) if customer_response_times else 0.0,
            "agent_response_assessment": self._assess_response_time(
                sum(agent_response_times) / len(agent_response_times) if agent_response_times else 0
            ),
            "response_times": response_details[:10]
        }
    
    def _assess_response_time(self, avg_time: float) -> str:
        if avg_time == 0:
            return "unknown"
        elif avg_time < 0.5:
            return "too_quick"
        elif avg_time <= 2.0:
            return "optimal"
        elif avg_time <= 4.0:
            return "acceptable"
        else:
            return "too_slow"
    
    def _analyze_questions(self, agent_turns: List[Dict], customer_turns: List[Dict]) -> Dict:
        def count_questions(turns: List[Dict]) -> int:
            count = 0
            for turn in turns:
                text = turn.get('text', '').lower()
                if '?' in text or any(text.startswith(q) for q in ['what', 'when', 'where', 'why', 'how', 'who', 'which', 'can', 'could', 'would', 'should', 'do', 'does', 'did', 'is', 'are']):
                    count += 1
            return count
        
        agent_questions = count_questions(agent_turns)
        customer_questions = count_questions(customer_turns)
        
        questions_per_agent_turn = agent_questions / len(agent_turns) if agent_turns else 0
        
        return {
            "agent_questions": agent_questions,
            "customer_questions": customer_questions,
            "agent_question_rate": round(questions_per_agent_turn, 2),
            "question_pattern": "good" if 0.2 <= questions_per_agent_turn <= 0.6 else "needs_improvement"
        }
    
    def _analyze_active_listening(self, agent_turns: List[Dict]) -> Dict:
        listening_phrases = [
            "i understand", "i see", "i hear you", "that makes sense",
            "i appreciate", "thank you for", "let me make sure", 
            "if i understand correctly", "so what you're saying",
            "got it", "okay", "alright", "right", "absolutely",
            "definitely", "certainly", "of course", "sure"
        ]
        
        acknowledgment_count = 0
        acknowledgment_turns = []
        
        for turn in agent_turns:
            text = turn.get('text', '').lower()
            if any(phrase in text for phrase in listening_phrases):
                acknowledgment_count += 1
                acknowledgment_turns.append({
                    "timestamp": turn.get('start_time', 0),
                    "text": turn.get('text', '')[:100]
                })
        
        acknowledgment_rate = acknowledgment_count / len(agent_turns) if agent_turns else 0
        
        return {
            "acknowledgment_count": acknowledgment_count,
            "acknowledgment_rate": round(acknowledgment_rate, 2),
            "listening_assessment": "excellent" if acknowledgment_rate >= 0.4 else "good" if acknowledgment_rate >= 0.2 else "needs_improvement",
            "acknowledgment_examples": acknowledgment_turns[:5]
        }
    
    def _calculate_behavioral_score(self, metrics: Dict) -> float:
        score = 100.0
        
        agent_pacing = metrics.get('agent_pacing', 'unknown')
        if agent_pacing == 'too_fast':
            score -= 15
        elif agent_pacing == 'too_slow':
            score -= 10
        elif agent_pacing == 'slightly_fast':
            score -= 5
        
        agent_interrupts = metrics.get('agent_interruptions', 0)
        score -= min(agent_interrupts * 5, 20)
        
        response_assessment = metrics.get('agent_response_assessment', 'unknown')
        if response_assessment == 'too_slow':
            score -= 15
        elif response_assessment == 'too_quick':
            score -= 10
        
        question_pattern = metrics.get('question_pattern', 'needs_improvement')
        if question_pattern == 'needs_improvement':
            score -= 10
        
        listening = metrics.get('listening_assessment', 'needs_improvement')
        if listening == 'needs_improvement':
            score -= 20
        elif listening == 'good':
            score -= 10
        
        silence_pct = metrics.get('silence_percentage', 0)
        if silence_pct > 30:
            score -= 10
        elif silence_pct > 20:
            score -= 5
        
        talk_ratio = metrics.get('talk_ratio', 1.0)
        if talk_ratio > 3.0:
            score -= 10
        elif talk_ratio > 2.0:
            score -= 5
        
        return max(round(score, 1), 0.0)
    
    def _generate_assessment(self, score: float) -> str:
        if score >= 90:
            return "excellent"
        elif score >= 75:
            return "good"
        elif score >= 60:
            return "acceptable"
        elif score >= 40:
            return "needs_improvement"
        else:
            return "poor"
    
    def _empty_analysis(self) -> Dict:
        return {
            "words_per_minute": {},
            "silence_analysis": {},
            "interruption_analysis": {},
            "response_time_analysis": {},
            "question_analysis": {},
            "active_listening": {},
            "behavioral_score": 0.0,
            "overall_assessment": "unknown"
        }


def behavioral_analyze_call(transcript):
    """
    Analyze behavioral patterns in a call transcript from WhisperX.
    
    Args:
        transcript: Dict with 'utterances' key containing list of utterance dicts,
                   or JSON string representation of the same
    
    Returns:
        Dict with behavioral metrics including WPM, silences, interruptions,
        response times, questions, active listening, and overall score
    """
    if isinstance(transcript, str):
        transcript = json.loads(transcript)
    
    if not isinstance(transcript, dict):
        raise ValueError("Transcript must be a dict or JSON string")
    
    utterances = transcript.get("utterances", [])
    if not utterances:
        raise ValueError("Transcript must contain 'utterances' key with list of utterances")
    
    turns = []
    for utt in utterances:
        turns.append({
            'speaker': utt.get('role', 'Unknown'),
            'text': utt.get('text', ''),
            'start_time': utt.get('start', 0.0),
            'end_time': utt.get('end', 0.0)
        })
    
    analyzer = BehavioralAnalyzer()
    return analyzer.analyze_call(turns)
