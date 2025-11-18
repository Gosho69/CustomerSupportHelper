import json
from typing import Dict, List, Optional
from collections import Counter


class BehavioralSummaryAnalyzer:
    
    def __init__(self):
        pass
    
    def analyze_call(self, behavioral_data: Dict) -> Dict:
        if not behavioral_data or behavioral_data.get("behavioral_score", 0) == 0:
            return self._empty_summary()
        
        wpm_summary = self._summarize_wpm(behavioral_data.get("words_per_minute", {}))
        silence_summary = self._summarize_silences(behavioral_data.get("silence_analysis", {}))
        interruption_summary = self._summarize_interruptions(behavioral_data.get("interruption_analysis", {}))
        response_summary = self._summarize_response_times(behavioral_data.get("response_time_analysis", {}))
        question_summary = self._summarize_questions(behavioral_data.get("question_analysis", {}))
        listening_summary = self._summarize_listening(behavioral_data.get("active_listening", {}))
        
        overall_score = behavioral_data.get("behavioral_score", 0)
        overall_assessment = behavioral_data.get("overall_assessment", "unknown")
        
        narrative = self._generate_narrative(
            wpm_summary,
            silence_summary,
            interruption_summary,
            response_summary,
            question_summary,
            listening_summary,
            overall_score,
            overall_assessment
        )
        
        issues = self._identify_issues(behavioral_data)
        strengths = self._identify_strengths(behavioral_data)
        
        return {
            "summary": narrative,
            "overall_score": overall_score,
            "overall_assessment": overall_assessment,
            "pacing_analysis": wpm_summary,
            "silence_analysis": silence_summary,
            "interruption_analysis": interruption_summary,
            "response_analysis": response_summary,
            "question_analysis": question_summary,
            "listening_analysis": listening_summary,
            "key_issues": issues,
            "key_strengths": strengths
        }
    
    def _summarize_wpm(self, wpm_data: Dict) -> Dict:
        if not wpm_data:
            return {"status": "unknown", "description": "No pacing data available"}
        
        agent_wpm = wpm_data.get("agent_wpm", 0)
        customer_wpm = wpm_data.get("customer_wpm", 0)
        agent_pacing = wpm_data.get("agent_pacing", "unknown")
        talk_ratio = wpm_data.get("talk_ratio", 0)
        
        status = "good" if agent_pacing == "optimal" else "needs_attention"
        
        description_parts = []
        if agent_pacing == "optimal":
            description_parts.append(f"Agent speaking pace is optimal at {agent_wpm} WPM")
        elif agent_pacing == "too_fast":
            description_parts.append(f"Agent speaking too fast at {agent_wpm} WPM (should be 120-180)")
        elif agent_pacing == "too_slow":
            description_parts.append(f"Agent speaking too slow at {agent_wpm} WPM (should be 120-180)")
        else:
            description_parts.append(f"Agent WPM: {agent_wpm}")
        
        if talk_ratio > 3.0:
            description_parts.append(f"Agent dominating conversation (talk ratio: {talk_ratio}:1)")
        elif talk_ratio > 2.0:
            description_parts.append(f"Agent talking significantly more (talk ratio: {talk_ratio}:1)")
        
        return {
            "status": status,
            "agent_wpm": agent_wpm,
            "customer_wpm": customer_wpm,
            "agent_pacing": agent_pacing,
            "talk_ratio": talk_ratio,
            "description": ". ".join(description_parts)
        }
    
    def _summarize_silences(self, silence_data: Dict) -> Dict:
        if not silence_data:
            return {"status": "unknown", "description": "No silence data available"}
        
        silence_count = silence_data.get("silence_count", 0)
        silence_percentage = silence_data.get("silence_percentage", 0)
        avg_silence = silence_data.get("avg_silence_duration", 0)
        longest_silence = silence_data.get("longest_silence", 0)
        
        if silence_percentage > 30:
            status = "concerning"
        elif silence_percentage > 20:
            status = "moderate"
        else:
            status = "normal"
        
        description_parts = []
        if silence_percentage > 30:
            description_parts.append(f"High silence percentage ({silence_percentage}%) indicates awkward pauses")
        elif silence_percentage > 20:
            description_parts.append(f"Moderate silence percentage ({silence_percentage}%)")
        else:
            description_parts.append(f"Normal silence levels ({silence_percentage}%)")
        
        if longest_silence > 5.0:
            description_parts.append(f"Longest pause was {longest_silence} seconds")
        
        return {
            "status": status,
            "silence_count": silence_count,
            "silence_percentage": silence_percentage,
            "avg_silence_duration": avg_silence,
            "longest_silence": longest_silence,
            "description": ". ".join(description_parts)
        }
    
    def _summarize_interruptions(self, interruption_data: Dict) -> Dict:
        if not interruption_data:
            return {"status": "unknown", "description": "No interruption data available"}
        
        total_interruptions = interruption_data.get("interruption_count", 0)
        agent_interruptions = interruption_data.get("agent_interruptions", 0)
        customer_interruptions = interruption_data.get("customer_interruptions", 0)
        
        if agent_interruptions >= 5:
            status = "problematic"
        elif agent_interruptions >= 3:
            status = "concerning"
        else:
            status = "acceptable"
        
        description_parts = []
        if agent_interruptions >= 5:
            description_parts.append(f"Agent interrupted customer {agent_interruptions} times (excessive)")
        elif agent_interruptions >= 3:
            description_parts.append(f"Agent interrupted customer {agent_interruptions} times (needs improvement)")
        elif agent_interruptions > 0:
            description_parts.append(f"Agent interrupted customer {agent_interruptions} time(s)")
        else:
            description_parts.append("No agent interruptions detected")
        
        if customer_interruptions > 0:
            description_parts.append(f"Customer interrupted {customer_interruptions} time(s)")
        
        return {
            "status": status,
            "total_interruptions": total_interruptions,
            "agent_interruptions": agent_interruptions,
            "customer_interruptions": customer_interruptions,
            "description": ". ".join(description_parts)
        }
    
    def _summarize_response_times(self, response_data: Dict) -> Dict:
        if not response_data:
            return {"status": "unknown", "description": "No response time data available"}
        
        avg_agent_response = response_data.get("avg_agent_response_time", 0)
        response_assessment = response_data.get("agent_response_assessment", "unknown")
        
        if response_assessment == "optimal":
            status = "good"
        elif response_assessment in ["acceptable", "too_quick"]:
            status = "moderate"
        else:
            status = "needs_attention"
        
        description_parts = []
        if response_assessment == "optimal":
            description_parts.append(f"Agent response time is optimal ({avg_agent_response}s)")
        elif response_assessment == "too_slow":
            description_parts.append(f"Agent responds too slowly ({avg_agent_response}s, should be <2s)")
        elif response_assessment == "too_quick":
            description_parts.append(f"Agent responds very quickly ({avg_agent_response}s, may not be listening)")
        else:
            description_parts.append(f"Agent average response time: {avg_agent_response}s")
        
        return {
            "status": status,
            "avg_agent_response_time": avg_agent_response,
            "response_assessment": response_assessment,
            "description": ". ".join(description_parts)
        }
    
    def _summarize_questions(self, question_data: Dict) -> Dict:
        if not question_data:
            return {"status": "unknown", "description": "No question data available"}
        
        agent_questions = question_data.get("agent_questions", 0)
        question_rate = question_data.get("agent_question_rate", 0)
        question_pattern = question_data.get("question_pattern", "unknown")
        
        status = "good" if question_pattern == "good" else "needs_improvement"
        
        description_parts = []
        if question_pattern == "good":
            description_parts.append(f"Good question pattern ({agent_questions} questions asked)")
        else:
            if question_rate < 0.2:
                description_parts.append(f"Agent asked too few questions ({agent_questions} questions)")
            else:
                description_parts.append(f"Agent asked too many questions ({agent_questions} questions)")
        
        return {
            "status": status,
            "agent_questions": agent_questions,
            "question_rate": question_rate,
            "question_pattern": question_pattern,
            "description": ". ".join(description_parts)
        }
    
    def _summarize_listening(self, listening_data: Dict) -> Dict:
        if not listening_data:
            return {"status": "unknown", "description": "No listening data available"}
        
        acknowledgment_count = listening_data.get("acknowledgment_count", 0)
        acknowledgment_rate = listening_data.get("acknowledgment_rate", 0)
        listening_assessment = listening_data.get("listening_assessment", "unknown")
        
        if listening_assessment == "excellent":
            status = "excellent"
        elif listening_assessment == "good":
            status = "good"
        else:
            status = "needs_improvement"
        
        description_parts = []
        if listening_assessment == "excellent":
            description_parts.append(f"Excellent active listening ({acknowledgment_count} acknowledgments)")
        elif listening_assessment == "good":
            description_parts.append(f"Good active listening ({acknowledgment_count} acknowledgments)")
        else:
            description_parts.append(f"Poor active listening (only {acknowledgment_count} acknowledgments)")
        
        return {
            "status": status,
            "acknowledgment_count": acknowledgment_count,
            "acknowledgment_rate": acknowledgment_rate,
            "listening_assessment": listening_assessment,
            "description": ". ".join(description_parts)
        }
    
    def _generate_narrative(
        self,
        wpm_summary: Dict,
        silence_summary: Dict,
        interruption_summary: Dict,
        response_summary: Dict,
        question_summary: Dict,
        listening_summary: Dict,
        overall_score: float,
        overall_assessment: str
    ) -> str:
        narrative_parts = []
        
        narrative_parts.append(f"Overall behavioral score: {overall_score}/100 ({overall_assessment})")
        
        narrative_parts.append(wpm_summary.get("description", ""))
        
        if interruption_summary.get("status") in ["concerning", "problematic"]:
            narrative_parts.append(interruption_summary.get("description", ""))
        
        if response_summary.get("status") != "good":
            narrative_parts.append(response_summary.get("description", ""))
        
        if listening_summary.get("status") == "excellent":
            narrative_parts.append(listening_summary.get("description", ""))
        elif listening_summary.get("status") == "needs_improvement":
            narrative_parts.append(listening_summary.get("description", ""))
        
        if silence_summary.get("status") in ["concerning", "moderate"]:
            narrative_parts.append(silence_summary.get("description", ""))
        
        if question_summary.get("status") == "needs_improvement":
            narrative_parts.append(question_summary.get("description", ""))
        
        return " ".join(narrative_parts)
    
    def _identify_issues(self, behavioral_data: Dict) -> List[Dict]:
        issues = []
        
        wpm = behavioral_data.get("words_per_minute", {})
        if wpm.get("agent_pacing") == "too_fast":
            issues.append({
                "type": "pacing",
                "severity": "medium",
                "description": f"Agent speaking too fast at {wpm.get('agent_wpm', 0)} WPM"
            })
        elif wpm.get("agent_pacing") == "too_slow":
            issues.append({
                "type": "pacing",
                "severity": "low",
                "description": f"Agent speaking too slow at {wpm.get('agent_wpm', 0)} WPM"
            })
        
        if wpm.get("talk_ratio", 0) > 3.0:
            issues.append({
                "type": "talk_ratio",
                "severity": "high",
                "description": f"Agent dominating conversation (talk ratio: {wpm.get('talk_ratio', 0)}:1)"
            })
        
        interruptions = behavioral_data.get("interruption_analysis", {})
        if interruptions.get("agent_interruptions", 0) >= 5:
            issues.append({
                "type": "interruptions",
                "severity": "high",
                "description": f"Agent interrupted customer {interruptions.get('agent_interruptions', 0)} times"
            })
        elif interruptions.get("agent_interruptions", 0) >= 3:
            issues.append({
                "type": "interruptions",
                "severity": "medium",
                "description": f"Agent interrupted customer {interruptions.get('agent_interruptions', 0)} times"
            })
        
        response = behavioral_data.get("response_time_analysis", {})
        if response.get("agent_response_assessment") == "too_slow":
            issues.append({
                "type": "response_time",
                "severity": "medium",
                "description": f"Agent responds too slowly ({response.get('avg_agent_response_time', 0)}s)"
            })
        
        listening = behavioral_data.get("active_listening", {})
        if listening.get("listening_assessment") == "needs_improvement":
            issues.append({
                "type": "active_listening",
                "severity": "high",
                "description": f"Poor active listening (only {listening.get('acknowledgment_count', 0)} acknowledgments)"
            })
        
        silence = behavioral_data.get("silence_analysis", {})
        if silence.get("silence_percentage", 0) > 30:
            issues.append({
                "type": "silences",
                "severity": "medium",
                "description": f"High silence percentage ({silence.get('silence_percentage', 0)}%)"
            })
        
        questions = behavioral_data.get("question_analysis", {})
        if questions.get("question_pattern") == "needs_improvement":
            issues.append({
                "type": "questions",
                "severity": "low",
                "description": f"Suboptimal questioning pattern ({questions.get('agent_questions', 0)} questions)"
            })
        
        return sorted(issues, key=lambda x: {"high": 0, "medium": 1, "low": 2}.get(x["severity"], 3))
    
    def _identify_strengths(self, behavioral_data: Dict) -> List[Dict]:
        strengths = []
        
        wpm = behavioral_data.get("words_per_minute", {})
        if wpm.get("agent_pacing") == "optimal":
            strengths.append({
                "type": "pacing",
                "description": f"Optimal speaking pace at {wpm.get('agent_wpm', 0)} WPM"
            })
        
        if wpm.get("talk_ratio", 0) <= 1.5:
            strengths.append({
                "type": "talk_ratio",
                "description": f"Balanced conversation (talk ratio: {wpm.get('talk_ratio', 0)}:1)"
            })
        
        interruptions = behavioral_data.get("interruption_analysis", {})
        if interruptions.get("agent_interruptions", 0) == 0:
            strengths.append({
                "type": "interruptions",
                "description": "No interruptions - agent let customer speak"
            })
        
        response = behavioral_data.get("response_time_analysis", {})
        if response.get("agent_response_assessment") == "optimal":
            strengths.append({
                "type": "response_time",
                "description": f"Optimal response timing ({response.get('avg_agent_response_time', 0)}s)"
            })
        
        listening = behavioral_data.get("active_listening", {})
        if listening.get("listening_assessment") == "excellent":
            strengths.append({
                "type": "active_listening",
                "description": f"Excellent active listening ({listening.get('acknowledgment_count', 0)} acknowledgments)"
            })
        elif listening.get("listening_assessment") == "good":
            strengths.append({
                "type": "active_listening",
                "description": f"Good active listening ({listening.get('acknowledgment_count', 0)} acknowledgments)"
            })
        
        questions = behavioral_data.get("question_analysis", {})
        if questions.get("question_pattern") == "good":
            strengths.append({
                "type": "questions",
                "description": f"Good questioning pattern ({questions.get('agent_questions', 0)} questions)"
            })
        
        return strengths
    
    def _empty_summary(self) -> Dict:
        return {
            "summary": "No behavioral data available.",
            "overall_score": 0.0,
            "overall_assessment": "unknown",
            "pacing_analysis": {"status": "unknown", "description": "No data"},
            "silence_analysis": {"status": "unknown", "description": "No data"},
            "interruption_analysis": {"status": "unknown", "description": "No data"},
            "response_analysis": {"status": "unknown", "description": "No data"},
            "question_analysis": {"status": "unknown", "description": "No data"},
            "listening_analysis": {"status": "unknown", "description": "No data"},
            "key_issues": [],
            "key_strengths": []
        }


def summarize_behavioral_call(behavioral_results):
    analyzer = BehavioralSummaryAnalyzer()
    return analyzer.analyze_call(behavioral_results)
