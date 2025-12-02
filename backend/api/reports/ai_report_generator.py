import json
import requests
from typing import Dict, List
import os


class AIReportGenerator:
    
    def __init__(self, ollama_url=os.getenv("OLLAMA_URL", "http://localhost:11434")):
        self.ollama_url = ollama_url
        self.model = "llama3.2"
        self.temperature = 0.7
        self.max_tokens = 2000
        
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code != 200:
                raise ValueError(f"Ollama not available at {self.ollama_url}")
        except Exception as e:
            raise ValueError(f"Cannot connect to Ollama: {e}")
    
    def generate_comprehensive_summary(
        self, 
        metrics: Dict,
        agent_name: str,
        report_type: str,
        start_date: str,
        end_date: str
    ) -> str:
        
        prompt = self._build_summary_prompt(metrics, agent_name, report_type, start_date, end_date)
        
        system_prompt = """You are an expert performance analyst specializing in customer service evaluation. 
Generate detailed, professional, and actionable performance reports. 
Use clear language, specific metrics, and constructive feedback. 
Structure your response with clear sections and bullet points where appropriate.
Be encouraging but honest about areas needing improvement."""
        
        full_prompt = f"{system_prompt}\n\n{prompt}"
        
        try:
            response = self._call_ollama(full_prompt)
            return response.strip()
        
        except Exception as e:
            return self._generate_fallback_summary(metrics, agent_name, report_type)
    
    def generate_detailed_strengths(self, metrics: Dict, agent_name: str) -> List[str]:
        behavioral_score = f"{metrics['behavioral_metrics']['average_score']:.2f}" if metrics['behavioral_metrics'].get('average_score') is not None else 'N/A'
        emotional_score = f"{metrics['emotional_metrics']['average_score']:.2f}" if metrics['emotional_metrics'].get('average_score') is not None else 'N/A'
        
        # Format call evidence
        call_evidence = metrics.get('call_evidence', {})
        evidence_summary = ""
        if call_evidence:
            evidence_summary = f"\n\nCall Evidence ({call_evidence.get('total_calls', 0)} calls analyzed):"
            if call_evidence.get('positive_moments'):
                evidence_summary += f"\n- Positive outcomes: {'; '.join(call_evidence['positive_moments'][:3])}"
            if call_evidence.get('calls_analyzed'):
                resolved_calls = sum(1 for c in call_evidence['calls_analyzed'] if c.get('emotional_outcome', {}).get('resolution') == 'resolved')
                if resolved_calls > 0:
                    evidence_summary += f"\n- {resolved_calls} calls successfully resolved"
        
        prompt = f"""Based on the following performance metrics for {agent_name}, 
        identify and describe their key strengths with SPECIFIC evidence from the metrics and actual calls. 
        Every strength must directly reference a metric value or call evidence.
        
        Metrics:
        - Behavioral Score: {behavioral_score} (0.0-1.0, 0.8+ is excellent)
        - Emotional Score: {emotional_score}
        - Positive Call Rate: {metrics['emotional_metrics']['positive_percentage']:.1f}%
        - Consistency Score: {metrics['consistency']['consistency_score'] or 'N/A'} (100.0 is perfect)
        - Team Percentile: {metrics['comparison']['percentile'] or 'N/A'}
        - Trend: {metrics['behavioral_metrics']['trend']}
        - Total Calls: {metrics['call_volume']['total_calls']}{evidence_summary}
        
        IMPORTANT: Each strength MUST cite specific metrics or call evidence. For example:
        - "Excellent behavioral performance with a score of 0.85, demonstrated across 15 calls including successful resolutions"
        - "Perfect consistency at 100.0, showing reliable performance"
        
        Provide 3-5 detailed strength statements as a JSON array of strings. Be specific and evidence-based."""
        
        system_prompt = "You are a performance analyst. Respond only with a valid JSON array of strings."
        full_prompt = f"{system_prompt}\n\n{prompt}"
        
        try:
            content = self._call_ollama(full_prompt, max_tokens=500)
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            
            strengths = json.loads(content)
            return strengths if isinstance(strengths, list) else [content]
        
        except Exception as e:
            return self._generate_rule_based_strengths(metrics)
    
    def generate_detailed_weaknesses(self, metrics: Dict, agent_name: str) -> List[str]:
        # Pre-format values to avoid f-string format specifier issues
        behavioral_score = f"{metrics['behavioral_metrics']['average_score']:.2f}" if metrics['behavioral_metrics'].get('average_score') is not None else 'N/A'
        emotional_score = f"{metrics['emotional_metrics']['average_score']:.2f}" if metrics['emotional_metrics'].get('average_score') is not None else 'N/A'
        
        # Format call evidence
        call_evidence = metrics.get('call_evidence', {})
        evidence_summary = ""
        if call_evidence:
            evidence_summary = f"\n\nCall Evidence ({call_evidence.get('total_calls', 0)} calls analyzed):"
            if call_evidence.get('issues_found'):
                evidence_summary += f"\n- Issues identified: {'; '.join(call_evidence['issues_found'][:3])}"
            else:
                evidence_summary += "\n- No significant issues found in calls"
        
        prompt = f"""Based on the following performance metrics for {agent_name}, 
        identify ONLY areas for improvement that are CLEARLY indicated by the metrics or actual call evidence. 
        DO NOT make up generic suggestions like "team dynamics", "delegation", or "emotional intelligence" unless there is specific evidence in the metrics.
        
        Metrics:
        - Behavioral Score: {behavioral_score} (0.0-1.0 scale, 0.8+ is excellent)
        - Emotional Score: {emotional_score} (0.0-1.0 scale, higher is better)
        - Negative Call Rate: {metrics['emotional_metrics']['negative_percentage']:.1f}%
        - Consistency Score: {metrics['consistency']['consistency_score'] or 'N/A'} (100.0 is perfect)
        - Variance: {metrics['consistency']['variance'] or 'N/A'} (0.0 is perfect consistency)
        - Trend: {metrics['behavioral_metrics']['trend']}
        - Total Calls: {metrics['call_volume']['total_calls']}{evidence_summary}
        
        IMPORTANT RULES:
        - If Behavioral Score >= 0.8 AND Consistency >= 90 AND Negative Rate < 20% AND no issues found in calls, return an empty array [].
        - Only identify weaknesses that have clear evidence in the metrics or call evidence above.
        - If call evidence shows "no significant issues", do NOT invent problems.
        - DO NOT suggest things like team collaboration, delegation, or emotional intelligence unless explicitly shown in metrics.
        - If you must suggest improvements, cite the specific call ID or metric that proves the issue exists.
        
        Provide 0-2 specific areas for improvement as a JSON array of strings, or [] if performance is genuinely excellent."""
        
        system_prompt = "You are a performance analyst. Respond only with a valid JSON array of strings. Be constructive and supportive."
        full_prompt = f"{system_prompt}\n\n{prompt}"
        
        try:
            content = self._call_ollama(full_prompt, max_tokens=500)
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            
            weaknesses = json.loads(content)
            return weaknesses if isinstance(weaknesses, list) else [content]
        
        except Exception as e:
            return self._generate_rule_based_weaknesses(metrics)
    
    def generate_actionable_recommendations(
        self, 
        metrics: Dict, 
        strengths: List[str], 
        weaknesses: List[str],
        agent_name: str
    ) -> List[str]:
        
        # Format call evidence for context
        call_evidence = metrics.get('call_evidence', {})
        evidence_context = ""
        if call_evidence:
            evidence_context = f"\n\nCall Performance Evidence ({call_evidence.get('total_calls', 0)} calls):"
            if call_evidence.get('issues_found'):
                evidence_context += f"\n- Issues observed: {'; '.join(call_evidence['issues_found'][:3])}"
            if call_evidence.get('positive_moments'):
                evidence_context += f"\n- Positive outcomes: {'; '.join(call_evidence['positive_moments'][:3])}"
            if call_evidence.get('calls_analyzed'):
                high_scores = sum(1 for c in call_evidence['calls_analyzed'] if c.get('behavioral_score', 0) >= 85)
                low_scores = sum(1 for c in call_evidence['calls_analyzed'] if c.get('behavioral_score', 0) < 70)
                evidence_context += f"\n- High performing calls (85+): {high_scores}"
                if low_scores > 0:
                    evidence_context += f"\n- Calls needing improvement (<70): {low_scores}"
        
        weekly_context = ""
        weekly_evidence = metrics.get('weekly_evidence', {})
        if weekly_evidence:
            weekly_context = f"\n\nWeekly Performance Patterns ({weekly_evidence.get('weeks_analyzed', 0)} weeks):"
            if weekly_evidence.get('trends_observed'):
                weekly_context += f"\n- {'; '.join(weekly_evidence['trends_observed'])}"
        
        prompt = f"""Based on {agent_name}'s actual call performance and monthly activity:

Strengths:
{chr(10).join(f'- {s}' for s in strengths)}

Areas for Improvement:
{chr(10).join(f'- {w}' for w in weaknesses) if weaknesses else '- Performance is excellent overall'}

Key Metrics:
- Total Calls: {metrics['call_volume']['total_calls']}
- Behavioral Trend: {metrics['behavioral_metrics']['trend']}
- Consistency: {metrics['consistency']['consistency_score'] or 'N/A'}
- Most Common Topics: {list(metrics['topic_metrics']['most_common_topics'].keys())[:3] if metrics['topic_metrics']['most_common_topics'] else 'None'}{evidence_context}{weekly_context}

Generate 4-6 specific, actionable recommendations based on the actual call evidence and performance patterns above. 
Each recommendation should:
1. Reference specific calls, issues, or patterns from the evidence (e.g., "In calls where customer frustration was high...")
2. Be concrete and implementable with clear actions
3. Include suggested timeframes or frequency when relevant
4. If no significant issues exist, focus on maintaining excellence and expanding strengths to new topics/situations
5. Be supportive and motivating

IMPORTANT: Base recommendations on the actual call evidence and weekly patterns provided. Do NOT make generic suggestions.

Provide as a JSON array of strings."""
        
        system_prompt = "You are an expert performance coach. Provide specific, evidence-based, actionable recommendations. Respond only with a valid JSON array."
        full_prompt = f"{system_prompt}\n\n{prompt}"
        
        try:
            content = self._call_ollama(full_prompt, max_tokens=800)
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            
            recommendations = json.loads(content)
            return recommendations if isinstance(recommendations, list) else [content]
        
        except Exception as e:
            return self._generate_rule_based_recommendations(weaknesses)
    
    def generate_executive_summary(self, metrics: Dict, agent_name: str, report_type: str) -> str:
        # Pre-format values to avoid f-string format specifier issues
        behavioral_score = f"{metrics['behavioral_metrics']['average_score']:.2f}" if metrics['behavioral_metrics'].get('average_score') is not None else 'N/A'
        
        prompt = f"""Create a brief executive summary (2-3 sentences) for {agent_name}'s {report_type} performance report.

Key Metrics:
- Overall Rating: {metrics['assessment']['overall_rating']}
- Total Calls: {metrics['call_volume']['total_calls']}
- Behavioral Score: {behavioral_score}
- Performance Trend: {metrics['behavioral_metrics']['trend']}
- Team Percentile: {metrics['comparison']['percentile'] or 'N/A'}

Highlight the most important points concisely and professionally."""
        
        system_prompt = "You are a performance analyst. Create brief, impactful executive summaries."
        full_prompt = f"{system_prompt}\n\n{prompt}"
        
        try:
            return self._call_ollama(full_prompt, max_tokens=150).strip()
        
        except Exception as e:
            return metrics['assessment']['summary']
    
    def _build_summary_prompt(
        self, 
        metrics: Dict, 
        agent_name: str, 
        report_type: str,
        start_date: str,
        end_date: str
    ) -> str:
        
        # Format call evidence
        call_evidence = metrics.get('call_evidence', {})
        evidence_context = ""
        if call_evidence and call_evidence.get('calls_analyzed'):
            evidence_context = f"\n\nCall Evidence ({call_evidence.get('total_calls', 0)} calls analyzed):"
            if call_evidence.get('positive_moments'):
                evidence_context += f"\n- Positive outcomes: {len(call_evidence['positive_moments'])} instances"
            if call_evidence.get('issues_found'):
                evidence_context += f"\n- Issues identified: {len(call_evidence['issues_found'])} instances"
        
        weekly_analysis = ""
        weekly_evidence = metrics.get('weekly_evidence', {})
        if metrics.get('weekly_analysis'):
            wa = metrics['weekly_analysis']
            weekly_analysis = f"""

Weekly Analysis (for monthly report):
- Total Weeks Analyzed: {wa['total_weeks_analyzed']}
- Best Week: {wa['best_week']['start_date']} to {wa['best_week']['end_date']} (Score: {wa['best_week']['behavioral_score']:.2f}, Calls: {wa['best_week']['total_calls']})
- Worst Week: {wa['worst_week']['start_date']} to {wa['worst_week']['end_date']} (Score: {wa['worst_week']['behavioral_score']:.2f}, Calls: {wa['worst_week']['total_calls']})
"""
            if weekly_evidence and weekly_evidence.get('trends_observed'):
                weekly_analysis += f"\n- Trends: {'; '.join(weekly_evidence['trends_observed'])}"
        
        emotional_score = f"{metrics['emotional_metrics']['average_score']:.2f}" if metrics['emotional_metrics'].get('average_score') is not None else 'N/A'
        behavioral_score = f"{metrics['behavioral_metrics']['average_score']:.2f}" if metrics['behavioral_metrics'].get('average_score') is not None else 'N/A'
        avg_duration = f"{metrics['call_volume']['average_duration']:.1f}" if metrics['call_volume']['average_duration'] else 'N/A'
        
        return f"""Generate a comprehensive performance report for {agent_name} covering the {report_type} period from {start_date} to {end_date}.

Performance Metrics:

Call Volume:
- Total Calls: {metrics['call_volume']['total_calls']}
- Average Call Duration: {avg_duration} seconds

Emotional Metrics:
- Average Emotional Score: {emotional_score}
- Positive Calls: {metrics['emotional_metrics']['positive_percentage']:.1f}%
- Negative Calls: {metrics['emotional_metrics']['negative_percentage']:.1f}%
- Emotional Trend: {metrics['emotional_metrics']['trend']}

Behavioral Metrics:
- Average Behavioral Score: {behavioral_score}
- Behavioral Trend: {metrics['behavioral_metrics']['trend']}

Consistency:
- Consistency Score: {metrics['consistency']['consistency_score'] or 'N/A'}
- Performance Variance: {metrics['consistency']['variance'] or 'N/A'}

Team Comparison:
- Percentile Ranking: {metrics['comparison']['percentile'] or 'N/A'}

Topic Analysis:
- Most Common Topics: {', '.join(f"{k} ({v} calls)" for k, v in list(metrics['topic_metrics']['most_common_topics'].items())[:5]) if metrics['topic_metrics']['most_common_topics'] else 'No topics recorded'}
{weekly_analysis}{evidence_context}

IMPORTANT: Base all observations on the metrics and call evidence provided above. Do not make generic statements without evidence.

Please provide:
1. An opening paragraph summarizing overall performance based on the actual metrics
2. Detailed analysis of key strengths with specific examples from the metrics and call evidence
3. Areas for improvement ONLY if supported by metrics or call issues (if call evidence shows no issues, acknowledge excellent performance)
4. Pattern analysis (trends, consistency, topic expertise) using the actual data provided
5. Comparison to team performance (if percentile available)
6. A motivating conclusion with forward-looking perspective

Use professional language, cite specific numbers and call evidence, and structure the report clearly with headers and sections."""
    
    def _call_ollama(self, prompt: str, max_tokens: int = None) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": max_tokens or self.max_tokens
            }
        }
        
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=120
            )
            
            if response.status_code != 200:
                raise Exception(f"Ollama API error: {response.status_code}")
            
            result = response.json()
            return result.get("response", "")
        
        except Exception as e:
            raise Exception(f"Error calling Ollama: {e}")
    
    def _generate_fallback_summary(self, metrics: Dict, agent_name: str, report_type: str) -> str:
        
        total_calls = metrics['call_volume']['total_calls']
        rating = metrics['assessment']['overall_rating']
        trend = metrics['behavioral_metrics']['trend']
        
        return f"""Performance Report for {agent_name}

This {report_type} report covers {total_calls} calls with an overall rating of {rating}.

Performance is currently {trend}. {"The agent demonstrates strong capabilities in customer service." if rating in ['excellent', 'good'] else "There are opportunities for performance improvement."}

Key metrics and detailed analysis are available in the structured data sections below."""
    
    def _generate_rule_based_strengths(self, metrics: Dict) -> List[str]:
        strengths = []
        
        if metrics['behavioral_metrics']['average_score'] and metrics['behavioral_metrics']['average_score'] > 0.7:
            strengths.append("Demonstrates strong overall behavioral performance in customer interactions")
        
        if metrics['emotional_metrics']['positive_percentage'] > 60:
            strengths.append(f"Achieves high positive customer sentiment rate of {metrics['emotional_metrics']['positive_percentage']:.1f}%")
        
        if metrics['consistency']['consistency_score'] and metrics['consistency']['consistency_score'] > 80:
            strengths.append("Maintains consistent performance quality across all customer interactions")
        
        if metrics['behavioral_metrics']['trend'] == 'improving':
            strengths.append("Shows positive performance trajectory with improving trend")
        
        return strengths if strengths else ["Completed all assigned customer interactions"]
    
    def _generate_rule_based_weaknesses(self, metrics: Dict) -> List[str]:
        weaknesses = []
        
        if metrics['behavioral_metrics']['average_score'] and metrics['behavioral_metrics']['average_score'] < 0.5:
            weaknesses.append("Overall behavioral performance below target standards")
        
        if metrics['emotional_metrics']['negative_percentage'] > 40:
            weaknesses.append(f"High rate of negative customer sentiment ({metrics['emotional_metrics']['negative_percentage']:.1f}%)")
        
        if metrics['consistency']['consistency_score'] and metrics['consistency']['consistency_score'] < 60:
            weaknesses.append("Performance consistency varies significantly between calls")
        
        if metrics['behavioral_metrics']['trend'] == 'declining':
            weaknesses.append("Performance trend showing decline - requires attention")
        
        return weaknesses if weaknesses else ["Minor opportunities for performance optimization"]
    
    def _generate_rule_based_recommendations(self, weaknesses: List[str]) -> List[str]:
        recommendations = []
        
        for weakness in weaknesses:
            if 'behavioral' in weakness.lower():
                recommendations.append("Review customer service best practices and company guidelines")
            if 'sentiment' in weakness.lower() or 'negative' in weakness.lower():
                recommendations.append("Practice de-escalation techniques and empathy building")
            if 'consistency' in weakness.lower() or 'varies' in weakness.lower():
                recommendations.append("Focus on maintaining steady performance standards across all interactions")
            if 'declining' in weakness.lower() or 'trend' in weakness.lower():
                recommendations.append("Schedule coaching session to identify and address performance barriers")
        
        if not recommendations:
            recommendations.append("Continue current performance practices and seek opportunities for growth")
        
        return recommendations
