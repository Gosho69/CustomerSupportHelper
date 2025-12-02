import statistics
from datetime import timedelta
from typing import Dict, List
from django.utils import timezone

from calls.models import Call
from users.models import MyUser


class PerformanceCalculator:
    
    def __init__(self, agent: MyUser, start_date, end_date, report_type='weekly'):
        self.agent = agent
        self.start_date = start_date
        self.end_date = end_date
        self.report_type = report_type
        self.calls = Call.objects.filter(
            agent=agent,
            call_date__gte=start_date,
            call_date__lte=end_date
        )
        self.weekly_reports = None
        
        if report_type == 'monthly':
            from .models import PerformanceReport
            self.weekly_reports = PerformanceReport.objects.filter(
                agent=agent,
                report_type='weekly',
                start_date__gte=start_date,
                end_date__lte=end_date
            ).order_by('start_date')
    
    def calculate_metrics(self) -> Dict:
        if not self.calls.exists():
            return self._empty_report()
        
        metrics = {
            'call_volume': self._calculate_call_volume(),
            'emotional_metrics': self._calculate_emotional_metrics(),
            'behavioral_metrics': self._calculate_behavioral_metrics(),
            'topic_metrics': self._calculate_topic_metrics(),
            'consistency': self._calculate_consistency(),
            'comparison': self._calculate_team_comparison(),
            'assessment': self._generate_assessment(),
        }
        
        if self.report_type == 'monthly' and self.weekly_reports and self.weekly_reports.exists():
            metrics = self._enhance_with_weekly_data(metrics)
        
        return metrics
    
    def _calculate_call_volume(self) -> Dict:
        total = self.calls.count()
        durations = [c.duration for c in self.calls if c.duration]
        avg_duration = statistics.mean(durations) if durations else None
        
        return {
            'total_calls': total,
            'average_duration': avg_duration,
        }
    
    def _calculate_emotional_metrics(self) -> Dict:
        empathy_scores = []
        frustration_scores = []
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        
        for call in self.calls:
            if call.emotional_summary:
                empathy = call.emotional_summary.get('agent_empathy_score', 0)
                frustration = call.emotional_summary.get('customer_frustration_level', 0)
                empathy_scores.append(empathy)
                frustration_scores.append(frustration)
                
                call_tone = call.emotional_summary.get('call_tone', 'neutral')
                if call_tone == 'positive':
                    positive_count += 1
                elif call_tone == 'negative':
                    negative_count += 1
                else:
                    neutral_count += 1
        
        total = len(empathy_scores)
        emotional_scores = [(emp - frust) for emp, frust in zip(empathy_scores, frustration_scores)]
        avg_score = statistics.mean(emotional_scores) if emotional_scores else None
        positive_pct = (positive_count / total * 100) if total > 0 else 0
        negative_pct = (negative_count / total * 100) if total > 0 else 0
        
        trend = self._calculate_trend(emotional_scores)
        
        return {
            'average_score': avg_score,
            'positive_percentage': positive_pct,
            'negative_percentage': negative_pct,
            'trend': trend,
        }
    
    def _calculate_behavioral_metrics(self) -> Dict:
        overall_scores = []
        
        for call in self.calls:
            if call.behavioral_analysis:
                score = call.behavioral_analysis.get('behavioral_score', 0)
                normalized_score = score / 100.0
                overall_scores.append(normalized_score)
        
        trend = self._calculate_trend(overall_scores)
        
        avg_score = statistics.mean(overall_scores) if overall_scores else None
        
        return {
            'average_score': avg_score,
            'empathy': avg_score,
            'professionalism': avg_score,
            'problem_solving': avg_score,
            'trend': trend,
        }
    
    def _calculate_topic_metrics(self) -> Dict:
        topic_counts = {}
        
        for call in self.calls:
            if call.topic_analysis:
                primary = call.topic_analysis.get('primary_topic')
                if primary:
                    topic_counts[primary] = topic_counts.get(primary, 0) + 1
        
        sorted_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)
        most_common = dict(sorted_topics[:5])
        
        return {
            'most_common_topics': most_common,
        }
    
    def _calculate_consistency(self) -> Dict:
        behavioral_scores = []
        
        for call in self.calls:
            if call.behavioral_analysis:
                score = call.behavioral_analysis.get('behavioral_score', 0)
                normalized_score = score / 100.0
                behavioral_scores.append(normalized_score)
        
        if len(behavioral_scores) < 2:
            return {'consistency_score': None, 'variance': None}
        
        variance = statistics.stdev(behavioral_scores)
        
        consistency = max(0, 100 - (variance * 100))
        
        return {
            'consistency_score': round(consistency, 2),
            'variance': round(variance, 4),
        }
    
    def _calculate_team_comparison(self) -> Dict:
        agent_scores = []
        for call in self.calls:
            if call.behavioral_analysis:
                score = call.behavioral_analysis.get('behavioral_score', 0)
                normalized_score = score / 100.0
                agent_scores.append(normalized_score)
        
        agent_avg = statistics.mean(agent_scores) if agent_scores else 0
        
        if not self.agent.company:
            return {'percentile': None}
        
        team_agents = MyUser.objects.filter(
            company=self.agent.company,
            role='agent'
        ).exclude(id=self.agent.id)
        
        team_scores = []
        for teammate in team_agents:
            teammate_calls = Call.objects.filter(
                agent=teammate,
                call_date__gte=self.start_date,
                call_date__lte=self.end_date
            )
            for call in teammate_calls:
                if call.behavioral_analysis:
                    score = call.behavioral_analysis.get('behavioral_score', 0)
                    normalized_score = score / 100.0
                    team_scores.append(normalized_score)
        
        if not team_scores:
            return {'percentile': None}
        
        below_agent = sum(1 for score in team_scores if score < agent_avg)
        percentile = (below_agent / len(team_scores) * 100) if team_scores else 50
        
        return {
            'percentile': round(percentile, 2),
        }
    
    def _generate_assessment(self) -> Dict:
        behavioral = self._calculate_behavioral_metrics()
        emotional = self._calculate_emotional_metrics()
        consistency = self._calculate_consistency()
        
        strengths = []
        weaknesses = []
        recommendations = []
        
        if behavioral['empathy'] and behavioral['empathy'] > 0.7:
            strengths.append("High empathy - connects well with customers")
        elif behavioral['empathy'] and behavioral['empathy'] < 0.5:
            weaknesses.append("Low empathy - needs to show more understanding")
            recommendations.append("Practice active listening and acknowledgment phrases")
        
        if behavioral['professionalism'] and behavioral['professionalism'] > 0.7:
            strengths.append("Maintains professional demeanor")
        elif behavioral['professionalism'] and behavioral['professionalism'] < 0.5:
            weaknesses.append("Professionalism needs improvement")
            recommendations.append("Review company communication guidelines")
        
        if behavioral['problem_solving'] and behavioral['problem_solving'] > 0.7:
            strengths.append("Strong problem-solving skills")
        elif behavioral['problem_solving'] and behavioral['problem_solving'] < 0.5:
            weaknesses.append("Problem-solving could be improved")
            recommendations.append("Study common issue resolution procedures")
        
        if emotional['positive_percentage'] > 60:
            strengths.append("Creates positive customer experiences")
        elif emotional['negative_percentage'] > 40:
            weaknesses.append("High rate of negative customer sentiment")
            recommendations.append("Focus on de-escalation techniques")
        
        if consistency['consistency_score'] and consistency['consistency_score'] > 80:
            strengths.append("Consistent performance across calls")
        elif consistency['consistency_score'] and consistency['consistency_score'] < 60:
            weaknesses.append("Inconsistent performance - varies significantly between calls")
            recommendations.append("Work on maintaining steady performance standards")
        
        avg_behavioral = behavioral['average_score'] or 0
        if avg_behavioral >= 0.8:
            rating = "excellent"
            summary = "Outstanding performance - consistently exceeds expectations"
        elif avg_behavioral >= 0.65:
            rating = "good"
            summary = "Solid performance - meets expectations with room for growth"
        elif avg_behavioral >= 0.5:
            rating = "needs_improvement"
            summary = "Performance below expectations - requires focused development"
        else:
            rating = "poor"
            summary = "Significant improvement needed - immediate coaching recommended"
        
        return {
            'strengths': strengths,
            'weaknesses': weaknesses,
            'recommendations': recommendations,
            'overall_rating': rating,
            'summary': summary,
        }
    
    def _calculate_trend(self, scores: List[float]) -> str:
        if len(scores) < 3:
            return "stable"
        
        mid = len(scores) // 2
        first_half_avg = statistics.mean(scores[:mid])
        second_half_avg = statistics.mean(scores[mid:])
        
        diff = second_half_avg - first_half_avg
        
        if diff > 0.1:
            return "improving"
        elif diff < -0.1:
            return "declining"
        else:
            return "stable"
    
    def _enhance_with_weekly_data(self, metrics: Dict) -> Dict:
        """
        Enhance monthly metrics with insights from weekly reports.
        This provides week-over-week trends and pattern analysis.
        """
        weekly_behavioral_scores = []
        weekly_emotional_scores = []
        weekly_consistency_scores = []
        
        for week_report in self.weekly_reports:
            if week_report.average_behavioral_score:
                weekly_behavioral_scores.append(week_report.average_behavioral_score)
            if week_report.average_emotional_score:
                weekly_emotional_scores.append(week_report.average_emotional_score)
            if week_report.performance_consistency_score:
                weekly_consistency_scores.append(week_report.performance_consistency_score)
        
        # Calculate week-over-week trend
        if len(weekly_behavioral_scores) >= 2:
            week_trend = self._calculate_trend(weekly_behavioral_scores)
            metrics['behavioral_metrics']['week_over_week_trend'] = week_trend
            
            # Calculate improvement rate
            first_week = weekly_behavioral_scores[0]
            last_week = weekly_behavioral_scores[-1]
            improvement_rate = ((last_week - first_week) / first_week * 100) if first_week > 0 else 0
            metrics['behavioral_metrics']['improvement_rate'] = round(improvement_rate, 2)
        
        # Identify best and worst weeks
        if weekly_behavioral_scores:
            best_week_idx = weekly_behavioral_scores.index(max(weekly_behavioral_scores))
            worst_week_idx = weekly_behavioral_scores.index(min(weekly_behavioral_scores))
            
            best_week_report = list(self.weekly_reports)[best_week_idx]
            worst_week_report = list(self.weekly_reports)[worst_week_idx]
            
            metrics['weekly_analysis'] = {
                'best_week': {
                    'start_date': str(best_week_report.start_date),
                    'end_date': str(best_week_report.end_date),
                    'behavioral_score': best_week_report.average_behavioral_score,
                    'total_calls': best_week_report.total_calls
                },
                'worst_week': {
                    'start_date': str(worst_week_report.start_date),
                    'end_date': str(worst_week_report.end_date),
                    'behavioral_score': worst_week_report.average_behavioral_score,
                    'total_calls': worst_week_report.total_calls
                },
                'total_weeks_analyzed': self.weekly_reports.count()
            }
        
        if len(weekly_consistency_scores) >= 2:
            avg_weekly_consistency = statistics.mean(weekly_consistency_scores)
            metrics['consistency']['average_weekly_consistency'] = round(avg_weekly_consistency, 2)
        
        if 'weekly_analysis' in metrics:
            best_week_score = metrics['weekly_analysis']['best_week']['behavioral_score']
            worst_week_score = metrics['weekly_analysis']['worst_week']['behavioral_score']
            score_variance = best_week_score - worst_week_score
            
            if score_variance > 0.2:
                metrics['assessment']['weaknesses'].append(
                    f"Significant performance variation between weeks (range: {worst_week_score:.2f} - {best_week_score:.2f})"
                )
                metrics['assessment']['recommendations'].append(
                    "Focus on maintaining consistent performance standards throughout the month"
                )
            elif score_variance < 0.1:
                metrics['assessment']['strengths'].append(
                    "Maintains consistent performance across all weeks of the month"
                )
        
        return metrics
    
    def _empty_report(self) -> Dict:
        return {
            'call_volume': {'total_calls': 0, 'average_duration': None},
            'emotional_metrics': {
                'average_score': None,
                'positive_percentage': 0,
                'negative_percentage': 0,
                'trend': 'stable'
            },
            'behavioral_metrics': {
                'average_score': None,
                'empathy': None,
                'professionalism': None,
                'problem_solving': None,
                'trend': 'stable'
            },
            'topic_metrics': {'most_common_topics': {}},
            'consistency': {'consistency_score': None, 'variance': None},
            'comparison': {'percentile': None},
            'assessment': {
                'strengths': [],
                'weaknesses': [],
                'recommendations': ['No calls in this period'],
                'overall_rating': 'no_data',
                'summary': 'No calls recorded in this period'
            },
        }
