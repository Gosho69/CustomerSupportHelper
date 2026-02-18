"""
Evidence extraction utilities for performance reports.
Extracts call-level and weekly-level evidence from raw data.
"""
from typing import Dict, Optional

from calls.models import Call


def extract_call_evidence(calls) -> Dict:
    """Extract evidence from individual calls for AI report generation."""
    evidence = {
        'total_calls': calls.count(),
        'calls_analyzed': [],
        'common_patterns': [],
        'issues_found': [],
        'positive_moments': []
    }

    for call in calls[:10]:
        call_data = {
            'call_id': call.id,
            'date': str(call.call_date.date()) if call.call_date else 'N/A',
            'behavioral_score': call.behavioral_analysis.get('behavioral_score', 0) if call.behavioral_analysis else 0,
            'emotional_outcome': None,
            'topics': [],
            'coaching_feedback': None
        }

        if call.emotional_summary:
            call_data['emotional_outcome'] = {
                'resolution': call.emotional_summary.get('resolution_status'),
                'satisfaction': call.emotional_summary.get('customer_satisfaction'),
                'agent_empathy': call.emotional_summary.get('agent_empathy_score'),
                'customer_frustration': call.emotional_summary.get('customer_frustration_level')
            }

            if call.emotional_summary.get('resolution_status') == 'resolved':
                evidence['positive_moments'].append(f"Call {call.id}: Issue resolved successfully")

            if call.emotional_summary.get('customer_frustration_level', 0) > 0.5:
                evidence['issues_found'].append(
                    f"Call {call.id}: High customer frustration ({call.emotional_summary.get('customer_frustration_level'):.2f})"
                )

        if call.topic_analysis:
            primary = call.topic_analysis.get('primary_topic')
            if primary:
                call_data['topics'].append(primary)

        if call.coaching_tips:
            if isinstance(call.coaching_tips, dict):
                call_data['coaching_feedback'] = call.coaching_tips.get('message', '')
                quality = call.coaching_tips.get('quality_status')
                if quality and quality != 'excellent':
                    evidence['issues_found'].append(f"Call {call.id}: Quality status '{quality}'")

        evidence['calls_analyzed'].append(call_data)

    return evidence


def extract_weekly_evidence(weekly_reports) -> Optional[Dict]:
    """Extract evidence from weekly reports for monthly report generation."""
    if not weekly_reports or not weekly_reports.exists():
        return None

    evidence = {
        'weeks_analyzed': weekly_reports.count(),
        'weekly_performance': [],
        'trends_observed': []
    }

    weekly_scores = []
    for week in weekly_reports:
        week_data = {
            'period': f"{week.start_date} to {week.end_date}",
            'behavioral_score': week.average_behavioral_score,
            'total_calls': week.total_calls,
            'consistency': week.performance_consistency_score,
            'strengths_count': len(week.strengths) if week.strengths else 0,
            'weaknesses_count': len(week.weaknesses) if week.weaknesses else 0
        }
        evidence['weekly_performance'].append(week_data)
        if week.average_behavioral_score:
            weekly_scores.append(week.average_behavioral_score)

    if len(weekly_scores) >= 2:
        if weekly_scores[-1] > weekly_scores[0]:
            evidence['trends_observed'].append(
                f"Performance improved from {weekly_scores[0]:.2f} to {weekly_scores[-1]:.2f}"
            )
        elif weekly_scores[-1] < weekly_scores[0]:
            evidence['trends_observed'].append(
                f"Performance declined from {weekly_scores[0]:.2f} to {weekly_scores[-1]:.2f}"
            )
        else:
            evidence['trends_observed'].append(
                f"Consistent performance maintained at {weekly_scores[0]:.2f}"
            )

    return evidence
