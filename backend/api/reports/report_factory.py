"""
Single source of truth for creating a PerformanceReport from a metrics dict.

All callers (management commands, views, auto-generator) should go through
`create_report_from_metrics` so that adding a new field only requires one
change here rather than in every call site.
"""
from typing import Dict, Optional

from users.models import MyUser
from .models import PerformanceReport
from .performance_calculator import PerformanceCalculator


def calculate_and_create_report(
    agent: MyUser,
    report_type: str,
    start_date,
    end_date,
    generated_by: Optional[MyUser] = None,
) -> PerformanceReport:
    """
    Run PerformanceCalculator for the given agent / period and persist the
    resulting PerformanceReport.  Returns the saved report object.
    """
    calculator = PerformanceCalculator(agent, start_date, end_date, report_type=report_type)
    metrics = calculator.calculate_metrics()
    return create_report_from_metrics(
        agent=agent,
        report_type=report_type,
        start_date=start_date,
        end_date=end_date,
        metrics=metrics,
        generated_by=generated_by,
    )


def create_report_from_metrics(
    agent: MyUser,
    report_type: str,
    start_date,
    end_date,
    metrics: Dict,
    generated_by: Optional[MyUser] = None,
) -> PerformanceReport:
    """
    Persist a PerformanceReport from a pre-computed metrics dict.
    Use this when you have already called calculate_metrics() yourself.
    """
    assessment = metrics["assessment"]
    return PerformanceReport.objects.create(
        agent=agent,
        report_type=report_type,
        start_date=start_date,
        end_date=end_date,
        # Call volume
        total_calls=metrics["call_volume"]["total_calls"],
        average_call_duration=metrics["call_volume"]["average_duration"],
        # Emotional
        average_emotional_score=metrics["emotional_metrics"]["average_score"],
        positive_calls_percentage=metrics["emotional_metrics"]["positive_percentage"],
        negative_calls_percentage=metrics["emotional_metrics"]["negative_percentage"],
        emotional_trend=metrics["emotional_metrics"]["trend"],
        # Behavioural
        average_behavioral_score=metrics["behavioral_metrics"]["average_score"],
        empathy_score=metrics["behavioral_metrics"]["empathy"],
        professionalism_score=metrics["behavioral_metrics"]["professionalism"],
        problem_solving_score=metrics["behavioral_metrics"]["problem_solving"],
        behavioral_trend=metrics["behavioral_metrics"]["trend"],
        # Topics & consistency & comparison
        most_common_topics=metrics["topic_metrics"]["most_common_topics"],
        performance_consistency_score=metrics["consistency"]["consistency_score"],
        variance_from_average=metrics["consistency"]["variance"],
        percentile_score=metrics["comparison"]["percentile"],
        # Assessment
        strengths=assessment["strengths"],
        weaknesses=assessment["weaknesses"],
        recommendations=assessment["recommendations"],
        overall_rating=assessment["overall_rating"],
        summary=assessment["summary"],
        executive_summary=assessment.get("executive_summary", assessment["summary"]),
        ai_generated=assessment.get("ai_generated", False),
        # Monthly-only weekly breakdown
        weekly_analysis=metrics.get("weekly_analysis"),
        # Metadata
        generated_by=generated_by,
    )
