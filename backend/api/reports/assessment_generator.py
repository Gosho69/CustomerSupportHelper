"""
Assessment generation for performance reports.
Handles both AI-powered and rule-based assessment generation.
"""
import sys
import traceback
from typing import Dict, Optional

from .evidence_extractor import extract_call_evidence, extract_weekly_evidence


def generate_assessment(
    calls,
    agent,
    report_type: str,
    start_date,
    end_date,
    behavioral_metrics: Dict,
    emotional_metrics: Dict,
    consistency_metrics: Dict,
    all_metrics: Dict,
    ai_generator,
    use_ai: bool,
    weekly_reports=None,
) -> Dict:
    """
    Generate a performance assessment using AI or rule-based fallback.
    
    Returns a dict with keys: strengths, weaknesses, recommendations,
    overall_rating, summary, executive_summary, ai_generated.
    """
    avg_behavioral = behavioral_metrics.get('average_score') or 0

    if avg_behavioral >= 0.8:
        rating = "excellent"
        base_summary = "Outstanding performance - consistently exceeds expectations"
    elif avg_behavioral >= 0.65:
        rating = "good"
        base_summary = "Solid performance - meets expectations with room for growth"
    elif avg_behavioral >= 0.5:
        rating = "needs_improvement"
        base_summary = "Performance below expectations - requires focused development"
    else:
        rating = "poor"
        base_summary = "Significant improvement needed - immediate coaching recommended"

    # Try AI-powered generation first
    if use_ai and ai_generator and all_metrics:
        result = _generate_ai_assessment(
            calls=calls,
            agent=agent,
            report_type=report_type,
            start_date=start_date,
            end_date=end_date,
            all_metrics=all_metrics,
            rating=rating,
            ai_generator=ai_generator,
            weekly_reports=weekly_reports,
        )
        if result:
            return result

    # Rule-based fallback
    return _generate_rule_based_assessment(
        behavioral_metrics=behavioral_metrics,
        emotional_metrics=emotional_metrics,
        consistency_metrics=consistency_metrics,
        rating=rating,
        base_summary=base_summary,
    )


def _generate_ai_assessment(
    calls,
    agent,
    report_type: str,
    start_date,
    end_date,
    all_metrics: Dict,
    rating: str,
    ai_generator,
    weekly_reports=None,
) -> Optional[Dict]:
    """Attempt AI-powered assessment generation. Returns None on failure."""
    try:
        agent_name = f"{agent.first_name} {agent.last_name}".strip() or agent.username

        call_evidence = extract_call_evidence(calls)
        weekly_evidence = (
            extract_weekly_evidence(weekly_reports)
            if report_type == 'monthly' and weekly_reports
            else None
        )

        metrics_with_rating = all_metrics.copy()
        metrics_with_rating['assessment'] = {'overall_rating': rating}
        metrics_with_rating['call_evidence'] = call_evidence
        if weekly_evidence:
            metrics_with_rating['weekly_evidence'] = weekly_evidence

        strengths = ai_generator.generate_detailed_strengths(metrics_with_rating, agent_name)
        weaknesses = ai_generator.generate_detailed_weaknesses(metrics_with_rating, agent_name)
        recommendations = ai_generator.generate_actionable_recommendations(
            metrics_with_rating, strengths, weaknesses, agent_name
        )
        summary = ai_generator.generate_comprehensive_summary(
            metrics_with_rating,
            agent_name,
            report_type,
            str(start_date),
            str(end_date),
        )
        executive_summary = ai_generator.generate_executive_summary(
            metrics_with_rating, agent_name, report_type
        )

        return {
            'strengths': strengths,
            'weaknesses': weaknesses,
            'recommendations': recommendations,
            'overall_rating': rating,
            'summary': summary,
            'executive_summary': executive_summary,
            'ai_generated': True,
        }

    except Exception as e:
        print(f"[ERROR] AI generation failed: {str(e)}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return None


def _generate_rule_based_assessment(
    behavioral_metrics: Dict,
    emotional_metrics: Dict,
    consistency_metrics: Dict,
    rating: str,
    base_summary: str,
) -> Dict:
    """Generate assessment using simple rule-based logic."""
    strengths = []
    weaknesses = []
    recommendations = []

    empathy = behavioral_metrics.get('empathy')
    professionalism = behavioral_metrics.get('professionalism')
    problem_solving = behavioral_metrics.get('problem_solving')

    if empathy and empathy > 0.7:
        strengths.append("High empathy - connects well with customers")
    elif empathy and empathy < 0.5:
        weaknesses.append("Low empathy - needs to show more understanding")
        recommendations.append("Practice active listening and acknowledgment phrases")

    if professionalism and professionalism > 0.7:
        strengths.append("Maintains professional demeanor")
    elif professionalism and professionalism < 0.5:
        weaknesses.append("Professionalism needs improvement")
        recommendations.append("Review company communication guidelines")

    if problem_solving and problem_solving > 0.7:
        strengths.append("Strong problem-solving skills")
    elif problem_solving and problem_solving < 0.5:
        weaknesses.append("Problem-solving could be improved")
        recommendations.append("Study common issue resolution procedures")

    if emotional_metrics.get('positive_percentage', 0) > 60:
        strengths.append("Creates positive customer experiences")
    elif emotional_metrics.get('negative_percentage', 0) > 40:
        weaknesses.append("High rate of negative customer sentiment")
        recommendations.append("Focus on de-escalation techniques")

    consistency_score = consistency_metrics.get('consistency_score')
    if consistency_score and consistency_score > 80:
        strengths.append("Consistent performance across calls")
    elif consistency_score and consistency_score < 60:
        weaknesses.append("Inconsistent performance - varies significantly between calls")
        recommendations.append("Work on maintaining steady performance standards")

    return {
        'strengths': strengths,
        'weaknesses': weaknesses,
        'recommendations': recommendations,
        'overall_rating': rating,
        'summary': base_summary,
        'executive_summary': base_summary,
        'ai_generated': False,
    }
