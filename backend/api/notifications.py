"""
notifications.py
────────────────
Transactional email sending via Resend.

Local dev  (RESEND_API_KEY is empty or 'test'):
    Emails are logged to the console — nothing is actually sent.

Production (RESEND_API_KEY starts with 're_'):
    Emails are dispatched through the Resend HTTP API.

Public API
──────────
    send_report_notification(report)
        → notifies the agent that their weekly/monthly report is ready.

    send_call_analyzed_notification(call)
        → notifies the agent that one of their calls has been fully analyzed.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from django.conf import settings

if TYPE_CHECKING:
    from calls.models import Call
    from reports.models import PerformanceReport

logger = logging.getLogger(__name__)


# ── Core send ─────────────────────────────────────────────────────────────────

def _send(*, to: str, subject: str, html: str) -> bool:
    """
    Dispatch a single email.  Returns True on success, False on any failure.
    Never raises — caller should not have to handle email errors.
    """
    api_key: str = getattr(settings, "RESEND_API_KEY", "")
    from_addr: str = getattr(settings, "RESEND_FROM_EMAIL", "noreply@agentsights.com")

    # Dev / test mode — just log it
    if not api_key or api_key.lower() == "test":
        logger.info(
            "\n"
            "╔══ [EMAIL — DEV MODE] ══════════════════════════════════╗\n"
            "  To:      %s\n"
            "  From:    %s\n"
            "  Subject: %s\n"
            "  (HTML body suppressed — set RESEND_API_KEY to send for real)\n"
            "╚════════════════════════════════════════════════════════╝",
            to, from_addr, subject,
        )
        return True

    try:
        import resend  # noqa: PLC0415
        resend.api_key = api_key
        resend.Emails.send({
            "from": from_addr,
            "to": [to],
            "subject": subject,
            "html": html,
        })
        logger.info("Email sent → %s | %s", to, subject)
        return True
    except Exception:
        logger.exception("Failed to send email to %s (subject: %s)", to, subject)
        return False


# ── Shared HTML primitives ────────────────────────────────────────────────────

def _score_color(score: float) -> str:
    if score >= 85:
        return "#0caf60"
    if score >= 70:
        return "#e68a00"
    return "#e25950"


def _rating_badge(rating: str | None) -> str:
    MAP = {
        "excellent":        ("Excellent",        "#d1fae5", "#065f46"),
        "good":             ("Good",             "#fef3c7", "#92400e"),
        "needs_improvement":("Needs Improvement","#fee2e2", "#991b1b"),
        "poor":             ("Poor",             "#fee2e2", "#991b1b"),
    }
    label, bg, color = MAP.get(rating or "", ("No Data", "#f3f4f6", "#6b7280"))
    return (
        f'<span style="display:inline-block;padding:3px 10px;border-radius:12px;'
        f'font-size:12px;font-weight:700;background:{bg};color:{color};">'
        f'{label}</span>'
    )


def _bullet_list(items: list[str], color: str = "#635bff") -> str:
    if not items:
        return ""
    lis = "".join(
        f'<li style="margin:4px 0;color:#374151;">'
        f'<span style="color:{color};margin-right:6px;">•</span>{item}</li>'
        for item in items[:4]
    )
    return f'<ul style="margin:8px 0;padding-left:0;list-style:none;">{lis}</ul>'


def _cta_button(url: str, label: str) -> str:
    return (
        f'<a href="{url}" style="display:inline-block;margin-top:20px;padding:12px 28px;'
        f'background:#635bff;color:#ffffff;font-weight:700;font-size:14px;'
        f'border-radius:8px;text-decoration:none;">{label} →</a>'
    )


def _email_wrapper(content: str) -> str:
    """Wrap content in a consistent email shell."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AgentSights</title></head>
<body style="margin:0;padding:0;background:#f9fafb;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;">
  <table width="100%" cellpadding="0" cellspacing="0" style="background:#f9fafb;padding:32px 16px;">
    <tr><td align="center">
      <table width="600" cellpadding="0" cellspacing="0" style="max-width:600px;width:100%;">

        <!-- Logo / brand bar -->
        <tr>
          <td style="background:#635bff;border-radius:12px 12px 0 0;padding:20px 32px;">
            <span style="color:#ffffff;font-size:20px;font-weight:800;letter-spacing:-0.5px;">
              AgentSights
            </span>
          </td>
        </tr>

        <!-- Body -->
        <tr>
          <td style="background:#ffffff;padding:32px;border-left:1px solid #e5e7eb;
                     border-right:1px solid #e5e7eb;">
            {content}
          </td>
        </tr>

        <!-- Footer -->
        <tr>
          <td style="background:#f3f4f6;border:1px solid #e5e7eb;border-top:none;
                     border-radius:0 0 12px 12px;padding:16px 32px;
                     font-size:12px;color:#9ca3af;text-align:center;">
            AgentSights · Automated notification · Do not reply to this email.
          </td>
        </tr>

      </table>
    </td></tr>
  </table>
</body>
</html>"""


# ── Report notification ───────────────────────────────────────────────────────

def _build_report_html(report: "PerformanceReport") -> str:
    frontend_url: str = getattr(settings, "FRONTEND_URL", "http://localhost:3000")
    reports_url = f"{frontend_url}/dashboard/my-reports"

    agent_name = (
        f"{report.agent.first_name} {report.agent.last_name}".strip()
        or report.agent.username
    )

    score = round((report.average_behavioral_score or 0) * 100)
    score_color = _score_color(score)

    period_type = report.report_type.capitalize()

    start = report.start_date.strftime("%-d %b %Y") if report.start_date else ""
    end = report.end_date.strftime("%-d %b %Y") if report.end_date else ""
    period_str = f"{start} – {end}" if start and end else start or end

    rating_html = _rating_badge(report.overall_rating)

    exec_summary = ""
    if report.executive_summary:
        exec_summary = (
            f'<div style="background:#f5f3ff;border-left:3px solid #635bff;'
            f'padding:12px 16px;border-radius:0 8px 8px 0;margin:20px 0;">'
            f'<p style="margin:0;font-size:14px;color:#374151;line-height:1.6;">'
            f'{report.executive_summary}</p></div>'
        )

    strengths_html = ""
    if report.strengths:
        strengths_html = (
            f'<p style="font-size:13px;font-weight:700;color:#065f46;margin:20px 0 6px;">'
            f'✓ Strengths</p>'
            + _bullet_list(list(report.strengths)[:3], "#0caf60")
        )

    improvements_html = ""
    if report.weaknesses:
        improvements_html = (
            f'<p style="font-size:13px;font-weight:700;color:#92400e;margin:16px 0 6px;">'
            f'→ Areas for Improvement</p>'
            + _bullet_list(list(report.weaknesses)[:3], "#e68a00")
        )

    consistency = ""
    if report.performance_consistency_score is not None:
        c = round(report.performance_consistency_score)
        consistency = (
            f'<span style="margin-left:16px;font-size:13px;color:#6b7280;">'
            f'Consistency <strong style="color:#374151;">{c}/100</strong></span>'
        )

    percentile = ""
    if report.percentile_score is not None:
        p = round(report.percentile_score)
        percentile = (
            f'<span style="margin-left:16px;font-size:13px;color:#6b7280;">'
            f'Team Percentile <strong style="color:#374151;">{p}th</strong></span>'
        )

    content = f"""
    <h2 style="margin:0 0 4px;font-size:22px;color:#111827;">
      Your {period_type} Report is Ready
    </h2>
    <p style="margin:0 0 20px;font-size:14px;color:#6b7280;">Hi {agent_name},</p>

    <!-- Score card -->
    <div style="background:#f9fafb;border:1px solid #e5e7eb;border-radius:10px;
                padding:20px 24px;margin-bottom:20px;">
      <div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;">
        <span style="font-size:48px;font-weight:800;color:{score_color};
                     line-height:1;">{score}</span>
        <div>
          <p style="margin:0 0 4px;font-size:12px;color:#6b7280;">
            Overall Performance Score
          </p>
          {rating_html}
        </div>
      </div>
      <p style="margin:10px 0 0;font-size:13px;color:#6b7280;">
        Period: <strong style="color:#374151;">{period_str}</strong>
        {consistency}{percentile}
      </p>
      <p style="margin:6px 0 0;font-size:13px;color:#6b7280;">
        Calls analysed: <strong style="color:#374151;">{report.total_calls}</strong>
      </p>
    </div>

    {exec_summary}
    {strengths_html}
    {improvements_html}

    {_cta_button(reports_url, "View Full Report")}
    """
    return _email_wrapper(content)


def send_report_notification(report: "PerformanceReport") -> bool:
    """Send a 'report ready' email to the report's agent. Fire-and-forget safe."""
    try:
        email = report.agent.email
        if not email:
            logger.warning("send_report_notification: agent %s has no email — skipping", report.agent_id)
            return False

        period_type = report.report_type.capitalize()
        start = report.start_date.strftime("%-d %b") if report.start_date else ""
        end = report.end_date.strftime("%-d %b %Y") if report.end_date else ""
        period_str = f"{start}–{end}" if start and end else start or end

        subject = f"Your {period_type} Performance Report — {period_str}"
        html = _build_report_html(report)
        return _send(to=email, subject=subject, html=html)
    except Exception:
        logger.exception("send_report_notification: unexpected error for report %s", getattr(report, "id", "?"))
        return False


# ── Call analyzed notification ────────────────────────────────────────────────

def _build_call_html(call: "Call") -> str:
    frontend_url: str = getattr(settings, "FRONTEND_URL", "http://localhost:3000")
    calls_url = f"{frontend_url}/dashboard/calls"

    agent_name = (
        f"{call.agent.first_name} {call.agent.last_name}".strip()
        or call.agent.username
    )

    # Duration
    duration_str = "—"
    if call.duration:
        m, s = divmod(int(call.duration), 60)
        duration_str = f"{m}:{str(s).zfill(2)} min"

    # Behavioural score
    score_html = ""
    if call.behavioral_analysis:
        raw_score = call.behavioral_analysis.get("behavioral_score")
        if raw_score is not None:
            score = round(float(raw_score))
            color = _score_color(score)
            score_html = (
                f'<p style="margin:6px 0 0;font-size:13px;color:#6b7280;">'
                f'Behavioural Score '
                f'<strong style="color:{color};font-size:16px;">{score}</strong>/100</p>'
            )

    # Call tone
    tone_html = ""
    if call.emotional_summary:
        tone = call.emotional_summary.get("call_tone", "").capitalize()
        tone_colors = {"Positive": "#0caf60", "Negative": "#e25950", "Neutral": "#6b7280"}
        tone_color = tone_colors.get(tone, "#6b7280")
        if tone:
            tone_html = (
                f'<p style="margin:6px 0 0;font-size:13px;color:#6b7280;">'
                f'Call Tone <strong style="color:{tone_color};">{tone}</strong></p>'
            )

    # Coaching tip
    coaching_html = ""
    if call.coaching_tips:
        tip = ""
        if isinstance(call.coaching_tips, dict):
            tip = call.coaching_tips.get("message") or call.coaching_tips.get("tip", "")
        elif isinstance(call.coaching_tips, str):
            tip = call.coaching_tips
        if tip:
            coaching_html = (
                f'<div style="background:#fefce8;border-left:3px solid #fde047;'
                f'padding:12px 16px;border-radius:0 8px 8px 0;margin:20px 0;">'
                f'<p style="margin:0 0 4px;font-size:12px;font-weight:700;color:#92400e;">'
                f'💡 Coaching Tip</p>'
                f'<p style="margin:0;font-size:14px;color:#78350f;line-height:1.5;">'
                f'{str(tip)[:300]}</p></div>'
            )

    # Primary topic
    topic_html = ""
    if call.topic_analysis:
        topic = call.topic_analysis.get("primary_topic", "")
        if topic:
            topic_html = (
                f'<p style="margin:6px 0 0;font-size:13px;color:#6b7280;">'
                f'Topic <strong style="color:#374151;">{topic}</strong></p>'
            )

    # Call date
    call_date_str = ""
    if call.call_date:
        call_date_str = call.call_date.strftime("%-d %b %Y, %H:%M")

    content = f"""
    <h2 style="margin:0 0 4px;font-size:22px;color:#111827;">
      Call Analysis Complete
    </h2>
    <p style="margin:0 0 20px;font-size:14px;color:#6b7280;">Hi {agent_name},</p>
    <p style="font-size:14px;color:#374151;margin:0 0 20px;">
      Your call recording has been fully analysed. Here's a quick summary:
    </p>

    <!-- Call summary card -->
    <div style="background:#f9fafb;border:1px solid #e5e7eb;border-radius:10px;
                padding:20px 24px;margin-bottom:20px;">
      <p style="margin:0;font-size:13px;color:#6b7280;">
        Date &amp; Time <strong style="color:#374151;">{call_date_str}</strong>
      </p>
      <p style="margin:6px 0 0;font-size:13px;color:#6b7280;">
        Duration <strong style="color:#374151;">{duration_str}</strong>
      </p>
      {topic_html}
      {score_html}
      {tone_html}
    </div>

    {coaching_html}

    {_cta_button(calls_url, "View All My Calls")}
    """
    return _email_wrapper(content)


def send_call_analyzed_notification(call: "Call") -> bool:
    """Send a 'call analysis complete' email to the call's agent. Fire-and-forget safe."""
    try:
        email = call.agent.email
        if not email:
            logger.warning("send_call_analyzed_notification: agent %s has no email — skipping", call.agent_id)
            return False

        date_str = call.call_date.strftime("%-d %b %Y") if call.call_date else ""
        subject = f"Call Analysis Complete — {date_str}"
        html = _build_call_html(call)
        return _send(to=email, subject=subject, html=html)
    except Exception:
        logger.exception("send_call_analyzed_notification: unexpected error for call %s", getattr(call, "id", "?"))
        return False
