"""
Data migration: expand the refusal_signals phrase list with Comcast-style
interrogation / stalling / retention phrases that the previous list missed.

These phrases caused refusal_hit=False on a call with request_hits=31 because
the agent used "my job is to have a conversation with you about keeping your
service" — none of which matched the original refusal signal list.

This migration updates the existing 'refusal_signals' DB row by appending the
new phrases.  Rule 0 in _apply_outcome_correction handles extreme-volume calls
(request_hits >= 20) independently, but updating the list ensures moderate-
volume cases (5–19 hits) are also caught correctly.
"""

from django.db import migrations


_NEW_PHRASES = [
    # Comcast-style interrogation / stalling tactics
    "keeping your service",
    "keeping your account",
    "keep your service",
    "keep your account",
    "my job is to",
    "have a conversation with you about",
    "help me understand why you",
    "why is it that you're looking",
    "what would it take to keep",
    "before i can do that for you",
    "understand the reason",
]


def update_refusal_signals(apps, schema_editor):
    PhraseList = apps.get_model("phrase_config", "PhraseList")
    try:
        obj = PhraseList.objects.get(name="refusal_signals")
        existing = obj.data if isinstance(obj.data, list) else []
        existing_set = set(existing)
        additions = [p for p in _NEW_PHRASES if p not in existing_set]
        if additions:
            obj.data = existing + additions
            obj.save(update_fields=["data"])
    except PhraseList.DoesNotExist:
        pass  # populated by 0002 — should always exist


def reverse_noop(apps, schema_editor):
    pass


class Migration(migrations.Migration):

    dependencies = [
        ("phrase_config", "0003_add_sentiment_and_behavioral_lists"),
    ]

    operations = [
        migrations.RunPython(update_refusal_signals, reverse_code=reverse_noop),
    ]
