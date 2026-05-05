"""
Data migration: add remaining phrase/word lists used in sentiment analysis
(_analyze_sentiment) and behavioral analysis (_analyze_active_listening,
_analyze_questions).

Total new entries: 9 — bringing the full count to 25.
"""

from django.db import migrations


def populate_extra_phrase_lists(apps, schema_editor):
    PhraseList = apps.get_model("phrase_config", "PhraseList")

    entries = [
        # ── Emotion_analyzation/emotion_analyzer.py — _analyze_sentiment() ──

        {
            "name": "recording_announcement_phrases",
            "list_type": "phrase_list",
            "description": "Phrases that indicate a 'call is being recorded' announcement. When detected, _analyze_sentiment returns neutral/0.5 immediately.",
            "data": [
                "call is now being recorded",
                "now being recorded",
            ],
        },
        {
            "name": "question_starters",
            "list_type": "phrase_list",
            "description": "Words/phrases that indicate the utterance is a question. Used by _analyze_sentiment to switch to a question-specific sentiment path.",
            "data": [
                "did you",
                "do you",
                "can you",
                "could you",
                "would you",
                "will you",
                "what",
                "where",
                "when",
                "why",
                "how",
                "is that",
                "are you",
                "have you",
            ],
        },
        {
            "name": "sentiment_clear_positive",
            "list_type": "phrase_list",
            "description": "Positive-sentiment words used in the question branch of _analyze_sentiment. If any are present (with no negatives), returns 'positive'.",
            "data": [
                "great",
                "excellent",
                "wonderful",
                "perfect",
                "fantastic",
                "happy",
            ],
        },
        {
            "name": "sentiment_clear_negative",
            "list_type": "phrase_list",
            "description": "Negative-sentiment words used in the question branch of _analyze_sentiment. If any are present (with no positives), returns 'negative'.",
            "data": [
                "problem",
                "issue",
                "wrong",
                "bad",
                "terrible",
                "upset",
                "angry",
                "hate",
            ],
        },
        {
            "name": "sentiment_strong_positive",
            "list_type": "phrase_list",
            "description": "Strong positive words used for short-text (≤5 words) sentiment in _analyze_sentiment.",
            "data": [
                "thank",
                "thanks",
                "excellent",
                "great",
                "perfect",
                "yes",
                "wonderful",
                "welcome",
            ],
        },
        {
            "name": "sentiment_strong_negative",
            "list_type": "phrase_list",
            "description": "Strong negative words used for short-text (≤5 words) sentiment in _analyze_sentiment.",
            "data": [
                "sorry",
                "no",
                "wrong",
                "upset",
                "angry",
                "hate",
                "disappointed",
                "frustrated",
            ],
        },
        {
            "name": "procedural_indicators",
            "list_type": "phrase_list",
            "description": "Phrases that mark a turn as procedural/neutral (agent reading back info, verifying details). _analyze_sentiment returns neutral/0.7 when any are present.",
            "data": [
                "let me",
                "i will",
                "i can",
                "what is",
                "my name is",
                "the item",
                "the number",
                "zip code",
                "located in",
                "order number",
                "item number",
                "customer number",
            ],
        },

        # ── behaviour_analyzer/behavioral_analyzer.py ──────────────────────

        {
            "name": "listening_phrases",
            "list_type": "phrase_list",
            "description": "Phrases that indicate active listening by the agent. Used by _analyze_active_listening to count acknowledgment turns and compute the acknowledgment rate.",
            "data": [
                "i understand",
                "i see",
                "i hear you",
                "that makes sense",
                "i appreciate",
                "thank you for",
                "let me make sure",
                "if i understand correctly",
                "so what you're saying",
                "got it",
                "okay",
                "alright",
                "right",
                "absolutely",
                "definitely",
                "certainly",
                "of course",
                "sure",
            ],
        },
        {
            "name": "question_words",
            "list_type": "phrase_list",
            "description": "Words used to detect questions in _analyze_questions. A turn counts as a question if it contains '?' or starts with one of these words.",
            "data": [
                "what",
                "when",
                "where",
                "why",
                "how",
                "who",
                "which",
                "can",
                "could",
                "would",
                "should",
                "do",
                "does",
                "did",
                "is",
                "are",
            ],
        },
    ]

    PhraseList.objects.bulk_create(
        [PhraseList(**entry) for entry in entries],
        ignore_conflicts=True,
    )


def reverse_noop(apps, schema_editor):
    pass


class Migration(migrations.Migration):

    dependencies = [
        ("phrase_config", "0002_populate_defaults"),
    ]

    operations = [
        migrations.RunPython(populate_extra_phrase_lists, reverse_code=reverse_noop),
    ]
