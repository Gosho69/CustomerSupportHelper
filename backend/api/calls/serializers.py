from rest_framework import serializers
from .models import Call


class CallUploadSerializer(serializers.ModelSerializer):
    summarization_model = serializers.ChoiceField(
        choices=['gpt4', 'local'],
        default='gpt4',
        required=False,
        help_text="Choose summarization model: 'gpt4' or 'local'"
    )
    
    class Meta:
        model = Call
        fields = ['audio_file', 'summarization_model']
        extra_kwargs = {
            'audio_file': {'required': True}
        }
    
    def validate_audio_file(self, value):
        if not value:
            raise serializers.ValidationError("Audio file is required")
        allowed_extensions = ['.wav', '.mp3', '.m4a', '.flac', '.ogg', '.opus']
        ext = value.name.lower().split('.')[-1]
        if f'.{ext}' not in allowed_extensions:
            raise serializers.ValidationError(
                f"Unsupported file format. Allowed: {', '.join(allowed_extensions)}"
            )
        
        if value.size > 100 * 1024 * 1024:
            raise serializers.ValidationError("File size cannot exceed 100MB")
        
        return value


class CallSerializer(serializers.ModelSerializer):
    agent_name = serializers.SerializerMethodField()
    
    class Meta:
        model = Call
        fields = [
            'id',
            'agent',
            'agent_name',
            'call_date',
            'duration',
            'audio_file',
            'transcript',
            'transcript_summary',
            'emotional_analysis',
            'emotional_summary',
            'behavioral_analysis',
            'behavioral_summary',
            'coaching_tips',
            'topic_analysis',
            'created_at',
            'updated_at'
        ]
        read_only_fields = [
            'id',
            'agent',
            'call_date',
            'transcript',
            'transcript_summary',
            'emotional_analysis',
            'emotional_summary',
            'behavioral_analysis',
            'behavioral_summary',
            'coaching_tips',
            'topic_analysis',
            'created_at',
            'updated_at'
        ]
    
    def get_agent_name(self, obj):
        return f"{obj.agent.first_name} {obj.agent.last_name}".strip() or obj.agent.username


class CallListSerializer(serializers.ModelSerializer):
    agent_name = serializers.SerializerMethodField()
    behavioral_score = serializers.SerializerMethodField()
    skill_scores = serializers.SerializerMethodField()

    class Meta:
        model = Call
        fields = [
            'id',
            'agent',
            'agent_name',
            'call_date',
            'duration',
            'behavioral_score',
            'skill_scores',
            'created_at'
        ]

    def get_agent_name(self, obj):
        return f"{obj.agent.first_name} {obj.agent.last_name}".strip() or obj.agent.username

    def get_behavioral_score(self, obj):
        if obj.behavioral_analysis:
            return obj.behavioral_analysis.get('behavioral_score', None)
        return None

    def get_skill_scores(self, obj):
        if not obj.behavioral_analysis:
            return None
        ba = obj.behavioral_analysis

        def pacing_to_score(pacing):
            return {
                "optimal": 100, "slightly_fast": 80,
                "too_fast": 60, "too_slow": 65, "unknown": 50,
            }.get(pacing, 50)

        def response_to_score(assessment):
            return {
                "optimal": 100, "acceptable": 75,
                "too_quick": 70, "too_slow": 50, "unknown": 50,
            }.get(assessment, 50)

        al = ba.get("active_listening", {})
        wpm = ba.get("words_per_minute", {})
        interruptions = ba.get("interruption_analysis", {})
        response = ba.get("response_time_analysis", {})
        questions = ba.get("question_analysis", {})
        silence = ba.get("silence_analysis", {})

        listening_score = min(round(al.get("acknowledgment_rate", 0) * 200), 100)
        pacing_score = pacing_to_score(wpm.get("agent_pacing", "unknown"))
        agent_ints = interruptions.get("agent_interruptions", 0)
        communication_score = max(100 - agent_ints * 10, 40)
        response_score = response_to_score(response.get("agent_response_assessment", "unknown"))
        question_pattern = questions.get("question_pattern", "needs_improvement")
        engagement_score = 100 if question_pattern == "good" else 60
        silence_pct = silence.get("silence_percentage", 0)
        composure_score = max(round(100 - max(silence_pct - 10, 0) * 2), 40)

        return {
            "active_listening": listening_score,
            "pacing": pacing_score,
            "communication": communication_score,
            "response_time": response_score,
            "engagement": engagement_score,
            "composure": composure_score,
        }
