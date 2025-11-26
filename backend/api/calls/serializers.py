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
        # Check if file is provided
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
            'created_at',
            'updated_at'
        ]
    
    def get_agent_name(self, obj):
        return f"{obj.agent.first_name} {obj.agent.last_name}".strip() or obj.agent.username


class CallListSerializer(serializers.ModelSerializer):
    agent_name = serializers.SerializerMethodField()
    
    class Meta:
        model = Call
        fields = [
            'id',
            'agent_name',
            'call_date',
            'duration',
            'created_at'
        ]
    
    def get_agent_name(self, obj):
        return f"{obj.agent.first_name} {obj.agent.last_name}".strip() or obj.agent.username
