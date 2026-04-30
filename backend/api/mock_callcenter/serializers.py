from rest_framework import serializers
from .models import ExternalCall

ALLOWED_EXTENSIONS = {'.wav', '.mp3', '.m4a', '.flac', '.ogg', '.opus'}
MAX_UPLOAD_SIZE = 100 * 1024 * 1024  # 100 MB


class ExternalCallUploadSerializer(serializers.Serializer):
    audio_file = serializers.FileField()
    agent_email = serializers.EmailField()

    def validate_audio_file(self, value):
        import os
        ext = os.path.splitext(value.name)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise serializers.ValidationError(
                f"Unsupported file format '{ext}'. "
                f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
            )
        if value.size > MAX_UPLOAD_SIZE:
            raise serializers.ValidationError(
                f"File too large ({value.size / 1024 / 1024:.1f} MB). Maximum is 100 MB."
            )
        return value


class ExternalCallListSerializer(serializers.ModelSerializer):
    external_id = serializers.UUIDField(read_only=True)

    class Meta:
        model = ExternalCall
        fields = [
            'external_id',
            'agent_email',
            'caller_id',
            'call_started_at',
            'duration_seconds',
            'analyzed',
            'imported_at',
        ]
        read_only_fields = fields
