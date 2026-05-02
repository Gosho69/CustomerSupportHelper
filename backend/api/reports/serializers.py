from rest_framework import serializers
from .models import PerformanceReport


class PerformanceReportSerializer(serializers.ModelSerializer):
    agent_name = serializers.SerializerMethodField()
    agent_username = serializers.CharField(source='agent.username', read_only=True)
    agent_email = serializers.CharField(source='agent.email', read_only=True)
    generated_by_name = serializers.SerializerMethodField()
    
    class Meta:
        model = PerformanceReport
        fields = [
            'id', 'agent', 'agent_name', 'agent_username', 'agent_email',
            'report_type', 'start_date', 'end_date',
            'total_calls', 'average_call_duration',
            'average_emotional_score', 'positive_calls_percentage', 'negative_calls_percentage', 'emotional_trend',
            'average_behavioral_score', 'empathy_score', 'professionalism_score', 'problem_solving_score', 'behavioral_trend',
            'average_csat_score',
            'most_common_topics', 'topic_resolution_rate',
            'performance_consistency_score', 'variance_from_average',
            'ranking_in_team', 'percentile_score',
            'strengths', 'weaknesses', 'recommendations', 'weekly_analysis',
            'overall_rating', 'summary', 'executive_summary', 'ai_generated',
            'generated_at', 'generated_by', 'generated_by_name'
        ]
        read_only_fields = ['id', 'generated_at', 'agent_name', 'agent_username', 'agent_email', 'generated_by_name']
    
    def get_agent_name(self, obj):
        if obj.agent.first_name or obj.agent.last_name:
            return f"{obj.agent.first_name} {obj.agent.last_name}".strip()
        return obj.agent.username
    
    def get_generated_by_name(self, obj):
        if obj.generated_by:
            if obj.generated_by.first_name or obj.generated_by.last_name:
                return f"{obj.generated_by.first_name} {obj.generated_by.last_name}".strip()
            return obj.generated_by.username
        return "System"


class ReportListSerializer(serializers.ModelSerializer):
    agent_name = serializers.SerializerMethodField()
    agent_username = serializers.CharField(source='agent.username', read_only=True)
    
    def get_agent_name(self, obj):
        if obj.agent.first_name or obj.agent.last_name:
            return f"{obj.agent.first_name} {obj.agent.last_name}".strip()
        return obj.agent.username
    
    class Meta:
        model = PerformanceReport
        fields = [
            'id', 'agent', 'agent_name', 'agent_username',
            'report_type', 'start_date', 'end_date',
            'total_calls', 'overall_rating', 'average_behavioral_score',
            'generated_at'
        ]
        read_only_fields = fields
