from django.db import models
from users.models import MyUser


class PerformanceReport(models.Model):
    REPORT_TYPE_CHOICES = [
        ('weekly', 'Weekly'),
        ('monthly', 'Monthly'),
    ]
    
    RATING_CHOICES = [
        ('excellent', 'Excellent'),
        ('good', 'Good'),
        ('needs_improvement', 'Needs Improvement'),
        ('poor', 'Poor'),
        ('no_data', 'No Data'),
    ]
    
    TREND_CHOICES = [
        ('improving', 'Improving'),
        ('stable', 'Stable'),
        ('declining', 'Declining'),
    ]
    
    agent = models.ForeignKey(MyUser, on_delete=models.CASCADE, related_name='performance_reports', limit_choices_to={'role': 'agent'})
    report_type = models.CharField(max_length=10, choices=REPORT_TYPE_CHOICES)
    start_date = models.DateField()
    end_date = models.DateField()
    
    # Call Volume Metrics
    total_calls = models.IntegerField(default=0)
    average_call_duration = models.FloatField(null=True, blank=True, help_text="Average call duration in seconds")
    
    # Emotional Performance
    average_emotional_score = models.FloatField(null=True, blank=True, help_text="Average emotional sentiment score")
    positive_calls_percentage = models.FloatField(null=True, blank=True, help_text="Percentage of positive calls")
    negative_calls_percentage = models.FloatField(null=True, blank=True, help_text="Percentage of negative calls")
    emotional_trend = models.CharField(max_length=20, choices=TREND_CHOICES, null=True, blank=True)
    
    # Behavioral Performance
    average_behavioral_score = models.FloatField(null=True, blank=True, help_text="Overall behavioral score")
    empathy_score = models.FloatField(null=True, blank=True, help_text="Average empathy score")
    professionalism_score = models.FloatField(null=True, blank=True, help_text="Average professionalism score")
    problem_solving_score = models.FloatField(null=True, blank=True, help_text="Average problem solving score")
    behavioral_trend = models.CharField(max_length=20, choices=TREND_CHOICES, null=True, blank=True)
    
    # CSAT
    average_csat_score = models.FloatField(null=True, blank=True, help_text="Average predicted CSAT score (1.0–5.0 raw)")

    # Topic Analysis
    most_common_topics = models.JSONField(default=dict, help_text="Dictionary of topic names and their counts")
    topic_resolution_rate = models.JSONField(default=dict, help_text="Dictionary of topic resolution rates")
    
    # Consistency Metrics
    performance_consistency_score = models.FloatField(null=True, blank=True, help_text="Score 0-100, higher is more consistent")
    variance_from_average = models.FloatField(null=True, blank=True, help_text="Standard deviation of performance")
    
    # Comparison Metrics
    ranking_in_team = models.IntegerField(null=True, blank=True, help_text="Ranking position in team")
    percentile_score = models.FloatField(null=True, blank=True, help_text="Percentile score 0-100")
    
    # Improvement Areas
    strengths = models.JSONField(default=list, help_text="List of identified strengths")
    weaknesses = models.JSONField(default=list, help_text="List of identified weaknesses")
    recommendations = models.JSONField(default=list, help_text="List of coaching recommendations")
    
    # Weekly Analysis (for monthly reports)
    weekly_analysis = models.JSONField(null=True, blank=True, help_text="Week-by-week analysis for monthly reports")
    
    # Overall Assessment
    overall_rating = models.CharField(max_length=20, choices=RATING_CHOICES, null=True, blank=True)
    summary = models.TextField(blank=True, help_text="Comprehensive AI-generated or rule-based performance summary")
    executive_summary = models.TextField(blank=True, help_text="Brief executive summary (2-3 sentences)")
    ai_generated = models.BooleanField(default=False, help_text="Whether report used AI generation")
    
    # Metadata
    generated_at = models.DateTimeField(auto_now_add=True)
    generated_by = models.ForeignKey(MyUser, on_delete=models.SET_NULL, null=True, blank=True, related_name='reports_generated')
    
    class Meta:
        ordering = ['-start_date', '-end_date']
        unique_together = ['agent', 'report_type', 'start_date', 'end_date']
        verbose_name = 'Performance Report'
        verbose_name_plural = 'Performance Reports'
    
    def __str__(self):
        return f"{self.agent.username} - {self.report_type} - {self.start_date} to {self.end_date}"
