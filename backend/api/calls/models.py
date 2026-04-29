from django.db import models
from users.models import MyUser


class Call(models.Model):
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('processing', 'Processing'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]

    agent = models.ForeignKey(MyUser, on_delete=models.CASCADE, related_name='calls', limit_choices_to={'role': 'agent'})
    call_date = models.DateTimeField(auto_now_add=True)
    duration = models.IntegerField(null=True, blank=True, help_text="Call duration in seconds")
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending', db_index=True)
    error_message = models.TextField(null=True, blank=True)
    
    audio_file = models.FileField(upload_to='call_recordings/', null=True, blank=True)
    
    transcript = models.JSONField(null=True, blank=True, help_text="Full transcript of the call")
    transcript_summary = models.JSONField(null=True, blank=True, help_text="Summary of the transcript")
    
    emotional_analysis = models.JSONField(null=True, blank=True, help_text="Detailed emotional analysis")
    emotional_summary = models.JSONField(null=True, blank=True, help_text="Summary of emotional analysis")
    
    behavioral_analysis = models.JSONField(null=True, blank=True, help_text="Detailed behavioral analysis")
    behavioral_summary = models.JSONField(null=True, blank=True, help_text="Summary of behavioral analysis")
    
    coaching_tips = models.JSONField(null=True, blank=True, help_text="AI-generated coaching tips")
    topic_analysis = models.JSONField(null=True, blank=True, help_text="Topic analysis of the call")
    vocal_analysis = models.JSONField(null=True, blank=True, help_text="Audio-level vocal stress and prosody analysis per utterance")

    predicted_csat = models.FloatField(
        null=True, blank=True,
        help_text="Predicted CSAT score 1.0–5.0"
    )
    predicted_csat_label = models.CharField(
        max_length=30, null=True, blank=True,
        help_text="very_satisfied / satisfied / neutral / dissatisfied / very_dissatisfied"
    )
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-call_date']
        verbose_name = 'Call'
        verbose_name_plural = 'Calls'
    
    def __str__(self):
        return f"Call by {self.agent.username} on {self.call_date.strftime('%Y-%m-%d %H:%M')}"
    
