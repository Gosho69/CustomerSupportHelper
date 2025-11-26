from django.db import models
from users.models import MyUser


class Call(models.Model):
    agent = models.ForeignKey(MyUser, on_delete=models.CASCADE, related_name='calls', limit_choices_to={'role': 'agent'})
    call_date = models.DateTimeField(auto_now_add=True)
    duration = models.IntegerField(null=True, blank=True, help_text="Call duration in seconds")
    
    audio_file = models.FileField(upload_to='call_recordings/', null=True, blank=True)
    
    transcript = models.JSONField(null=True, blank=True, help_text="Full transcript of the call")
    transcript_summary = models.JSONField(null=True, blank=True, help_text="Summary of the transcript")
    
    emotional_analysis = models.JSONField(null=True, blank=True, help_text="Detailed emotional analysis")
    emotional_summary = models.JSONField(null=True, blank=True, help_text="Summary of emotional analysis")
    
    behavioral_analysis = models.JSONField(null=True, blank=True, help_text="Detailed behavioral analysis")
    behavioral_summary = models.JSONField(null=True, blank=True, help_text="Summary of behavioral analysis")
    
    coaching_tips = models.JSONField(null=True, blank=True, help_text="AI-generated coaching tips")
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-call_date']
        verbose_name = 'Call'
        verbose_name_plural = 'Calls'
    
    def __str__(self):
        return f"Call by {self.agent.username} on {self.call_date.strftime('%Y-%m-%d %H:%M')}"
    
