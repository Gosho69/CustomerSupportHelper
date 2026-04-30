import uuid
from django.db import models


class ExternalCall(models.Model):
    external_id = models.UUIDField(
        default=uuid.uuid4,
        unique=True,
        db_index=True,
        editable=False,
    )
    agent_email = models.EmailField(
        help_text="Maps to MyUser.email with role=agent in the main platform."
    )
    caller_id = models.CharField(max_length=30, default='unknown')
    call_started_at = models.DateTimeField()
    duration_seconds = models.PositiveIntegerField(default=0)
    audio_file = models.FileField(upload_to='mock_callcenter/recordings/')
    analyzed = models.BooleanField(
        default=False,
        db_index=True,
        help_text="Set to True once the main platform has imported and enqueued this call.",
    )
    imported_at = models.DateTimeField(
        null=True,
        blank=True,
        help_text="Timestamp when the main platform imported this call.",
    )
    platform_call = models.OneToOneField(
        'calls.Call',
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name='external_source',
        help_text="The Call record created in the main platform for this external call.",
    )

    class Meta:
        ordering = ['-call_started_at']
        verbose_name = 'External Call'
        verbose_name_plural = 'External Calls'

    def __str__(self):
        status = 'analyzed' if self.analyzed else 'pending'
        return f"ExternalCall {self.external_id} | {self.agent_email} | {status}"
