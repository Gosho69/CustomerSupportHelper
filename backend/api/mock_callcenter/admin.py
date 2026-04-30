from django.contrib import admin
from .models import ExternalCall


@admin.register(ExternalCall)
class ExternalCallAdmin(admin.ModelAdmin):
    list_display = [
        'external_id',
        'agent_email',
        'caller_id',
        'call_started_at',
        'duration_seconds',
        'analyzed',
        'imported_at',
        'platform_call',
    ]
    list_filter = ['analyzed']
    search_fields = ['agent_email', 'caller_id']
    readonly_fields = ['external_id', 'imported_at', 'platform_call']
    ordering = ['-call_started_at']
