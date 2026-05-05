from django.contrib import admin
from .models import PhraseList


@admin.register(PhraseList)
class PhraseListAdmin(admin.ModelAdmin):
    list_display   = ["name", "list_type", "is_active", "updated_at"]
    list_filter    = ["list_type", "is_active"]
    search_fields  = ["name", "description"]
    readonly_fields = ["updated_at"]
