from django.db import models


class PhraseList(models.Model):
    LIST_TYPE_CHOICES = [
        ("phrase_list", "Phrase List"),   # data = list[str]
        ("mapping",     "Mapping"),       # data = dict[str, str | float]
        ("topic_group", "Topic Group"),   # data = dict[str, {keywords, phrases, weight}]
    ]

    name        = models.CharField(max_length=100, unique=True)
    list_type   = models.CharField(max_length=20, choices=LIST_TYPE_CHOICES, default="phrase_list")
    data        = models.JSONField()
    description = models.TextField(blank=True)
    is_active   = models.BooleanField(default=True)
    updated_at  = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "Phrase List"

    def __str__(self):
        return self.name
