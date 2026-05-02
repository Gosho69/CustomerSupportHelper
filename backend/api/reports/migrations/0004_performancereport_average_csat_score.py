from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('reports', '0003_performancereport_ai_generated_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='performancereport',
            name='average_csat_score',
            field=models.FloatField(blank=True, help_text='Average predicted CSAT score (1.0–5.0 raw)', null=True),
        ),
    ]
