from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('calls', '0002_call_topic_analysis'),
    ]

    operations = [
        migrations.AddField(
            model_name='call',
            name='status',
            field=models.CharField(
                choices=[
                    ('pending', 'Pending'),
                    ('processing', 'Processing'),
                    ('completed', 'Completed'),
                    ('failed', 'Failed'),
                ],
                db_index=True,
                default='pending',
                max_length=20,
            ),
        ),
        migrations.AddField(
            model_name='call',
            name='error_message',
            field=models.TextField(blank=True, null=True),
        ),
    ]
