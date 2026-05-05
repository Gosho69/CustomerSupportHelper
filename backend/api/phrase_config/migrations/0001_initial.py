from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='PhraseList',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=100, unique=True)),
                ('list_type', models.CharField(
                    choices=[
                        ('phrase_list', 'Phrase List'),
                        ('mapping', 'Mapping'),
                        ('topic_group', 'Topic Group'),
                    ],
                    default='phrase_list',
                    max_length=20,
                )),
                ('data', models.JSONField()),
                ('description', models.TextField(blank=True)),
                ('is_active', models.BooleanField(default=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
            ],
            options={
                'verbose_name': 'Phrase List',
            },
        ),
    ]
