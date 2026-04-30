from django.apps import AppConfig


class CallsConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'calls'

    def ready(self):
        # Celery's autodiscover_tasks() only scans for tasks.py.
        # Explicitly import every non-standard task module so their
        # @shared_task decorators register with the Celery app.
        import calls.csat_tasks  # noqa: F401
        import calls.integration_tasks  # noqa: F401
