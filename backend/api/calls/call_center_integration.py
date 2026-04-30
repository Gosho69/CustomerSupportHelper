"""
Call Center Integration Service Layer

Provides an abstraction between the sync task and the actual call center data source.
The mock implementation uses direct ORM queries (no HTTP round-trip), which is correct
for the simulation. A real integration would subclass BaseCallCenterIntegration and
use requests/urllib to talk to an external vendor API.

To add a real integration:
  1. Subclass BaseCallCenterIntegration and implement the three abstract methods.
  2. Register the new backend name in get_integration_service().
  3. Set CALL_CENTER_INTEGRATION=<your_backend_name> in the environment.
"""
import abc
import logging

from django.conf import settings

logger = logging.getLogger(__name__)


class BaseCallCenterIntegration(abc.ABC):
    """
    Abstract interface that every call center integration must implement.
    The sync task calls these three methods — nothing more.
    """

    @abc.abstractmethod
    def get_unanalyzed_calls(self) -> list[dict]:
        """
        Return a list of metadata dicts for calls that have not yet been
        imported into the main platform.

        Each dict must contain at minimum:
          - external_id (str): unique identifier in the external system
          - agent_email (str): email of the agent who handled the call
          - call_started_at (str): ISO 8601 timestamp
          - duration_seconds (int)
          - caller_id (str)
        """

    @abc.abstractmethod
    def get_audio_content(self, external_id: str) -> bytes:
        """
        Return the raw audio bytes for the given external call ID.
        The sync task writes these to a temp file before passing to analyze_call_task.
        """

    @abc.abstractmethod
    def mark_as_analyzed(self, external_id: str) -> None:
        """
        Signal the external system that this call has been imported.
        Called after analyze_call_task has been enqueued (not after analysis completes).
        """


class MockCallCenterIntegration(BaseCallCenterIntegration):
    """
    Simulation backend — uses Django ORM directly.

    No HTTP round-trip needed because the mock call center is part of the same
    Django project. This means the integration works even when the web server
    is not running (e.g. in a Celery worker). A real integration would replace
    these ORM calls with HTTP requests to an external vendor API.
    """

    def get_unanalyzed_calls(self) -> list[dict]:
        from mock_callcenter.models import ExternalCall
        calls = ExternalCall.objects.filter(analyzed=False).only(
            'external_id', 'agent_email', 'caller_id',
            'call_started_at', 'duration_seconds',
        )
        return [
            {
                'external_id': str(c.external_id),
                'agent_email': c.agent_email,
                'caller_id': c.caller_id,
                'call_started_at': c.call_started_at.isoformat(),
                'duration_seconds': c.duration_seconds,
            }
            for c in calls
        ]

    def get_audio_content(self, external_id: str) -> bytes:
        from mock_callcenter.models import ExternalCall
        ec = ExternalCall.objects.get(external_id=external_id)
        with ec.audio_file.open('rb') as f:
            return f.read()

    def mark_as_analyzed(self, external_id: str) -> None:
        from mock_callcenter.models import ExternalCall
        from django.utils import timezone
        ExternalCall.objects.filter(external_id=external_id).update(
            analyzed=True,
            imported_at=timezone.now(),
        )


def get_integration_service() -> BaseCallCenterIntegration | None:
    """
    Factory function. Returns None when integration is disabled so callers
    can short-circuit cleanly without raising an exception.

    Controlled by settings.CALL_CENTER_INTEGRATION (env: CALL_CENTER_INTEGRATION):
      'mock'     → MockCallCenterIntegration (ORM-based simulation)
      'disabled' → None (sync task will no-op)

    To add a real backend, add a new branch here and in CALL_CENTER_INTEGRATION docs.
    """
    backend = getattr(settings, 'CALL_CENTER_INTEGRATION', 'disabled')

    if backend == 'mock':
        logger.debug("get_integration_service: returning MockCallCenterIntegration")
        return MockCallCenterIntegration()

    if backend == 'disabled':
        return None

    raise ValueError(
        f"Unknown CALL_CENTER_INTEGRATION backend: {backend!r}. "
        "Valid values: 'mock', 'disabled'."
    )
