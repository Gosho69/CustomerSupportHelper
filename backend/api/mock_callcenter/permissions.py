from rest_framework.permissions import BasePermission
from django.conf import settings
import hmac


class ApiKeyPermission(BasePermission):
    """
    Validates the X-API-Key header against settings.MOCK_CALLCENTER_API_KEY.
    Used instead of JWT authentication for the mock call center endpoints,
    simulating an external system's API key auth scheme.
    """
    message = "Invalid or missing API key. Provide a valid X-API-Key header."

    def has_permission(self, request, view):
        api_key = request.META.get('HTTP_X_API_KEY')
        expected = getattr(settings, 'MOCK_CALLCENTER_API_KEY', None)
        if not api_key or not expected:
            return False
        try:
            api_key_s = str(api_key)
            expected_s = str(expected)
        except Exception:
            return False
        return hmac.compare_digest(api_key_s, expected_s)
