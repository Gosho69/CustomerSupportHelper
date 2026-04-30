from django.urls import path
from .views import (
    ExternalCallUploadView,
    ExternalCallListView,
    ExternalCallAudioView,
    ExternalCallMarkAnalyzedView,
)

urlpatterns = [
    # Primary entry point: upload a new call recording to the mock platform
    path('calls/upload/', ExternalCallUploadView.as_view(), name='mock-callcenter-upload'),
    # List calls (supports ?analyzed=false to filter unanalyzed)
    path('calls/', ExternalCallListView.as_view(), name='mock-callcenter-list'),
    # Download audio for a specific call
    path('calls/<uuid:external_id>/audio/', ExternalCallAudioView.as_view(), name='mock-callcenter-audio'),
    # Mark a call as analyzed/imported
    path('calls/<uuid:external_id>/', ExternalCallMarkAnalyzedView.as_view(), name='mock-callcenter-mark-analyzed'),
]
