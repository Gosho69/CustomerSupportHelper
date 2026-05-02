from django.urls import path
from .views import UploadCallView, AgentCallsListView, CallDetailView, CallStatusView, QueueStatusView

urlpatterns = [
    path('upload/', UploadCallView.as_view(), name='upload-call'),
    path('my-calls/', AgentCallsListView.as_view(), name='agent-calls-list'),
    path('queue-status/', QueueStatusView.as_view(), name='call-queue-status'),
    path('<int:pk>/', CallDetailView.as_view(), name='call-detail'),
    path('<int:pk>/status/', CallStatusView.as_view(), name='call-status'),
]
