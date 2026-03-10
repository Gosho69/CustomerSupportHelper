from django.urls import path
from .views import UploadCallView, AgentCallsListView, CallDetailView, CallStatusView

urlpatterns = [
    path('upload/', UploadCallView.as_view(), name='upload-call'),
    path('my-calls/', AgentCallsListView.as_view(), name='agent-calls-list'),
    path('<int:pk>/', CallDetailView.as_view(), name='call-detail'),
    path('<int:pk>/status/', CallStatusView.as_view(), name='call-status'),
]
