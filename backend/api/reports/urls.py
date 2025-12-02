from django.urls import path
from .views import GenerateReportView, AgentReportsListView, ReportDetailView, MyReportsView

urlpatterns = [
    path('generate/', GenerateReportView.as_view(), name='generate-report'),
    path('agent/<int:agent_id>/', AgentReportsListView.as_view(), name='agent-reports'),
    path('my-reports/', MyReportsView.as_view(), name='my-reports'),
    path('all/', AgentReportsListView.as_view(), name='all-reports'),
    path('<int:report_id>/', ReportDetailView.as_view(), name='report-detail'),
]
