from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated

from .models import PerformanceReport
from .serializers import PerformanceReportSerializer, ReportListSerializer
from .performance_calculator import PerformanceCalculator
from .report_factory import create_report_from_metrics
from .date_utils import (
    get_week_date_range, 
    get_month_date_range,
    get_previous_week_date_range,
    get_previous_month_date_range
)
from users.models import MyUser


class GenerateReportView(APIView):
    permission_classes = [IsAuthenticated]
    
    def post(self, request):
        if request.user.role != 'head_of_department':
            return Response(
                {'error': 'Only heads of department can manually generate reports'},
                status=status.HTTP_403_FORBIDDEN
            )
        
        agent_id = request.data.get('agent_id')
        report_type = request.data.get('report_type')
        use_previous_period = request.data.get('use_previous_period', False)
        
        if not agent_id or not report_type:
            return Response(
                {'error': 'agent_id and report_type are required'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        if report_type not in ['weekly', 'monthly']:
            return Response(
                {'error': 'Invalid report_type. Use "weekly" or "monthly"'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            agent = MyUser.objects.get(id=agent_id, role='agent')
        except MyUser.DoesNotExist:
            return Response(
                {'error': 'Agent not found'},
                status=status.HTTP_404_NOT_FOUND
            )
        
        # Head of department can only generate reports for their subordinates
        if agent.reporting_to != request.user:
            return Response(
                {'error': 'You can only generate reports for your subordinates'},
                status=status.HTTP_403_FORBIDDEN
            )
        
        if report_type == 'weekly':
            if use_previous_period:
                start_date, end_date = get_previous_week_date_range()
            else:
                start_date, end_date = get_week_date_range()
        else:
            if use_previous_period:
                start_date, end_date = get_previous_month_date_range()
            else:
                start_date, end_date = get_month_date_range()
        
        if start_date < agent.created_at.date():
            return Response(
                {'error': 'Cannot generate a report for a period before the agent was created'},
                status=status.HTTP_400_BAD_REQUEST
            )

        existing_report = PerformanceReport.objects.filter(
            agent=agent,
            report_type=report_type,
            start_date=start_date,
            end_date=end_date
        ).first()
        
        if existing_report:
            serializer = PerformanceReportSerializer(existing_report)
            return Response({
                'message': 'Report already exists',
                'report': serializer.data
            }, status=status.HTTP_200_OK)
        
        calculator = PerformanceCalculator(agent, start_date, end_date, report_type=report_type)
        metrics = calculator.calculate_metrics()

        report = create_report_from_metrics(
            agent=agent,
            report_type=report_type,
            start_date=start_date,
            end_date=end_date,
            metrics=metrics,
            generated_by=request.user,
        )

        # Best-effort email — never block the API response.
        try:
            from notifications import send_report_notification
            send_report_notification(report)
        except Exception:
            import logging
            logging.getLogger(__name__).exception(
                "GenerateReportView: email notification failed for report %s", report.id
            )

        serializer = PerformanceReportSerializer(report)
        return Response({
            'message': 'Report generated successfully',
            'report': serializer.data
        }, status=status.HTTP_201_CREATED)


class AgentReportsListView(APIView):
    permission_classes = [IsAuthenticated]
    
    def get(self, request, agent_id=None):
        if request.user.role == 'agent':
            reports = PerformanceReport.objects.filter(
                agent=request.user,
                start_date__gte=request.user.created_at.date(),
            )

        elif request.user.role == 'head_of_department':
            if agent_id:
                try:
                    agent = MyUser.objects.get(id=agent_id, reporting_to=request.user)
                    reports = PerformanceReport.objects.filter(
                        agent=agent,
                        start_date__gte=agent.created_at.date(),
                    )
                except MyUser.DoesNotExist:
                    return Response(
                        {'error': 'Agent not found or not your subordinate'},
                        status=status.HTTP_404_NOT_FOUND
                    )
            else:
                from django.db.models import F
                from django.db.models.functions import TruncDate
                subordinates = MyUser.objects.filter(reporting_to=request.user, role='agent')
                reports = PerformanceReport.objects.filter(
                    agent__in=subordinates,
                ).annotate(
                    agent_created=TruncDate('agent__created_at'),
                ).filter(start_date__gte=F('agent_created'))

        elif request.user.role == 'admin':
            if agent_id:
                try:
                    agent = MyUser.objects.get(id=agent_id)
                    reports = PerformanceReport.objects.filter(
                        agent=agent,
                        start_date__gte=agent.created_at.date(),
                    )
                except MyUser.DoesNotExist:
                    return Response(
                        {'error': 'Agent not found'},
                        status=status.HTTP_404_NOT_FOUND
                    )
            else:
                from django.db.models import F
                from django.db.models.functions import TruncDate
                reports = PerformanceReport.objects.annotate(
                    agent_created=TruncDate('agent__created_at'),
                ).filter(start_date__gte=F('agent_created'))
        
        else:
            return Response(
                {'error': 'Unauthorized'},
                status=status.HTTP_403_FORBIDDEN
            )
        
        serializer = ReportListSerializer(reports, many=True)
        return Response({
            'count': reports.count(),
            'reports': serializer.data
        })


class ReportDetailView(APIView):
    permission_classes = [IsAuthenticated]
    
    def get(self, request, report_id):
        try:
            report = PerformanceReport.objects.get(id=report_id)
        except PerformanceReport.DoesNotExist:
            return Response(
                {'error': 'Report not found'},
                status=status.HTTP_404_NOT_FOUND
            )
        
        if request.user.role == 'agent' and report.agent != request.user:
            return Response(
                {'error': 'You can only view your own reports'},
                status=status.HTTP_403_FORBIDDEN
            )
        
        if request.user.role == 'head_of_department' and report.agent.reporting_to != request.user:
            return Response(
                {'error': 'You can only view reports for your subordinates'},
                status=status.HTTP_403_FORBIDDEN
            )
        
        serializer = PerformanceReportSerializer(report)
        return Response(serializer.data)


class MyReportsView(APIView):
    permission_classes = [IsAuthenticated]
    
    def get(self, request):
        if request.user.role == 'agent':
            reports = PerformanceReport.objects.filter(
                agent=request.user,
                start_date__gte=request.user.created_at.date(),
            )
        elif request.user.role == 'head_of_department':
            from django.db.models import F
            from django.db.models.functions import TruncDate
            subordinates = MyUser.objects.filter(reporting_to=request.user, role='agent')
            reports = PerformanceReport.objects.filter(
                agent__in=subordinates,
            ).annotate(
                agent_created=TruncDate('agent__created_at'),
            ).filter(start_date__gte=F('agent_created'))
        elif request.user.role == 'admin':
            from django.db.models import F
            from django.db.models.functions import TruncDate
            reports = PerformanceReport.objects.annotate(
                agent_created=TruncDate('agent__created_at'),
            ).filter(start_date__gte=F('agent_created'))
        else:
            return Response(
                {'error': 'Unauthorized'},
                status=status.HTTP_403_FORBIDDEN
            )
        
        serializer = ReportListSerializer(reports, many=True)
        
        return Response({
            'count': reports.count(),
            'reports': serializer.data
        })
