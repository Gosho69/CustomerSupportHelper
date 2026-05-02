import os
from django.utils import timezone
from rest_framework import status, generics
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from django.conf import settings
from .models import Call
from .serializers import CallUploadSerializer, CallSerializer, CallListSerializer, CallStatusSerializer
from users.views import IsAdminOrHeadOfDepartment
import tempfile


class UploadCallView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        if request.user.role != 'agent':
            return Response(
                {'error': 'Only agents can upload calls'},
                status=status.HTTP_403_FORBIDDEN
            )

        serializer = CallUploadSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        summarization_model = serializer.validated_data.get('summarization_model', 'gpt4')
        audio_file = serializer.validated_data['audio_file']

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file.name)[1], dir=settings.MEDIA_ROOT)
        temp_path = temp_file.name

        try:
            for chunk in audio_file.chunks():
                temp_file.write(chunk)
            temp_file.close()

            local_model_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                'AI_modules', 'model', 'final'
            )
            whisper_model_size = os.environ.get("WHISPER_MODEL", "medium")

            # Save the call record immediately — analysis runs in background
            call = Call.objects.create(
                agent=request.user,
                audio_file=audio_file,
                status='pending',
            )

            from .tasks import analyze_call_task
            analyze_call_task.delay(
                call_id=call.id,
                temp_path=temp_path,
                summarization_model=summarization_model,
                whisper_model_size=whisper_model_size,
                local_model_path=local_model_path,
                gpt4_max_tokens=1500,
            )

            return Response(
                {'call_id': call.id, 'status': 'pending'},
                status=status.HTTP_202_ACCEPTED
            )

        except Exception as e:
            temp_file.close()
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            return Response(
                {'error': f'Upload failed: {str(e)}'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class CallStatusView(generics.RetrieveAPIView):
    permission_classes = [IsAuthenticated]
    serializer_class = CallStatusSerializer

    def get_queryset(self):
        if self.request.user.role == 'agent':
            return Call.objects.filter(agent=self.request.user)
        elif self.request.user.role == 'head_of_department':
            subordinates = self.request.user.subordinates.all()
            return Call.objects.filter(agent__in=subordinates)
        return Call.objects.none()


class AgentCallsListView(generics.ListAPIView):
    permission_classes = [IsAuthenticated]
    serializer_class = CallListSerializer

    def get_queryset(self):
        # Only surface calls whose analysis has fully finished.
        # pending/processing calls are still being analyzed and have no meaningful
        # results yet; showing them would confuse agents.
        # failed calls are included so agents know when a recording couldn't be processed.
        VISIBLE_STATUSES = ['completed', 'failed']

        if self.request.user.role == 'agent':
            return Call.objects.filter(
                agent=self.request.user,
                status__in=VISIBLE_STATUSES,
            ).order_by('-call_date')
        elif self.request.user.role == 'head_of_department':
            subordinates = self.request.user.subordinates.all()
            return Call.objects.filter(
                agent__in=subordinates,
                status__in=VISIBLE_STATUSES,
            ).order_by('-call_date')
        return Call.objects.none()


class CallDetailView(generics.RetrieveAPIView):
    permission_classes = [IsAuthenticated]
    serializer_class = CallSerializer

    def get_queryset(self):
        if self.request.user.role == 'agent':
            return Call.objects.filter(agent=self.request.user)
        elif self.request.user.role == 'head_of_department':
            subordinates = self.request.user.subordinates.all()
            return Call.objects.filter(agent__in=subordinates)
        return Call.objects.none()


class QueueStatusView(APIView):
    """
    GET /api/calls/queue-status/

    Returns queue depth and today's processing stats scoped to the mock callcenter
    workflow. Used by the Call Center Portal so operators can see live progress.

    Fields:
      awaiting_import  — ExternalCall records not yet imported (analyzed=False).
                         This is the true "pending" count for the mock callcenter page.
      in_queue         — Call records imported from the callcenter but not yet analyzed.
      processing       — Call records currently being analyzed by the Celery worker.
      completed_today  — Calls that finished analysis today (UTC).
      failed_today     — Calls that failed analysis today (UTC).
    """
    permission_classes = [IsAuthenticated]

    def get(self, request):
        from mock_callcenter.models import ExternalCall
        today = timezone.now().date()
        return Response({
            'awaiting_import': ExternalCall.objects.filter(analyzed=False).count(),
            'in_queue':        Call.objects.filter(status='pending', external_source__isnull=False).count(),
            'processing':      Call.objects.filter(status='processing').count(),
            'completed_today': Call.objects.filter(status='completed', updated_at__date=today).count(),
            'failed_today':    Call.objects.filter(status='failed',    updated_at__date=today).count(),
        })
