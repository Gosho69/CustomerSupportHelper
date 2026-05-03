import os
import logging
import mimetypes
import urllib.parse
from django.utils import timezone
from django.http import FileResponse
from rest_framework import generics, status
from rest_framework.views import APIView
from rest_framework.response import Response

from rest_framework import serializers as drf_serializers

from .models import ExternalCall
from .permissions import ApiKeyPermission
from .serializers import (
    ExternalCallListSerializer,
    ExternalCallUploadSerializer,
    MAX_BATCH_UPLOAD_SIZE,
    validate_single_audio_file,
)

logger = logging.getLogger(__name__)


class ExternalCallUploadView(APIView):
    """
    POST /api/mock-callcenter/calls/upload/

    Simulates a call center operator uploading a new call recording to the
    external platform. The main platform's sync task will discover and import
    this call automatically on its next run.

    Required fields (multipart/form-data):
      - audio_file: the recording (.wav, .mp3, .m4a, .flac, .ogg, .opus; max 100 MB)
      - agent_email: email address of the agent who handled the call
    """
    authentication_classes = []
    permission_classes = [ApiKeyPermission]

    def post(self, request):
        serializer = ExternalCallUploadSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        audio_file = serializer.validated_data['audio_file']
        agent_email = serializer.validated_data['agent_email']

        ext_call = ExternalCall(
            agent_email=agent_email,
            call_started_at=timezone.now(),
        )
        ext_call.audio_file.save(audio_file.name, audio_file, save=False)
        ext_call.save()

        logger.info(
            f"ExternalCallUploadView: new call {ext_call.external_id} "
            f"uploaded for agent {agent_email}"
        )

        return Response(
            {
                'external_id': str(ext_call.external_id),
                'status': 'pending',
                'message': (
                    "Call uploaded to the mock call center. "
                    "The main platform will import and analyze it automatically."
                ),
            },
            status=status.HTTP_201_CREATED,
        )


class ExternalCallBulkUploadView(APIView):
    """
    POST /api/mock-callcenter/calls/bulk-upload/

    Upload multiple call recordings in a single request. Each file is validated
    and stored independently so a single bad file does not reject the whole batch.
    After any successful imports the sync task is triggered immediately (countdown=5s)
    rather than waiting for the next Celery Beat cycle.

    Required fields (multipart/form-data):
      - audio_files: one or more recordings (.wav, .mp3, .m4a, .flac, .ogg, .opus; max 100 MB each)
      - agent_email: email address of the agent who handled all calls in this batch
    """
    authentication_classes = []
    permission_classes = [ApiKeyPermission]

    def post(self, request):
        from calls.integration_tasks import sync_external_calls

        # Validate agent_email manually — DRF's ListField(child=FileField()) does not
        # reliably validate InMemoryUploadedFile objects when passed through a data dict,
        # so we skip the BulkUploadSerializer and validate directly.
        agent_email = (request.data.get('agent_email') or '').strip()
        if not agent_email:
            return Response(
                {'agent_email': ['This field is required.']},
                status=status.HTTP_400_BAD_REQUEST,
            )
        try:
            agent_email = drf_serializers.EmailField().run_validation(agent_email)
        except drf_serializers.ValidationError as exc:
            return Response({'agent_email': exc.detail}, status=status.HTTP_400_BAD_REQUEST)

        raw_files = request.FILES.getlist('audio_files')
        if not raw_files:
            return Response(
                {'audio_files': ['No files provided.']},
                status=status.HTTP_400_BAD_REQUEST,
            )
        if len(raw_files) > MAX_BATCH_UPLOAD_SIZE:
            return Response(
                {'audio_files': [f'Too many files. Maximum is {MAX_BATCH_UPLOAD_SIZE} per request.']},
                status=status.HTTP_400_BAD_REQUEST,
            )
        imported = []
        failed = []

        for f in raw_files:
            try:
                validate_single_audio_file(f)
            except Exception as exc:
                # exc.detail is a list when raised via serializers.ValidationError
                error_msg = exc.detail[0] if hasattr(exc, 'detail') else str(exc)
                failed.append({'filename': f.name, 'error': str(error_msg)})
                continue

            try:
                ext_call = ExternalCall(
                    agent_email=agent_email,
                    call_started_at=timezone.now(),
                )
                ext_call.audio_file.save(f.name, f, save=False)
                ext_call.save()
                imported.append({
                    'external_id': str(ext_call.external_id),
                    'filename': f.name,
                    'status': 'pending',
                })
                logger.info(
                    f"ExternalCallBulkUploadView: created {ext_call.external_id} "
                    f"for agent {agent_email} ({f.name})"
                )
            except Exception as exc:
                logger.error(f"ExternalCallBulkUploadView: DB save failed for {f.name}: {exc}")
                failed.append({'filename': f.name, 'error': 'Internal error saving record.'})

        if imported:
            try:
                sync_external_calls.apply_async(countdown=5)
                logger.info(
                    f"ExternalCallBulkUploadView: triggered sync_external_calls "
                    f"(countdown=5s) after {len(imported)} imports"
                )
            except Exception as exc:
                logger.warning(f"ExternalCallBulkUploadView: could not trigger sync task: {exc}")

        return Response(
            {
                'total': len(raw_files),
                'imported': len(imported),
                'failed': len(failed),
                'calls': imported + failed,
            },
            status=status.HTTP_201_CREATED if imported else status.HTTP_400_BAD_REQUEST,
        )


class ExternalCallListView(generics.ListAPIView):
    """
    GET /api/mock-callcenter/calls/
    GET /api/mock-callcenter/calls/?analyzed=false

    Returns metadata for calls in the mock platform.
    The sync task uses ?analyzed=false to discover calls that need importing.
    """
    authentication_classes = []
    permission_classes = [ApiKeyPermission]
    serializer_class = ExternalCallListSerializer

    def get_queryset(self):
        qs = ExternalCall.objects.all()
        analyzed_param = self.request.query_params.get('analyzed')
        if analyzed_param is not None:
            # 'false' → not analyzed; anything else → analyzed
            qs = qs.filter(analyzed=(analyzed_param.lower() != 'false'))
        agent_email = self.request.query_params.get('agent_email', '').strip()
        if agent_email:
            qs = qs.filter(agent_email__iexact=agent_email)
        return qs


class ExternalCallAudioView(APIView):
    """
    GET /api/mock-callcenter/calls/<external_id>/audio/

    Returns the raw audio file for a given external call.
    Used by the sync task to download audio before importing it.
    """
    authentication_classes = []
    permission_classes = [ApiKeyPermission]

    def get(self, request, external_id):
        try:
            ext_call = ExternalCall.objects.get(external_id=external_id)
        except ExternalCall.DoesNotExist:
            return Response({'detail': 'Not found.'}, status=status.HTTP_404_NOT_FOUND)

        if not ext_call.audio_file:
            return Response({'detail': 'No audio file attached.'}, status=status.HTTP_404_NOT_FOUND)

        file_handle = ext_call.audio_file.open('rb')
        filename = os.path.basename(ext_call.audio_file.name)
        # Determine a sensible content type: prefer storage-provided value
        content_type = None
        try:
            content_type = getattr(ext_call.audio_file.file, 'content_type', None)
        except Exception:
            content_type = None
        if not content_type:
            content_type, _ = mimetypes.guess_type(filename)
        if not content_type:
            content_type = 'application/octet-stream'

        # Sanitize filename: remove control characters and quotes, fallback to safe name
        safe_fname = ''.join(ch for ch in filename if 32 <= ord(ch) <= 126 and ch not in ('"', '\\'))
        if not safe_fname:
            safe_fname = 'download'
        quoted = urllib.parse.quote(safe_fname)

        response = FileResponse(file_handle, content_type=content_type)
        # Provide both plain filename and RFC5987 encoded filename* for broad compatibility
        response['Content-Disposition'] = f"attachment; filename=\"{safe_fname}\"; filename*=UTF-8''{quoted}"
        return response


class ExternalCallMarkAnalyzedView(APIView):
    """
    PATCH /api/mock-callcenter/calls/<external_id>/

    Marks an external call as analyzed (imported by the main platform).
    Called by the sync task after successfully creating a Call record.
    """
    authentication_classes = []
    permission_classes = [ApiKeyPermission]

    def patch(self, request, external_id):
        try:
            ext_call = ExternalCall.objects.get(external_id=external_id)
        except ExternalCall.DoesNotExist:
            return Response({'detail': 'Not found.'}, status=status.HTTP_404_NOT_FOUND)

        ext_call.analyzed = True
        ext_call.imported_at = timezone.now()
        ext_call.save(update_fields=['analyzed', 'imported_at'])

        logger.info(f"ExternalCallMarkAnalyzedView: {external_id} marked as analyzed")
        return Response({'status': 'marked as analyzed', 'imported_at': ext_call.imported_at})
