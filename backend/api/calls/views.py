import os
import sys
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from rest_framework import status, generics
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from django.conf import settings
from .models import Call
from .serializers import CallUploadSerializer, CallSerializer, CallListSerializer
from users.views import IsAdminOrHeadOfDepartment
import tempfile

from AI_modules.orchestrator.orchestrator import analyze_call


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
        
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file.name)[1])
        temp_path = temp_file.name
        
        for chunk in audio_file.chunks():
            temp_file.write(chunk)
        temp_file.close()
        
        local_model_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'AI_modules', 'model', 'final'
        )
        
        custom_keywords = None
        if request.user.company and request.user.company.custom_keywords:
            custom_keywords = request.user.company.custom_keywords
        
        try:
            analysis_results = analyze_call(
                audio_path=temp_path,
                summarization_model=summarization_model,
                whisper_model_size="small",
                device="cpu",
                compute_type="int8",
                local_model_path=local_model_path,
                gpt4_max_tokens=1500,
                custom_keywords=custom_keywords
            )
            
            call = Call.objects.create(
                agent=request.user,
                audio_file=audio_file,
                transcript=analysis_results['transcript'],
                transcript_summary=analysis_results['call_summary'],
                emotional_analysis=analysis_results['emotion_analysis'],
                emotional_summary=analysis_results['emotion_summary'],
                behavioral_analysis=analysis_results['behavioral_analysis'],
                behavioral_summary=analysis_results['behavioral_summary'],
                coaching_tips=analysis_results['coaching_tips'],
                topic_analysis=analysis_results['topic_analysis']
            )
            
            # Extract duration from transcript
            duration = 0
            transcript = analysis_results.get('transcript', {})
            
            # First try: duration_sec field from whisperer
            if 'duration_sec' in transcript:
                duration = int(transcript['duration_sec'])
            # Second try: calculate from utterances
            elif 'utterances' in transcript and transcript['utterances']:
                last_utterance = max(transcript['utterances'], key=lambda u: u.get('end', 0))
                duration = int(last_utterance.get('end', 0))
            # Third try: calculate from segments (legacy support)
            elif 'segments' in transcript and transcript['segments']:
                last_segment = transcript['segments'][-1]
                duration = int(last_segment.get('end', 0))
            
            if duration > 0:
                call.duration = duration
                call.save()
            
            os.unlink(temp_path)
            
            return Response({
                'message': 'Call uploaded and analyzed successfully',
                'call': CallSerializer(call).data
            }, status=status.HTTP_201_CREATED)
            
        except ValueError as e:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            return Response(
                {'error': f'Invalid audio file: {str(e)}'},
                status=status.HTTP_400_BAD_REQUEST
            )
        except FileNotFoundError as e:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            return Response(
                {'error': 'Audio file not found or could not be read'},
                status=status.HTTP_400_BAD_REQUEST
            )
        except Exception as e:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            return Response(
                {'error': f'Analysis failed: {str(e)}'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class AgentCallsListView(generics.ListAPIView):
    permission_classes = [IsAuthenticated]
    serializer_class = CallListSerializer
    
    def get_queryset(self):
        if self.request.user.role == 'agent':
            return Call.objects.filter(agent=self.request.user).order_by('-call_date')
        elif self.request.user.role == 'head_of_department':
            subordinates = self.request.user.subordinates.all()
            return Call.objects.filter(agent__in=subordinates).order_by('-call_date')
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
