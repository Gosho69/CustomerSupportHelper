from rest_framework import status, generics, permissions
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework_simplejwt.tokens import RefreshToken
from django.contrib.auth import authenticate
from .models import MyUser
from .serializers import (
    UserSerializer, 
    LoginSerializer, 
    CreateAgentSerializer,
    CreateHeadOfDepartmentSerializer,
    CreateAdminSerializer
)


class IsAdmin(permissions.BasePermission):
    def has_permission(self, request, view):
        return request.user and request.user.is_authenticated and request.user.role == 'admin'


class IsHeadOfDepartment(permissions.BasePermission):
    def has_permission(self, request, view):
        return request.user and request.user.is_authenticated and request.user.role == 'head_of_department'


class IsAdminOrHeadOfDepartment(permissions.BasePermission):
    def has_permission(self, request, view):
        return request.user and request.user.is_authenticated and request.user.role in ['admin', 'head_of_department']


class LoginView(APIView):
    permission_classes = [permissions.AllowAny]
    
    def post(self, request):
        serializer = LoginSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        user = authenticate(
            username=serializer.validated_data['username'],
            password=serializer.validated_data['password']
        )
        
        if not user:
            return Response({'error': 'Invalid credentials'}, status=status.HTTP_401_UNAUTHORIZED)
        
        if not user.is_active:
            return Response({'error': 'User account is disabled'}, status=status.HTTP_403_FORBIDDEN)
        
        refresh = RefreshToken.for_user(user)
        return Response({
            'access': str(refresh.access_token),
            'refresh': str(refresh),
            'user': UserSerializer(user).data
        })


class CreateAdminView(APIView):
    permission_classes = [IsAdmin]
    
    def post(self, request):
        serializer = CreateAdminSerializer(data=request.data)
        if serializer.is_valid():
            admin = MyUser.objects.create_user(
                username=serializer.validated_data['username'],
                email=serializer.validated_data['email'],
                password=serializer.validated_data['password'],
                first_name=serializer.validated_data.get('first_name', ''),
                last_name=serializer.validated_data.get('last_name', ''),
                role='admin',
                is_staff=True,
                is_superuser=True
            )
            
            return Response({
                'message': 'Admin created successfully',
                'user': UserSerializer(admin).data
            }, status=status.HTTP_201_CREATED)
        
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class CreateHeadOfDepartmentView(APIView):
    permission_classes = [IsAdmin]
    
    def post(self, request):
        serializer = CreateHeadOfDepartmentSerializer(data=request.data)
        if serializer.is_valid():
            head = MyUser.objects.create_user(
                username=serializer.validated_data['username'],
                email=serializer.validated_data['email'],
                password=serializer.validated_data['password'],
                first_name=serializer.validated_data.get('first_name', ''),
                last_name=serializer.validated_data.get('last_name', ''),
                role='head_of_department'
            )
            
            return Response({
                'message': 'Head of Department created successfully',
                'user': UserSerializer(head).data
            }, status=status.HTTP_201_CREATED)
        
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class CreateAgentView(APIView):
    permission_classes = [IsHeadOfDepartment]
    
    def post(self, request):
        serializer = CreateAgentSerializer(data=request.data)
        if serializer.is_valid():
            agent = MyUser.objects.create_user(
                username=serializer.validated_data['username'],
                email=serializer.validated_data['email'],
                password=serializer.validated_data['password'],
                first_name=serializer.validated_data.get('first_name', ''),
                last_name=serializer.validated_data.get('last_name', ''),
                role='agent',
                reporting_to=request.user,
                company=request.user.company
            )
            
            return Response({
                'message': 'Agent created successfully',
                'agent': UserSerializer(agent).data
            }, status=status.HTTP_201_CREATED)
        
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class SubordinatesListView(generics.ListAPIView):
    permission_classes = [IsHeadOfDepartment]
    serializer_class = UserSerializer
    
    def get_queryset(self):
        return self.request.user.subordinates.all()


class AllUsersListView(generics.ListAPIView):
    permission_classes = [IsAdmin]
    serializer_class = UserSerializer
    queryset = MyUser.objects.all()
    
    def get_queryset(self):
        queryset = MyUser.objects.all()
        role = self.request.query_params.get('role', None)
        if role:
            queryset = queryset.filter(role=role)
        return queryset.order_by('-created_at')


class AllHeadsOfDepartmentListView(generics.ListAPIView):
    permission_classes = [IsAdmin]
    serializer_class = UserSerializer
    
    def get_queryset(self):
        return MyUser.objects.filter(role='head_of_department').order_by('-created_at')


class CurrentUserView(APIView):
    permission_classes = [permissions.IsAuthenticated]
    
    def get(self, request):
        serializer = UserSerializer(request.user)
        return Response(serializer.data, status=status.HTTP_200_OK)


class UserDetailView(generics.RetrieveUpdateDestroyAPIView):
    permission_classes = [IsAdminOrHeadOfDepartment]
    serializer_class = UserSerializer
    queryset = MyUser.objects.all()
    
    def get_queryset(self):
        user = self.request.user
        if user.role == 'admin':
            # Admin can access all users across all companies
            return MyUser.objects.all()
        elif user.role == 'head_of_department':
            # Head of department can only manage their subordinates
            return MyUser.objects.filter(reporting_to=user)
        return MyUser.objects.none()
