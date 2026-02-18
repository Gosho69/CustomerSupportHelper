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
                phone=serializer.validated_data.get('phone', ''),
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
                phone=serializer.validated_data.get('phone', ''),
                role='head_of_department'
            )
            
            return Response({
                'message': 'Head of Department created successfully',
                'user': UserSerializer(head).data
            }, status=status.HTTP_201_CREATED)
        
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class CreateAgentView(APIView):
    permission_classes = [IsAdminOrHeadOfDepartment]
    
    def post(self, request):
        serializer = CreateAgentSerializer(data=request.data)
        if serializer.is_valid():
            # Determine reporting_to: admin provides it explicitly, HoD defaults to self
            reporting_to_id = request.data.get('reporting_to')
            if reporting_to_id:
                try:
                    reporting_to = MyUser.objects.get(id=reporting_to_id, role='head_of_department')
                except MyUser.DoesNotExist:
                    return Response({'error': 'Head of department not found'}, status=status.HTTP_400_BAD_REQUEST)
            elif request.user.role == 'head_of_department':
                reporting_to = request.user
            else:
                return Response({'error': 'reporting_to is required when admin creates an agent'}, status=status.HTTP_400_BAD_REQUEST)
            
            # Determine company: use explicit value, or fall back to the manager's company
            company_id = request.data.get('company')
            company = None
            if company_id:
                from company.models import Company
                try:
                    company = Company.objects.get(id=company_id)
                except Company.DoesNotExist:
                    return Response({'error': 'Company not found'}, status=status.HTTP_400_BAD_REQUEST)
            elif reporting_to and reporting_to.company:
                company = reporting_to.company
            
            agent = MyUser.objects.create_user(
                username=serializer.validated_data['username'],
                email=serializer.validated_data['email'],
                password=serializer.validated_data['password'],
                first_name=serializer.validated_data.get('first_name', ''),
                last_name=serializer.validated_data.get('last_name', ''),
                phone=serializer.validated_data.get('phone', ''),
                role='agent',
                reporting_to=reporting_to,
                company=company
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
    
    def patch(self, request):
        allowed_fields = ['first_name', 'last_name', 'email', 'phone']
        update_data = {k: v for k, v in request.data.items() if k in allowed_fields}
        serializer = UserSerializer(request.user, data=update_data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


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
