from rest_framework import status, generics
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from .models import Company
from .serializers import CompanySerializer, CreateCompanySerializer, AssignHeadToCompanySerializer, ManageCompanyKeywordsSerializer
from users.models import MyUser
from users.views import IsAdmin


class CreateCompanyView(APIView):
    permission_classes = [IsAdmin]
    
    def post(self, request):
        serializer = CreateCompanySerializer(data=request.data)
        if serializer.is_valid():
            company = Company.objects.create(
                name=serializer.validated_data['name'],
                industry=serializer.validated_data.get('industry', ''),
                purpose=serializer.validated_data.get('purpose', ''),
                address=serializer.validated_data.get('address', ''),
                phone_number=serializer.validated_data.get('phone_number', '')
            )
            
            return Response({
                'message': 'Company created successfully',
                'company': CompanySerializer(company).data
            }, status=status.HTTP_201_CREATED)
        
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class AssignHeadToCompanyView(APIView):
    permission_classes = [IsAdmin]
    
    def post(self, request, pk):
        try:
            company = Company.objects.get(pk=pk)
        except Company.DoesNotExist:
            return Response({'error': 'Company not found'}, status=status.HTTP_404_NOT_FOUND)
        
        serializer = AssignHeadToCompanySerializer(data=request.data)
        if serializer.is_valid():
            head_id = serializer.validated_data['head_of_department_id']
            head = MyUser.objects.get(id=head_id)
            
            # Assign head to company
            head.company = company
            head.save()
            
            # Auto-assign company to all existing agents reporting to this head
            agents = head.subordinates.filter(role='agent')
            agents.update(company=company)
            
            return Response({
                'message': 'Head of department assigned successfully',
                'company': CompanySerializer(company).data,
                'agents_updated': agents.count()
            }, status=status.HTTP_200_OK)
        
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class AllCompaniesListView(generics.ListAPIView):
    permission_classes = [IsAdmin]
    serializer_class = CompanySerializer
    queryset = Company.objects.all().order_by('-created_at')


class CompanyDetailView(generics.RetrieveUpdateDestroyAPIView):
    permission_classes = [IsAdmin]
    serializer_class = CompanySerializer
    queryset = Company.objects.all()


class CompanyEmployeesListView(APIView):
    permission_classes = [IsAdmin]
    
    def get(self, request, pk):
        try:
            company = Company.objects.get(pk=pk)
        except Company.DoesNotExist:
            return Response({'error': 'Company not found'}, status=status.HTTP_404_NOT_FOUND)
        
        from users.serializers import UserSerializer
        employees = company.employees.all().order_by('role', '-created_at')
        serializer = UserSerializer(employees, many=True)
        
        return Response({
            'company': CompanySerializer(company).data,
            'employees': serializer.data
        }, status=status.HTTP_200_OK)


class ManageCompanyKeywordsView(APIView):
    permission_classes = [IsAuthenticated]
    
    def get(self, request):
        """Get custom keywords for head's company"""
        if request.user.role != 'head_of_department':
            return Response(
                {'error': 'Only heads of department can access this endpoint'},
                status=status.HTTP_403_FORBIDDEN
            )
        
        if not request.user.company:
            return Response(
                {'error': 'You are not assigned to a company'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        return Response({
            'company_id': request.user.company.id,
            'company_name': request.user.company.name,
            'custom_keywords': request.user.company.custom_keywords or {}
        }, status=status.HTTP_200_OK)
    
    def post(self, request):
        """Add or update custom keywords for head's company"""
        if request.user.role != 'head_of_department':
            return Response(
                {'error': 'Only heads of department can manage keywords'},
                status=status.HTTP_403_FORBIDDEN
            )
        
        if not request.user.company:
            return Response(
                {'error': 'You are not assigned to a company'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        serializer = ManageCompanyKeywordsSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        company = request.user.company
        company.custom_keywords = serializer.validated_data['custom_keywords']
        company.save()
        
        return Response({
            'message': 'Custom keywords updated successfully',
            'company_id': company.id,
            'company_name': company.name,
            'custom_keywords': company.custom_keywords
        }, status=status.HTTP_200_OK)
    
    def delete(self, request):
        """Remove custom keywords from head's company"""
        if request.user.role != 'head_of_department':
            return Response(
                {'error': 'Only heads of department can manage keywords'},
                status=status.HTTP_403_FORBIDDEN
            )
        
        if not request.user.company:
            return Response(
                {'error': 'You are not assigned to a company'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        company = request.user.company
        company.custom_keywords = None
        company.save()
        
        return Response({
            'message': 'Custom keywords removed successfully',
            'company_id': company.id,
            'company_name': company.name
        }, status=status.HTTP_200_OK)
