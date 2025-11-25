from rest_framework import status, generics
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from .models import Company
from .serializers import CompanySerializer, CreateCompanySerializer, AssignHeadToCompanySerializer
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
