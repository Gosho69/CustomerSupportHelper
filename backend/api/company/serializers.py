from rest_framework import serializers
from .models import Company
from users.models import MyUser


class CompanySerializer(serializers.ModelSerializer):
    head_of_department = serializers.SerializerMethodField()
    employees_count = serializers.SerializerMethodField()
    
    class Meta:
        model = Company
        fields = ['id', 'name', 'industry', 'purpose', 'address', 'phone_number', 'created_at', 'head_of_department', 'employees_count']
        read_only_fields = ['id', 'created_at']
    
    def get_head_of_department(self, obj):
        head = obj.employees.filter(role='head_of_department').first()
        if head:
            return {
                'id': head.id,
                'username': head.username,
                'email': head.email,
                'first_name': head.first_name,
                'last_name': head.last_name
            }
        return None
    
    def get_employees_count(self, obj):
        return obj.employees.count()


class CreateCompanySerializer(serializers.Serializer):
    name = serializers.CharField(required=True, max_length=255)
    industry = serializers.CharField(required=False, allow_blank=True)
    purpose = serializers.CharField(required=False, allow_blank=True)
    address = serializers.CharField(required=False, allow_blank=True)
    phone_number = serializers.CharField(required=False, allow_blank=True, max_length=20)
    
    def validate_name(self, value):
        if Company.objects.filter(name=value).exists():
            raise serializers.ValidationError("Company with this name already exists")
        return value


class AssignHeadToCompanySerializer(serializers.Serializer):
    head_of_department_id = serializers.IntegerField(required=True)
    
    def validate_head_of_department_id(self, value):
        try:
            user = MyUser.objects.get(id=value)
            if user.role != 'head_of_department':
                raise serializers.ValidationError("User must be a head of department")
            if user.company is not None:
                raise serializers.ValidationError("This head of department is already assigned to a company")
            return value
        except MyUser.DoesNotExist:
            raise serializers.ValidationError("User not found")
