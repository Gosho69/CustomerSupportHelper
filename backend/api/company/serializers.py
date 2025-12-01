from rest_framework import serializers
from .models import Company
from users.models import MyUser


class CompanySerializer(serializers.ModelSerializer):
    head_of_department = serializers.SerializerMethodField()
    employees_count = serializers.SerializerMethodField()
    
    class Meta:
        model = Company
        fields = ['id', 'name', 'industry', 'purpose', 'address', 'phone_number', 'custom_keywords', 'created_at', 'head_of_department', 'employees_count']
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


class ManageCompanyKeywordsSerializer(serializers.Serializer):
    custom_keywords = serializers.JSONField(required=True)
    
    def validate_custom_keywords(self, value):
        if not isinstance(value, dict):
            raise serializers.ValidationError("custom_keywords must be a dictionary")
        
        for topic_name, topic_data in value.items():
            if not isinstance(topic_data, dict):
                raise serializers.ValidationError(f"Topic '{topic_name}' must be a dictionary")
            
            if 'keywords' not in topic_data and 'phrases' not in topic_data:
                raise serializers.ValidationError(f"Topic '{topic_name}' must have at least 'keywords' or 'phrases'")
            
            if 'keywords' in topic_data and not isinstance(topic_data['keywords'], list):
                raise serializers.ValidationError(f"Keywords for topic '{topic_name}' must be a list")
            
            if 'phrases' in topic_data and not isinstance(topic_data['phrases'], list):
                raise serializers.ValidationError(f"Phrases for topic '{topic_name}' must be a list")
            
            if 'weight' in topic_data:
                try:
                    weight = float(topic_data['weight'])
                    if weight < 0 or weight > 2:
                        raise serializers.ValidationError(f"Weight for topic '{topic_name}' must be between 0 and 2")
                except (ValueError, TypeError):
                    raise serializers.ValidationError(f"Weight for topic '{topic_name}' must be a number")
        
        return value
