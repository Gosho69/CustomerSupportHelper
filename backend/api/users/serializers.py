from rest_framework import serializers
from .models import MyUser


class UserSerializer(serializers.ModelSerializer):
    reporting_to_username = serializers.CharField(source='reporting_to.username', read_only=True)
    reporting_to_email = serializers.EmailField(source='reporting_to.email', read_only=True)
    company_name = serializers.CharField(source='company.name', read_only=True)
    subordinates_count = serializers.SerializerMethodField()
    
    class Meta:
        model = MyUser
        fields = [
            'id', 'username', 'email', 'first_name', 'last_name', 'phone',
            'role', 'is_active', 'is_staff', 'reporting_to', 
            'reporting_to_username', 'reporting_to_email', 
            'company', 'company_name', 'subordinates_count', 'created_at'
        ]
        read_only_fields = ['id', 'created_at']
    
    def get_subordinates_count(self, obj):
        return obj.subordinates.count()


class LoginSerializer(serializers.Serializer):
    username = serializers.CharField(required=True)
    password = serializers.CharField(required=True, write_only=True)


class CreateAdminSerializer(serializers.Serializer):
    username = serializers.CharField(required=True)
    email = serializers.EmailField(required=True)
    password = serializers.CharField(required=True, write_only=True, min_length=8)
    first_name = serializers.CharField(required=False, allow_blank=True)
    last_name = serializers.CharField(required=False, allow_blank=True)
    
    def validate_username(self, value):
        if MyUser.objects.filter(username=value).exists():
            raise serializers.ValidationError("Username already exists")
        return value
    
    def validate_email(self, value):
        if MyUser.objects.filter(email=value).exists():
            raise serializers.ValidationError("Email already exists")
        return value


class CreateHeadOfDepartmentSerializer(serializers.Serializer):
    username = serializers.CharField(required=True)
    email = serializers.EmailField(required=True)
    password = serializers.CharField(required=True, write_only=True, min_length=8)
    first_name = serializers.CharField(required=False, allow_blank=True)
    last_name = serializers.CharField(required=False, allow_blank=True)
    
    def validate_username(self, value):
        if MyUser.objects.filter(username=value).exists():
            raise serializers.ValidationError("Username already exists")
        return value
    
    def validate_email(self, value):
        if MyUser.objects.filter(email=value).exists():
            raise serializers.ValidationError("Email already exists")
        return value


class CreateAgentSerializer(serializers.Serializer):
    username = serializers.CharField(required=True)
    email = serializers.EmailField(required=True)
    password = serializers.CharField(required=True, write_only=True, min_length=8)
    first_name = serializers.CharField(required=False, allow_blank=True)
    last_name = serializers.CharField(required=False, allow_blank=True)
    
    def validate_username(self, value):
        if MyUser.objects.filter(username=value).exists():
            raise serializers.ValidationError("Username already exists")
        return value
    
    def validate_email(self, value):
        if MyUser.objects.filter(email=value).exists():
            raise serializers.ValidationError("Email already exists")
        return value
