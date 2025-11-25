from django.urls import path
from . import views

urlpatterns = [
    path('create/', views.CreateCompanyView.as_view(), name='create_company'),
    path('all/', views.AllCompaniesListView.as_view(), name='all_companies'),
    path('<int:pk>/', views.CompanyDetailView.as_view(), name='company_detail'),
    path('<int:pk>/assign-head/', views.AssignHeadToCompanyView.as_view(), name='assign_head_to_company'),
    path('<int:pk>/employees/', views.CompanyEmployeesListView.as_view(), name='company_employees'),
]
