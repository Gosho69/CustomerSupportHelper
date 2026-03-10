from django.urls import path
from . import views

urlpatterns = [
    path('login/', views.LoginView.as_view(), name='login'),
    path('logout/', views.LogoutView.as_view(), name='logout'),
    path('token/refresh/', views.CookieTokenRefreshView.as_view(), name='token_refresh'),
    path('me/', views.CurrentUserView.as_view(), name='current_user'),

    path('create-admin/', views.CreateAdminView.as_view(), name='create_admin'),
    path('create-head/', views.CreateHeadOfDepartmentView.as_view(), name='create_head_of_department'),
    path('all/', views.AllUsersListView.as_view(), name='all_users'),
    path('heads/', views.AllHeadsOfDepartmentListView.as_view(), name='all_heads'),

    path('create-agent/', views.CreateAgentView.as_view(), name='create_agent'),
    path('subordinates/', views.SubordinatesListView.as_view(), name='subordinates'),

    path('<int:pk>/', views.UserDetailView.as_view(), name='user_detail'),
]
