from django.urls import path
from .views import predict_role_view

urlpatterns = [
    path('', predict_role_view, name='predict_role'),
]