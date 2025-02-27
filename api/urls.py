from django.urls import path
from .views import main, create_user

urlpatterns = [
    path('', main, name='main'),
    path('create_user/', create_user, name='create_user'),
]

