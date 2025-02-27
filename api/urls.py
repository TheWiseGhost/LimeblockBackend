from django.urls import path
from .views import main, create_user, sign_in

urlpatterns = [
    path('', main, name='main'),
    path('create_user/', create_user, name='create_user'),
    path('sign_in/', sign_in, name='create_user'),
]

