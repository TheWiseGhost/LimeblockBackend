from django.urls import path
from .views import main, create_user, sign_in, user_details, update_frontend, frontend_details, update_backend, backend_details

urlpatterns = [
    path('', main, name='main'),
    path('create_user/', create_user, name='create_user'),
    path('sign_in/', sign_in, name='sign_in'),
    path('user_details/', user_details, name='user_details'),
    path('update_frontend/', update_frontend, name='update_frontend'),
    path('frontend_details/', frontend_details, name='frontend_details'),
    path('update_backend/', update_backend, name='update_backend'),
    path('backend_details/', backend_details, name='backend_details'),
]

