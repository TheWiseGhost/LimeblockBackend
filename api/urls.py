from django.urls import path
from .views import main, create_user, sign_in, user_details, update_frontend, frontend_details, update_backend, backend_details, process_prompt, public_frontend_details
from .views import create_checkout_session, stripe_webhook, update_emails, get_mau_stats, commit_backend_action, are_maus_remaining, test_endpoint

urlpatterns = [
    path('', main, name='main'),
    path('create_user/', create_user, name='create_user'),
    path('sign_in/', sign_in, name='sign_in'),
    path('user_details/', user_details, name='user_details'),
    path('update_frontend/', update_frontend, name='update_frontend'),
    path('frontend_details/', frontend_details, name='frontend_details'),
    path('public_frontend_details/', public_frontend_details, name='public_frontend_details'),
    path('update_backend/', update_backend, name='update_backend'),
    path('backend_details/', backend_details, name='backend_details'),
    path('process_prompt/', process_prompt, name='process_prompt'),
    path('test_endpoint/', test_endpoint, name='test_endpoint'),
    path("create_checkout_session/", create_checkout_session, name="create_checkout_session"),
    path("stripe/webhook/", stripe_webhook, name="stripe-webhook"),
    path('update_emails/', update_emails, name='update_emails'),
    path('get_mau_stats/', get_mau_stats, name='get_mau_stats'),
    path('commit_backend_action/', commit_backend_action, name='commit_backend_action'),
    path('are_maus_remaining/', are_maus_remaining, name="are_maus_remaining"),
]

