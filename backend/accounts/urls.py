from django.urls import path
# from django.contrib.auth.views import LogoutView
from .views import (
    RegistrationAPIView, LoginAPIView, UserRetrieveUpdateAPIView,
    LogoutView, ChangePasswordView, ResetPasswordView
    )

app_name = 'accounts'
urlpatterns = [
    path('register/', RegistrationAPIView.as_view()),
    path('login', LoginAPIView.as_view()),
    path('user', UserRetrieveUpdateAPIView.as_view()),
    path('logout', LogoutView.as_view()),
    path('passwordchange', ChangePasswordView.as_view()),
    path('passwordreset', ResetPasswordView.as_view()),

]