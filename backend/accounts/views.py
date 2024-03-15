from django.shortcuts import render

# Create your views here.

from rest_framework import status
from rest_framework.permissions import AllowAny, IsAuthenticated, IsAdminUser
from rest_framework.response import Response
from rest_framework.views import APIView
from accounts.renderers import UserJSONRenderer
from rest_framework.generics import RetrieveUpdateAPIView, UpdateAPIView
from rest_framework.permissions import AllowAny, IsAuthenticated
from .serializers import (
    RegistrationSerializer, LoginSerializer, UserRetrieveSerializer,
    ChangePasswordSerializer, UserSerializer, ResetPasswordSerializer, SystemModuleUpdateSerializer,
    SystemModuleCreateSerializer)
import json
from .backends import  JWTAuthentication
from .signals import user_logged_out
from .models import User, SystemModules
from rest_framework import viewsets, serializers, views
from myutils import sql_server

# from api.views_new import getcitybyid, getcountrybyid, getRegion

RunSqlServerQuery = sql_server.SqlServer.runQuery

class SystemModuleUpdateviewset(viewsets.ModelViewSet):
    queryset = SystemModules.objects.all()
    serializer_class = SystemModuleUpdateSerializer

class SystemModuleCreateviewset(viewsets.ModelViewSet):
    queryset = SystemModules.objects.all()
    serializer_class = SystemModuleCreateSerializer


class RegistrationAPIView(APIView):
    # Allow any user (authenticated or not) to hit this endpoint.
    permission_classes = (IsAuthenticated,)
    # permission_classes = (AllowAny,)
    # renderer_classes = (UserJSONRenderer,)
    serializer_class = RegistrationSerializer

    def post(self, request):
        user = request.data
    
        # The create serializer, validate serializer, save serializer pattern
        # below is common and you will see it a lot throughout this course and
        # your own work later on. Get familiar with it.
        serializer = self.serializer_class(data=user)
        serializer.is_valid(raise_exception=True)
        serializer.save()

        return Response(serializer.data, status=status.HTTP_201_CREATED)



class LoginAPIView(APIView):
    permission_classes = (AllowAny,)
    # renderer_classes = (UserJSONRenderer,)
    serializer_class = LoginSerializer

    def post(self, request):

        user = request.data
        # print("post-data::::::::::::::::::::::::::::::::::::::::::::", user)
        # Notice here that we do not call `serializer.save()` like we did for
        # the registration endpoint. This is because we don't  have
        # anything to save. Instead, the `validate` method on our serializer
        # handles everything we need.
        serializer = self.serializer_class(data=user)
        serializer.is_valid(raise_exception=True)
        # print("details:::::::::::::::::::::::::::::::::::::::::::::::", serializer.data)
        
        a= serializer.data
        if 'route_access' in a:
            a['route_access'] = json.loads(a['route_access'])
        return Response(a, status=status.HTTP_200_OK)



class UserRetrieveUpdateAPIView(RetrieveUpdateAPIView):

    permission_classes = (IsAuthenticated,)
    # renderer_classes = (UserJSONRenderer,)
    serializer_class = UserRetrieveSerializer

    def retrieve(self, request, *args, **kwargs):
        # There is nothing to validate or save here. Instead, we just want the
        # serializer to handle turning our `User` object into something that
        # can be JSONified and sent to the client.
        serializer = self.serializer_class(request.user)
        a = serializer.data
        a['route_access'] = json.loads(a['route_access'])
        return Response(a, status=status.HTTP_200_OK)

    def update(self, request, *args, **kwargs):
        serializer_data = request.data.get('user', {})

        # Here is that serialize, validate, save pattern we talked about
        # before.
        serializer = self.serializer_class(
            request.user, data=serializer_data, partial=True
        )
        serializer.is_valid(raise_exception=True)
        serializer.save()

        return Response(serializer.data, status=status.HTTP_200_OK)


class LogoutView(APIView):
    authentication_classes = (JWTAuthentication,)
    permission_classes = (IsAuthenticated,)

    def post(self, request, format=None):
        request._auth.delete()
        user_logged_out.send(sender=request.user.__class__,
                             request=request, user=request.user)
        return Response(None, status=status.HTTP_204_NO_CONTENT)



class ChangePasswordView(APIView):
        
    """
    An endpoint for changing password.
    """
    serializer_class = ChangePasswordSerializer
    model = User
    permission_classes = (IsAuthenticated,)
    
    def get_object(self, queryset=None):
        obj = self.request.user
        return obj
    
    def post(self, request, *args, **kwargs):
        self.object = self.get_object()
        serializer = self.serializer_class(data=request.data)
        if serializer.is_valid():
            # Check old password
            if not self.object.check_password(serializer.data.get("old_password")):
                return Response({"old_password": ["Wrong password."]}, status=status.HTTP_400_BAD_REQUEST)
            # set_password also hashes the password that the user will get
            self.object.set_password(serializer.data.get("new_password"))
            self.object.is_resetpwd = False
            self.object.save()
            response = {
                'status': 'success',
                'code': status.HTTP_200_OK,
                'message': 'Password updated successfully',
                'data': []
            }
            return Response(response)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class ResetPasswordView(APIView):
        
    """
    An endpoint for changing password.
    """
    serializer_class = ResetPasswordSerializer
    model = User
    permission_classes = (IsAuthenticated,)
    def get_object(self, queryset=None):
        obj = self.request.user.password
        # print("getobj", obj)
        return obj
    def post(self, request, *args, **kwargs):
        self.object = self.get_object()
        serializer = self.serializer_class(data=request.data)
        if serializer.is_valid():
            # Check old password
            # if not self.object.check_password(serializer.data.get("old_password")):
            #     return Response({"old_password": ["Wrong password."]}, status=status.HTTP_400_BAD_REQUEST)
            # set_password also hashes the password that the user will get
            self.object.set_password(serializer.data.get("new_password"))
            self.object.save()
            response = {
                'status': 'success',
                'code': status.HTTP_200_OK,
                'message': 'Password reset successfully',
                'data': []
            }
            return Response(response)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class UserViewSet(viewsets.ModelViewSet):
    """
    API endpoint that allows users to be viewed or edited.
    """
    permission_classes = (IsAuthenticated,)
    queryset = User.objects.all()
    queryset.is_resetpwd = True
    serializer_class = UserSerializer
   

    