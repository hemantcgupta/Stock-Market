from django.db import models

# Create your models here.
from django.db import models
from django.contrib.auth.models import (
    BaseUserManager, AbstractBaseUser, PermissionsMixin
)
import jwt
from datetime import datetime, timedelta
# from django.core.validators import validate_comma_separated_integer_list
from django.conf import settings    
import json
import enum

class SystemModules(models.Model):
    name = models.CharField(max_length=255, null=False)
    path = models.CharField(max_length=255, null=True)
    create_date = models.DateTimeField(auto_now_add=True)
    is_deleted = models.BooleanField(default=False)
    parentid = models.CharField(null=True, max_length=128)
    module_type = models.CharField(null=True, max_length=16)
    event_code = models.CharField(null=True, max_length=128)
    visibility = models.CharField(max_length=10)  


class UserManager(BaseUserManager):

    def create_user(self, email, username, role, access, route_access, systemmodule, password=None):
        # """
        # Creates and saves a User with the given email and password.
        # """
       
        if not email:
            raise ValueError('Users must have an email address')
        if not username:
            raise ValueError('Users must have an username')
        if not role:
            raise ValueError('Users must have a Role')
        if not access:
            raise ValueError('Users must have a POS Access')
        if not route_access:
            raise ValueError('Users must have a Route Access')
        if not systemmodule:
            raise ValueError('Users must be alloted System Modules')
        
        
        # print(route_access)
        user = self.model(
            email=self.normalize_email(email),
            username=username,
            role=role,
            access=access,
            route_access=route_access,
            systemmodule=systemmodule,
        )
        user.set_password(password)
        user.save(using=self._db)
        return user


    def create_staffuser(self, email, username, role, access, route_access, systemmodule, password):
        """
        Creates and saves a staff user with the given email, password, role and access
        """
        user = self.create_user(
            email = email,
            username = username,
            role = role,
            access = access,
            route_access = route_access,
            systemmodule=systemmodule,
            password=password,
        )
        user.staff = True
        user.save(using=self._db)
        return user

    def create_superuser(self, email, username, role, access, route_access, systemmodule, password):
        """
        Creates and saves a staff user with the given email, password, role and access
        """
        user = self.create_user(
            email = email,
            username = username,
            role = role,
            access = access,
            route_access = route_access,
            systemmodule=systemmodule,
            password=password,
        )
        user.is_staff = True
        user.is_admin = True
        user.save(using=self._db)
        return user


class User(AbstractBaseUser):
    class Meta:
        db_table = 'auth_users'

    email = models.CharField(max_length=255,unique=True)
    username = models.CharField(max_length=255)
    is_active = models.BooleanField(default=True)
    is_staff = models.BooleanField(default=False) # a admin user; non super-user
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    is_admin = models.BooleanField(default=False) # a superuser
    role = models.CharField(max_length=255)
    access = models.CharField(max_length=255)
    route_access = models.CharField(max_length=255)
    is_resetpwd = models.BooleanField(default=True)
    systemmodule = models.CharField(max_length=255, null=True)
    # id = models.IntegerField(primary_key=True)
    
    # notice the absence of a "Password field", that is built in.

    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = ['username', 'role', 'access', 'route_access', 'systemmodule',] # Email & Password are required by default.  'username', 'role', 'access',
    # print(route_access, "")
    objects = UserManager()

    @property
    def token(self):
        """
        Allows us to get a user's token by calling `user.token` instead of
        `user.generate_jwt_token().

        The `@property` decorator above makes this possible. `token` is called
        a "dynamic property".
        """

        # print("at def token", self)
        return self._generate_jwt_token()

    def get_access(self):
        accesslist = self.access.split('#')
        return {
            'regionId' : (accesslist[1] if len(accesslist) >= 2 else '*'),
            'countryId' : (accesslist[2] if len(accesslist) >= 3 else '*'),
            'cityId' : (accesslist[3] if len(accesslist) >= 4 else '*'),
            'commonOD' : (accesslist[4] if len(accesslist) >= 5 else '*')
        }

    def get_route_access(self):
        route_accesslist = json.loads(self.route_access)
        return route_accesslist
    
    def get_systemmodules(self):
        sysmlist = self.systemmodule
        return sysmlist
    
    def get_email(self):
        return self.email
    
    def get_full_name(self):
        # The user is identified by their email address
        return self.email

    def get_short_name(self):
        # The user is identified by their email address
        return self.email
  
    def __str__(self): 
        # print("Pasword ",self.is_active)
        return self.email

    def _generate_jwt_token(self):
        """
        Generates a JSON Web Token that stores this user's ID and has an expiry
        date set to 60 minutes into the future.
        """
        dt = datetime.utcnow() + timedelta(hours=6)
        # print("generate token self:::::::::::::::::::::::::::::::::::::::::", self)
        token = jwt.encode({
            'id': self.pk,
            'exp': dt,
            'username': self.username,
            'role': self.role,
            'access': self.access,
            'route_access':self.route_access,
            'is_resetpwd': self.is_resetpwd,
            # 'systemmodule': self.systemmodule,
        }, settings.SECRET_KEY)

        # print('token=:::::::::::::::::::::::::::::::::::::::::::::',token)
        return token.decode('utf-8')


    @property
    def is_role(self):
        "What is User rsole?"
        return self.role

    
    @property
    def is_access(self):
        "What is User access?"
        return self.access
    

    @property
    def is_username(self):
        "What is Username?"
        return self.username
