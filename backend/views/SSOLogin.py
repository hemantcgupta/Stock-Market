import base64
import json
import os

from django.conf import settings
from django.shortcuts import redirect
from rest_framework import status
from rest_framework import generics
from rest_framework.response import Response
from datetime import datetime
from dateutil.relativedelta import relativedelta
from datetime import datetime, timedelta, timezone

from myutils import sql_server
from myutils import encryption
from myutils import sso_details

RunSqlServerQuery = sql_server.SqlServer.runQuery

CERT_FILE = os.path.join( os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cacert.pem')


class InitiateLogin (generics.GenericAPIView):
    def get(self, request):
        idpid = settings.SSO_LOGIN_CONFIGURATION["idpid"]
        saasid = settings.SSO_LOGIN_CONFIGURATION["saasid"]
        ssoUrl = "{0}?saasid={1}&idpid={2}".format(settings.SSO_LOGIN_CONFIGURATION["ssoUserInitURL"], saasid, idpid)       
        return redirect(ssoUrl)

class InitiateLogout (generics.GenericAPIView):
    def get(self, request):
        return redirect(settings.SSO_LOGIN_CONFIGURATION["ssoLogoutURL"])

class InvalidLogin (generics.GenericAPIView):
    def get(self, request):
        errorCode = request.GET.get('errorCode', "")
        message = request.GET.get('message', "")
        # print(message, "before redirection")
        redirectURL = "{0}?error={1}".format(settings.SSO_LOGIN_CONFIGURATION["frontendDashboardURL"] , message)
        return redirect(redirectURL)
        
class SuccessLogin (generics.GenericAPIView):
    def post(self, request):
        # print("before try")
        try:
            tokenid = request.POST.get('tokenid', "")
            agentid = request.POST.get('agentid', "")
            
            finalTextResponse = sso_details.getSSOUserDetails(agentid, tokenid)
            # print("finalTextResponse", finalTextResponse)

            if finalTextResponse is not None:
                attributes = json.loads(finalTextResponse)
                armsGroup = attributes['group_access']
                # print("Hello checking access", armsGroup)
                n = 0
                for item in armsGroup:
                    subItem = item.split(",")
                    # print(subItem)
                    if "CN=ARMS_GROUP" in subItem:
                        # print("Yes you are a valid user------------------------------>")
                        n = 1
                if n != 1:
                    return redirect(settings.SSO_LOGIN_CONFIGURATION["frontendDashboardURL"])

            if finalTextResponse is not None:
                attributes = json.loads(finalTextResponse)
                userID = attributes['pingone.subject.from.idp']
                userName = attributes['pingone.subject']
                email = attributes['email_address']
                userNameWithIdArray = userName.split(',')

                if len(userNameWithIdArray) == 2:
                    userName = userNameWithIdArray[1]
                else:
                    userName = userNameWithIdArray[2]
                # print(userName, "-------------------------------------------> nusername")

                queryforAllusers = """Select * from auth_users Where email = '{0}'""".format(email)
                responseQueryforAllUsers = RunSqlServerQuery(queryforAllusers, True, True)
                # print("Query",queryforAllusers,"\n Response", responseQueryforAllUsers)
                systemModules = settings.DEFAULT_SYSTEM_MODULES["default"]
                if len(responseQueryforAllUsers) < 1 :
                    insertQuery = """INSERT INTO dbo.auth_users (email, password, username, is_active, is_staff, created_at, updated_at, is_admin, role, access, is_resetpwd, systemmodule, route_access)
                    VALUES ('{0}', '{1}', '{2}', '{3}', '{4}', {5}, {6}, '{7}', '{8}', '{9}', '{10}', '{11}', '{12}' ) """.format(email, tokenid, userName, 1, 1, 'CURRENT_TIMESTAMP', 'CURRENT_TIMESTAMP', 0, 'Super Admin', '#*', 0, systemModules, {})
                    # print(insertQuery)
                    insertUserinDB = RunSqlServerQuery(insertQuery, True, True)
                elif len(responseQueryforAllUsers) == 1 :
                    updateQuery = """ UPDATE auth_users SET password = '{}' WHERE email = '{}'""".format(tokenid, email)
                    # print(updateQuery)
                    updateUserinDB = RunSqlServerQuery(updateQuery, True, True)
                
                systemmodulesQuery = """ SELECT systemmodule, id FROM auth_users WHERE email = '{}'""".format(email)
                # print(systemmodulesQuery)
                systemmodules = RunSqlServerQuery(systemmodulesQuery, True, True)

                
                
                modules = systemmodules[0]['systemmodule']
                modules = modules.split(",")         

                frontEndDashboardName = ''
                if '10' in modules:
                    frontEndDashboardName = 'posDashboard'
                elif '11' in modules:
                    frontEndDashboardName = 'routeDashboard'
                elif '12' in modules:
                    frontEndDashboardName = 'decisionMakingDashboard'

                finalTokenDict = {
                    "tokenID": tokenid,
                    "email": email,
                    "userID": userID,
                    "agentid": agentid,
                    "id": systemmodules[0]['id'],
                    "exp": datetime.utcnow() + timedelta(seconds=1800)
                }
                encryptedToken = encryption.generateAuthToken(finalTokenDict)

                # print(encryptedToken)

                if encryptedToken is not None:
                    encryptedToken = encryptedToken.decode("utf-8") 

                redirectURL = "{0}/proceed?whereto={1}&token={2}".format(settings.SSO_LOGIN_CONFIGURATION["frontendDashboardURL"] , frontEndDashboardName, encryptedToken)
                return redirect(redirectURL)
            else:
                redirectURL = "{0}?error={1}".format(settings.SSO_LOGIN_CONFIGURATION["frontendDashboardURL"] , "Something went wrong please try after sometime.")
                return redirect(redirectURL)

        except Exception as err:
            # print(err)
            redirectURL = "{0}?error={1}".format(settings.SSO_LOGIN_CONFIGURATION["frontendDashboardURL"] , "Something went wrong please try after sometime.")
            return redirect(redirectURL)


