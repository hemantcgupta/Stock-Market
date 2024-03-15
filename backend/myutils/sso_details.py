import base64
import json
import requests

from django.conf import settings

def isJson(text):
    try:
        jsonObject = json.loads(text)
        # print(jsonObject)
    except ValueError as e:
        # print(e)
        return False
    return True

def getSSOUserDetails(agentid, tokenid):
    if tokenid is None or tokenid == '' or agentid is None or agentid == '':
        return None

    restApiKey = settings.SSO_LOGIN_CONFIGURATION["restAPIPassword"]
    credential = base64.b64encode((settings.SSO_LOGIN_CONFIGURATION["resAPIUsername"]+":" + restApiKey).encode('utf-8')).decode('utf-8')
    ssoServiceURL = settings.SSO_LOGIN_CONFIGURATION["ssoUserDetailsURL"] + tokenid

    payload = {
        "Accept": "application/json",
        "Authorization": "Basic " + credential,
        "Cookie": "agentid=" + agentid
    }

    response = requests.post(ssoServiceURL, data=json.dumps(payload), headers=payload, auth=(settings.SSO_LOGIN_CONFIGURATION["resAPIUsername"], restApiKey))
    finalTextResponse = response.text
    # print(finalTextResponse)
    #finalTextResponse = """{"pingone.subject.from.idp":"V102645","subject":"V102645,,Vishal Dharmawat","pingone.saas.id":"a3441c69-0232-4ca9-8e87-1348d24a0359","pingone.subject":"V102645,,Vishal Dharmawat","pingone.authninstant":"1598959858970","pingone.assertion.id":"XUcGuCQhGEJiZLSRPA-28RevldH","pingone.nameid.format":"urn:oasis:names:tc:SAML:1.1:nameid-format:unspecified","pingone.idp.id":"malaysiaairlines.com","pingone.authn.context":"urn:oasis:names:tc:SAML:2.0:ac:classes:unspecified","pingone.nameid.qualifier":"","group_access":["CN=Domain Users,CN=Users,DC=mh,DC=corp,DC=net","CN=ARMS_GROUP_UAT,OU=Applications IDs,OU=MH,DC=mh,DC=corp,DC=net","CN=ARMS_GROUP,OU=Applications IDs,OU=MH,DC=mh,DC=corp,DC=net","CN=GS_ZPA,OU=MH,DC=mh,DC=corp,DC=net"]}"""

    if isJson(finalTextResponse):
        return finalTextResponse
    else:
        return None
