import jwt
from cryptography.fernet import Fernet
from django.conf import settings

key = Fernet.generate_key()
cipherSuite = Fernet(key)

def encrypt(text):
    if text is None or text is '':
        return None

    byteText = bytes(text, 'utf-8')
    print(byteText)
    return cipherSuite.encrypt(byteText)

def decrypt(cipherText):
    if cipherText is None or cipherText is '':
        return None
    
    return cipherSuite.decrypt(cipherText)

def generateAuthToken(data): 
    if data is None or data is '':
        return None
    encodedToken = jwt.encode(data, settings.SECRET_KEY, algorithm='HS256')
    return encodedToken

def decodeAuthToken(data): 
    if data is None or data is '':
        return None
    decodedToken = jwt.decode(data, settings.SECRET_KEY, algorithms=['HS256'])
    return decodedToken
