# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 19:41:29 2024

@author: Hemant
"""

import pyodbc
import sqlalchemy
import urllib.request
import time
import os
from functools import lru_cache

# =============================================================================
# Connecting To Database
# =============================================================================
# def cnxn(dbName):
#     cnxn_str = ("Driver=ODBC Driver 17 for SQL Server;"
#                 "Server=HEMANT\SQLEXPRESS;"
#                 f"Database={dbName};"
#                 "UID=hemantcgupta;"
#                 "PWD=hemantcgupta")
#     return pyodbc.connect(cnxn_str)

@lru_cache(maxsize=None, typed=False)
def cnxn(dbName):
    cnxn_str = (
        "Driver={ODBC Driver 18 for SQL Server};"
        "Server=localhost,1433;"
        f"Database={dbName};"
        "UID=hemantcgupta;"
        "PWD=Bluebird951@;"
        "Encrypt=no;"   # useful for local non-certified server
        "TrustServerCertificate=yes;"
    )
    return pyodbc.connect(cnxn_str)




# =============================================================================
# Creating Database if not exist
# =============================================================================
def create_database(dbName):
    cnxn_str = (
        "Driver={ODBC Driver 18 for SQL Server};"
        "Server=localhost,1433;"
        f"Database={dbName};"
        "UID=hemantcgupta;"
        "PWD=Bluebird951@;"
        "Encrypt=no;"   # useful for local non-certified server
        "TrustServerCertificate=yes;"
    )
    cnxn = pyodbc.connect(cnxn_str)
    cursor = cnxn.cursor()
    cnxn.autocommit = True
    query = f"IF DB_ID('{dbName}') IS NULL CREATE DATABASE {dbName};"
    cursor.execute(query)
 
   

# =============================================================================
# Data Inserting into tables
# =============================================================================
@lru_cache(maxsize=None, typed=False)
def get_engine(dbName):
    params = urllib.parse.quote_plus('''DRIVER=ODBC Driver 18 for SQL Server;
                                        SERVER=localhost,1433;
                                        DATABASE={};
                                        UID=hemantcgupta;
                                        PWD=Bluebird951@;
                                        Encrypt=no;
                                        TrustServerCertificate=yes'''.format(dbName))   
    engine = sqlalchemy.create_engine('mssql+pyodbc:///?odbc_connect={}'.format(params))
    return engine

def Data_Inserting_Into_DB(df, dbName, Table_Name, insertMethod):
    try:
        
        engine = get_engine(dbName)
        @sqlalchemy.event.listens_for(engine, 'before_cursor_execute')
        def receive_before_cursor_execute(conn, cursor, statement, params, context, executemany):
            if executemany:
                cursor.fast_executemany = True  
        start = time.time()
        chunksize = int(len(df)/10)       
        if len(df) <= 1000:
            chunksize = len(df)
        df.to_sql(Table_Name, engine, if_exists = insertMethod, index=False, chunksize = chunksize)
        return {'dbName': dbName, Table_Name: 'Successful'}
    except Exception as e:
        print(f"Data Insert Into DB Unsuccessful In: {round(time.time()-start)} Sec; Error: {str(e)}")
        return {'dbName': dbName, Table_Name: e}