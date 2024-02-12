# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 19:41:29 2024

@author: Hemant
"""

import pyodbc
import sqlalchemy
import urllib.request
import time


# =============================================================================
# Connecting To Database
# =============================================================================
def cnxn(dbName):
    cnxn_str = ("Driver=SQL Server Native Client 11.0;"
                "Server=DESKTOP-4ABRK6A\SQLEXPRESS;"
                f"Database={dbName};"
                "UID=hemantcgupta;"
                "PWD=hemantcgupta")
    return pyodbc.connect(cnxn_str)


# =============================================================================
# Data Inserting into tables
# =============================================================================
def Data_Inserting_Into_DB(df, dbName, Table_Name, insertMethod):
    try:
        params = urllib.parse.quote_plus('''DRIVER=SQL Server Native Client 11.0;
                                            SERVER=DESKTOP-4ABRK6A\SQLEXPRESS;
                                            DATABASE={};
                                            UID=hemantcgupta;
                                            PWD=hemantcgupta'''.format(dbName))   
        engine = sqlalchemy.create_engine('mssql+pyodbc:///?odbc_connect={}'.format(params))
        @sqlalchemy.event.listens_for(engine, 'before_cursor_execute')
        def receive_before_cursor_execute(conn, cursor, statement, params, context, executemany):
            if executemany:
                cursor.fast_executemany = True     
        chunksize = int(len(df)/10)       
        if len(df) <= 1000:
            chunksize = len(df)
        df.to_sql(Table_Name, engine, if_exists = insertMethod, index=False, chunksize = chunksize)
        return {Table_Name: 'successful'}
    except Exception as e:
        return {Table_Name: e}