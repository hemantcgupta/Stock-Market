# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 16:54:28 2024

@author: Hemant
"""

import pyodbc
import pandas as pd
from tqdm import tqdm

# =============================================================================
# Connecting To Database
# =============================================================================
def cnxn(dbName):
    cnxn_str = ("Driver=ODBC Driver 17 for SQL Server;"
                "Server=DESKTOP-4ABRK6A\SQLEXPRESS;"
                f"Database={dbName};"
                "UID=hemantcgupta;"
                "PWD=hemantcgupta")
    return pyodbc.connect(cnxn_str)


query = "SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE = 'BASE TABLE'"
tables = pd.read_sql(query, cnxn('mkintervalmaster'))['TABLE_NAME'].tolist()


for table_name in tqdm(tables):
    query = f"""
    WITH CTE AS (
        SELECT 
            Datetime,
            ROW_NUMBER() OVER (PARTITION BY Datetime ORDER BY (SELECT NULL)) AS row_num
        FROM mkintervalmaster.dbo.[{table_name}]
    )
    DELETE FROM CTE WHERE row_num > 1
    """
    try:
        conn = cnxn('master')
        cursor = conn.cursor()
        cursor.execute(query)
        conn.commit()
        print(f"Duplicates removed from table: {table_name}")
    except Exception as e:
        print(f"Failed to remove duplicates from table: {table_name}. Error: {str(e)}")

cursor.close()
conn.close()
