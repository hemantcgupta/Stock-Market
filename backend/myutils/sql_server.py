from django.db import connection
from datetime import datetime
from django.conf import settings
import pandas as pd
import pyodbc
from sqlalchemy import create_engine

class SqlServer: 
    @staticmethod
    def runQuery(query, start, stop): 
        if query == None:
            return "Query can't be empty"
        
        cursor_connection = connection.cursor()
        # df = pd.DataFrame()
        # resp = pd.read_sql(que, connection)
        getConn = cursor_connection.execute(query)
        print("query executed", datetime.now())

        if cursor_connection.description == None:
            return 'Query Executed'
        # print(results)
        field_names = [item[0] for item in cursor_connection.description]
        results = cursor_connection.fetchall()
        # print(results)
        
        fetchdata = []
        for row in results:
            objDict = {}
            for index, value in enumerate(row):
                objDict[field_names[index]] = value
            fetchdata.append(objDict)

        # if stop == True:
        cursor_connection.close()
        return fetchdata

    @staticmethod
    def runProcedureQuery(query):
        if query == None:
            return "Query can't be empty"
        try: 
            cursor_connection = connection.cursor()
            # df = pd.DataFrame()
            # resp = pd.read_sql(que, connection)
            getConn = cursor_connection.execute(query)
            print("query executed", datetime.now())

            # if cursor_connection.description == None:
            #     return 'Query Executed'

            while cursor_connection.nextset():   # NB: This always skips the first resultset
                try:
                    results = cursor_connection.fetchall()
                    break
                except Exception as err:
                    print(err)
                    continue

            field_names = [item[0] for item in cursor_connection.description]

            # print(results)
            
            fetchdata = []
            for row in results:
                objDict = {}
                for index, value in enumerate(row):
                    objDict[field_names[index]] = value
                fetchdata.append(objDict)

            # if stop == True:
            cursor_connection.close()
                # if stop == True:
            return fetchdata
        except Exception as err:
            print(err)

    @staticmethod
    def runUpdateQuery(dataFrame, file_name):
        
        user = settings.DATABASES["default"]["USER"]
        password = settings.DATABASES["default"]["PASSWORD"]
        host = settings.DATABASES["default"]["HOST"]
        port = settings.DATABASES["default"]["PORT"]
        schema = settings.DATABASES["default"]["NAME"]
        options = settings.DATABASES["default"]["OPTIONS"]["driver"]
        options = options[12:14]
        string = 'mssql+pyodbc://{0}:{1}@{2}:{3}/{4}?driver=ODBC+Driver+{5}+for+SQL+Server'.format(user, password, host, port, schema, options)
        # print(options, string)
        engine = create_engine('mssql+pyodbc://{0}:{1}@{2}:{3}/{4}?driver=ODBC+Driver+{5}+for+SQL+Server'.format(user, password, host, port, schema, options))
        # print("hiiiii")
        dataFrame.to_sql(name='{}'.format(file_name), con=engine, if_exists = 'append', index=False)

        return True

