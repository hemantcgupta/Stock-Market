# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 18:30:08 2024

@author: Hemant
"""

# import pyodbc
# import pandas as pd
# import os
# import zipfile
# from tqdm import tqdm

# class DatabaseExporter:
#     def __init__(self, dbName):
#         self.dbName = dbName
#         self.connection = self._connect_db()

#     def _connect_db(self):
#         cnxn_str = ("Driver=ODBC Driver 17 for SQL Server;"
#                     "Server=DESKTOP-4ABRK6A\\SQLEXPRESS;"
#                     f"Database={self.dbName};"
#                     "UID=hemantcgupta;"
#                     "PWD=hemantcgupta")
#         return pyodbc.connect(cnxn_str)

#     def export_table_to_excel(self, table_name, excel_file):
#         query = f"SELECT * FROM [{table_name}]"
#         df = pd.read_sql(query, self.connection)
#         df.to_excel(excel_file, index=False)

#     def export_all_tables_to_excel_and_zip(self, output_dir):
#         if not os.path.exists(output_dir):
#             os.makedirs(output_dir)
        
#         cursor = self.connection.cursor()
#         cursor.execute("SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE = 'BASE TABLE'")
#         tables = cursor.fetchall()
        
#         for table in tqdm(tables):
#             table_name = table.TABLE_NAME
#             excel_file = os.path.join(output_dir, f"{table_name}.xlsx")
#             self.export_table_to_excel(table_name, excel_file)
        
#         zip_file = os.path.join(output_dir, "database_tables.zip")
#         with zipfile.ZipFile(zip_file, 'w') as zipf:
#             for root, dirs, files in os.walk(output_dir):
#                 for file in files:
#                     zipf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), output_dir))

# # Example usage:
# if __name__ == "__main__":
#     dbName = "stockmarketinterval"
#     output_dir = r"C:\Users\heman\Desktop\Stock-Market\Jobs\SSMS-DB\mkintervalmaster"
#     exporter = DatabaseExporter(dbName)
#     exporter.export_all_tables_to_excel_and_zip(output_dir)



# =============================================================================
# 
# =============================================================================
# import pyodbc
# import pandas as pd
# import os
# import zipfile
# import shutil
# from tqdm import tqdm
# from multiprocessing import Pool, cpu_count
# import warnings
# warnings.filterwarnings('ignore')

# class DatabaseExporter:
#     def __init__(self, db_name):
#         self.db_name = db_name

#     def _connect_db(self, db_name):
#         cnxn_str = ("Driver=ODBC Driver 17 for SQL Server;"
#                     "Server=DESKTOP-4ABRK6A\\SQLEXPRESS;"
#                     f"Database={db_name};"
#                     "UID=hemantcgupta;"
#                     "PWD=hemantcgupta")
#         return pyodbc.connect(cnxn_str)

#     def export_table_to_excel(self, connection, table_name, excel_file):
#         query = f"SELECT * FROM [{table_name}]"
#         df = pd.read_sql(query, connection)
#         df.to_excel(excel_file, index=False)

#     def export_db_tables_to_excel_and_zip(self, output_dir):
#         db_folder = os.path.join(output_dir, db_name)
#         os.makedirs(db_folder, exist_ok=True)
        
#         connection = self._connect_db(db_name)
#         cursor = connection.cursor()
#         cursor.execute("SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE = 'BASE TABLE'")
#         tables = cursor.fetchall()
        
#         for table in tqdm(tables):
#             table_name = table.TABLE_NAME
#             excel_file = os.path.join(db_folder, f"{table_name}.xlsx")
#             self.export_table_to_excel(connection, table_name, excel_file)
        
#         zip_file = os.path.join(output_dir, f"{db_name}.zip")
#         with zipfile.ZipFile(zip_file, 'w') as zipf:
#             for root, dirs, files in os.walk(db_folder):
#                 for file in files:
#                     zipf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), db_folder))
        
#         shutil.rmtree(db_folder)
            


# db_name = "stockmarketinterval"
# output_dir =  r"C:\Users\heman\Desktop\Stock-Market\Jobs\SSMS-DB\mkintervalmaster"
# exporter = DatabaseExporter(db_name)
# exporter.export_db_tables_to_excel_and_zip(output_dir)











# =============================================================================
# 
# =============================================================================
# import pyodbc
# import pandas as pd
# import os
# import zipfile
# import shutil
# from multiprocessing import Pool, cpu_count
# from tqdm import tqdm

# class DatabaseExporter:
#     def __init__(self, db_names):
#         self.db_names = db_names
#         self.cpu_count = cpu_count()

#     def _connect_db(self, db_name):
#         cnxn_str = ("Driver=ODBC Driver 17 for SQL Server;"
#                     "Server=DESKTOP-4ABRK6A\\SQLEXPRESS;"
#                     f"Database={db_name};"
#                     "UID=hemantcgupta;"
#                     "PWD=hemantcgupta")
#         return pyodbc.connect(cnxn_str)

#     def export_table_to_excel(self, args):
#         table_name, connection, excel_file = args
#         query = f"SELECT * FROM [{table_name}]"
#         df = pd.read_sql(query, connection)
#         df.to_excel(excel_file, index=False)

#     def _export_db_tables_to_excel(self, args):
#         db_name, output_dir = args
#         db_folder = os.path.join(output_dir, db_name)
#         os.makedirs(db_folder, exist_ok=True)
        
#         connection = self._connect_db(db_name)
#         cursor = connection.cursor()
#         cursor.execute("SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE = 'BASE TABLE'")
#         tables = cursor.fetchall()
        
#         for table in tables:
#             table_name = table.TABLE_NAME
#             excel_file = os.path.join(db_folder, f"{table_name}.xlsx")
#             self.export_table_to_excel((table_name, connection, excel_file))
        
#         return db_folder

#     def export_all_dbs_tables_to_excel(self, output_dir):
#         with Pool(processes=self.cpu_count) as pool:
#             db_folders = list(tqdm(pool.imap_unordered(self._export_db_tables_to_excel, [(db_name, output_dir) for db_name in self.db_names]), total=len(self.db_names)))
#         return db_folders

#     def _zip_db_folder(self, db_folder):
#         zip_file = db_folder + '.zip'
#         with zipfile.ZipFile(zip_file, 'w') as zipf:
#             for root, dirs, files in os.walk(db_folder):
#                 for file in files:
#                     zipf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), db_folder))
#         return zip_file

#     def zip_all_db_folders(self, db_folders):
#         with Pool(processes=self.cpu_count) as pool:
#             zip_files = list(tqdm(pool.imap_unordered(self._zip_db_folder, db_folders), total=len(db_folders)))
#         return zip_files


# if __name__ == "__main__":
#     db_names = ["stockmarketinterval"]  # List of database names
#     output_dir = r"C:\Users\heman\Desktop\Stock-Market\Jobs\SSMS-DB\mkintervalmaster"
#     exporter = DatabaseExporter(db_names)
    
#     # Export tables to Excel
#     db_folders = exporter.export_all_dbs_tables_to_excel(output_dir)
    
#     # Zip exported folders
#     zip_files = exporter.zip_all_db_folders(db_folders)

    

# =============================================================================
# 
# =============================================================================
import pandas as pd
import os
import zipfile
import io
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import warnings
warnings.filterwarnings('ignore')
from Scripts.dbConnection import cnxn

class DatabaseExporter:
    def __init__(self, db_name):
        self.db_name = db_name
        self.cpu_count = int(cpu_count() * 0.8)
        
    def export_table_to_excel(self, args):
        table_name, excel_file = args
        query = f"SELECT * FROM [{table_name}]"
        df = pd.read_sql(query, cnxn(self.db_name))
        return {excel_file: df}
    
    def write_excel_to_zip(self, dct):
        for file_path, df in dct.items():
            excel_buffer = io.BytesIO()
            df.to_excel(excel_buffer, index=False)
            zip_file.writestr(os.path.basename(file_path), excel_buffer.getvalue())
        

    def export_db_tables_to_excel_and_zip(self, output_dir):
        db_folder = os.path.join(output_dir, self.db_name)
        os.makedirs(db_folder, exist_ok=True)
        query = "SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE = 'BASE TABLE'"
        stockSymbols = pd.read_sql(query, cnxn(self.db_name))['TABLE_NAME'].tolist()
        args_list = [(tableName, os.path.join(db_folder, f"{tableName}.xlsx")) for tableName in stockSymbols]
        with Pool(processes=self.cpu_count) as pool:
            result = list(tqdm(pool.imap(self.export_table_to_excel, args_list), total=len(args_list)))
        # return result
        with io.BytesIO() as zip_buffer:
            with zipfile.ZipFile(zip_buffer, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zip_file:
                with Pool(processes=self.cpu_count) as pool:
                    result = list(tqdm(pool.imap(self.write_excel_to_zip, result), total=len(args_list)))
                for dct in tqdm(result, desc="Writing to Zip"):
                    for file_path, df in dct.items():
                        excel_buffer = io.BytesIO()
                        df.to_excel(excel_buffer, index=False)
                        zip_file.writestr(os.path.basename(file_path), excel_buffer.getvalue())
            with open('output.zip', 'wb') as f:
                f.write(zip_buffer.getvalue())
        
        # with io.BytesIO() as zip_buffer:
        #     with zipfile.ZipFile(zip_buffer, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zip_file:
        #         for dct in tqdm(result, desc="Writing to Zip"):
        #             for file_path, df in dct.items():
        #                 excel_buffer = io.BytesIO()
        #                 df.to_excel(excel_buffer, index=False)
        #                 zip_file.writestr(os.path.basename(file_path), excel_buffer.getvalue())
        #     with open('output.zip', 'wb') as f:
        #         f.write(zip_buffer.getvalue())
        # def write_excel_to_zip(args):
        #     dct, file_path = args.key
        #     excel_buffer = io.BytesIO()
        #     df = list(dct.values())[0]  # Assuming there's only one key-value pair in each dictionary
        #     df.to_excel(excel_buffer, index=False)
        #     return os.path.basename(file_path), excel_buffer.getvalue()
        # with Pool(cpu_count()) as pool:
        #     zip_data = write_zip(result)
        # with io.BytesIO() as zip_buffer:
        #     with zipfile.ZipFile(zip_buffer, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zip_file:
        #         for file_path, data in tqdm(pool.imap_unordered(write_excel_to_zip, result), total=len(result), desc="Writing to Zip"):
        #             zip_file.writestr(file_path, data)
        #     return zip_buffer.getvalue()     
                
                
if __name__ == "__main__":
    db_name = "stockmarketinterval"
    output_dir =  r"C:\Users\heman\Desktop\Stock-Market\Jobs\SSMS-DB\mkintervalmaster"
    exporter = DatabaseExporter(db_name)
    result = exporter.export_db_tables_to_excel_and_zip(output_dir)

