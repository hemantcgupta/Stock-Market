# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 20:32:36 2024

@author: Hemant
"""

import subprocess
import os
from datetime import datetime

class SqlDatabase:
    def __init__(self):
        self.server_name = 'localhost\\SQLEXPRESS'
        self.sqlpackage_path = r"C:\Program Files\Microsoft SQL Server\140\DAC\bin\SqlPackage.exe"

    def backup(self, db_name, file_path):
        try:
            command = [
                self.sqlpackage_path,
                "/a:Export",
                f"/ssn:{self.server_name}",
                f"/sdn:{db_name}",
                f"/tf:{file_path}"
            ]
            subprocess.run(command, check=True)
            print("Backup successful.")
        except subprocess.CalledProcessError as e:
            print(f"Backup failed: {e}")

    def restore(self, db_name, file_path):
        try:
            command = [
                self.sqlpackage_path,
                "/a:Import",
                f"/tsn:{self.server_name}",
                f"/tdn:{db_name}",
                f"/sf:{file_path}"
            ]
            subprocess.run(command, check=True)
            print("Restore successful.")
        except subprocess.CalledProcessError as e:
            print(f"Restore failed: {e}")

def DBBACKUP(db_name, file_path):
    database = SqlDatabase()
    db_folder = os.path.join(file_path, db_name)
    os.makedirs(db_folder, exist_ok=True)
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    backup_file_path = os.path.join(db_folder, f"{db_name}_{current_datetime}.bacpac")
    database.backup(db_name, backup_file_path)
    print(f"Database '{db_name}' exported on backup file: {backup_file_path}") 

def DBRESTORE(db_name, file_path):
    database = SqlDatabase()  
    db_folder = os.path.join(file_path, db_name)
    os.makedirs(db_folder, exist_ok=True)
    backup_files = [f for f in os.listdir(db_folder) if f.endswith('.bacpac')]
    if backup_files:
        latest_backup_file = max(backup_files, key=lambda f: os.path.getmtime(os.path.join(db_folder, f)))
        backup_file_path = os.path.join(db_folder, latest_backup_file)
        database.restore(db_name, backup_file_path)
        print(f"Database '{db_name}' imported from the latest backup file: {backup_file_path}") 
    else:
        print("No backup files found in the specified directory.") 
    
# DBBACKUP('mkintervalmaster', './SSMSDB')    
# DBRESTORE('mkintervalmaster', './SSMSDB')