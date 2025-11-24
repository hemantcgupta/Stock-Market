# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 20:32:36 2024

@author: Hemant
"""

import subprocess
import shutil
import os
from datetime import datetime


# =============================================================================
# MacOs With Docker --Platform Linux
# =============================================================================


class SqlDatabase:
    def __init__(
        self,
        container: str = "sql2022",
        docker_path: str = '/usr/local/bin/docker',
        sqlpackage_path: str = "/opt/sqlpackage/sqlpackage",
        server_name: str = "localhost,1433",
        username: str = "hemantcgupta",
        password: str = "Bluebird951@",
    ):
    
        self.docker = docker_path or shutil.which("docker")
        if not self.docker:
            raise FileNotFoundError(
                "docker binary not found in PATH. Install Docker or pass docker_path explicitly."
            )

        self.container = container
        self.sqlpackage_path = sqlpackage_path
        self.server_name = server_name
        self.username = username
        self.password = password 

        if not self.password:
            raise ValueError(
                "SA password not provided. Pass password to SqlDatabase(...) or set MSSQL_SA_PASSWORD env var."
            )

        if not self._is_container_running():
            raise RuntimeError(
                f"Container '{self.container}' is not running. Start it before using this class."
            )

        if not self._sqlpackage_exists_in_container():
            raise RuntimeError(
                f"sqlpackage not found at '{self.sqlpackage_path}' inside container '{self.container}'.\n"
                "Install sqlpackage inside the container (see README)."
            )

    def _run_cmd(self, cmd, capture_output=True):
        print("Running:", " ".join(cmd))
        try:
            completed = subprocess.run(
                cmd,
                check=True,
                capture_output=capture_output,
                text=True,
            )
            if completed.stdout:
                print("STDOUT:", completed.stdout.strip())
            if completed.stderr:
                print("STDERR:", completed.stderr.strip())
            return completed
        except subprocess.CalledProcessError as ex:
            msg = (
                f"Command failed (rc={ex.returncode}): {' '.join(cmd)}\n"
                f"stdout:\n{ex.stdout}\n\nstderr:\n{ex.stderr}"
            )
            raise RuntimeError(msg)

    def _is_container_running(self) -> bool:
        cmd = [self.docker, "inspect", "-f", "{{.State.Running}}", self.container]
        try:
            res = self._run_cmd(cmd)
            return res.stdout.strip().lower() == "true"
        except Exception:
            return False

    def _sqlpackage_exists_in_container(self) -> bool:
        cmd = [
            self.docker,
            "exec",
            self.container,
            "bash",
            "-lc",
            f"test -x {self.sqlpackage_path} && echo OK || echo MISSING",
        ]
        try:
            res = self._run_cmd(cmd)
            return res.stdout.strip() == "OK"
        except Exception:
            return False

    def backup(self, db_name: str, backup_local_path: str):
        os.makedirs(os.path.dirname(os.path.abspath(backup_local_path)), exist_ok=True)
        container_path = f"/tmp/{os.path.basename(backup_local_path)}"
        cmd_export = [
            self.docker,
            "exec",
            self.container,
            self.sqlpackage_path,
            "/a:Export",
            f"/ssn:{self.server_name}",
            f"/sdn:{db_name}",
            f"/tf:{container_path}",
            f"/su:{self.username}",
            f"/sp:{self.password}",
            "/SourceEncryptConnection:False",
            "/SourceTrustServerCertificate:True"
        ]
        self._run_cmd(cmd_export)
        cmd_cp = [self.docker, "cp", f"{self.container}:{container_path}", backup_local_path]
        self._run_cmd(cmd_cp)
        print(f"✅ Backup complete: {backup_local_path}")
        return backup_local_path

    def restore(self, db_name: str, backup_local_path: str):
        if not os.path.isfile(backup_local_path):
            raise FileNotFoundError(f"Local bacpac not found: {backup_local_path}")

        container_path = f"/tmp/{os.path.basename(backup_local_path)}"
        cmd_cp_in = [self.docker, "cp", backup_local_path, f"{self.container}:{container_path}"]
        self._run_cmd(cmd_cp_in)
        
        cmd_import = [
            self.docker,
            "exec",
            self.container,
            self.sqlpackage_path,
            "/a:Import",
            f"/tsn:{self.server_name}",
            f"/tdn:{db_name}",
            f"/sf:{container_path}",
            f"/tu:{self.username}",
            f"/tp:{self.password}",
            "/TargetEncryptConnection:False",
            "/TargetTrustServerCertificate:True"
        ]
        self._run_cmd(cmd_import)
        print(f"✅ Restore complete: {backup_local_path} -> {db_name}")
        return db_name


def DBBACKUP(db_name, file_path):
    database = SqlDatabase()
    db_folder = os.path.join(file_path, db_name)
    os.makedirs(db_folder, exist_ok=True)
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    backup_file_path = os.path.join(db_folder, f"{db_name}_{current_datetime}.bacpac")
    database.backup(db_name, backup_file_path)
    print(f"Database '{db_name}' exported to: {backup_file_path}")


def DBRESTORE(db_name, file_path):
    database = SqlDatabase()
    db_folder = os.path.join(file_path, db_name)
    os.makedirs(db_folder, exist_ok=True)
    backup_files = [f for f in os.listdir(db_folder) if f.endswith('.bacpac')]
    if backup_files:
        latest_backup_file = max(
            backup_files,
            key=lambda f: os.path.getmtime(os.path.join(db_folder, f))
        )
        backup_file_path = os.path.join(db_folder, latest_backup_file)
        database.restore(db_name, backup_file_path)
        print(f"Database '{db_name}' imported from: {backup_file_path}")
    else:
        print("No .bacpac backup files found.")



# # =============================================================================
# # For MacOS
# # =============================================================================
# import subprocess
# import os
# from datetime import datetime

# class SqlDatabase:
#     def __init__(self):
#         # Your Docker container name
#         self.container = "sql2022"  

#         # SQL Server inside container
#         self.server_name = "localhost"
#         self.username = "sa"
#         self.password = "Bluebird951@"

#         # Path inside container
#         self.sqlpackage_path = "/opt/sqlpackage/sqlpackage"

#     def _run_cmd(self, cmd):
#         """Run a terminal command and print output/errors."""
#         print("Running:", " ".join(cmd))
#         subprocess.run(cmd, check=True)

#     def backup(self, db_name, backup_local_path):
#         # 1) Copy file FROM mac → TO container
#         container_path = f"/tmp/{os.path.basename(backup_local_path)}"

#         # 2) Run sqlpackage export inside container
#         command = [
#             "docker", "exec", self.container,
#             self.sqlpackage_path,
#             "/a:Export",
#             f"/ssn:{self.server_name}",
#             f"/sdn:{db_name}",
#             f"/tf:{container_path}",
#             f"/su:{self.username}",
#             f"/sp:{self.password}"
#         ]

#         self._run_cmd(command)

#         # 3) Copy exported file back to macOS
#         command = [
#             "docker", "cp",
#             f"{self.container}:{container_path}",
#             backup_local_path
#         ]
#         self._run_cmd(command)

#         print("✅ Backup successful:", backup_local_path)

#     def restore(self, db_name, backup_local_path):
#         # 1) Copy bacpac into container
#         container_path = f"/tmp/{os.path.basename(backup_local_path)}"
#         self._run_cmd(["docker", "cp", backup_local_path, f"{self.container}:{container_path}"])

#         # 2) Run sqlpackage Import inside container
#         command = [
#             "docker", "exec", self.container,
#             self.sqlpackage_path,
#             "/a:Import",
#             f"/tsn:{self.server_name}",
#             f"/tdn:{db_name}",
#             f"/sf:{container_path}",
#             f"/tu:{self.username}",
#             f"/tp:{self.password}"
#         ]

#         self._run_cmd(command)
#         print("✅ Restore successful:", backup_local_path)


# def DBBACKUP(db_name, file_path):
#     database = SqlDatabase()
#     db_folder = os.path.join(file_path, db_name)
#     os.makedirs(db_folder, exist_ok=True)
#     current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#     backup_file_path = os.path.join(db_folder, f"{db_name}_{current_datetime}.bacpac")
#     database.backup(db_name, backup_file_path)
#     print(f"Database '{db_name}' exported to: {backup_file_path}")


# def DBRESTORE(db_name, file_path):
#     database = SqlDatabase()
#     db_folder = os.path.join(file_path, db_name)
#     os.makedirs(db_folder, exist_ok=True)
#     backup_files = [f for f in os.listdir(db_folder) if f.endswith('.bacpac')]
#     if backup_files:
#         latest_backup_file = max(
#             backup_files,
#             key=lambda f: os.path.getmtime(os.path.join(db_folder, f))
#         )
#         backup_file_path = os.path.join(db_folder, latest_backup_file)
#         database.restore(db_name, backup_file_path)
#         print(f"Database '{db_name}' imported from: {backup_file_path}")
#     else:
#         print("No .bacpac backup files found.")



# DBRESTORE('mkgrowwdaymaster', './Backup')



# =============================================================================
# 
# =============================================================================

# class SqlDatabase:
#     def __init__(self):
#         # For macOS + Docker SQL Server
#         self.server_name = 'localhost,1433'
#         self.sqlpackage_path = '/Users/hemantcgupta/sqlpackage/sqlpackage'  # or wherever sqlpackage installs
#         self.username = 'hemantcgupta'  # or hemantcgupta if created
#         self.password = 'Bluebird951@'

#     def backup(self, db_name, file_path):
#         try:
#             command = [
#                 self.sqlpackage_path,
#                 '/a:Export',
#                 f'/ssn:{self.server_name}',
#                 f'/sdn:{db_name}',
#                 f'/tf:{file_path}',
#                 f'/su:{self.username}',
#                 f'/sp:{self.password}',
#                 '/TargetFileFormat:BACPAC',
#                 "/TargetEncryptConnection:False",
#                 "/TargetTrustServerCertificate:True"
#             ]
#             subprocess.run(command, check=True)
#             print("✅ Backup successful.")
#         except subprocess.CalledProcessError as e:
#             print(f"❌ Backup failed: {e}")

#     def restore(self, db_name, file_path):
#         try:
#             command = [
#                 self.sqlpackage_path,
#                 "/a:Import",
#                 f"/tsn:{self.server_name}",
#                 f"/tdn:{db_name}",
#                 f"/sf:{file_path}",
#                 "/tu:hemantcgupta",
#                 "/tp:Bluebird951@",
#                 "/TargetEncryptConnection:False",
#                 "/TargetTrustServerCertificate:True"
#             ]
#             subprocess.run(command, check=True)
#             print("✅ Restore successful.")
#         except subprocess.CalledProcessError as e:
#             print(f"❌ Restore failed: {e}")


# def DBBACKUP(db_name, file_path):
#     database = SqlDatabase()
#     db_folder = os.path.join(file_path, db_name)
#     os.makedirs(db_folder, exist_ok=True)
#     current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#     backup_file_path = os.path.join(db_folder, f"{db_name}_{current_datetime}.bacpac")
#     database.backup(db_name, backup_file_path)
#     print(f"Database '{db_name}' exported to: {backup_file_path}")


# def DBRESTORE(db_name, file_path):
#     database = SqlDatabase()
#     db_folder = os.path.join(file_path, db_name)
#     os.makedirs(db_folder, exist_ok=True)
#     backup_files = [f for f in os.listdir(db_folder) if f.endswith('.bacpac')]
#     if backup_files:
#         latest_backup_file = max(
#             backup_files,
#             key=lambda f: os.path.getmtime(os.path.join(db_folder, f))
#         )
#         backup_file_path = os.path.join(db_folder, latest_backup_file)
#         database.restore(db_name, backup_file_path)
#         print(f"Database '{db_name}' imported from: {backup_file_path}")
#     else:
#         print("No .bacpac backup files found.")



# =============================================================================
# For Windows
# =============================================================================
# import subprocess
# import os
# from datetime import datetime

# class SqlDatabase:
#     def __init__(self):
#         self.server_name = 'localhost\\SQLEXPRESS'
#         self.sqlpackage_path = r"C:\Program Files\Microsoft SQL Server\140\DAC\bin\SqlPackage.exe"

#     def backup(self, db_name, file_path):
#         try:
#             command = [
#                 self.sqlpackage_path,
#                 "/a:Export",
#                 f"/ssn:{self.server_name}",
#                 f"/sdn:{db_name}",
#                 f"/tf:{file_path}"
#             ]
#             subprocess.run(command, check=True)
#             print("Backup successful.")
#         except subprocess.CalledProcessError as e:
#             print(f"Backup failed: {e}")

#     def restore(self, db_name, file_path):
#         try:
#             command = [
#                 self.sqlpackage_path,
#                 "/a:Import",
#                 f"/tsn:{self.server_name}",
#                 f"/tdn:{db_name}",
#                 f"/sf:{file_path}"
#             ]
#             subprocess.run(command, check=True)
#             print("Restore successful.")
#         except subprocess.CalledProcessError as e:
#             print(f"Restore failed: {e}")

# def DBBACKUP(db_name, file_path):
#     database = SqlDatabase()
#     db_folder = os.path.join(file_path, db_name)
#     os.makedirs(db_folder, exist_ok=True)
#     current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#     backup_file_path = os.path.join(db_folder, f"{db_name}_{current_datetime}.bacpac")
#     database.backup(db_name, backup_file_path)
#     print(f"Database '{db_name}' exported on backup file: {backup_file_path}") 

# def DBRESTORE(db_name, file_path):
#     database = SqlDatabase()  
#     db_folder = os.path.join(file_path, db_name)
#     os.makedirs(db_folder, exist_ok=True)
#     backup_files = [f for f in os.listdir(db_folder) if f.endswith('.bacpac')]
#     if backup_files:
#         latest_backup_file = max(backup_files, key=lambda f: os.path.getmtime(os.path.join(db_folder, f)))
#         backup_file_path = os.path.join(db_folder, latest_backup_file)
#         database.restore(db_name, backup_file_path)
#         print(f"Database '{db_name}' imported from the latest backup file: {backup_file_path}") 
#     else:
#         print("No backup files found in the specified directory.") 
    
# DBBACKUP('mkintervalmaster', './SSMSDB')    
# DBRESTORE('mkintervalmaster', './SSMSDB')


