import subprocess
import shutil
import os
from datetime import datetime
from typing import Optional


class SqlDatabase:
    """
    Backup/Restore wrapper that runs sqlpackage inside a Docker container.
    Designed for macOS (M1/M2/M3/M4) running an x64 SQL Server container (sql2022).
    """

    def __init__(
        self,
        container: str = "sql2022",
        docker_path: str = '/usr/local/bin/docker',
        sqlpackage_path: str = "/opt/sqlpackage/sqlpackage",
        server_name: str = "localhost,1433",
        username: str = "hemantcgupta",
        password: str = "Bluebird951@",
    ):
        """
        Parameters:
          container        - Docker container name (must already be running).
          docker_path      - Full path to docker binary (auto-detected if omitted).
          sqlpackage_path  - Path to sqlpackage binary inside the container.
          server_name      - Hostname used by sqlpackage (usually localhost).
          username         - SQL login username (usually 'sa').
          password         - SQL login password (required).
        """
        # resolve docker binary path
        self.docker = docker_path or shutil.which("docker")
        if not self.docker:
            raise FileNotFoundError(
                "docker binary not found in PATH. Install Docker or pass docker_path explicitly."
            )

        # basic configuration
        self.container = container
        self.sqlpackage_path = sqlpackage_path
        self.server_name = server_name
        self.username = username
        self.password = password 

        if not self.password:
            raise ValueError(
                "SA password not provided. Pass password to SqlDatabase(...) or set MSSQL_SA_PASSWORD env var."
            )

        # quick runtime checks
        if not self._is_container_running():
            raise RuntimeError(
                f"Container '{self.container}' is not running. Start it before using this class."
            )

        if not self._sqlpackage_exists_in_container():
            raise RuntimeError(
                f"sqlpackage not found at '{self.sqlpackage_path}' inside container '{self.container}'.\n"
                "Install sqlpackage inside the container (see README)."
            )

    # -------------------------
    # Internal helpers
    # -------------------------
    def _run_cmd(self, cmd, capture_output=True):
        """Run a command (list) and return completed process or raise with stdout/stderr."""
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
            # show helpful debug info
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
        # use a short test to see if the file exists & is executable inside container
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

    # -------------------------
    # Public operations
    # -------------------------
    def backup(self, db_name: str, backup_local_path: str):
        """
        Export a database to a local .bacpac path.
        - backup_local_path: absolute or relative path where a .bacpac will be written locally.
        """
        # ensure destination folder exists
        os.makedirs(os.path.dirname(os.path.abspath(backup_local_path)), exist_ok=True)

        # remote path inside container
        container_path = f"/tmp/{os.path.basename(backup_local_path)}"

        # run export inside container (creates the file at container_path)
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
            "/TargetEncryptConnection:False",
            "/TargetTrustServerCertificate:True",
        ]
        self._run_cmd(cmd_export)

        # copy file from container -> host
        cmd_cp = [self.docker, "cp", f"{self.container}:{container_path}", backup_local_path]
        self._run_cmd(cmd_cp)

        print(f"✅ Backup complete: {backup_local_path}")
        return backup_local_path

    def restore(self, db_name: str, backup_local_path: str):
        """
        Import (restore) a local .bacpac file into the container's SQL Server as `db_name`.
        """
        # verify local file exists
        if not os.path.isfile(backup_local_path):
            raise FileNotFoundError(f"Local bacpac not found: {backup_local_path}")

        container_path = f"/tmp/{os.path.basename(backup_local_path)}"

        # copy file into container
        cmd_cp_in = [self.docker, "cp", backup_local_path, f"{self.container}:{container_path}"]
        self._run_cmd(cmd_cp_in)

        # run import inside container
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
            "/TargetTrustServerCertificate:True",
        ]
        self._run_cmd(cmd_import)

        print(f"✅ Restore complete: {backup_local_path} -> {db_name}")
        return db_name

    # convenience helpers for folder-based workflows
    # def backup_latest_filename(self, db_name: str, out_folder: str) -> str:
    #     os.makedirs(out_folder, exist_ok=True)
    #     ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    #     filename = f"{db_name}_{ts}.bacpac"
    #     return os.path.join(out_folder, filename)

    # def restore_latest_in_folder(self, db_name: str, folder: str) -> str:
    #     if not os.path.isdir(folder):
    #         raise FileNotFoundError(f"Backup folder not found: {folder}")
    #     db_folder = os.path.join(folder, db_name)
    #     os.makedirs(db_folder, exist_ok=True)
    #     backup_files = [f for f in os.listdir(db_folder) if f.lower().endswith(".bacpac")]
    #     if not backup_files:
    #         raise FileNotFoundError(f"No .bacpac files found in {folder}")
    #     latest_backup_file = max(
    #                 backup_files,
    #                 key=lambda f: os.path.getmtime(os.path.join(db_folder, f))
    #             )
    #     backup_file_path = os.path.join(db_folder, latest_backup_file)
    #     return self.restore(db_name, backup_file_path)

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

# -------------------------
# Example usage
# -------------------------
# if __name__ == "__main__":
    

#     db = SqlDatabase()  # or pass password param if you prefer
#     # Create a backup file path & run backup:
#     # local_backup = db.backup_latest_filename("mkgrowwdaymaster", "./Backup")
#     # db.backup("mkgrowwdaymaster", local_backup)
#     db.restore_latest_in_folder("mkgrowwdaymaster", "./Backup")

    # To restore latest bacpac in ./Backup into database name "mkgrowwdaymaster":
    # db.restore_latest_in_folder("mkgrowwdaymaster", "./Backup")


# # -*- coding: utf-8 -*-
# """
# Created on Sun Mar 31 20:32:36 2024

# @author: Hemant
# """


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

# # class SqlDatabase:
# #     def __init__(self):
# #         # For macOS + Docker SQL Server
# #         self.server_name = 'localhost,1433'
# #         self.sqlpackage_path = '/Users/hemantcgupta/sqlpackage/sqlpackage'  # or wherever sqlpackage installs
# #         self.username = 'hemantcgupta'  # or hemantcgupta if created
# #         self.password = 'Bluebird951@'

# #     def backup(self, db_name, file_path):
# #         try:
# #             command = [
# #                 self.sqlpackage_path,
# #                 '/a:Export',
# #                 f'/ssn:{self.server_name}',
# #                 f'/sdn:{db_name}',
# #                 f'/tf:{file_path}',
# #                 f'/su:{self.username}',
# #                 f'/sp:{self.password}',
# #                 '/TargetFileFormat:BACPAC',
# #                 "/TargetEncryptConnection:False",
# #                 "/TargetTrustServerCertificate:True"
# #             ]
# #             subprocess.run(command, check=True)
# #             print("✅ Backup successful.")
# #         except subprocess.CalledProcessError as e:
# #             print(f"❌ Backup failed: {e}")

# #     def restore(self, db_name, file_path):
# #         try:
# #             command = [
# #                 self.sqlpackage_path,
# #                 "/a:Import",
# #                 f"/tsn:{self.server_name}",
# #                 f"/tdn:{db_name}",
# #                 f"/sf:{file_path}",
# #                 "/tu:hemantcgupta",
# #                 "/tp:Bluebird951@",
# #                 "/TargetEncryptConnection:False",
# #                 "/TargetTrustServerCertificate:True"
# #             ]
# #             subprocess.run(command, check=True)
# #             print("✅ Restore successful.")
# #         except subprocess.CalledProcessError as e:
# #             print(f"❌ Restore failed: {e}")


# # def DBBACKUP(db_name, file_path):
# #     database = SqlDatabase()
# #     db_folder = os.path.join(file_path, db_name)
# #     os.makedirs(db_folder, exist_ok=True)
# #     current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# #     backup_file_path = os.path.join(db_folder, f"{db_name}_{current_datetime}.bacpac")
# #     database.backup(db_name, backup_file_path)
# #     print(f"Database '{db_name}' exported to: {backup_file_path}")


# # def DBRESTORE(db_name, file_path):
# #     database = SqlDatabase()
# #     db_folder = os.path.join(file_path, db_name)
# #     os.makedirs(db_folder, exist_ok=True)
# #     backup_files = [f for f in os.listdir(db_folder) if f.endswith('.bacpac')]
# #     if backup_files:
# #         latest_backup_file = max(
# #             backup_files,
# #             key=lambda f: os.path.getmtime(os.path.join(db_folder, f))
# #         )
# #         backup_file_path = os.path.join(db_folder, latest_backup_file)
# #         database.restore(db_name, backup_file_path)
# #         print(f"Database '{db_name}' imported from: {backup_file_path}")
# #     else:
# #         print("No .bacpac backup files found.")

# # =============================================================================
# # For Windows
# # =============================================================================
# # import subprocess
# # import os
# # from datetime import datetime

# # class SqlDatabase:
# #     def __init__(self):
# #         self.server_name = 'localhost\\SQLEXPRESS'
# #         self.sqlpackage_path = r"C:\Program Files\Microsoft SQL Server\140\DAC\bin\SqlPackage.exe"

# #     def backup(self, db_name, file_path):
# #         try:
# #             command = [
# #                 self.sqlpackage_path,
# #                 "/a:Export",
# #                 f"/ssn:{self.server_name}",
# #                 f"/sdn:{db_name}",
# #                 f"/tf:{file_path}"
# #             ]
# #             subprocess.run(command, check=True)
# #             print("Backup successful.")
# #         except subprocess.CalledProcessError as e:
# #             print(f"Backup failed: {e}")

# #     def restore(self, db_name, file_path):
# #         try:
# #             command = [
# #                 self.sqlpackage_path,
# #                 "/a:Import",
# #                 f"/tsn:{self.server_name}",
# #                 f"/tdn:{db_name}",
# #                 f"/sf:{file_path}"
# #             ]
# #             subprocess.run(command, check=True)
# #             print("Restore successful.")
# #         except subprocess.CalledProcessError as e:
# #             print(f"Restore failed: {e}")

# # def DBBACKUP(db_name, file_path):
# #     database = SqlDatabase()
# #     db_folder = os.path.join(file_path, db_name)
# #     os.makedirs(db_folder, exist_ok=True)
# #     current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# #     backup_file_path = os.path.join(db_folder, f"{db_name}_{current_datetime}.bacpac")
# #     database.backup(db_name, backup_file_path)
# #     print(f"Database '{db_name}' exported on backup file: {backup_file_path}") 

# # def DBRESTORE(db_name, file_path):
# #     database = SqlDatabase()  
# #     db_folder = os.path.join(file_path, db_name)
# #     os.makedirs(db_folder, exist_ok=True)
# #     backup_files = [f for f in os.listdir(db_folder) if f.endswith('.bacpac')]
# #     if backup_files:
# #         latest_backup_file = max(backup_files, key=lambda f: os.path.getmtime(os.path.join(db_folder, f)))
# #         backup_file_path = os.path.join(db_folder, latest_backup_file)
# #         database.restore(db_name, backup_file_path)
# #         print(f"Database '{db_name}' imported from the latest backup file: {backup_file_path}") 
# #     else:
# #         print("No backup files found in the specified directory.") 
    
# # DBBACKUP('mkintervalmaster', './SSMSDB')    
# # DBRESTORE('mkintervalmaster', './SSMSDB')


