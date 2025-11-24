----------- delete login user 

USE master;
GO
-- Kill all sessions using this login
DECLARE @spid INT;
DECLARE c CURSOR FOR
SELECT session_id FROM sys.dm_exec_sessions WHERE login_name = 'hemantcgupta';
OPEN c; FETCH NEXT FROM c INTO @spid;
WHILE @@FETCH_STATUS = 0
BEGIN
    EXEC('KILL ' + @spid);
    FETCH NEXT FROM c INTO @spid;
END
CLOSE c; DEALLOCATE c;
GO

-- Remove database users mapped to this login
EXEC sp_MSforeachdb 'USE ?; IF EXISTS (SELECT 1 FROM sys.database_principals WHERE name = ''hemantcgupta'') DROP USER [hemantcgupta]';
GO

-- Finally DROP the login
DROP LOGIN [hemantcgupta];
GO



-------------------------------- Check Ownder shift database 
SELECT name, suser_sname(owner_sid) AS owner
FROM sys.databases
WHERE suser_sname(owner_sid) = 'hemantcgupta';


---------- Step 1 — Create login & user (if not exists)

IF NOT EXISTS (SELECT 1 FROM sys.sql_logins WHERE name = 'hemantcgupta')
    CREATE LOGIN [hemantcgupta] WITH PASSWORD = 'Bluebird951@';

-- Create user in master (needed for owner)
USE master;
IF NOT EXISTS (SELECT 1 FROM sys.database_principals WHERE name = 'hemantcgupta')
    CREATE USER [hemantcgupta] FOR LOGIN [hemantcgupta];

-------------- Step 2 — Change Owner for ALL Databases Owned by sa

DECLARE @sql NVARCHAR(MAX) = N'';

SELECT @sql = @sql + '
ALTER AUTHORIZATION ON DATABASE::[' + name + '] TO [hemantcgupta];
'
FROM sys.databases
WHERE suser_sname(owner_sid) = 'sa'
  AND name NOT IN ('master','msdb','model','tempdb');  -- avoid system DBs

PRINT @sql;   -- for verification
EXEC sp_executesql @sql;



------ removing hemantc =gupta maping from the db 
USE mkprediction;
SELECT name, sid FROM sys.database_principals WHERE name = 'hemantcgupta';

USE mkprediction;
DROP USER IF EXISTS [hemantcgupta];


-------- drop all the db 
use master
DECLARE @sql NVARCHAR(MAX) = N'';

SELECT @sql = @sql + '
ALTER DATABASE [' + name + '] SET SINGLE_USER WITH ROLLBACK IMMEDIATE;
DROP DATABASE [' + name + '];'
FROM sys.databases
WHERE name NOT IN ('master', 'model', 'msdb', 'tempdb');  -- protect system DBs

PRINT @sql;   -- show what will run (VERY IMPORTANT)
EXEC sp_executesql @sql;   -- ← UNCOMMENT THIS TO ACTUALLY DROP ALL DATABASES
