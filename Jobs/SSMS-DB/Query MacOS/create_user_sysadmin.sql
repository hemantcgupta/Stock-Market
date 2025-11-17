-- create_user_sysadmin.sql
SET NOCOUNT ON;
BEGIN TRY
    USE [master];
    -- 1) create login (if it doesn't exist)
    IF NOT EXISTS (SELECT 1 FROM sys.server_principals WHERE name = N'hemantcgupta')
    BEGIN
        CREATE LOGIN [hemantcgupta] WITH PASSWORD = N'Bluebird951@', 
            DEFAULT_DATABASE = [master], 
            CHECK_POLICY = ON, 
            CHECK_EXPIRATION = OFF;
        PRINT 'Login created';
    END
    ELSE
    BEGIN
        PRINT 'Login already exists';
    END

    -- 2) add to sysadmin server role (gives admin across server)
    IF NOT EXISTS (
        SELECT 1 FROM sys.server_role_members rm
        JOIN sys.server_principals r ON rm.role_principal_id = r.principal_id
        JOIN sys.server_principals p ON rm.member_principal_id = p.principal_id
        WHERE r.name = N'sysadmin' AND p.name = N'hemantcgupta')
    BEGIN
        ALTER SERVER ROLE [sysadmin] ADD MEMBER [hemantcgupta];
        PRINT 'Added to sysadmin';
    END
    ELSE
    BEGIN
        PRINT 'Already member of sysadmin';
    END

    -- 3) (Optional) create a database user and grant db_owner in all non-system DBs
    DECLARE @dbname sysname, @sql nvarchar(max);
    DECLARE db_cursor CURSOR FOR
        SELECT name FROM sys.databases
        WHERE name NOT IN ('master','tempdb','model','msdb') -- skip system DBs
          AND state = 0; -- only online DBs

    OPEN db_cursor;
    FETCH NEXT FROM db_cursor INTO @dbname;
    WHILE @@FETCH_STATUS = 0
    BEGIN
        SET @sql = N'USE [' + QUOTENAME(@dbname) + N'];' + CHAR(13) +
                   N'IF NOT EXISTS (SELECT 1 FROM sys.database_principals WHERE name = N''hemantcgupta'') ' +
                   N'BEGIN CREATE USER [hemantcgupta] FOR LOGIN [hemantcgupta]; END;' + CHAR(13) +
                   N'ALTER ROLE [db_owner] ADD MEMBER [hemantcgupta];';
        EXEC sp_executesql @sql;
        PRINT 'Processed DB: ' + @dbname;
        FETCH NEXT FROM db_cursor INTO @dbname;
    END

    CLOSE db_cursor;
    DEALLOCATE db_cursor;
END TRY
BEGIN CATCH
    PRINT 'ERROR: ' + ERROR_MESSAGE();
    THROW;
END CATCH;
