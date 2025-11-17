-------- Delete all tables from DB and Only Keeps Three tables 'BLS', 'SIEMENS', 'TATAMOTORS' ------------
USE master;
GO

DECLARE @dbList TABLE (DbName NVARCHAR(100));
INSERT INTO @dbList (DbName)
VALUES 
    ('mkdaymaster'),
    ('mkintervalmaster'),
    ('mkgrowwdaymaster'),
    ('mkgrowwintervalmaster'),
    ('mkintervalanalyzer');

DECLARE @dbName NVARCHAR(100);
DECLARE @sql NVARCHAR(MAX);

DECLARE db_cursor CURSOR FOR
SELECT DbName FROM @dbList;

OPEN db_cursor;
FETCH NEXT FROM db_cursor INTO @dbName;

WHILE @@FETCH_STATUS = 0
BEGIN
    SET @sql = '
    USE [' + @dbName + '];

    DECLARE @innerSql NVARCHAR(MAX) = N'''';

    SELECT @innerSql += ''DROP TABLE ['' + SCHEMA_NAME(schema_id) + ''].['' + name + ''];'' + CHAR(13)
    FROM sys.tables
    WHERE name NOT IN (''BLS'', ''SIEMENS'', ''TATAMOTORS'');

    EXEC sp_executesql @innerSql;
    ';

    EXEC sp_executesql @sql;

    FETCH NEXT FROM db_cursor INTO @dbName;
END;

CLOSE db_cursor;
DEALLOCATE db_cursor;


------------------ Delete All Other Records (Except BLS, SIEMENS, TATAMOTORS): ------------------
USE master;
DELETE FROM mkgrowwinfo.dbo.mkGrowwInfo
WHERE nseScriptCode NOT IN ('BLS', 'SIEMENS', 'TATAMOTORS');

SELECT * FROM mkgrowwinfo.dbo.mkGrowwInfo
WHERE nseScriptCode NOT IN ('BLS', 'SIEMENS', 'TATAMOTORS');

------------------ Delete All Other Records (Except BLS, SIEMENS, TATAMOTORS) from mkanalyzer: ------------------
USE mkanalyzer;
GO

DECLARE @tableList TABLE (TableName NVARCHAR(128));
INSERT INTO @tableList (TableName)
VALUES 
('mkDayFeature'),
('mkDayMA'),
('mkDayPrediction'),
('mkDayProbability'),
('mkDaySeasonality'),
('mkIntervalFeature'),
('mkMonthFeature'),
('mkMonthSeasonality'),
('mkMonthSummary'),
('mkTopPrediction'),
('simulationPrediction'),
('simulationPredMSE'),
('topAccurateStats'),
('topAccurateStats1'),
('topAccurateTickerDetails'),
('topPriorityStats');

DECLARE @tableName NVARCHAR(128);
DECLARE @sql NVARCHAR(MAX);

DECLARE table_cursor CURSOR FOR
SELECT TableName FROM @tableList;

OPEN table_cursor;
FETCH NEXT FROM table_cursor INTO @tableName;

WHILE @@FETCH_STATUS = 0
BEGIN
    -- Construct dynamic DELETE SQL to retain only specific tickerNames
    SET @sql = '
    IF EXISTS (
        SELECT 1 FROM INFORMATION_SCHEMA.COLUMNS 
        WHERE TABLE_NAME = ''' + @tableName + ''' AND COLUMN_NAME = ''tickerName''
    )
    BEGIN
        DELETE FROM [' + @tableName + '] 
        WHERE tickerName NOT IN (''BLS'', ''SIEMENS'', ''TATAMOTORS'');
    END
    ';

    PRINT @sql; -- Optional: To preview the query
    EXEC sp_executesql @sql;

    FETCH NEXT FROM table_cursor INTO @tableName;
END;

CLOSE table_cursor;
DEALLOCATE table_cursor;


