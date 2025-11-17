select * from BLS order by Datetime desc
select * from SIEMENS order by Datetime desc
select * from TATAMOTORS order by Datetime desc

------------------ Make altered date -x minute to adjust ----------------------
USE mkgrowwintervalmaster;

DECLARE @tableName NVARCHAR(128);
DECLARE @sql NVARCHAR(MAX);

DECLARE table_cursor CURSOR FOR
SELECT TABLE_NAME
FROM INFORMATION_SCHEMA.COLUMNS
WHERE COLUMN_NAME = 'Datetime'
GROUP BY TABLE_NAME;

OPEN table_cursor;
FETCH NEXT FROM table_cursor INTO @tableName;

WHILE @@FETCH_STATUS = 0
BEGIN
    -- Build dynamic SQL to handle VARCHAR or DATETIME formats safely
    SET @sql = '
    UPDATE [' + @tableName + ']
    SET Datetime = CONVERT(VARCHAR(19), DATEADD(MINUTE, -DATEPART(MINUTE, 
        CAST(Datetime AS DATETIME)) % 5, CAST(Datetime AS DATETIME)), 120)
    WHERE ISDATE(Datetime) = 1 AND DATEPART(MINUTE, CAST(Datetime AS DATETIME)) % 5 != 0;
    ';

    PRINT @sql;  -- Shows the query for debugging (optional)
    EXEC sp_executesql @sql;

    FETCH NEXT FROM table_cursor INTO @tableName;
END;

CLOSE table_cursor;
DEALLOCATE table_cursor;



