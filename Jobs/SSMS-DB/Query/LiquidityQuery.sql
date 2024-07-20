use mkdaymaster;

select top 10 [Open] from BLS
order by Datetime desc


SELECT 
    Datetime,
    ((High - Low) / Close) AS BASR_Interval,
    Volume,
    ((High - Low) / Close) * LOG(Volume) AS LS_Interval
FROM 
    BLS
ORDER BY 
    Datetime
LIMIT 10;



SELECT 
    CONVERT(date, Datetime) AS Date, [Open], [High], [Low], [Close], [Volume],
    ((MAX(High) - MIN(Low)) / MAX([Close])) AS BASR_Day,
    SUM(Volume) / COUNT(DISTINCT CONVERT(date, Datetime)) AS ATV_Day,
    ((MAX(High) - MIN(Low)) / MAX([Close])) * LOG(SUM(Volume) / COUNT(DISTINCT CONVERT(date, Datetime))) AS LS_Day
FROM 
    TATAPOWER
GROUP BY 
    CONVERT(date, Datetime), [Open], [High], [Low], [Close], [Volume]
ORDER BY 
    Date DESC
