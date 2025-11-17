DECLARE @ticker NVARCHAR(100);
SET @ticker = 'MOL';  -- Set the desired ticker name dynamically

DECLARE @sql NVARCHAR(MAX);

SET @sql = '
SELECT 
    sp.Datetime,
    sp.tickerName, 
    sp.Entry2 AS ActualEntry, 
    sp.Exit2 AS ActualExit, 
    sp.[High] AS ActualHigh,
    sp.predTmEntry2 AS PredEntry, 
    sp.predTmExit2 AS PredExit, 
    sp.[Close],
    CASE 
        WHEN (sp.Entry2 <= sp.predTmEntry2 AND (sp.Exit2 >= sp.predTmExit2 OR sp.predTmEntry2 < sp.[Close])) 
        THEN 1 
        ELSE 0 
    END AS TmPL,
    CASE 
        WHEN (sp.Entry2 <= sp.predTmEntry2) OR (sp.predTmEntry2 >= sp.[Close])
        THEN 1 
        ELSE 0 
    END AS gotEntry,
    CASE 
        WHEN (sp.Entry2 <= sp.predTmEntry2 AND (sp.Exit2 >= sp.predTmExit2 OR sp.predTmEntry2 < sp.[Close])) 
        THEN 0
        WHEN (sp.Entry2 <= sp.predTmEntry2 AND sp.predTmEntry2 >= sp.[Close]) OR (sp.predTmEntry2 >= sp.[Close])
        THEN 1 
        ELSE 0 
    END AS gotLoss,
    CASE 
        WHEN sp.[High] >= sp.predTmExit2 
        THEN 1
        ELSE 0 
    END AS gotSell,
    CASE 
        WHEN (sp.Entry2 <= sp.predTmEntry2 AND sp.Exit2 >= sp.predTmExit2) 
        THEN ROUND(((sp.predTmExit2 - sp.predTmEntry2) / sp.predTmExit2) * 100, 2)
        ELSE ROUND(((sp.[Close] - sp.predTmEntry2) / sp.[Close]) * 100, 2)
    END AS ActualProfit, 
    sp.EtEx2Profit AS PredProfit,
    y.Date,
    y.tt
FROM 
    mkanalyzer.dbo.simulationPrediction sp
LEFT JOIN 
    (
        SELECT 
            CAST(Datetime AS DATE) AS Date,
            SUM(ROUND((High - Low) / High, 2)) AS tt
        FROM ' + QUOTENAME(@ticker) + ' 
        GROUP BY 
            CAST(Datetime AS DATE)
    ) y
ON CAST(sp.Datetime AS DATE) = y.Date
WHERE sp.tickerName = @ticker
ORDER BY sp.Datetime DESC;';

-- Execute the dynamic SQL
EXEC sp_executesql @sql, N'@ticker NVARCHAR(100)', @ticker;
