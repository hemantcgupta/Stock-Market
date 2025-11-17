WITH cte1 AS (
    SELECT 
        CAST(Datetime AS Date) AS Date, 
        tickerName, 
        successCount, 
        PredEntry, 
        PredExit, 
        ActualEntry, 
        ActualExit, 
        ActualProfit, 
        PredProfit
    FROM mkTopPrediction
),
cte2 AS (
    SELECT
        CAST(Datetime AS Date) AS Date,
        tickerName,
        [Open],
        [High],
        [Low],
		[Close],
        LAG([Low], 1) OVER (PARTITION BY tickerName ORDER BY Datetime) AS Prev1DayLow,
        LAG([Low], 2) OVER (PARTITION BY tickerName ORDER BY Datetime) AS Prev2DayLow
    FROM mkanalyzer.dbo.mkDayFeature
    WHERE Datetime >= DATEADD(MONTH, -1, (SELECT MIN(CAST(Datetime AS Date)) FROM mkTopPrediction))
)
, cte3 AS (
    SELECT 
        c1.*, 
        CASE
            WHEN c2.[Low] > COALESCE(c2.Prev1DayLow, 0) 
                AND COALESCE(c2.Prev1DayLow, 0) > COALESCE(c2.Prev2DayLow, 0) 
            THEN 1
            ELSE 0
        END AS PrevDayNotBreak,
		CASE 
            WHEN (c1.ActualEntry <= c1.PredEntry) OR (c1.PredEntry >= c2.[Close])
            THEN 1 
            ELSE 0 
        END AS gotEntry,
		CASE 
			WHEN (c1.ActualEntry <= c1.PredEntry AND (c1.ActualExit >= c1.PredExit OR c1.PredEntry < c2.[Close])) 
			THEN 0
			WHEN (c1.ActualEntry <= c1.PredEntry AND c1.ActualEntry >= c2.[Close]) OR (c1.ActualEntry >= c2.[Close])
			THEN 1 
			ELSE 0 
		END AS gotLoss,
        ROW_NUMBER() OVER (
            PARTITION BY c1.Date 
            ORDER BY 
                c1.Date DESC, 
                CASE
                    WHEN c2.[Low] > COALESCE(c2.Prev1DayLow, 0) 
                        AND COALESCE(c2.Prev1DayLow, 0) > COALESCE(c2.Prev2DayLow, 0) 
                    THEN 1
                    ELSE 0
                END DESC, 
                c1.successCount DESC
        ) AS row_num
    FROM cte1 c1
    LEFT JOIN cte2 c2 
        ON c1.Date = c2.Date 
        AND c1.tickerName = c2.tickerName
)
SELECT *, month(Date) as months
FROM cte3
--WHERE row_num = 1 --and gotEntry = 1 --and month(Date) = 9
WHERE gotEntry = 1 --and month(Date) = 9
ORDER BY cte3.PrevDayNotBreak DESC, cte3.Date DESC
