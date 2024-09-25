WITH PreviousHighs AS (
    SELECT
        Datetime,
        tickerName,
        [Open],
        [High],
        [Low],
        [Close],
        LAG([High], 1) OVER (PARTITION BY tickerName ORDER BY Datetime) AS PrevDayHigh,
        LAG([High], 2) OVER (PARTITION BY tickerName ORDER BY Datetime) AS Prev2DayHigh,
        LAG([High], 3) OVER (PARTITION BY tickerName ORDER BY Datetime) AS Prev3DayHigh
    FROM mkDayFeature
    -- Filter for the last month based on the maximum datetime in the dataset
    WHERE Datetime >= DATEADD(MONTH, -1, (SELECT MAX(Datetime) FROM mkDayFeature))
),
FilteredResults AS (
    SELECT
        Datetime,
        tickerName,
        [Open],
        [High],
        [Low],
        [Close],
        CASE
            WHEN COALESCE(PrevDayHigh, 0) > COALESCE(Prev2DayHigh, 0) AND COALESCE(Prev2DayHigh, 0) > COALESCE(Prev3DayHigh, 0) THEN 1
            ELSE 0
        END AS PrevDayBreak,
        ROW_NUMBER() OVER (PARTITION BY tickerName ORDER BY Datetime DESC) AS row_num -- Identify the latest record for each ticker
    FROM PreviousHighs
)
-- Count the occurrences of PrevDayBreak and get the latest break status
SELECT
    tickerName,
    COUNT(CASE WHEN PrevDayBreak = 1 THEN 1 ELSE NULL END) AS PrevDayBreakCount,
    MAX(CASE WHEN row_num = 1 THEN PrevDayBreak ELSE NULL END) AS latestBreak -- Get the break status for the latest record
FROM FilteredResults
GROUP BY tickerName
ORDER BY latestBreak DESC, PrevDayBreakCount DESC;
