DECLARE @Date DATE;
DECLARE @Date30 DATE;
DECLARE @PredDate DATE;

SET @Date = '2024-09-02'; 
SET @Date30 = '2024-08-02'; 
SET @PredDate = '2024-09-03';

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
    FROM mkanalyzer.dbo.mkDayFeature
    -- Filter for the last month based on the maximum datetime in the dataset
    WHERE Datetime >= DATEADD(MONTH, -1, @Date)
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
),
-- Count the occurrences of PrevDayBreak and get the latest break status
cte as (SELECT
    tickerName,
    COUNT(CASE WHEN PrevDayBreak = 1 THEN 1 ELSE NULL END) AS PrevDayBreakCount,
    MAX(CASE WHEN row_num = 1 THEN PrevDayBreak ELSE NULL END) AS latestBreak -- Get the break status for the latest record
FROM FilteredResults
GROUP BY tickerName
--ORDER BY latestBreak DESC, PrevDayBreakCount DESC
)

select m.tickerName, sum(m.gotProfit) as counts, a.Date15, a.gotProfit, aa.gotProfit as p, c.PrevDayBreakCount, c.latestBreak, aa.gotEntry,
(sum(m.gotProfit)+ c.PrevDayBreakCount) as sam
from algo7515 m
left join (select tickerName, Date15, gotProfit, gotEntry from algo7515 where Date15 = @Date and gotProfit = 1 and gotEntry = 1) a on a.tickerName = m.tickerName
left join (select tickerName, Date15, gotProfit, gotEntry from algo7515 where Date15 = @PredDate and gotProfit = 1 and gotEntry = 1) aa on aa.tickerName = m.tickerName
left join cte c on c.tickerName = m.tickerName
where m.Date15 >= @Date30 and m.Date15 <= @Date
group by m.tickerName, a.Date15, a.gotProfit, aa.gotProfit,  c.PrevDayBreakCount, c.latestBreak, aa.gotEntry
--order by counts DESC
--ORDER BY latestBreak DESC, PrevDayBreakCount DESC
--ORDER BY latestBreak DESC, counts DESC
ORDER BY latestBreak DESC, PrevDayBreakCount DESC
