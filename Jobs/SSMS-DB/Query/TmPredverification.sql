select tas.*, CASE 
        WHEN (Entry2 <= predTmEntry2 AND (Exit2 >= predTmExit2 OR predTmEntry2 < [Close])) 
        THEN 1 
        ELSE 0 
    END AS TmPL
from topAccurateStats AS tas
LEFT JOIN simulationPrediction AS sp ON tas.tickerName = sp.tickerName AND tas.Date = sp.Datetime
where tas.Date = '2024-07-25' and tas.TmPredPL = 1
order by [AccuracyScore:Mode] DESC


------------------------------------------------------------
select tas.Date, tas.tickerName, tas.[AccuracyScore:Mode],
CASE 
    WHEN (Entry2 <= predTmEntry2 AND (Exit2 >= predTmExit2 OR predTmEntry2 < [Close])) 
    THEN 1 
    ELSE 0 
END AS TmPL, tas.TmPredPL,
CASE 
    WHEN (Entry2 <= predTmEntry2 AND predTmEntry2 >= [Close]) 
    THEN 1 
    ELSE 0 
END AS gotLoss,
CASE 
    WHEN (Entry2 <= predTmEntry2 AND Exit2 >= predTmExit2) 
    THEN round(((predTmExit2-predTmEntry2)/predTmExit2) * 100, 2)
    ELSE round((([Close]-predTmEntry2)/[Close]) * 100, 2)
END AS ProfitPercent,
CASE 
    WHEN Entry2 <= predTmEntry2
    THEN 1 
    ELSE 0 
END AS gotEntry
from topAccurateStats AS tas
LEFT JOIN simulationPrediction AS sp ON tas.tickerName = sp.tickerName AND tas.Date = sp.Datetime
where tas.Date = '2024-07-25' and tas.TmPredPL = 1
order by [AccuracyScore:Mode] DESC


---------------------------------------------------------------------
select tas.Date, tas.tickerName, tas.[AccuracyScore:Mode],
CASE 
    WHEN (Entry2 <= predTmEntry2 AND (Exit2 >= predTmExit2 OR predTmEntry2 < [Close])) 
    THEN 1 
    ELSE 0 
END AS TmPL, tas.TmPredPL,
CASE 
    WHEN (Entry2 <= predTmEntry2 AND predTmEntry2 >= [Close]) 
    THEN 1 
    ELSE 0 
END AS gotLoss,
CASE 
    WHEN (Entry2 <= predTmEntry2 AND Exit2 >= predTmExit2) 
    THEN round(((predTmExit2-predTmEntry2)/predTmExit2) * 100, 2)
    ELSE round((([Close]-predTmEntry2)/[Close]) * 100, 2)
END AS ProfitPercent,
CASE 
    WHEN Entry2 <= predTmEntry2
    THEN 1 
    ELSE 0 
END AS gotEntry
from topAccurateStats AS tas
LEFT JOIN simulationPrediction AS sp ON tas.tickerName = sp.tickerName AND tas.Date = sp.Datetime
where tas.Date = '2024-07-25' and tas.TmPredPL = 1 AND Entry2 <= predTmEntry2 
        AND Exit2 >= predTmExit2
order by [AccuracyScore:Mode] DESC



------
with cte as (select tas.Date, tas.tickerName, tas.[AccuracyScore:Mode],
CASE 
    WHEN (Entry2 <= predTmEntry2 AND (Exit2 >= predTmExit2 OR predTmEntry2 < [Close])) 
    THEN 1 
    ELSE 0 
END AS TmPL, tas.TmPredPL,
CASE 
    WHEN (Entry2 <= predTmEntry2 AND predTmEntry2 >= [Close]) 
    THEN 1 
    ELSE 0 
END AS gotLoss,
CASE 
    WHEN (Entry2 <= predTmEntry2 AND Exit2 >= predTmExit2) 
    THEN round(((predTmExit2-predTmEntry2)/predTmExit2) * 100, 2)
    ELSE round((([Close]-predTmEntry2)/[Close]) * 100, 2)
END AS ProfitPercent,
CASE 
    WHEN Entry2 <= predTmEntry2
    THEN 1 
    ELSE 0 
END AS gotEntry
from topAccurateStats AS tas
LEFT JOIN simulationPrediction AS sp ON tas.tickerName = sp.tickerName AND tas.Date = sp.Datetime
where tas.Date = '2024-08-02' and tas.TmPredPL = 1
),
cte1 as (
SELECT tickerName, COUNT(Datetime) AS counts 
FROM simulationPrediction
WHERE Entry2 <= predTmEntry2 AND Exit2 >= predTmExit2
	AND CAST(Datetime AS DATE) >= CAST(DATEADD(MONTH, -1, '2024-08-02') AS DATE)
	AND CAST(Datetime AS DATE) <= CAST('2024-08-02' AS DATE)
GROUP BY tickerName
)
select * from cte c1 left join cte1 c2 on c1.tickerName = c2.tickerName
order by c2.counts DESC

-----
with cte as (select tas.Date, tas.tickerName, tas.[AccuracyScore:Mode],
CASE 
    WHEN (Entry2 <= predTmEntry2 AND (Exit2 >= predTmExit2 OR predTmEntry2 < [Close])) 
    THEN 1 
    ELSE 0 
END AS TmPL, tas.TmPredPL,
CASE 
    WHEN (Entry2 <= predTmEntry2 AND predTmEntry2 >= [Close]) 
    THEN 1 
    ELSE 0 
END AS gotLoss,
CASE 
    WHEN (Entry2 <= predTmEntry2 AND Exit2 >= predTmExit2) 
    THEN round(((predTmExit2-predTmEntry2)/predTmExit2) * 100, 2)
    ELSE round((([Close]-predTmEntry2)/[Close]) * 100, 2)
END AS ProfitPercent,
CASE 
    WHEN Entry2 <= predTmEntry2
    THEN 1 
    ELSE 0 
END AS gotEntry,
round(((sp.[predTmOpen]-sp.[Open])/sp.[Open])*100, 2) As 'pO-sO-Diff'
from topAccurateStats AS tas
LEFT JOIN simulationPrediction AS sp ON tas.tickerName = sp.tickerName AND tas.Date = sp.Datetime
where tas.Date = '2024-08-02' and tas.TmPredPL = 1
),
cte1 as (
SELECT tickerName, COUNT(Datetime) AS counts 
FROM simulationPrediction
WHERE Entry2 <= predTmEntry2 AND Exit2 >= predTmExit2
	AND CAST(Datetime AS DATE) >= CAST(DATEADD(MONTH, -1, '2024-08-02') AS DATE)
	AND CAST(Datetime AS DATE) <= CAST('2024-08-02' AS DATE)
GROUP BY tickerName
)
select * from cte c1 left join cte1 c2 on c1.tickerName = c2.tickerName
order by c2.counts DESC, [AccuracyScore:Mode] DESC
--order by c1.[AccuracyScore:Mode] DESC


--select top 10 * from simulationPrediction where tickerName = 'HINDOILEXP'
--order by Datetime Desc




------------ FInal Query Use ---------------
DECLARE @inputDate DATE;
SET @inputDate = '2024-07-19';  -- Replace this with your desired date

WITH cte AS (
    SELECT 
        tas.Date, 
        tas.tickerName, 
        tas.[AccuracyScore:Mode],
        CASE 
            WHEN (Entry2 <= predTmEntry2 AND (Exit2 >= predTmExit2 OR predTmEntry2 < [Close])) 
            THEN 1 
            ELSE 0 
        END AS TmPL, 
        tas.TmPredPL,
        CASE 
            WHEN (Entry2 <= predTmEntry2 AND predTmEntry2 >= [Close]) 
            THEN 1 
            ELSE 0 
        END AS gotLoss,
        CASE 
            WHEN (Entry2 <= predTmEntry2 AND Exit2 >= predTmExit2) 
            THEN round(((predTmExit2-predTmEntry2)/predTmExit2) * 100, 2)
            ELSE round((([Close]-predTmEntry2)/[Close]) * 100, 2)
        END AS ProfitPercent,
        CASE 
            WHEN Entry2 <= predTmEntry2
            THEN 1 
            ELSE 0 
        END AS gotEntry,
        round(((sp.[Open]-sp.[predTmOpen])/sp.[predTmOpen])*100, 2) AS 'AOpen/POpen-Diff',
		Entry2 As ActualEntry, predTmEntry2 as PredEntry, Exit2 as ActulExit, predTmExit2 as PredExit, 
		[predTmOpen] as PredOpen, [Open] as ActualOpen, [Close] as ActualClose
    FROM topAccurateStats AS tas
    LEFT JOIN simulationPrediction AS sp 
    ON tas.tickerName = sp.tickerName AND tas.Date = sp.Datetime
    WHERE tas.Date = @inputDate AND tas.TmPredPL = 1
),
cte1 AS (
    SELECT tickerName, COUNT(Datetime) AS counts 
    FROM simulationPrediction
    WHERE Entry2 <= predTmEntry2 
      AND Exit2 >= predTmExit2
      AND CAST(Datetime AS DATE) >= CAST(DATEADD(MONTH, -1, @inputDate) AS DATE)
      AND CAST(Datetime AS DATE) <= @inputDate
    GROUP BY tickerName
)
SELECT * 
FROM cte c1 
LEFT JOIN cte1 c2 ON c1.tickerName = c2.tickerName
ORDER BY c2.counts DESC, [AccuracyScore:Mode] DESC;
