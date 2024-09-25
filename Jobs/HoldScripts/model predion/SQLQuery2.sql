
with cte as(
select CAST(predDatetime AS DATE) as Datetime, tickerName,  Entry2 as ActualEntry, Exit2 AS ActualExit, [High] AS ActualHigh,
predTmEntry2 AS PredExtry, predTmExit2 AS PredExit, [Close],
CASE 
    WHEN (Entry2 <= predTmEntry2 AND (Exit2 >= predTmExit2 OR predTmEntry2 < [Close])) 
    THEN 1 
    ELSE 0 
END AS TmPL,
CASE 
    WHEN (Entry2 <= predTmEntry2) OR (predTmEntry2 >= [Close])
    THEN 1 
    ELSE 0 
END AS gotEntry,
CASE 
    WHEN (Entry2 <= predTmEntry2 AND (Exit2 >= predTmExit2 OR predTmEntry2 < [Close])) 
    THEN 0
    WHEN (Entry2 <= predTmEntry2 AND predTmEntry2 >= [Close]) OR (predTmEntry2 >= [Close])
    THEN 1 
    ELSE 0 
END AS gotLoss,
CASE 
    WHEN [High] >= predTmExit2 
    THEN 1
    ELSE 0 
END AS gotSell,
CASE 
    WHEN (Entry2 <= predTmEntry2 AND Exit2 >= predTmExit2) 
    THEN ROUND(((predTmExit2 - predTmEntry2) / predTmExit2) * 100, 2)
    ELSE ROUND((([Close] - predTmEntry2) / [Close]) * 100, 2)
END AS ActualProfit, EtEx2Profit as PredProfit
from simulationPrediction where tickerName = 'MOL' and predDatetime <= '2024-08-26'
)
select *,  
ROUND((ActualEntry-PredExtry)/PredExtry *100, 2) AS diffEntry,
ROUND((ActualExit-PredExit)/PredExit*100, 2) AS diffExit,
ROUND((ActualHigh-PredExit)/PredExit*100, 2) AS diffHigh,
LAG(CASE WHEN gotSell = 1 THEN ROUND(([Close]-PredExit)/PredExit*100, 2) ELSE 0 END, 1) OVER (ORDER BY Datetime DESC) AS diffClose,
LAG(gotSell, 1) OVER (ORDER BY Datetime DESC) AS sell,
LAG(
    CASE 
        WHEN (TmPL = 1 AND ActualProfit = PredProfit) THEN 'ETEX' 
        WHEN (TmPL = 1 AND ActualProfit != PredProfit) THEN 'ETCL' 
        WHEN (gotEntry = 1 AND gotLoss = 1) THEN 'ETLS'
        ELSE 'NOET' 
    END, 1
) OVER (ORDER BY Datetime DESC) AS TmPred
from cte
order by Datetime DESC


