WITH cte AS (
SELECT 
    tas1.Datetime as Date, 
    tas1.tickerName, 
	tas1.successCount,
    tas1.modelAccuracyPL,
	tas1.epochLossPL,
	tas1.modelAccuracygotLoss,
	tas1.epochLossgotLoss,
	tas1.pMomentum,
	tas1.nMomentum,
	tas1.buySignal,
	tas1.sellSignal,
	tas1.holdingSignal,
	sp.EtEx2Profit as PredProfit,
    CASE 
        WHEN (Entry2 <= predTmEntry2 AND (Exit2 >= predTmExit2 OR predTmEntry2 < [Close])) 
        THEN 1 
        ELSE 0 
    END AS TmPL, 
    tas1.TmPredPL,
	tas1.TmPredgotLoss,
	tas1.TmPredPL5Summary, tas1.TmPredgotLoss5Summary,
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
        WHEN (Entry2 <= predTmEntry2 AND Exit2 >= predTmExit2) 
        THEN round(((predTmExit2-predTmEntry2)/predTmExit2) * 100, 2)
        ELSE round((([Close]-predTmEntry2)/[Close]) * 100, 2)
    END AS ActualProfit,
    round(((sp.[Open]-sp.[predTmOpen])/sp.[predTmOpen])*100, 2) AS 'AOpen/POpen-Diff',
    Entry2 As ActualEntry, predTmEntry2 as PredEntry, Exit2 as ActulExit, predTmExit2 as PredExit, 
    [predTmOpen] as PredOpen, [Open] as ActualOpen, [Close] as ActualClose, 
	ROW_NUMBER() OVER (PARTITION BY tas1.Datetime ORDER BY successCount DESC, epochLossPL ASC) as rn
FROM mkTopPrediction AS tas1
LEFT JOIN simulationPrediction AS sp 
ON tas1.tickerName = sp.tickerName AND tas1.Datetime = sp.Datetime
--WHERE TmPredPL = 1 and TmPredgotLoss = 0
)
SELECT Date, tickerName, TmPL, TmPredPL, TmPredgotLoss, gotEntry, gotLoss, rn, PredEntry, PredExit, ActualProfit, PredProfit, PredOpen, 
ActualEntry, ActulExit, ActualOpen, ActualClose, [AOpen/POpen-Diff], 
successCount, modelAccuracyPL, epochLossPL, modelAccuracygotLoss, epochLossgotLoss, pMomentum, nMomentum, buySignal, sellSignal, holdingSignal, TmPredPL5Summary, TmPredgotLoss5Summary
FROM cte 
--WHERE rn = 1 AND TmPredPL = 1 AND gotEntry = 1
ORDER BY Date DESC;


SELECT month(cast(Date as Date)) as months, sum(ActualProfit) ActualProfit
FROM cte 
WHERE rn = 1 AND TmPredPL = 1 AND gotEntry = 1
group by month(cast(Date as Date)) 
ORDER BY month(cast(Date as Date)) DESC;

--select * from mkTopPrediction  