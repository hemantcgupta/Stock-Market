----11111--------------------- Top 1 based on epochLoss filter TmPredPL after the assigning row number -----------------------------------
WITH cte AS (
SELECT 
    tas1.Datetime as Date, 
    tas1.tickerName, 
	tas1.successCount,
    tas1.modelAccuracy,
	tas1.epochLoss,
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
	ROW_NUMBER() OVER (PARTITION BY tas1.Datetime ORDER BY successCount DESC, epochLoss ASC) as rn
FROM topAccurateStats1 AS tas1
LEFT JOIN simulationPrediction AS sp 
ON tas1.tickerName = sp.tickerName AND tas1.Datetime = sp.Datetime
)
SELECT Date, tickerName, TmPL, TmPredPL, gotEntry, gotLoss, PredEntry, PredExit, PredOpen, 
ActualEntry, ActulExit, ActualOpen, ActualClose, [AOpen/POpen-Diff], ActualProfit, PredProfit, 
successCount, modelAccuracy, epochLoss, pMomentum, nMomentum, buySignal, sellSignal, holdingSignal
FROM cte 
WHERE rn = 1 AND TmPredPL = 1 --AND gotEntry = 1
ORDER BY Date DESC;


-----22222222222-------------------- Top 1 based on epochLoss  filter TmPredPl before row number -----------------------------------
WITH cte AS (
SELECT 
    tas1.Datetime as Date, 
    tas1.tickerName, 
	tas1.successCount,
    tas1.modelAccuracy,
	tas1.epochLoss,
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
	ROW_NUMBER() OVER (PARTITION BY tas1.Datetime ORDER BY successCount DESC, epochLoss ASC) as rn
FROM topAccurateStats1 AS tas1
LEFT JOIN simulationPrediction AS sp 
ON tas1.tickerName = sp.tickerName AND tas1.Datetime = sp.Datetime
WHERE tas1.TmPredPL = 1
)
SELECT Date, tickerName, TmPL, TmPredPL, gotEntry, gotLoss, PredEntry, PredExit, PredOpen, 
ActualEntry, ActulExit, ActualOpen, ActualClose, [AOpen/POpen-Diff], ActualProfit, PredProfit, 
successCount, modelAccuracy, epochLoss, pMomentum, nMomentum, buySignal, sellSignal, holdingSignal
FROM cte 
WHERE rn = 1 AND TmPredPL = 1 AND gotEntry = 1
ORDER BY Date DESC;

 
 
 


----333333333--------------------- Top 1 based on modelAccuracy filter TmPredPL after row number -----------------------------------
WITH cte AS (
SELECT 
    tas1.Datetime as Date, 
    tas1.tickerName, 
	tas1.successCount,
    tas1.modelAccuracy,
	tas1.epochLoss,
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
	ROW_NUMBER() OVER (PARTITION BY tas1.Datetime ORDER BY successCount DESC, modelAccuracy DESC) as rn
FROM topAccurateStats1 AS tas1
LEFT JOIN simulationPrediction AS sp 
ON tas1.tickerName = sp.tickerName AND tas1.Datetime = sp.Datetime
)
SELECT Date, tickerName, TmPL, TmPredPL, gotEntry, gotLoss, PredEntry, PredExit, PredOpen, 
ActualEntry, ActulExit, ActualOpen, ActualClose, [AOpen/POpen-Diff], ActualProfit, PredProfit, 
successCount, modelAccuracy, epochLoss, pMomentum, nMomentum, buySignal, sellSignal, holdingSignal
FROM cte 
WHERE rn = 1 AND TmPredPL = 1 AND gotEntry = 1
ORDER BY Date DESC;

--4444444------------------------ Top 1 based on modelAccuracy filter TmPredPL before row number -----------------------------------
WITH cte AS (
SELECT 
    tas1.Datetime as Date, 
    tas1.tickerName, 
	tas1.successCount,
    tas1.modelAccuracy,
	tas1.epochLoss,
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
	ROW_NUMBER() OVER (PARTITION BY tas1.Datetime ORDER BY successCount DESC, modelAccuracy DESC) as rn
FROM topAccurateStats1 AS tas1
LEFT JOIN simulationPrediction AS sp 
ON tas1.tickerName = sp.tickerName AND tas1.Datetime = sp.Datetime
WHERE tas1.TmPredPL = 1
)
SELECT Date, tickerName, TmPL, TmPredPL, gotEntry, gotLoss, PredEntry, PredExit, PredOpen, 
ActualEntry, ActulExit, ActualOpen, ActualClose, [AOpen/POpen-Diff], ActualProfit, PredProfit, 
successCount, modelAccuracy, epochLoss, pMomentum, nMomentum, buySignal, sellSignal, holdingSignal
FROM cte 
WHERE rn = 1 AND TmPredPL = 1 AND gotEntry = 1
ORDER BY Date DESC;

 
-------------------------------------------

WITH cte AS (
SELECT 
    tas1.Datetime as Date, 
    tas1.tickerName, 
	tas1.successCount,
    tas1.modelAccuracy,
	tas1.epochLoss,
	tas1.pMomentum,
	tas1.nMomentum,
	tas1.buySignal,
	tas1.sellSignal,
	tas1.holdingSignal,
	sp.EtEx2Profit*0.7 as PredProfit,
    CASE 
        WHEN (Entry2 <= predTmEntry2 AND (Exit2 >= (predTmEntry2+((predTmExit2-predTmEntry2)/predTmExit2)*70) OR predTmEntry2 < [Close])) 
        THEN 1 
        ELSE 0 
    END AS TmPL, 
    tas1.TmPredPL,
        CASE 
        WHEN (Entry2 <= predTmEntry2) OR (predTmEntry2 >= [Close])
        THEN 1 
        ELSE 0 
    END AS gotEntry,
    CASE 
        WHEN (Entry2 <= predTmEntry2 AND (Exit2 >= (predTmEntry2+((predTmExit2-predTmEntry2)/predTmExit2)*70) OR predTmEntry2 < [Close])) 
        THEN 0
        WHEN (Entry2 <= predTmEntry2 AND predTmEntry2 >= [Close]) OR (predTmEntry2 >= [Close])
        THEN 1 
        ELSE 0 
    END AS gotLoss,
    CASE 
        WHEN (Entry2 <= predTmEntry2 AND Exit2 >= (predTmEntry2+((predTmExit2-predTmEntry2)/predTmExit2)*70)) 
        THEN round((((predTmEntry2+((predTmExit2-predTmEntry2)/predTmExit2)*70)-predTmEntry2)/(predTmEntry2+((predTmExit2-predTmEntry2)/predTmExit2)*70)) * 100, 2)
        ELSE round((([Close]-predTmEntry2)/[Close]) * 100, 2)
    END AS ActualProfit,
    round(((sp.[Open]-sp.[predTmOpen])/sp.[predTmOpen])*100, 2) AS 'AOpen/POpen-Diff',
    Entry2 As ActualEntry, predTmEntry2 as PredEntry, Exit2 as ActulExit, (predTmEntry2+((predTmExit2-predTmEntry2)/predTmExit2)*70) as PredExit, 
    [predTmOpen] as PredOpen, [Open] as ActualOpen, [Close] as ActualClose,
	(pMomentum * 0.2) +
    (nMomentum * -0.2) +
    (buySignal * 0.2) +
    (sellSignal * -0.1) +
    (holdingSignal * 0.1) +
	(successCount * 0.3) +
	(modelAccuracy * 0.4) +
	(epochLoss * -0.2) 
	AS priorityScore,
	ROW_NUMBER() OVER (PARTITION BY tas1.Datetime ORDER BY 
        (pMomentum * 0.2) +
		(nMomentum * -0.2) +
		(buySignal * 0.2) +
		(sellSignal * -0.1) +
		(holdingSignal * 0.1) +
		(successCount * 0.3) +
		(modelAccuracy * 0.4) +
		(epochLoss * -0.2) DESC) as rn
FROM topAccurateStats1 AS tas1
LEFT JOIN simulationPrediction AS sp 
ON tas1.tickerName = sp.tickerName AND tas1.Datetime = sp.Datetime
)
SELECT Date, tickerName, TmPL, TmPredPL, gotEntry, gotLoss, PredEntry, PredExit, PredOpen, 
ActualEntry, ActulExit, ActualOpen, ActualClose, [AOpen/POpen-Diff], ActualProfit, PredProfit, 
successCount, modelAccuracy, epochLoss, pMomentum, nMomentum, buySignal, sellSignal, holdingSignal, priorityScore, rn
FROM cte 
WHERE Date = '2024-08-29' and TmPredPL = 1 --AND gotEntry = 1 --and 
ORDER BY Date DESC;
 


 