use mkdaymaster;
SELECT 
    CONVERT(date, o.Datetime) AS Date, o.[Open], o.[High], o.[Low], o.[Close], o.[Volume],
    ((MAX(o.High) - MIN(o.Low)) / MAX(o.[Close])) AS BASR_Day,
    SUM(o.Volume) / COUNT(DISTINCT CONVERT(date, o.Datetime)) AS ATV_Day,
    ((MAX(o.High) - MIN(o.Low)) / MAX(o.[Close])) * LOG(SUM(o.Volume) / COUNT(DISTINCT CONVERT(date, o.Datetime))) AS LS_Day, 
	mif.nCandleBelowOpen, mif.pCandleAboveOpen, mif.nCandle, mif.pCandle, mif.Hits44MA,
	case when (sp.Entry2 <= sp.predTmEntry2 AND (sp.Exit2 >= sp.predTmExit2 OR sp.predTmEntry2 < sp.[Close])) then 'YES' else 'NO' end as entry_close,
	case when (sp.Entry2 <= sp.predTmEntry2 AND sp.Exit2 >= sp.predTmExit2) then 'YES' else 'NO' end as entry_exit,
	case when (sp.Entry2 <= sp.predTmEntry2 AND sp.predTmEntry2 >= sp.[Close]) then 'YES' else 'NO' end as entry_loss,
	sp.Entry2, sp.Exit2, sp.predTmEntry2, sp.predTmExit2, round(sp.Entry2-sp.predTmEntry2, 2) AS diff, round(sp.predTmEntry2-sp.[Close], 2) AS loss,
	round(sp.predTmEntry2-sp.[Low], 2) AS low_loss
FROM 
	KOTAKBANK AS o
LEFT JOIN (select * from mkanalyzer.dbo.mkIntervalFeature where tickerName = 'KOTAKBANK') AS mif ON mif.Datetime = o.Datetime
LEFT JOIN (select * from mkanalyzer.dbo.simulationPrediction where tickerName = 'KOTAKBANK') AS sp ON sp.predDatetime = o.Datetime
GROUP BY 
    CONVERT(date, o.Datetime), o.[Open], o.[High], o.[Low], o.[Close], o.[Volume], 
	mif.nCandleBelowOpen, mif.pCandleAboveOpen, mif.nCandle, mif.pCandle, mif.Hits44MA,
	sp.Entry2, sp.predTmEntry2, sp.Exit2, sp.predTmExit2, sp.[Close], sp.[Low]
ORDER BY
    Date DESC

