from ScheduledJobs.stockdatadownloader import *
from ScheduledJobs.mkdayanalyzer import *
from ScheduledJobs.mkmonthanalyzer import *
from ScheduledJobs.mkdayprobability import *
from ScheduledJobs.mkdayma import *
from ScheduledJobs.mkintervalanalyzer import *
from ScheduledJobs.mkdayprediction import *
from ScheduledJobs.JobTomorrowAnalyzer import *
from ScheduledJobs.simulationPred import *
from ScheduledJobs.JobSimPredMSE import *

if __name__ == "__main__":
    resultJob1 = JobStockDataDownloader()
    resultJob2 = JobmkDayAnalyzer() 
    resultJob3 = JobmkMonthAnalyzer()
    resultJob4 = JobmkDayProbability()
    resultJob5 = JobmkDayMa()
    resultJob6 = JobmkIntervalAnalyzer()
    resultJob7 = JobmkDayPrediction()
    resultJob8 = JobTomorrowAnalyzer()
    resultJob9 = JobSimulationPrediction()
    resultJob10 = JobSimPredMSE()



  # SELECT 
  #       CASE 
  #           WHEN unpvt.ActualDatetime IS NULL THEN DATEADD(DAY, 1, unpvt.Datetime)
  #           ELSE unpvt.ActualDatetime 
  #       END AS ActualDatetime, 
  #       CASE 
  #           WHEN unpvt.predDatetime IS NULL THEN DATEADD(DAY, 1, unpvt.Datetime)
  #           ELSE unpvt.predDatetime 
  #       END AS predDatetime, 
  #       YEAR(unpvt.Datetime) AS [Year], 
 	#     MONTH(unpvt.Datetime) AS [MonthNumber], 
 	#     DAY(unpvt.Datetime) AS [DayNumber],
  #       unpvt.tickerName, 
  #       unpvt.Features, 
  #       unpvt.Value
  #   FROM 
  #       (
  #           SELECT 
  #               dp.*, 
  #               df.Datetime AS ActualDatetime,
  #               ip.Entry1, 
  #               ip.Exit1, 
  #               ip.EtExProfit1, 
  #               ip.Entry2, 
  #               ip.Exit2, 
  #               ip.EtExProfit2, 
  #               df.[Open], 
  #               df.[High], 
  #               df.[Low], 
  #               df.[Close], 
  #               df.[OC-P/L]
  #           FROM 
  #               (SELECT *, LEAD(Datetime, 1) OVER (ORDER BY Datetime) AS predDatetime FROM mkDayPrediction WHERE tickerName = '"& TickerFilterP &"') dp
  #           LEFT JOIN 
  #               (SELECT Datetime, Entry1, Exit1, EtExProfit1, Entry2, Exit2, EtExProfit2 FROM mkIntervalFeature WHERE tickerName = '"& TickerFilterP &"') ip 
  #           ON 
  #               dp.predDatetime = ip.Datetime
  #           LEFT JOIN 
  #               (SELECT Datetime, [Open], [High], [Low], [Close], [OC-P/L] FROM mkDayFeature WHERE tickerName = '"& TickerFilterP &"') df 
  #           ON 
  #               dp.predDatetime = df.Datetime
  #       ) AS merged
  #   UNPIVOT (
  #       Value FOR Features IN (
  #           [predTmOpen], [predTmEntry1], [predTmExit1], [predTmEntry2], [predTmExit2], [predTmClose], 
  #           [predTmMaxhigh], [predTmMaxlow], [EtEx1Profit], [EtEx2Profit], [predTmP/L], 
  #           [Entry1], [Exit1], [EtExProfit1], [Entry2], [Exit2], [EtExProfit2], [Open], [High], [Low], [Close], [OC-P/L]
  #       )
  #   ) AS unpvt