from ScheduledJobs.stockdatadownloader import *
from ScheduledJobs.mkdayanalyzer import *
from ScheduledJobs.mkmonthanalyzer import *
from ScheduledJobs.mkdayprobability import *
from ScheduledJobs.mkdayma import *
from ScheduledJobs.mkintervalanalyzer import *
from ScheduledJobs.mkdayprediction import *
from ScheduledJobs.JobTomorrowAnalyzer import *

if __name__ == "__main__":
    resultJob1 = JobStockDataDownloader()
    resultJob2 = JobmkDayAnalyzer() 
    resultJob3 = JobmkMonthAnalyzer()
    resultJob4 = JobmkDayProbability()
    resultJob5 = JobmkDayMa()
    resultJob6 = JobmkIntervalAnalyzer()
    resultJob7 = JobmkDayPrediction()
    resultJob8 = JobTomorrowAnalyzer()
    
# use mkanalyzer;
# with main as (
# 	select tickerName, max(Datetime) as Datetime from mkDayFeature
# 	group by tickerName
# 	)
# select m.[tickerName], m.[Datetime], f.[Open], f.[High], f.[Low], f.[Close], 
# f.[PvClose], f.[OC-P/L], f.[PvCC-P/L], f.[maxHigh], f.[maxLow], f.[closeTolerance], f.[priceBand],
# pb.[BuyInProfit MP::HP::MP::HP], pb.[SellInLoss MP::MP::LP::LP], pb.[BuyInLoss MP::HP::LP::HP], pb.[SellInProfit MP::HP::LP::LP],
# pb.[ProbabilityOfProfitMT2Percent], pb.[ProbabilityOfLoss1ratio3Percent], pb.[ProbabilityOfProfitTomorrow], pb.[ProbabilityOfLossTomorrow], 
# pb.[ProbabilityOfProfitLoss], pb.[ProbabilityOfmaxHigh], pb.[ProbabilityOfmaxLow], pb.[ProbabilityOfpriceBand], pb.[ProbabilityOfCloseTolerance],
# pd.[predTmOpen], pd.[predTmEntry1], pd.[predTmExit1], pd.[predTmEntry2], 
# pd.[predTmExit2], pd.[predTmClose], pd.[predTmMaxhigh], pd.[predTmMaxlow],
# pd.[EtEx1Profit], pd.[EtEx2Profit], pd.[predTmP/L]
# from main m
# left join mkDayFeature f on m.tickerName=f.tickerName and m.Datetime=f.Datetime
# left join mkDayProbability pb on m.tickerName=pb.tickerName and m.Datetime=pb.Datetime
# left join mkDayPrediction pd on m.tickerName=pd.tickerName and m.Datetime=pd.Datetime
# order by pb.[ProbabilityOfProfitMT2Percent] desc, pb.[ProbabilityOfProfitTomorrow] desc
