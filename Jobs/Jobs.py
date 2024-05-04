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
