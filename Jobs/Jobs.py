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
from models.MLModelsMain import jobPredictionModel

if __name__ == "__main__":
    resultJob1 = JobStockDataDownloader() # 45s + 1m = 2m
    resultJob2 = JobmkDayAnalyzer() # 45s 
    resultJob3 = JobmkMonthAnalyzer() # 15s 
    resultJob4 = JobmkDayProbability() # 5.20m
    resultJob5 = JobmkDayMa() # 2.20m
    resultJob6 = JobmkIntervalAnalyzer() # 40m
    resultJob7 = JobmkDayPrediction() # 2m
    resultJob8 = JobTomorrowAnalyzer() # 20s
    resultJob9 = JobSimulationPrediction() # 2m
    resultJob10 = JobSimPredMSE() # 30s
    result11 = jobPredictionModel() # 31m
