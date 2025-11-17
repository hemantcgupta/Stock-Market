# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 17:00:30 2024

@author: Hemant
"""
import pandas as pd
pd.options.display.float_format = '{:.2f}'.format
pd.set_option('display.max_columns', 30)
pd.set_option('display.max_rows', 40)
pd.set_option('expand_frame_repr', True)
from Scripts.TmPrediction import StockAnalyzer, StockMarketPhasesAnalyzer, StockPredictorAnalyzer, TopPredictor

if __name__ == "__main__":
    analyzer = StockAnalyzer()
    df = analyzer.analyze()
    df.head(10)
    df1, allProfit1, monthSummary1 = analyzer.top1(df)
    df2, allProfit2, monthSummary2 = analyzer.top2(df)
    df3, allProfit3, monthSummary3 = analyzer.top3(df)
    dfP = StockMarketPhasesAnalyzer.getMarketPhases(df)
    obj = StockPredictorAnalyzer()
    df = obj.analyze()
    df1, allProfit1, monthSummary1 = obj.top1(df)
    df2, allProfit2, monthSummary2 = obj.top2(df)
    df3, allProfit3, monthSummary3 = obj.top3(df)
    df = TopPredictor().analyze()
