select top 10 * from simulationPrediction


select top 10 * from simulationPrediction
where Entry2 <= predTmEntry2 and Exit2 >= predTmExit2 and Datetime = (
select max(Datetime) from simulationPrediction where tickerName = 'BLS'
)


-------------------------------
select *, round(((predTmExit2 - predTmEntry2)/predTmExit2)*100,2) as Profit from simulationPrediction
where Entry2 <= predTmEntry2 and Exit2 >= predTmExit2  and Datetime = '2024-07-18 00:00:00.000'
order by Datetime DESC, Profit Desc

------------------------------
select tickerName, count(Datetime) as counts from simulationPrediction
where Entry2 <= predTmEntry2 and Exit2 >= predTmExit2  and Datetime >= '2024-07-01 00:00:00.000' 
group by tickerName
order by counts DESC
-------------------
select *, round(((predTmExit2 - predTmEntry2)/predTmExit2)*100,2) as Profit from simulationPrediction
where Entry2 <= predTmEntry2 and Exit2 >= predTmExit2  and Datetime >= '2024-07-01 00:00:00.000' and tickerName = 'HINDOILEXP'
order by Datetime DESC, Profit Desc
---------------
select tickerName, datetime, predDatetime, Entry2, Exit2, 
predTmEntry2, predTmExit2, [close],
round(((predTmExit2 - predTmEntry2)/predTmExit2)*100,2) as predProfit,
round(((Exit2 - Entry2)/Exit2)*100,2) as actualProfit
from simulationPrediction
where Datetime >= '2024-07-01 00:00:00.000' and tickerName = 'HINDOILEXP'
order by Datetime DESC, predProfit Desc


