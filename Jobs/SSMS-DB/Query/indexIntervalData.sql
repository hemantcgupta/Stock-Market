WITH IntervalData AS (
    SELECT 
        DATEADD(MINUTE, DATEDIFF(MINUTE, 0, [DateTime]) / 15 * 15, 0) AS [IntervalTime],
        [DateTime], [Open], [High], [Low], [Close],
        ROW_NUMBER() OVER (PARTITION BY DATEADD(MINUTE, DATEDIFF(MINUTE, 0, [DateTime]) / 15 * 15, 0) ORDER BY [DateTime] ASC) AS [RowAsc],
        ROW_NUMBER() OVER (PARTITION BY DATEADD(MINUTE, DATEDIFF(MINUTE, 0, [DateTime]) / 15 * 15, 0) ORDER BY [DateTime] DESC) AS [RowDesc]
    FROM [^NSEBANK]
),
DataWithAggregates as (
	SELECT [IntervalTime] as Datetime,
		MAX(CASE WHEN [RowAsc] = 1 THEN [Open] END) AS [Open],
		MAX([High]) AS [High],
		MIN([Low]) AS [Low],
		MAX(CASE WHEN [RowDesc] = 1 THEN [Close] END) AS [Close],
		CAST([IntervalTime] AS DATE) AS [TradeDate]
	FROM IntervalData
	GROUP BY [IntervalTime]
),
DailyHigh AS (
    SELECT [TradeDate],
        round(0.001 * MAX(CASE WHEN [Datetime] = DATEADD(MINUTE, 9 * 60 + 15, CAST([TradeDate] AS DATETIME)) THEN [High] END) + MAX(CASE WHEN [Datetime] = DATEADD(MINUTE, 9 * 60 + 15, CAST([TradeDate] AS DATETIME)) THEN [High] END),2) AS [High01],
		round(-0.001 * MAX(CASE WHEN [Datetime] = DATEADD(MINUTE, 9 * 60 + 15, CAST([TradeDate] AS DATETIME)) THEN [Low] END) + MAX(CASE WHEN [Datetime] = DATEADD(MINUTE, 9 * 60 + 15, CAST([TradeDate] AS DATETIME)) THEN [Low] END),2) AS [Low01]
    FROM DataWithAggregates
    GROUP BY [TradeDate]
),
data as(
	SELECT D.[Datetime], DH.[TradeDate], D.[Open], D.[High], D.[Low], D.[Close], DH.[High01], DH.[Low01],
	round(DH.[High01]+DH.[High01]*0.004, 2) as P30,
	round(DH.[Low01]-DH.[Low01]*0.004, 2) as L30,
	case when [High] >= [High01] then 1 else 0 end as EP,
	case when [High] >= round(DH.[High01]+DH.[High01]*0.004, 2) then 1 else 0 end as PP,
	case when [Low] <= [Low01] then 1 else 0 end as EL,
	case when [Low] <= round(DH.[Low01]-DH.[Low01]*0.004, 2) then 1 else 0 end as LL
	FROM DataWithAggregates D
	JOIN DailyHigh DH ON D.[TradeDate] = DH.[TradeDate]
)
select TradeDate, max(EP) as EP,
max(PP) as PP,
max(EL) as EL,
max(LL) as LL
from data
group by TradeDate
order by [TradeDate] DESC