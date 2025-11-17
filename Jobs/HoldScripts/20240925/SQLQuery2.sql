with cte as(
	SELECT 
	CAST([Datetime] AS DATE) AS [Date], 
	LAG([Open]) OVER (ORDER BY [Datetime]) AS PreviousOpen,
	LAG([High]) OVER (ORDER BY [Datetime]) AS PreviousHigh,
    LAG([Low]) OVER (ORDER BY [Datetime]) AS PreviousLow,
	LAG([Close]) OVER (ORDER BY [Datetime]) AS PreviousClose, 
	case when LAG([Open]) OVER (ORDER BY [Datetime]) < LAG([Close]) OVER (ORDER BY [Datetime]) then 1 else 0 end as PreviousProfit,
	[Open], [High], [Low], [Close]
FROM BLS
)
SELECT *,
	-- Compare Previous value with Actual Low
	CASE 
        WHEN PreviousOpen > Low THEN 1
        ELSE 0
    END AS IsPrevOpenBrokenByLow,
	CASE 
        WHEN PreviousHigh > Low THEN 1 
        ELSE 0 
    END AS IsPrevHighBrokenByLow,
    CASE 
        WHEN PreviousLow > Low THEN 1 
        ELSE 0 
    END AS IsPrevLowBrokenByLow,
	CASE 
        WHEN PreviousClose > Low THEN 1 
        ELSE 0 
    END AS IsPrevCloseBrokenByLow,

    -- Compare Previous value with Actual High
    CASE 
        WHEN PreviousOpen < High THEN 1
        ELSE 0
    END AS IsPrevOpenBrokenByHigh,
	CASE 
        WHEN PreviousHigh < High THEN 1 
        ELSE 0 
    END AS IsPrevHighBrokenByHigh,
    CASE 
        WHEN PreviousLow < High THEN 1 
        ELSE 0 
    END AS IsPrevLowBrokenByHigh,
	CASE 
        WHEN PreviousClose < High THEN 1 
        ELSE 0 
    END AS IsPrevCloseBrokenByHigh
from cte
where Date >= '2024-06-05'
order by Date 
