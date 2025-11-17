WITH TimeFilteredData AS (
    SELECT 
        [Datetime],
        CAST([Datetime] AS DATE) AS TradeDate,
        [High], 
        [Low], 
        [Open], 
        [Close]
    FROM 
        BLS
    WHERE 
        (DATEPART(HOUR, [Datetime]) = 14 AND DATEPART(MINUTE, [Datetime]) BETWEEN 15 AND 59)
        OR 
        (DATEPART(HOUR, [Datetime]) = 15 AND DATEPART(MINUTE, [Datetime]) BETWEEN 0 AND 25)
),
AggregatedData AS (
    SELECT
        TradeDate,
        MAX(High) AS MaxHigh,
        MIN(Low) AS MinLow,
        MIN([Datetime]) AS MinDatetime,
        MAX([Datetime]) AS MaxDatetime
    FROM 
        TimeFilteredData
    GROUP BY 
        TradeDate
),
OpenCloseValues AS (
    SELECT
        tfd.TradeDate,
        ad.MaxHigh,
        ad.MinLow,
        MIN(CASE WHEN tfd.[Datetime] = ad.MinDatetime THEN tfd.[Open] END) AS OpenValue,
        MIN(CASE WHEN tfd.[Datetime] = ad.MaxDatetime THEN tfd.[Close] END) AS CloseValue
    FROM 
        TimeFilteredData tfd
    JOIN 
        AggregatedData ad 
    ON 
        tfd.TradeDate = ad.TradeDate
    GROUP BY 
        tfd.TradeDate, ad.MaxHigh, ad.MinLow, ad.MinDatetime, ad.MaxDatetime
)
SELECT
    TradeDate,
    MaxHigh,
    MinLow,
    OpenValue AS OpenPoint,
    CloseValue AS ClosePoint
FROM 
    OpenCloseValues
ORDER BY 
    TradeDate DESC;
