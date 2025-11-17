------------------------------ Previous 10 Day average ----------------------------
DECLARE @CurrentDate DATE = '2024-12-12 09:15:00.000';
WITH CTE AS (
	SELECT avg(Volume) as TenDayVolume FROM mkintervalmaster.dbo.[BLS]
	WHERE CAST(Datetime AS DATE) IN (
		SELECT TOP 10 CAST(Datetime AS DATE) FROM mkintervalmaster.dbo.[BLS]
		WHERE CAST(Datetime AS DATE) < CAST(@CurrentDate AS DATE)
		GROUP BY CAST(Datetime AS DATE)
		ORDER BY CAST(Datetime AS DATE) DESC
	)
)
SELECT *,ROUND((ISNULL((SELECT [VOLUME] FROM mkintervalmaster.dbo.[BLS] WHERE Datetime = @CurrentDate), 0) - TenDayVolume)/TenDayVolume, 2)*100 AS VolumeSpike FROM CTE

------------------------------ Previous Close 1 ----------------------------
DECLARE @CurrentDate DATE = '2024-12-12 09:15:00.000';
WITH CTE AS (
	SELECT [Close] as PreviousClose FROM mkdaymaster.dbo.[BLS] 
	WHERE CAST(Datetime AS DATE) IN (
		SELECT TOP 1 CAST(Datetime AS DATE) FROM mkdaymaster.dbo.[BLS]
		WHERE CAST(Datetime AS DATE) < CAST(@CurrentDate AS DATE)
		GROUP BY CAST(Datetime AS DATE)
		ORDER BY CAST(Datetime AS DATE) DESC
	)
)
SELECT ROUND((ISNULL((SELECT [Close] FROM mkintervalmaster.dbo.[BLS] WHERE Datetime = @CurrentDate), 0) - PreviousClose)/PreviousClose, 2)*100 AS VolumeSpike FROM CTE

------------------------------ Previous 14 Canddle ----------------------------
DECLARE @CurrentDate DATE = '2024-12-12 09:15:00.000';
WITH CTE AS (
SELECT TOP 14 *, LAG([Close]) OVER (ORDER BY Datetime) AS PreviousCloseInterval 
FROM mkintervalmaster.dbo.[BLS] 
WHERE CAST(Datetime AS DATE) <= CAST(@CurrentDate AS DATE)
ORDER BY Datetime DESC
),
ATR AS (
	SELECT 
	ROUND(SUM(Round(CASE 
		WHEN [High] - [Low] >= ABS([High] - PreviousCloseInterval) 
			 AND [High] - [Low] >= ABS([Low] - PreviousCloseInterval) THEN [High] - [Low]
		WHEN ABS([High] - PreviousCloseInterval) >= ABS([Low] - PreviousCloseInterval) THEN ABS([High] - PreviousCloseInterval)
		ELSE ABS([Low] - PreviousCloseInterval)
	END, 2))/14, 2) AS ATR
	FROM CTE
)
SELECT 
CASE 
	WHEN ATR > ROUND((SELECT [Close] FROM mkintervalmaster.dbo.[BLS] WHERE Datetime = @CurrentDate) * 0.015, 2) 
	THEN 'QUALIFIED' ELSE 'NOT QUALIFIED' 
END AS ATR
FROM ATR




------------------------------ VWAP 75 Canddle ----------------------------
DECLARE @CurrentDate DATE = '2024-12-12 09:15:00.000';
WITH CTE AS (
    SELECT TOP 75 *
    FROM mkintervalmaster.dbo.[BLS]
	WHERE CAST(Datetime AS DATE) <= CAST(@CurrentDate AS DATE)
    ORDER BY Datetime DESC
),
VWAP AS(
	SELECT ROUND(SUM([Close] * [Volume]) / SUM([Volume]), 2) AS VWAP FROM CTE
)
SELECT 
CASE 
	WHEN VWAP < ROUND(ISNULL((SELECT [Close] FROM mkintervalmaster.dbo.[BLS] WHERE Datetime = @CurrentDate), 0), 2) 
	THEN 'BULLISH' ELSE 'NOT BULLISH' 
END AS VWAP
FROM VWAP
