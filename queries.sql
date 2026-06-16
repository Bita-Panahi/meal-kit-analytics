-- queries.sql
-- ============
-- The SQL behind the "Overview" tab. Every query runs against data/data.duckdb,
-- which generate_data.py created. You can run these in the 01_overview_sql
-- notebook, or in any DuckDB client. Each query is commented so you can read it
-- aloud in an interview and explain exactly what it does.


-- Q1: How many customers do we have in total?
-- COUNT(*) counts the rows in the customers table (one row = one customer).
SELECT COUNT(*) AS total_customers
FROM customers;


-- Q2: Headline order numbers.
-- COUNT(*)         -> how many boxes were ordered all year
-- SUM(box_price_nok) -> total revenue in NOK
-- AVG(delivered_on_time::INT) -> on-time rate (TRUE becomes 1, FALSE becomes 0,
--                                so the average is the fraction delivered on time)
SELECT
    COUNT(*)                         AS total_orders,
    SUM(box_price_nok)               AS total_revenue_nok,
    AVG(delivered_on_time::INT)      AS on_time_rate
FROM orders;


-- Q3: Orders and revenue per brand.
-- We JOIN orders to customers so each order knows its brand, then GROUP BY brand
-- to get one summary row per brand. ORDER BY ... DESC puts the biggest first.
SELECT
    c.brand,
    COUNT(*)             AS orders,
    SUM(o.box_price_nok) AS revenue_nok
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
GROUP BY c.brand
ORDER BY revenue_nok DESC;


-- Q4: Revenue per region (same idea, grouped by city instead of brand).
SELECT
    c.region,
    COUNT(*)             AS orders,
    SUM(o.box_price_nok) AS revenue_nok
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
GROUP BY c.region
ORDER BY revenue_nok DESC;


-- Q5: Orders per week (the time series the Demand tab forecasts).
-- GROUP BY week gives one row per week with that week's order count.
SELECT
    week,
    COUNT(*) AS orders
FROM orders
GROUP BY week
ORDER BY week;
