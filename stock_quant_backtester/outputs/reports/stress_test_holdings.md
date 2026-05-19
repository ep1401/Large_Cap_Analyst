# Stress Test Holdings

- Back-tested performance is hypothetical.
- Snapshot analyst target models are excluded from the main historically safer ranking.
- Historical rating-count features use dated FMP grades-historical records available on or before each rebalance date.
- News sentiment depends on Alpha Vantage coverage and classification.
- Long/short is experimental and currently not recommended.
- Regime filters were tested and are not recommended for the main model based on current results.
- This is research/paper trading only, not financial advice.

- Best top_n by walk-forward average excess vs SPY: 10.
- top_n=10 stable within 1 percentage point of the best walk-forward average: True.
- top_n=10 concentration risk proxy (1/top_n): 10.00%.

| top_n | walk_forward_average_excess_vs_spy | 2025_excess_return_vs_spy | max_drawdown | average_turnover | average_holdings | concentration_risk |
| ----- | ---------------------------------- | ------------------------- | ------------ | ---------------- | ---------------- | ------------------ |
| 3.0   | -0.0562                            | -0.1684                   | -0.2974      | 0.98             | 2.9              | 0.3333             |
| 5.0   | -0.0417                            | -0.1063                   | -0.2205      | 0.8973           | 4.8333           | 0.2                |
| 8.0   | 0.0346                             | 0.0385                    | -0.2158      | 0.7417           | 7.7333           | 0.125              |
| 10.0  | 0.0755                             | 0.1395                    | -0.176       | 0.672            | 9.6667           | 0.1                |
| 15.0  | 0.0273                             | 0.0613                    | -0.1632      | 0.5382           | 14.5             | 0.0667             |
| 20.0  | 0.0164                             | 0.0207                    | -0.1689      | 0.418            | 19.3333          | 0.05               |
| 25.0  | 0.0143                             | 0.0267                    | -0.1668      | 0.3229           | 24.1667          | 0.04               |