# Signal Degradation Test

- Back-tested performance is hypothetical.
- Snapshot analyst target models are excluded from the main historically safer ranking.
- Historical rating-count features use dated FMP grades-historical records available on or before each rebalance date.
- News sentiment depends on Alpha Vantage coverage and classification.
- Long/short is experimental and currently not recommended.
- Regime filters were tested and are not recommended for the main model based on current results.
- This is research/paper trading only, not financial advice.

- Fragile to small score changes (0.05 noise): False.

| noise_std | average_excess_vs_spy | average_walk_forward_excess | pct_runs_beating_spy | drawdown_p25 | drawdown_median | drawdown_p75 |
| --------- | --------------------- | --------------------------- | -------------------- | ------------ | --------------- | ------------ |
| 0.05      | 0.1567                | 0.0459                      | 0.98                 | -0.198       | -0.1908         | -0.1793      |
| 0.1       | 0.0822                | 0.0273                      | 0.78                 | -0.2045      | -0.1931         | -0.1789      |
| 0.2       | -0.005                | 0.0056                      | 0.44                 | -0.2077      | -0.1923         | -0.1738      |
| 0.3       | -0.0714               | -0.0114                     | 0.32                 | -0.2092      | -0.1928         | -0.1801      |