# Rebalance Frequency Test

- Back-tested performance is hypothetical.
- Snapshot analyst target models are excluded from the main historically safer ranking.
- Historical rating-count features use dated FMP grades-historical records available on or before each rebalance date.
- News sentiment depends on Alpha Vantage coverage and classification.
- Long/short is experimental and currently not recommended.
- Regime filters were tested and are not recommended for the main model based on current results.
- The current model is cost-sensitive and should remain paper-trading only unless it survives realistic cost assumptions.
- This is research/paper trading only, not financial advice.

| variant_type           | rebalance_frequency_days | max_holding_days | total_cost_bps | walk_forward_average_excess_vs_spy | 2025_excess_return_vs_spy | max_drawdown | average_turnover |
| ---------------------- | ------------------------ | ---------------- | -------------- | ---------------------------------- | ------------------------- | ------------ | ---------------- |
| low_turnover_hold_band | 15                       | 21               | 10             | 0.162                              | 0.3587                    | -0.1365      | 0.7792           |
| low_turnover_hold_band | 15                       | 30               | 10             | 0.162                              | 0.3587                    | -0.1365      | 0.7792           |
| fixed                  | 15                       | 5                | 10             | 0.1566                             | 0.3353                    | -0.1525      | 0.975            |
| low_turnover_hold_band | 15                       | 5                | 10             | 0.1566                             | 0.3353                    | -0.1525      | 0.975            |
| fixed                  | 15                       | 10               | 10             | 0.1566                             | 0.3353                    | -0.1525      | 0.975            |
| low_turnover_hold_band | 15                       | 10               | 10             | 0.1566                             | 0.3353                    | -0.1525      | 0.975            |
| fixed                  | 15                       | 15               | 10             | 0.1566                             | 0.3353                    | -0.1525      | 0.975            |
| low_turnover_hold_band | 15                       | 15               | 10             | 0.1566                             | 0.3353                    | -0.1525      | 0.975            |
| fixed                  | 15                       | 21               | 10             | 0.1566                             | 0.3353                    | -0.1525      | 0.975            |
| fixed                  | 15                       | 30               | 10             | 0.1566                             | 0.3353                    | -0.1525      | 0.975            |
| fixed                  | 10                       | 5                | 10             | 0.1357                             | 0.3396                    | -0.1145      | 0.8583           |
| low_turnover_hold_band | 10                       | 5                | 10             | 0.1357                             | 0.3396                    | -0.1145      | 0.8583           |
| fixed                  | 10                       | 10               | 10             | 0.1357                             | 0.3396                    | -0.1145      | 0.8583           |
| low_turnover_hold_band | 10                       | 10               | 10             | 0.1357                             | 0.3396                    | -0.1145      | 0.8583           |
| fixed                  | 10                       | 15               | 10             | 0.1357                             | 0.3396                    | -0.1145      | 0.8583           |
| fixed                  | 10                       | 21               | 10             | 0.1357                             | 0.3396                    | -0.1145      | 0.8583           |
| fixed                  | 10                       | 30               | 10             | 0.1357                             | 0.3396                    | -0.1145      | 0.8583           |
| low_turnover_hold_band | 10                       | 15               | 10             | 0.1287                             | 0.3375                    | -0.1069      | 0.6861           |
| low_turnover_hold_band | 10                       | 21               | 10             | 0.1275                             | 0.3208                    | -0.1205      | 0.6083           |
| low_turnover_hold_band | 10                       | 30               | 10             | 0.1275                             | 0.3208                    | -0.1205      | 0.6083           |
| low_turnover_hold_band | 21                       | 30               | 10             | 0.0857                             | 0.1319                    | -0.1451      | 0.7714           |
| fixed                  | 21                       | 5                | 10             | 0.0809                             | 0.1322                    | -0.1697      | 1.0629           |
| low_turnover_hold_band | 21                       | 5                | 10             | 0.0809                             | 0.1322                    | -0.1697      | 1.0629           |
| fixed                  | 21                       | 10               | 10             | 0.0809                             | 0.1322                    | -0.1697      | 1.0629           |
| low_turnover_hold_band | 21                       | 10               | 10             | 0.0809                             | 0.1322                    | -0.1697      | 1.0629           |
| fixed                  | 21                       | 15               | 10             | 0.0809                             | 0.1322                    | -0.1697      | 1.0629           |
| low_turnover_hold_band | 21                       | 15               | 10             | 0.0809                             | 0.1322                    | -0.1697      | 1.0629           |
| fixed                  | 21                       | 21               | 10             | 0.0809                             | 0.1322                    | -0.1697      | 1.0629           |
| low_turnover_hold_band | 21                       | 21               | 10             | 0.0809                             | 0.1322                    | -0.1697      | 1.0629           |
| fixed                  | 21                       | 30               | 10             | 0.0809                             | 0.1322                    | -0.1697      | 1.0629           |