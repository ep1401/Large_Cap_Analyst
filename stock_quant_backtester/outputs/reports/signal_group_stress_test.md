# Signal Group Stress Test

- Back-tested performance is hypothetical.
- Snapshot analyst target models are excluded from the main historically safer ranking.
- Historical rating-count features use dated FMP grades-historical records available on or before each rebalance date.
- News sentiment depends on Alpha Vantage coverage and classification.
- Long/short is experimental and currently not recommended.
- Regime filters were tested and are not recommended for the main model based on current results.
- This is research/paper trading only, not financial advice.

- Essential signal groups (removed variant loses at least 2 percentage points of walk-forward average excess): ratings_removed, events_removed, technical_removed, sentiment_removed, risk_penalty_removed.
- Removable or low-impact signal groups (removed variant loses less than 0.5 percentage points): none flagged.

| variant_name             | walk_forward_average_excess_vs_spy | 2025_excess_return_vs_spy | max_drawdown | average_turnover | delta_vs_full_walk_forward |
| ------------------------ | ---------------------------------- | ------------------------- | ------------ | ---------------- | -------------------------- |
| full_incumbent           | 0.0755                             | 0.1395                    | -0.176       | 0.672            | 0.0                        |
| ratings_weight_half      | 0.036                              | 0.0609                    | -0.218       | 0.8147           | -0.0395                    |
| ratings_removed          | -0.0501                            | -0.0227                   | -0.2186      | 0.9213           | -0.1257                    |
| events_weight_half       | 0.0225                             | -0.0038                   | -0.2148      | 0.6733           | -0.0531                    |
| events_removed           | 0.0036                             | -0.0091                   | -0.1971      | 0.668            | -0.0719                    |
| technical_weight_half    | 0.023                              | 0.0336                    | -0.177       | 0.6853           | -0.0525                    |
| technical_removed        | -0.0143                            | -0.0001                   | -0.1984      | 0.6933           | -0.0898                    |
| sentiment_weight_half    | 0.0318                             | 0.079                     | -0.1976      | 0.568            | -0.0437                    |
| sentiment_removed        | 0.0156                             | 0.1002                    | -0.1889      | 0.5013           | -0.06                      |
| risk_penalty_weight_half | 0.0587                             | 0.072                     | -0.1855      | 0.668            | -0.0169                    |
| risk_penalty_removed     | 0.0472                             | 0.0708                    | -0.1938      | 0.656            | -0.0284                    |