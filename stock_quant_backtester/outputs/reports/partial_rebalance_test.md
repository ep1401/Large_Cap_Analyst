# Partial Rebalance Test

- Back-tested performance is hypothetical.
- Snapshot analyst target models are excluded from the main historically safer ranking.
- Historical rating-count features use dated FMP grades-historical records available on or before each rebalance date.
- News sentiment depends on Alpha Vantage coverage and classification.
- Long/short is experimental and currently not recommended.
- Regime filters were tested and are not recommended for the main model based on current results.
- The current model is cost-sensitive and should remain paper-trading only unless it survives realistic cost assumptions.
- This is research/paper trading only, not financial advice.

| variant_type           | max_turnover_per_rebalance | total_cost_bps | walk_forward_average_excess_vs_spy | 2025_excess_return_vs_spy | max_drawdown | average_turnover |
| ---------------------- | -------------------------- | -------------- | ---------------------------------- | ------------------------- | ------------ | ---------------- |
| base                   | nan                        | 10             | 0.0755                             | 0.1395                    | -0.176       | 0.6952           |
| low_turnover_hold_band | 0.25                       | 10             | 0.0622                             | 0.0614                    | -0.2234      | 0.6917           |
| low_turnover_hold_band | nan                        | 10             | 0.0369                             | 0.0793                    | -0.1786      | 0.4317           |
| low_turnover_hold_band | 0.5                        | 10             | 0.0341                             | 0.0528                    | -0.1841      | 0.6638           |
| low_turnover_hold_band | 0.75                       | 10             | -0.0009                            | 0.0041                    | -0.1996      | 0.5462           |
| base                   | 0.25                       | 10             | -0.0681                            | 0.1538                    | -0.2688      | 1.0              |
| base                   | 0.75                       | 10             | -0.0903                            | 0.0482                    | -0.1927      | 1.0              |
| base                   | 0.5                        | 10             | -0.1406                            | -0.0535                   | -0.2488      | 1.0              |
| base                   | nan                        | 20             | 0.0495                             | 0.1037                    | -0.1819      | 0.6952           |
| low_turnover_hold_band | 0.25                       | 20             | 0.0362                             | 0.0249                    | -0.2329      | 0.6917           |
| low_turnover_hold_band | nan                        | 20             | 0.0207                             | 0.0569                    | -0.1916      | 0.4317           |
| low_turnover_hold_band | 0.5                        | 20             | 0.0089                             | 0.0158                    | -0.1992      | 0.6638           |
| low_turnover_hold_band | 0.75                       | 20             | -0.0196                            | -0.0202                   | -0.2157      | 0.5462           |
| base                   | 0.25                       | 20             | -0.1052                            | 0.0903                    | -0.3269      | 1.0              |
| base                   | 0.75                       | 20             | -0.126                             | -0.0104                   | -0.2526      | 1.0              |
| base                   | 0.5                        | 20             | -0.1743                            | -0.1073                   | -0.3116      | 1.0              |
| base                   | nan                        | 30             | 0.024                              | 0.0689                    | -0.1877      | 0.6952           |
| low_turnover_hold_band | 0.25                       | 30             | 0.0108                             | -0.0106                   | -0.2424      | 0.6917           |
| low_turnover_hold_band | nan                        | 30             | 0.0048                             | 0.0349                    | -0.205       | 0.4317           |
| low_turnover_hold_band | 0.5                        | 30             | -0.0156                            | -0.0202                   | -0.2211      | 0.6638           |
| low_turnover_hold_band | 0.75                       | 30             | -0.0379                            | -0.044                    | -0.2315      | 0.5462           |
| base                   | 0.25                       | 30             | -0.1409                            | 0.0298                    | -0.3833      | 1.0              |
| base                   | 0.75                       | 30             | -0.1604                            | -0.0662                   | -0.3152      | 1.0              |
| base                   | 0.5                        | 30             | -0.2068                            | -0.1586                   | -0.3692      | 1.0              |