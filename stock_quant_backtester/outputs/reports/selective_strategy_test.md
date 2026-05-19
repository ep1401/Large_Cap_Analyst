# Selective Strategy Test

- Back-tested performance is hypothetical.
- Snapshot analyst target models are excluded from the main historically safer ranking.
- Historical rating-count features use dated FMP grades-historical records available on or before each rebalance date.
- News sentiment depends on Alpha Vantage coverage and classification.
- Long/short is experimental and currently not recommended.
- Regime filters were tested and are not recommended for the main model based on current results.
- This is research/paper trading only, not financial advice.

## Findings
- Best selective configuration: `final_quant_5d_selective_no_snapshot` threshold=0.5, top_n=10, cost_bps=10.
- Best selective walk-forward average excess vs SPY: 4.92%.
- Best selective 2025 excess vs SPY: 1.53%.
- Best selective windows beating SPY: 2/3.
- `score > 0.50` remains best: True.
- Allowing cash improves Sharpe/drawdown versus forced fill for the same setup: True.
- Average holdings for best selective setup: 9.21.
- Average percent invested for best selective setup: 92.13%.
- Selected stocks outperform non-selected stocks on average: True (0.17%).

## Results

| strategy_name                                         | display_name                                | top_n | total_cost_bps | min_score_threshold | allow_cash | walk_forward_average_excess_vs_spy | windows_beating_spy | 2025_excess_return_vs_spy | max_drawdown | average_holdings | average_percent_invested |
| ----------------------------------------------------- | ------------------------------------------- | ----- | -------------- | ------------------- | ---------- | ---------------------------------- | ------------------- | ------------------------- | ------------ | ---------------- | ------------------------ |
| final_quant_5d_no_snapshot_no_sma_filter              | Final Quant 5D - No SMA Filter              | 10    | 10.0           | nan                 | False      | 0.0544                             | 2                   | 0.091                     | -0.1738      | 9.26             | 0.9667                   |
| final_quant_5d_no_recent_downgrade_filter_no_snapshot | Final Quant 5D - No Recent Downgrade Filter | 10    | 10.0           | nan                 | False      | 0.0532                             | 2                   | 0.0276                    | -0.2018      | 9.2467           | 0.9667                   |
| final_quant_5d_selective_no_snapshot                  | Final Quant 5D Selective - No Snapshot      | 10    | 10.0           | 0.5                 | True       | 0.0492                             | 2                   | 0.0153                    | -0.2018      | 9.2133           | 0.9213                   |
| final_quant_5d_selective_no_snapshot                  | Final Quant 5D Selective - No Snapshot      | 10    | 10.0           | nan                 | True       | 0.0451                             | 2                   | 0.0153                    | -0.2018      | 9.2467           | 0.9247                   |
| final_quant_5d_selective_no_snapshot                  | Final Quant 5D Selective - No Snapshot      | 10    | 10.0           | 0.25                | True       | 0.0451                             | 2                   | 0.0153                    | -0.2018      | 9.2467           | 0.9247                   |
| historical_rating_counts_plus_events                  | historical_rating_counts_plus_events        | 10    | 10.0           | nan                 | False      | 0.042                              | 2                   | 0.1261                    | -0.17        | 9.6667           | 0.9667                   |
| historical_rating_score_selective_5d                  | Historical Rating Score Selective 5D        | 15    | 10.0           | nan                 | True       | 0.028                              | 3                   | 0.0317                    | -0.1944      | 14.5             | 0.9667                   |
| historical_rating_score_selective_5d                  | Historical Rating Score Selective 5D        | 15    | 10.0           | 0.25                | True       | 0.028                              | 3                   | 0.0317                    | -0.1944      | 14.5             | 0.9667                   |
| historical_rating_score_selective_5d                  | Historical Rating Score Selective 5D        | 15    | 10.0           | 0.5                 | True       | 0.028                              | 3                   | 0.0317                    | -0.1944      | 14.5             | 0.9667                   |
| final_quant_5d_no_snapshot_no_sma_filter              | Final Quant 5D - No SMA Filter              | 10    | 20.0           | nan                 | False      | 0.0272                             | 2                   | 0.0533                    | -0.1797      | 9.26             | 0.9667                   |
| historical_rating_counts_plus_events                  | historical_rating_counts_plus_events        | 10    | 20.0           | nan                 | False      | 0.0268                             | 1                   | 0.1032                    | -0.1739      | 9.6667           | 0.9667                   |
| historical_rating_score_selective_5d                  | Historical Rating Score Selective 5D        | 15    | 20.0           | nan                 | True       | 0.0263                             | 3                   | 0.0293                    | -0.1945      | 14.5             | 0.9667                   |
| historical_rating_score_selective_5d                  | Historical Rating Score Selective 5D        | 15    | 20.0           | 0.25                | True       | 0.0263                             | 3                   | 0.0293                    | -0.1945      | 14.5             | 0.9667                   |
| historical_rating_score_selective_5d                  | Historical Rating Score Selective 5D        | 15    | 20.0           | 0.5                 | True       | 0.0263                             | 3                   | 0.0293                    | -0.1945      | 14.5             | 0.9667                   |
| final_quant_5d_no_recent_downgrade_filter_no_snapshot | Final Quant 5D - No Recent Downgrade Filter | 10    | 20.0           | nan                 | False      | 0.0257                             | 1                   | -0.0091                   | -0.2087      | 9.2467           | 0.9667                   |
| final_quant_5d_selective_no_snapshot                  | Final Quant 5D Selective - No Snapshot      | 10    | 20.0           | 0.5                 | True       | 0.0237                             | 1                   | -0.019                    | -0.2087      | 9.2133           | 0.9213                   |
| historical_rating_score_selective_5d                  | Historical Rating Score Selective 5D        | 15    | 10.0           | 0.75                | True       | 0.0208                             | 3                   | 0.0077                    | -0.1964      | 13.84            | 0.9227                   |
| final_quant_5d_selective_no_snapshot                  | Final Quant 5D Selective - No Snapshot      | 15    | 10.0           | 0.5                 | True       | 0.02                               | 2                   | 0.0271                    | -0.1493      | 12.26            | 0.8173                   |
| final_quant_5d_selective_no_snapshot                  | Final Quant 5D Selective - No Snapshot      | 10    | 20.0           | nan                 | True       | 0.0195                             | 1                   | -0.019                    | -0.2087      | 9.2467           | 0.9247                   |
| final_quant_5d_selective_no_snapshot                  | Final Quant 5D Selective - No Snapshot      | 10    | 20.0           | 0.25                | True       | 0.0195                             | 1                   | -0.019                    | -0.2087      | 9.2467           | 0.9247                   |
| historical_rating_score_selective_5d                  | Historical Rating Score Selective 5D        | 15    | 20.0           | 0.75                | True       | 0.0193                             | 3                   | 0.0059                    | -0.1966      | 13.84            | 0.9227                   |
| final_quant_5d_selective_no_snapshot                  | Final Quant 5D Selective - No Snapshot      | 15    | 10.0           | nan                 | True       | 0.0152                             | 2                   | 0.0141                    | -0.1667      | 13.22            | 0.8813                   |
| final_quant_5d_selective_no_snapshot                  | Final Quant 5D Selective - No Snapshot      | 15    | 10.0           | 0.25                | True       | 0.0152                             | 2                   | 0.0141                    | -0.1667      | 13.22            | 0.8813                   |
| final_quant_5d_selective_no_snapshot                  | Final Quant 5D Selective - No Snapshot      | 10    | 10.0           | 0.75                | True       | 0.0045                             | 1                   | -0.0592                   | -0.1489      | 6.2267           | 0.6227                   |
| historical_rating_score_selective_5d                  | Historical Rating Score Selective 5D        | 10    | 10.0           | 1.0                 | True       | 0.0039                             | 2                   | 0.0647                    | -0.1626      | 9.48             | 0.948                    |