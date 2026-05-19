# Tuned Vs Baseline Comparison

- Tuned model created: True.
- Baseline strategy: `final_quant_5d_no_snapshot_no_sma_filter`.
- Tuned strategy included: final_quant_5d_weight_tuned_no_snapshot.

| strategy_name                            | display_name                              | walk_forward_average_excess_vs_spy | 2024_h1_excess_return_vs_spy | 2024_h2_excess_return_vs_spy | 2025_excess_return_vs_spy | windows_beating_spy | max_drawdown | average_turnover | delta_vs_baseline_walk_forward |
| ---------------------------------------- | ----------------------------------------- | ---------------------------------- | ---------------------------- | ---------------------------- | ------------------------- | ------------------- | ------------ | ---------------- | ------------------------------ |
| final_quant_5d_weight_tuned_no_snapshot  | Final Quant 5D - Weight Tuned No Snapshot | 0.0791                             | 0.1464                       | -0.0415                      | 0.1322                    | 2                   | -0.176       | 0.7236           | 0.0247                         |
| final_quant_5d_no_snapshot_no_sma_filter | Final Quant 5D - No SMA Filter            | 0.0544                             | 0.1333                       | -0.0611                      | 0.091                     | 2                   | -0.1738      | 0.7277           | 0.0                            |
| historical_rating_counts_plus_events     | historical_rating_counts_plus_events      | 0.042                              | 0.0014                       | -0.0015                      | 0.1261                    | 2                   | -0.17        | 0.3653           | -0.0124                        |
| historical_rating_score_only_5d          | Historical Rating Score Only 5D           | -0.0024                            | -0.0809                      | 0.0285                       | 0.0454                    | 2                   | -0.1773      | 0.052            | -0.0567                        |

## Recommendation Check
- Best walk-forward average excess vs SPY: 7.91%.
- Baseline walk-forward average excess vs SPY: 5.44%.
- Current top row: `final_quant_5d_weight_tuned_no_snapshot`.