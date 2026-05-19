# Regime Filter Test

- Back-tested performance is hypothetical.
- Snapshot analyst target models are excluded from the main historically safer ranking.
- Historical rating-count features use dated FMP grades-historical records available on or before each rebalance date.
- News sentiment depends on Alpha Vantage coverage and classification.
- Long/short is experimental and currently not recommended.
- This is research/paper trading only, not financial advice.

- Base model tested: Final Quant 5D - No SMA Filter
- QQQ regime variants available: False

## Results

| strategy_name                            | display_name                   | top_n | position_sizing | total_cost_bps | regime_name             | regime_exposure_when_blocked | test_period_excess_vs_spy | walk_forward_windows_beating_spy | max_drawdown | turnover | percent_periods_invested | average_exposure | recommended_setting | robustness_note       |
| ---------------------------------------- | ------------------------------ | ----- | --------------- | -------------- | ----------------------- | ---------------------------- | ------------------------- | -------------------------------- | ------------ | -------- | ------------------------ | ---------------- | ------------------- | --------------------- |
| final_quant_5d_no_snapshot_no_sma_filter | Final Quant 5D - No SMA Filter | 10    | equal_weight    | 10.0           | none                    | 1.0                          | 0.091564                  | 2                                | -0.173799    | 0.670667 | 0.966667                 | 0.966667         | True                | no_clear_improvement  |
| final_quant_5d_no_snapshot_no_sma_filter | Final Quant 5D - No SMA Filter | 10    | equal_weight    | 10.0           | spy_50d                 | 0.5                          | 0.072341                  | 2                                | -0.132877    | 0.618    | 0.966667                 | 0.856667         | False               | no_clear_improvement  |
| final_quant_5d_no_snapshot_no_sma_filter | Final Quant 5D - No SMA Filter | 10    | equal_weight    | 10.0           | spy_50d_return_positive | 0.5                          | 0.067271                  | 2                                | -0.126372    | 0.62     | 0.966667                 | 0.876667         | False               | no_clear_improvement  |
| final_quant_5d_no_snapshot_no_sma_filter | Final Quant 5D - No SMA Filter | 10    | equal_weight    | 10.0           | spy_21d_return_positive | 0.5                          | 0.062463                  | 2                                | -0.158191    | 0.629333 | 0.966667                 | 0.833333         | False               | no_clear_improvement  |
| final_quant_5d_no_snapshot_no_sma_filter | Final Quant 5D - No SMA Filter | 10    | equal_weight    | 10.0           | spy_200d                | 0.5                          | 0.048985                  | 2                                | -0.134723    | 0.566    | 0.966667                 | 0.82             | False               | no_clear_improvement  |
| final_quant_5d_no_snapshot_no_sma_filter | Final Quant 5D - No SMA Filter | 10    | equal_weight    | 10.0           | spy_50d                 | 0.0                          | 0.045987                  | 2                                | -0.119004    | 0.565333 | 0.746667                 | 0.746667         | False               | reduces_exposure_only |
| final_quant_5d_no_snapshot_no_sma_filter | Final Quant 5D - No SMA Filter | 10    | equal_weight    | 10.0           | spy_50d_return_positive | 0.0                          | 0.035566                  | 2                                | -0.105644    | 0.569333 | 0.786667                 | 0.786667         | False               | reduces_exposure_only |
| final_quant_5d_no_snapshot_no_sma_filter | Final Quant 5D - No SMA Filter | 10    | equal_weight    | 10.0           | spy_21d_return_positive | 0.0                          | 0.026524                  | 2                                | -0.170535    | 0.588    | 0.7                      | 0.7              | False               | reduces_exposure_only |
| final_quant_5d_no_snapshot_no_sma_filter | Final Quant 5D - No SMA Filter | 10    | equal_weight    | 10.0           | spy_200d                | 0.0                          | 0.001048                  | 2                                | -0.095919    | 0.461333 | 0.673333                 | 0.673333         | False               | reduces_exposure_only |