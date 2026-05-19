# Market Aware Model Comparison

- This is a research candidate workflow.
- 2026 forward data is not used for tuning or model selection.
- Back-tested performance is hypothetical.
- Snapshot analyst target fields are excluded.
- Historical rating-count features use dated FMP grades-historical records available on or before each rebalance date.
- News sentiment depends on Alpha Vantage coverage and classification.
- ML models may overfit and require future forward validation.
- This is research/paper trading only, not financial advice.

- Development window: 2023-01-01 to 2024-12-31
- Validation window: 2025-01-01 to 2025-12-31
- Best ML validation model loaded from artifact: `hist_gradient_boosting_regression`

## Comparison Table

| strategy_name                                                    | exposure_mode | validation_total_return | validation_spy_return | validation_excess_vs_spy | validation_sharpe  | validation_max_drawdown | validation_average_turnover | validation_windows_beating_spy | validation_rebalance_periods | top_decile_avg_forward_excess | bottom_decile_avg_forward_excess | top_minus_bottom_spread | rank_correlation |
| ---------------------------------------------------------------- | ------------- | ----------------------- | --------------------- | ------------------------ | ------------------ | ----------------------- | --------------------------- | ------------------------------ | ---------------------------- | ----------------------------- | -------------------------------- | ----------------------- | ---------------- |
| ml_ranker_5d_no_snapshot                                         | full          | 39.91%                  | 18.01%                | 21.90%                   | 2.6510084116339603 | -11.93%                 | 1.0353                      | 1                              | 17                           | 1.63%                         | -0.34%                           | 1.97%                   | 0.0715           |
| ml_ranker_5d_market_exposure_no_snapshot                         | discrete      | 26.60%                  | 18.01%                | 8.59%                    | 2.560720327495315  | -10.57%                 | 0.9088                      | 1                              | 17                           | 1.63%                         | -0.34%                           | 1.97%                   | 0.0715           |
| ml_ranker_5d_market_exposure_continuous_no_snapshot              | continuous    | 25.88%                  | 18.01%                | 7.87%                    | 2.697480970356517  | -9.42%                  | 0.8348                      | 1                              | 17                           | 1.63%                         | -0.34%                           | 1.97%                   | 0.0715           |
| final_quant_5d_market_aware_score_no_snapshot                    | full          | 16.54%                  | 18.01%                | -1.47%                   | 1.6712555685402513 | -14.43%                 | 0.7529                      | 0                              | 17                           | -0.02%                        | 1.57%                            | -1.60%                  | -0.0637          |
| final_quant_5d_weight_tuned_low_turnover_no_snapshot             | full          | 12.50%                  | 18.01%                | -5.51%                   | 1.3646088120475524 | -13.87%                 | 0.7294                      | 0                              | 17                           | -0.22%                        | 1.26%                            | -1.48%                  | -0.0561          |
| final_quant_5d_weight_tuned_market_regime_continuous_no_snapshot | continuous    | 9.19%                   | 18.01%                | -8.82%                   | 1.257554250983268  | -11.00%                 | 0.6108                      | 0                              | 17                           | -0.22%                        | 1.26%                            | -1.48%                  | -0.0561          |
| final_quant_5d_weight_tuned_market_regime_no_snapshot            | discrete      | 8.81%                   | 18.01%                | -9.20%                   | 1.098992322884025  | -12.35%                 | 0.6794                      | 0                              | 17                           | -0.22%                        | 1.26%                            | -1.48%                  | -0.0561          |
| SPY                                                              | buy_hold      | 18.01%                  | 18.01%                | 0.00%                    | nan                | n/a                     | 0.0                         | 0                              | 17                           | n/a                           | n/a                              | n/a                     | nan              |

## Research Readout

- Which model beats the current promoted model on 2025? ml_ranker_5d_no_snapshot, ml_ranker_5d_market_exposure_no_snapshot, ml_ranker_5d_market_exposure_continuous_no_snapshot, final_quant_5d_market_aware_score_no_snapshot
- Best drawdown among research candidates: `ml_ranker_5d_market_exposure_continuous_no_snapshot`
- Lowest turnover among research candidates: `SPY`
- Best top-minus-bottom rank spread: `ml_ranker_5d_no_snapshot`
- Does market sentiment/regime improve performance? yes
- Does ML improve performance? yes