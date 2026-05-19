# ML Ranker No Snapshot Report

- This is a research candidate workflow.
- 2026 forward data is not used for tuning or model selection.
- Back-tested performance is hypothetical.
- Snapshot analyst target fields are excluded.
- Historical rating-count features use dated FMP grades-historical records available on or before each rebalance date.
- News sentiment depends on Alpha Vantage coverage and classification.
- ML models may overfit and require future forward validation.
- This is research/paper trading only, not financial advice.

- Train window: 2023-01-01 to 2024-12-31
- Validation window: 2025-01-01 to 2025-12-31
- Target column: `future_5d_excess_return`
- Feature count: 28
- Best validation model: `hist_gradient_boosting_regression`
- Best validation excess vs SPY: 21.90%

## Validation Results

| model_name                        | rank_correlation | top_decile_avg_forward_excess | bottom_decile_avg_forward_excess | top_minus_bottom_spread | top_10_strategy_return | spy_return | excess_vs_spy | sharpe | max_drawdown | average_turnover | validation_rebalance_periods |
| --------------------------------- | ---------------- | ----------------------------- | -------------------------------- | ----------------------- | ---------------------- | ---------- | ------------- | ------ | ------------ | ---------------- | ---------------------------- |
| hist_gradient_boosting_regression | 0.0424           | 0.43%                         | 0.03%                            | 0.40%                   | 39.91%                 | 18.01%     | 21.90%        | 2.651  | -11.93%      | 1.0353           | 17                           |
| ridge_regression                  | 0.0573           | 0.32%                         | -0.39%                           | 0.71%                   | 37.57%                 | 18.01%     | 19.56%        | 2.2631 | -17.25%      | 0.8588           | 17                           |
| random_forest_regression          | 0.0355           | 0.29%                         | -0.15%                           | 0.45%                   | 18.73%                 | 18.01%     | 0.72%         | 1.397  | -16.57%      | 0.6941           | 17                           |
| logistic_outperform_classifier    | 0.036            | 0.02%                         | -0.28%                           | 0.30%                   | 16.00%                 | 18.01%     | -2.01%        | 1.1573 | -19.35%      | 0.6824           | 17                           |

## Notes

- The ML artifact stores only the best 2025 validation model for later research comparison.
- 2026 rows are excluded from training, validation, and model selection.