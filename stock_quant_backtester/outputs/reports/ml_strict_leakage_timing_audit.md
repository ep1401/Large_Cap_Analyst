# ML Strict Leakage Timing Audit

- This is a frozen ML research candidate.
- 2026 data was not used for training, tuning, or model selection.
- Conservative lag tests are used to detect possible timing leakage.
- Back-tested performance is hypothetical unless actually paper-tracked live.
- ML models may overfit and require extended forward validation.
- This is research/paper trading only, not financial advice.

- Audit status: PASS
- Frozen candidate: `ml_ranker_5d_no_snapshot` / `hist_gradient_boosting_regression`
- Forward window audited: 2026-01-02 to 2026-05-18

## Source Timestamp Checks

- Sentiment article rows checked: 300886
- Sentiment future-article violations: 0
- Rating rows with historical data: 5640
- Rating record-date future violations: 0
- Rating same-day rows requiring stricter lag treatment: 120
- Grade event rows checked: 16536
- Negative days_since_last_upgrade rows: 0
- Negative days_since_last_downgrade rows: 0

## Variant Results

| variant_name                  | ml_return | spy_return | excess_vs_spy | max_drawdown | turnover | average_holdings | rebalance_periods | fallback_future_return_uses | performance_drop_vs_normal |
| ----------------------------- | --------- | ---------- | ------------- | ------------ | -------- | ---------------- | ----------------- | --------------------------- | -------------------------- |
| normal_features               | 0.187027  | 0.084162   | 0.102865      | -0.04167     | 1.0      | 10.0             | 6                 | 0                           | 0.0                        |
| sentiment_lag_1d              | 0.162555  | 0.084162   | 0.078393      | -0.033951    | 1.0      | 10.0             | 6                 | 0                           | -0.024472                  |
| sentiment_lag_2d              | 0.153582  | 0.084162   | 0.069419      | -0.031685    | 1.0      | 10.0             | 6                 | 0                           | -0.033446                  |
| ratings_lag_1d                | 0.144243  | 0.084162   | 0.06008       | -0.040713    | 1.0      | 10.0             | 6                 | 0                           | -0.042785                  |
| grade_events_lag_1d           | 0.187027  | 0.084162   | 0.102865      | -0.04167     | 1.0      | 10.0             | 6                 | 0                           | 0.0                        |
| technical_lag_1d              | 0.203891  | 0.084162   | 0.119729      | -0.074363    | 1.0      | 10.0             | 6                 | 0                           | 0.016864                   |
| all_non_price_alt_data_lag_1d | 0.127548  | 0.084162   | 0.043386      | -0.041175    | 1.0      | 10.0             | 6                 | 0                           | -0.059479                  |
| all_features_lag_1d           | 0.180951  | 0.084162   | 0.096789      | -0.062648    | 1.0      | 10.0             | 6                 | 0                           | -0.006076                  |

## Key Questions

- ML still beats SPY with sentiment lagged 1 day: true (7.84%)
- ML still beats SPY with sentiment lagged 2 days: true (6.94%)
- ML still beats SPY with ratings lagged 1 day: true (6.01%)
- ML still beats SPY with grade events lagged 1 day: true (10.29%)
- ML still beats SPY with all alt-data lagged 1 day: true (4.34%)
- ML still beats SPY with all features lagged 1 day: true (9.68%)
- Largest performance drop vs normal came from: `all_non_price_alt_data_lag_1d` (-5.95%)