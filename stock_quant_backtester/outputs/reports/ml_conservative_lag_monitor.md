# ML Conservative Lag Monitor

- This is a frozen ML research candidate.
- 2026 data was not used for training, tuning, or model selection.
- Strict leakage timing audits passed, but ML can still overfit.
- Back-tested performance is hypothetical unless actually paper-tracked live.
- Snapshot analyst target fields are excluded.
- Historical rating-count features use dated FMP grades-historical records available on or before each rebalance date.
- News sentiment depends on Alpha Vantage coverage and classification.
- ML models may overfit and require extended forward validation.
- This is research/paper trading only, not financial advice.

- Strategy: `ml_ranker_5d_no_snapshot`

| variant_name                  | ml_return | spy_return | excess_vs_spy | max_drawdown | periods_beating_spy_pct | latest_date         |
| ----------------------------- | --------- | ---------- | ------------- | ------------ | ----------------------- | ------------------- |
| normal_features               | 0.187027  | 0.084162   | 0.102865      | -0.04167     | 0.666667                | 2026-05-18 00:00:00 |
| all_non_price_alt_data_lag_1d | 0.127548  | 0.084162   | 0.043386      | -0.041175    | 0.5                     | 2026-05-18 00:00:00 |
| all_features_lag_1d           | 0.180951  | 0.084162   | 0.096789      | -0.062648    | 0.666667                | 2026-05-18 00:00:00 |

## Readout

- Normal ML excess vs SPY: 10.29%
- Alt-data-lagged excess vs SPY: 4.34%
- All-features-lagged excess vs SPY: 9.68%
- Lagged variants still beat SPY: true
- Timing assumptions remain credible: true