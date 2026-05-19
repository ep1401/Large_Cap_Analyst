# Frozen ML Ranker vs SPY — 2025 Validation + 2026 Forward Summary

- This is a frozen ML research candidate.
- 2025 was used as validation/model-selection period.
- 2026 forward data was not used for training, tuning, or model selection.
- Back-tested performance is hypothetical unless actually paper-tracked live.
- Snapshot analyst target fields are excluded.
- Historical rating-count features use dated FMP grades-historical records available on or before each rebalance date.
- News sentiment depends on Alpha Vantage coverage and classification.
- ML models may overfit and require extended forward validation.
- This is research/paper trading only, not financial advice.

- Date range: 2025-01-02 to 2026-05-18
- Vertical split at 2026-01-01 marks the start of frozen forward monitoring.
- 2026 was not used for training, tuning, or model selection.

## Metrics

- ML total return: 66.31%
- SPY total return: 28.18%
- Excess return vs SPY: 38.13%
- ML max drawdown: -11.88%
- SPY max drawdown: -12.18%
- Sharpe: 1.717
- Average turnover: 1.000000
- Average holdings: 10.00
- Number of rebalance periods: 23

## Benchmark Validation

| period_name        | start_date          | end_date            | direct_spy_return | plotted_spy_return | absolute_difference | direct_spy_final_value | plotted_spy_final_value |
| ------------------ | ------------------- | ------------------- | ----------------- | ------------------ | ------------------- | ---------------------- | ----------------------- |
| 2025_2026_combined | 2025-01-02 00:00:00 | 2026-05-18 00:00:00 | 0.281775          | 0.281775           | 0.0                 | 12817.749698           | 12817.749698            |