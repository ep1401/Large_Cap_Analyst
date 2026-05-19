# Frozen ML Ranker vs SPY — 2025 Validation Summary

- This is a frozen ML research candidate.
- 2025 was used as validation/model-selection period.
- 2026 forward data was not used for training, tuning, or model selection.
- Back-tested performance is hypothetical unless actually paper-tracked live.
- Snapshot analyst target fields are excluded.
- Historical rating-count features use dated FMP grades-historical records available on or before each rebalance date.
- News sentiment depends on Alpha Vantage coverage and classification.
- ML models may overfit and require extended forward validation.
- This is research/paper trading only, not financial advice.

- Date range: 2025-01-02 to 2025-12-31
- Strategy: `ml_ranker_5d_no_snapshot`
- Model loaded from disk: `models/ml_ranker_no_snapshot.pkl`

## Metrics

- ML total return: 40.10%
- SPY total return: 18.01%
- Excess return vs SPY: 22.09%
- ML max drawdown: -11.88%
- SPY max drawdown: -12.18%
- Sharpe: 1.541
- Average turnover: 1.000000
- Average holdings: 10.00
- Number of rebalance periods: 17

## Benchmark Validation

| period_name     | start_date          | end_date            | direct_spy_return | plotted_spy_return | absolute_difference | direct_spy_final_value | plotted_spy_final_value |
| --------------- | ------------------- | ------------------- | ----------------- | ------------------ | ------------------- | ---------------------- | ----------------------- |
| 2025_validation | 2025-01-02 00:00:00 | 2025-12-31 00:00:00 | 0.180109          | 0.180109           | 0.0                 | 11801.089487           | 11801.089487            |