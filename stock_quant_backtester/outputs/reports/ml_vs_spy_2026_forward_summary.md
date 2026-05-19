# Frozen ML Ranker vs SPY — 2026 Forward Summary

- This is a frozen ML research candidate.
- 2025 was used as validation/model-selection period.
- 2026 forward data was not used for training, tuning, or model selection.
- Back-tested performance is hypothetical unless actually paper-tracked live.
- Snapshot analyst target fields are excluded.
- Historical rating-count features use dated FMP grades-historical records available on or before each rebalance date.
- News sentiment depends on Alpha Vantage coverage and classification.
- ML models may overfit and require extended forward validation.
- This is research/paper trading only, not financial advice.

- Forward start date: 2026-01-02
- Latest available date: 2026-05-18
- Strategy: `ml_ranker_5d_no_snapshot`
- Model loaded from disk: `models/ml_ranker_no_snapshot.pkl`
- Trading costs: 0.0120
- Current ML holdings: ABT, AMAT, AXP, BKNG, CAT, COST, CSCO, GE, INTU, WMT
- Latest buys: ABT, AXP, BKNG, CSCO, GE, INTU, WMT
- Latest sells: AVGO, DIS, JNJ, META, MSFT, NVDA, QCOM
- Latest holds: AMAT, CAT, COST

## Metrics

- ML total return: 18.70%
- SPY total return: 8.42%
- Excess return vs SPY: 10.29%
- ML max drawdown: -4.17%
- SPY max drawdown: -5.86%
- Sharpe: 2.250
- Average turnover: 1.000000
- Average holdings: 10.00
- Number of rebalance periods: 6

## Benchmark Validation

| period_name  | start_date          | end_date            | direct_spy_return | plotted_spy_return | absolute_difference | direct_spy_final_value | plotted_spy_final_value |
| ------------ | ------------------- | ------------------- | ----------------- | ------------------ | ------------------- | ---------------------- | ----------------------- |
| 2026_forward | 2026-01-02 00:00:00 | 2026-05-18 00:00:00 | 0.084162          | 0.084162           | 0.0                 | 10841.623497           | 10841.623497            |