# Research Candidate Summary

- This is a research candidate workflow.
- 2026 forward data was not used for ML training or model selection.
- Back-tested performance is hypothetical unless actually paper-tracked live.
- Snapshot analyst target fields are excluded.
- Historical rating-count features use dated FMP grades-historical records available on or before each rebalance date.
- News sentiment depends on Alpha Vantage coverage and classification.
- ML models may overfit and require future forward validation.
- This is research/paper trading only, not financial advice.

## Frozen Production/Paper Model

- Current frozen model: `final_quant_5d_weight_tuned_low_turnover_no_snapshot`
- 2026 forward return so far: -0.26%
- 2026 forward excess vs SPY: -4.65%

## Best 2025 ML Validation Candidate

- Model: `hist_gradient_boosting_regression`
- Strategy: `ml_ranker_5d_no_snapshot`
- 2026 forward data was not used for ML training or model selection.

## 2026 Forward Comparison

- ML return: 18.52%
- Rule-based return: -0.26%
- SPY return: 4.39%
- ML excess vs SPY: 14.14%
- ML excess vs rule-based: 18.78%
- ML max drawdown: -4.13%
- Rule-based max drawdown: -12.06%

## Forward Status

- ML should remain research candidate: true
- Move to extended paper monitoring: false
- recommended_strategy.yaml remains unchanged pending more forward evidence.