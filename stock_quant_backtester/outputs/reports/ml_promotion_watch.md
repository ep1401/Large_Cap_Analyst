# ML Promotion Watch

- This is a frozen ML research candidate.
- 2026 data was not used for training, tuning, or model selection.
- Strict leakage timing audits passed, but ML can still overfit.
- Back-tested performance is hypothetical unless actually paper-tracked live.
- Snapshot analyst target fields are excluded.
- Historical rating-count features use dated FMP grades-historical records available on or before each rebalance date.
- News sentiment depends on Alpha Vantage coverage and classification.
- ML models may overfit and require extended forward validation.
- This is research/paper trading only, not financial advice.

- Strategy name: `ml_ranker_5d_no_snapshot`
- Model type: `hist_gradient_boosting_regression`
- Forward months observed: 4.47
- Current status: `RESEARCH_CANDIDATE_MONITORING`
- Forward window: 2026-01-02 to 2026-05-18

## Current Forward Metrics

- ML return: 18.70%
- SPY return: 8.42%
- Rule-based return: -0.43%
- ML excess vs SPY: 10.29%
- ML excess vs rule-based: 19.13%
- ML max drawdown: -4.17%
- SPY max drawdown: -5.86%
- Periods beating SPY: 4 / 6 (66.67%)

## Status Criteria

| strategy_name            | model_type                        | forward_start_date  | latest_date         | months_forward_data | status                        | ml_total_return | spy_total_return | rule_total_return | ml_excess_vs_spy | ml_excess_vs_rule_based | ml_max_drawdown | spy_max_drawdown | periods_beating_spy_pct | ml_excess_vs_spy_positive | ml_excess_vs_rule_positive | drawdown_not_5pts_worse_than_spy | beats_spy_in_60pct_periods | beats_spy_at_20bps | strict_leakage_audit_pass | alt_data_lag_still_positive | all_features_lag_still_positive | no_preprocessing_leakage | no_2026_training_or_selection |
| ------------------------ | --------------------------------- | ------------------- | ------------------- | ------------------- | ----------------------------- | --------------- | ---------------- | ----------------- | ---------------- | ----------------------- | --------------- | ---------------- | ----------------------- | ------------------------- | -------------------------- | -------------------------------- | -------------------------- | ------------------ | ------------------------- | --------------------------- | ------------------------------- | ------------------------ | ----------------------------- |
| ml_ranker_5d_no_snapshot | hist_gradient_boosting_regression | 2026-01-02 00:00:00 | 2026-05-18 00:00:00 | 4.467806            | RESEARCH_CANDIDATE_MONITORING | 0.187027        | 0.084162         | -0.004261         | 0.102865         | 0.191288                | -0.04167        | -0.058629        | 0.666667                | True                      | True                       | True                             | True                       | True               | True                      | True                        | True                            | True                     | True                          |

## Watch Rules

- `<6 months`: `RESEARCH_CANDIDATE_MONITORING`
- `6-12 months` with all criteria passing: `EXTENDED_PAPER_MONITORING`
- `>=12 months` with all criteria passing: `PROMOTION_CANDIDATE`
- `>=12 months` with underperformance or materially worse drawdown: `FAILED_FORWARD_TEST`
- This report does not edit `recommended_strategy.yaml` automatically.