# ML Current Recommendations

- This is paper trading only, not financial advice.
- The ML model is frozen.
- New forward data is not used for retraining or tuning.
- Back-tested performance is hypothetical unless trades were actually paper-tracked live.
- This is a frozen ML research candidate.
- 2026 data is monitoring only and is not used for retraining or tuning.

- Strategy: `ml_ranker_5d_no_snapshot`
- Latest feature date: 2026-05-18
- Rebalance due: false
- Last rebalance date: 2026-05-18
- Next estimated rebalance date: 2026-06-08

## Current Holdings

| date                | period_end_date     | strategy_name            | ticker | action | rank | score    | weight | holding_days | latest_feature_date | model_type                        | status             | position_sizing | total_cost_bps | rebalance_frequency_days | enter_rank | hold_rank | max_holding_days | reason            |
| ------------------- | ------------------- | ------------------------ | ------ | ------ | ---- | -------- | ------ | ------------ | ------------------- | --------------------------------- | ------------------ | --------------- | -------------- | ------------------------ | ---------- | --------- | ---------------- | ----------------- |
| 2026-04-22 00:00:00 | 2026-05-18 00:00:00 | ml_ranker_5d_no_snapshot | BKNG   | HOLD   | 1    | 0.017932 | 0.1    | 15           | 2026-05-18 00:00:00 | hist_gradient_boosting_regression | research_candidate | equal_weight    | 20.0           | 15                       | 10         | 20        | 21               | rebalance_not_due |
| 2026-04-22 00:00:00 | 2026-05-18 00:00:00 | ml_ranker_5d_no_snapshot | ABT    | HOLD   | 2    | 0.013209 | 0.1    | 15           | 2026-05-18 00:00:00 | hist_gradient_boosting_regression | research_candidate | equal_weight    | 20.0           | 15                       | 10         | 20        | 21               | rebalance_not_due |
| 2026-04-22 00:00:00 | 2026-05-18 00:00:00 | ml_ranker_5d_no_snapshot | COST   | HOLD   | 3    | 0.013108 | 0.1    | 30           | 2026-05-18 00:00:00 | hist_gradient_boosting_regression | research_candidate | equal_weight    | 20.0           | 15                       | 10         | 20        | 21               | rebalance_not_due |
| 2026-04-22 00:00:00 | 2026-05-18 00:00:00 | ml_ranker_5d_no_snapshot | INTU   | HOLD   | 4    | 0.011983 | 0.1    | 15           | 2026-05-18 00:00:00 | hist_gradient_boosting_regression | research_candidate | equal_weight    | 20.0           | 15                       | 10         | 20        | 21               | rebalance_not_due |
| 2026-04-22 00:00:00 | 2026-05-18 00:00:00 | ml_ranker_5d_no_snapshot | GE     | HOLD   | 5    | 0.011814 | 0.1    | 15           | 2026-05-18 00:00:00 | hist_gradient_boosting_regression | research_candidate | equal_weight    | 20.0           | 15                       | 10         | 20        | 21               | rebalance_not_due |
| 2026-04-22 00:00:00 | 2026-05-18 00:00:00 | ml_ranker_5d_no_snapshot | CAT    | HOLD   | 6    | 0.010508 | 0.1    | 15           | 2026-05-18 00:00:00 | hist_gradient_boosting_regression | research_candidate | equal_weight    | 20.0           | 15                       | 10         | 20        | 21               | rebalance_not_due |
| 2026-04-22 00:00:00 | 2026-05-18 00:00:00 | ml_ranker_5d_no_snapshot | CSCO   | HOLD   | 7    | 0.010267 | 0.1    | 15           | 2026-05-18 00:00:00 | hist_gradient_boosting_regression | research_candidate | equal_weight    | 20.0           | 15                       | 10         | 20        | 21               | rebalance_not_due |
| 2026-04-22 00:00:00 | 2026-05-18 00:00:00 | ml_ranker_5d_no_snapshot | AMAT   | HOLD   | 8    | 0.008282 | 0.1    | 15           | 2026-05-18 00:00:00 | hist_gradient_boosting_regression | research_candidate | equal_weight    | 20.0           | 15                       | 10         | 20        | 21               | rebalance_not_due |
| 2026-04-22 00:00:00 | 2026-05-18 00:00:00 | ml_ranker_5d_no_snapshot | AXP    | HOLD   | 9    | 0.008165 | 0.1    | 15           | 2026-05-18 00:00:00 | hist_gradient_boosting_regression | research_candidate | equal_weight    | 20.0           | 15                       | 10         | 20        | 21               | rebalance_not_due |
| 2026-04-22 00:00:00 | 2026-05-18 00:00:00 | ml_ranker_5d_no_snapshot | WMT    | HOLD   | 10   | 0.007356 | 0.1    | 15           | 2026-05-18 00:00:00 | hist_gradient_boosting_regression | research_candidate | equal_weight    | 20.0           | 15                       | 10         | 20        | 21               | rebalance_not_due |

## Latest Sells

| ticker | action | reason                    |
| ------ | ------ | ------------------------- |
| AMAT   | SELL   | max_holding_days          |
| CAT    | SELL   | max_holding_days          |
| AVGO   | SELL   | recent_downgrade_flag_30d |
| QCOM   | SELL   | recent_downgrade_flag_30d |
| MSFT   | SELL   | rank>20:51                |
| NVDA   | SELL   | rank>20:41                |
| DIS    | SELL   | rank>20:39                |
| JNJ    | SELL   | rank>20:37                |
| META   | SELL   | rank>20:24                |