# Paper Trading Report

- Back-tested performance is hypothetical.
- Snapshot analyst target models are excluded from the main historically safer ranking.
- Historical rating-count features use dated FMP grades-historical records available on or before each rebalance date.
- News sentiment depends on Alpha Vantage coverage and classification.
- Long/short is experimental and currently not recommended.
- Regime filters were tested and are not recommended for the main model based on current results.
- The current model is cost-sensitive and should remain paper-trading only unless it survives realistic cost assumptions.
- This is research/paper trading only, not financial advice.

- Latest feature date in panel: 2026-05-18.
- Signal selection date used for recommendations: 2026-04-22.
- Suggested rebalance date: 2026-05-13.
- Strategy under paper trading: `final_quant_5d_weight_tuned_low_turnover_no_snapshot`.
- Estimated turnover at latest rebalance: 0.8000.
- Estimated trading cost at latest rebalance: 0.0016.

## Sells

| ticker | reason                    |
| ------ | ------------------------- |
| NFLX   | max_holding_days          |
| ORCL   | max_holding_days          |
| NOW    | recent_downgrade_flag_30d |
| BKNG   | max_holding_days          |

## New Buys

| ticker | reason         |
| ------ | -------------- |
| UNH    | enter_rank<=10 |
| AMD    | enter_rank<=10 |
| BAC    | enter_rank<=10 |
| GOOGL  | enter_rank<=10 |

## Current Holdings

| date                | ticker | sector                 | action | reason                | rank | score  | weight | strategy_name                                        | holding_period_days | position_sizing | total_cost_bps | min_score_threshold | allow_cash | cash_weight | top_signal_reasons                                                                                               | historical_rating_score | net_upgrade_score_30d | downgrade_count_30d | sentiment_7d | negative_news_ratio_7d | relative_strength_21d | volatility_21d | holding_days | rebalance_frequency_days | enter_rank | hold_rank | max_holding_days |
| ------------------- | ------ | ---------------------- | ------ | --------------------- | ---- | ------ | ------ | ---------------------------------------------------- | ------------------- | --------------- | -------------- | ------------------- | ---------- | ----------- | ---------------------------------------------------------------------------------------------------------------- | ----------------------- | --------------------- | ------------------- | ------------ | ---------------------- | --------------------- | -------------- | ------------ | ------------------------ | ---------- | --------- | ---------------- |
| 2026-04-22 00:00:00 | AMZN   | Consumer Discretionary | HOLD   | kept_within_hold_band | 2    | 0.7163 | 0.1    | final_quant_5d_weight_tuned_low_turnover_no_snapshot | 5                   | equal_weight    | 20.0           | None                | False      | 0.0         | historical_rating_score=+0.242, historical_positive_rating_ratio=+0.144, relative_strength_21d=+0.134            | 4.1429                  | 0.0                   | 0.0                 | 0.1898       | 0.0039                 | 0.13                  | 0.0216         | 30           | 15                       | 10         | 20        | 21               |
| 2026-04-22 00:00:00 | NVDA   | Information Technology | HOLD   | kept_within_hold_band | 14   | 0.3907 | 0.1    | final_quant_5d_weight_tuned_low_turnover_no_snapshot | 5                   | equal_weight    | 20.0           | None                | False      | 0.0         | historical_rating_score=+0.213, historical_positive_rating_ratio=+0.157, relevance_weighted_sentiment_7d=-0.116  | 4.0984                  | 0.0                   | 0.0                 | 0.0711       | 0.0139                 | 0.0677                | 0.0206         | 30           | 15                       | 10         | 20        | 21               |
| 2026-04-22 00:00:00 | INTU   | Information Technology | HOLD   | kept_within_hold_band | 19   | 0.3531 | 0.1    | final_quant_5d_weight_tuned_low_turnover_no_snapshot | 5                   | equal_weight    | 20.0           | None                | False      | 0.0         | relative_strength_21d=-0.149, historical_rating_score=+0.132, relevance_weighted_sentiment_7d=+0.123             | 3.9722                  | 0.0                   | 0.0                 | 0.3095       | 0.0227                 | -0.1882               | 0.0332         | 30           | 15                       | 10         | 20        | 21               |
| 2026-04-22 00:00:00 | KO     | Consumer Staples       | HOLD   | kept_within_hold_band | 20   | 0.332  | 0.1    | final_quant_5d_weight_tuned_low_turnover_no_snapshot | 5                   | equal_weight    | 20.0           | None                | False      | 0.0         | historical_rating_score=+0.203, historical_negative_rating_ratio=+0.097, historical_positive_rating_ratio=+0.070 | 4.0833                  | 0.0                   | 0.0                 | 0.1817       | 0.028                  | -0.0916               | 0.0095         | 30           | 15                       | 10         | 20        | 21               |
| 2026-04-22 00:00:00 | SPGI   | Financials             | HOLD   | kept_within_hold_band | 10   | 0.5068 | 0.1    | final_quant_5d_weight_tuned_low_turnover_no_snapshot | 5                   | equal_weight    | 20.0           | None                | False      | 0.0         | historical_rating_score=+0.257, historical_positive_rating_ratio=+0.161, historical_negative_rating_ratio=+0.097 | 4.1667                  | 0.0                   | 0.0                 | 0.1614       | 0.0053                 | -0.0381               | 0.0187         | 30           | 15                       | 10         | 20        | 21               |
| 2026-04-22 00:00:00 | TMO    | Health Care            | HOLD   | kept_within_hold_band | 4    | 0.6087 | 0.1    | final_quant_5d_weight_tuned_low_turnover_no_snapshot | 5                   | equal_weight    | 20.0           | None                | False      | 0.0         | historical_rating_score=+0.219, historical_positive_rating_ratio=+0.125, relevance_weighted_sentiment_7d=+0.109  | 4.1071                  | 0.0                   | 0.0                 | 0.2958       | 0.0                    | -0.0102               | 0.0187         | 30           | 15                       | 10         | 20        | 21               |
| 2026-04-22 00:00:00 | UNH    | Health Care            | BUY    | new_buy               | 1    | 0.806  | 0.1    | final_quant_5d_weight_tuned_low_turnover_no_snapshot | 5                   | equal_weight    | 20.0           | None                | False      | 0.0         | net_upgrade_score_30d=+0.415, relative_strength_21d=+0.220, historical_rating_score=+0.107                       | 3.9333                  | 2.0                   | 0.0                 | 0.2542       | 0.0                    | 0.2264                | 0.0275         | 15           | 15                       | 10         | 20        | 21               |
| 2026-04-22 00:00:00 | AMD    | Information Technology | BUY    | new_buy               | 3    | 0.6527 | 0.1    | final_quant_5d_weight_tuned_low_turnover_no_snapshot | 5                   | equal_weight    | 20.0           | None                | False      | 0.0         | relative_strength_21d=+0.333, historical_negative_rating_ratio=+0.097, relevance_weighted_sentiment_7d=+0.067    | 3.8431                  | 0.0                   | 0.0                 | 0.2534       | 0.0147                 | 0.412                 | 0.0351         | 15           | 15                       | 10         | 20        | 21               |
| 2026-04-22 00:00:00 | BAC    | Financials             | BUY    | new_buy               | 5    | 0.6042 | 0.1    | final_quant_5d_weight_tuned_low_turnover_no_snapshot | 5                   | equal_weight    | 20.0           | None                | False      | 0.0         | historical_rating_score=+0.245, historical_positive_rating_ratio=+0.123, historical_negative_rating_ratio=+0.097 | 4.1481                  | 0.0                   | 0.0                 | 0.1703       | 0.0026                 | 0.0327                | 0.0144         | 15           | 15                       | 10         | 20        | 21               |
| 2026-04-22 00:00:00 | GOOGL  | Communication Services | BUY    | new_buy               | 6    | 0.6031 | 0.1    | final_quant_5d_weight_tuned_low_turnover_no_snapshot | 5                   | equal_weight    | 20.0           | None                | False      | 0.0         | historical_rating_score=+0.205, historical_positive_rating_ratio=+0.129, historical_negative_rating_ratio=+0.097 | 4.0857                  | 0.0                   | 0.0                 | 0.2054       | 0.0                    | 0.0382                | 0.0236         | 15           | 15                       | 10         | 20        | 21               |