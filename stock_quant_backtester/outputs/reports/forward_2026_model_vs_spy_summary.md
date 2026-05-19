# Forward 2026 Model vs SPY Summary

- This is a forward/out-of-sample test using the frozen model configuration.
- No 2026 data was used to tune the model.
- Back-tested performance is hypothetical unless trades were actually paper-tracked live.
- Snapshot analyst target fields are excluded.
- Historical rating-count features use dated FMP grades-historical records available on or before each rebalance date.
- News sentiment depends on Alpha Vantage coverage and classification.
- This is research/paper trading only, not financial advice.

- Strategy name: `final_quant_5d_weight_tuned_low_turnover_no_snapshot`
- Display name: Final Quant 5D - Weight Tuned Low Turnover No Snapshot
- Base score model: `final_quant_5d_weight_tuned_no_snapshot`
- Execution mode: `low_turnover_hold_band`
- Forward test start date requested: 2026-01-01
- Forward test start date used: 2026-01-02
- Forward test end date: 2026-04-22
- Feature panel latest date: 2026-05-18
- Enter rank: 10
- Hold rank: 20
- Top N: 10
- Max holding days: 21
- Rebalance frequency: 15 trading days
- Cost assumption: 20 bps total
- Position sizing: equal_weight
- Allow cash: false
- Long/short: false
- Regime filter: none

## Metrics

- Number of rebalance periods: 6
- Model total return: -0.26%
- SPY total return: 4.39%
- Excess return vs SPY: -4.65%
- Annualized return: -2.16%
- Annualized volatility: 35.54%
- Sharpe: 0.115
- Max drawdown: -12.06%
- SPY max drawdown: -5.86%
- Average turnover: 0.866667
- Average holdings: 10.00
- Estimated trading costs: 0.0104
- Percent periods invested: 100.00%

## Benchmark Validation

- SPY start adjusted close: 681.3094
- SPY end adjusted close: 711.2100
- Direct SPY buy-and-hold return: 4.39%
- Plotted SPY final value: $10,438.87
- Benchmark validation difference: 0.0000%

## Latest Actions

- Latest buys: AMD, BAC, GOOGL, UNH
- Latest sells: BKNG, NFLX, NOW, ORCL
- Latest holds: AMZN, INTU, KO, NVDA, SPGI, TMO

## Current Holdings

| date                | ticker | action | reason_for_action     | rank | score  | target_weight | historical_rating_score | net_upgrade_score_30d | downgrade_count_30d | relevance_weighted_sentiment_7d | negative_news_ratio_7d | relative_strength_21d | volatility_21d | top_signal_reasons                                                                                               |
| ------------------- | ------ | ------ | --------------------- | ---- | ------ | ------------- | ----------------------- | --------------------- | ------------------- | ------------------------------- | ---------------------- | --------------------- | -------------- | ---------------------------------------------------------------------------------------------------------------- |
| 2026-04-22 00:00:00 | AMZN   | HOLD   | kept_within_hold_band | 2    | 0.7163 | 0.1           | 4.1429                  | 0.0                   | 0.0                 | 0.1898                          | 0.0039                 | 0.13                  | 0.0216         | historical_rating_score=+0.242, historical_positive_rating_ratio=+0.144, relative_strength_21d=+0.134            |
| 2026-04-22 00:00:00 | NVDA   | HOLD   | kept_within_hold_band | 14   | 0.3907 | 0.1           | 4.0984                  | 0.0                   | 0.0                 | 0.0711                          | 0.0139                 | 0.0677                | 0.0206         | historical_rating_score=+0.213, historical_positive_rating_ratio=+0.157, relevance_weighted_sentiment_7d=-0.116  |
| 2026-04-22 00:00:00 | INTU   | HOLD   | kept_within_hold_band | 19   | 0.3531 | 0.1           | 3.9722                  | 0.0                   | 0.0                 | 0.3095                          | 0.0227                 | -0.1882               | 0.0332         | relative_strength_21d=-0.149, historical_rating_score=+0.132, relevance_weighted_sentiment_7d=+0.123             |
| 2026-04-22 00:00:00 | KO     | HOLD   | kept_within_hold_band | 20   | 0.332  | 0.1           | 4.0833                  | 0.0                   | 0.0                 | 0.1817                          | 0.028                  | -0.0916               | 0.0095         | historical_rating_score=+0.203, historical_negative_rating_ratio=+0.097, historical_positive_rating_ratio=+0.070 |
| 2026-04-22 00:00:00 | SPGI   | HOLD   | kept_within_hold_band | 10   | 0.5068 | 0.1           | 4.1667                  | 0.0                   | 0.0                 | 0.1614                          | 0.0053                 | -0.0381               | 0.0187         | historical_rating_score=+0.257, historical_positive_rating_ratio=+0.161, historical_negative_rating_ratio=+0.097 |
| 2026-04-22 00:00:00 | TMO    | HOLD   | kept_within_hold_band | 4    | 0.6087 | 0.1           | 4.1071                  | 0.0                   | 0.0                 | 0.2958                          | 0.0                    | -0.0102               | 0.0187         | historical_rating_score=+0.219, historical_positive_rating_ratio=+0.125, relevance_weighted_sentiment_7d=+0.109  |
| 2026-04-22 00:00:00 | UNH    | BUY    | new_buy               | 1    | 0.806  | 0.1           | 3.9333                  | 2.0                   | 0.0                 | 0.2542                          | 0.0                    | 0.2264                | 0.0275         | net_upgrade_score_30d=+0.415, relative_strength_21d=+0.220, historical_rating_score=+0.107                       |
| 2026-04-22 00:00:00 | AMD    | BUY    | new_buy               | 3    | 0.6527 | 0.1           | 3.8431                  | 0.0                   | 0.0                 | 0.2534                          | 0.0147                 | 0.412                 | 0.0351         | relative_strength_21d=+0.333, historical_negative_rating_ratio=+0.097, relevance_weighted_sentiment_7d=+0.067    |
| 2026-04-22 00:00:00 | BAC    | BUY    | new_buy               | 5    | 0.6042 | 0.1           | 4.1481                  | 0.0                   | 0.0                 | 0.1703                          | 0.0026                 | 0.0327                | 0.0144         | historical_rating_score=+0.245, historical_positive_rating_ratio=+0.123, historical_negative_rating_ratio=+0.097 |
| 2026-04-22 00:00:00 | GOOGL  | BUY    | new_buy               | 6    | 0.6031 | 0.1           | 4.0857                  | 0.0                   | 0.0                 | 0.2054                          | 0.0                    | 0.0382                | 0.0236         | historical_rating_score=+0.205, historical_positive_rating_ratio=+0.129, historical_negative_rating_ratio=+0.097 |

## Outputs

- Feature panel: `data/final/features_panel_2026_forward.csv`
- Forward returns table: `outputs/tables/forward_2026_model_vs_spy_returns.csv`
- Forward benchmark validation: `outputs/tables/forward_2026_benchmark_validation.csv`
- Current recommendations table: `outputs/tables/current_recommendations_2026_forward.csv`
- Equity curve: `outputs/charts/forward_2026_model_vs_spy_equity_curve.png`
- Drawdown chart: `outputs/charts/forward_2026_model_vs_spy_drawdown.png`
- Frozen config copy: `outputs/reports/recommended_strategy_used_2026_forward.yaml`
- Paper trading performance history: `data/paper_trading/performance_history.csv`