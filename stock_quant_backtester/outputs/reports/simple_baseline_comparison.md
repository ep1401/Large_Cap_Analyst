# Simple Baseline Comparison

- 5-day comparison uses the current honest 5D model: `final_quant_5d_no_snapshot_no_sma_filter`.
- 21-day comparison uses the current 21D no-snapshot sibling: `final_quant_21d_no_snapshot_sector_capped`.
- Conclusions below are based on the full 2023-2025 window, with walk-forward columns included in the table.

## Headline Answers
- Final 5D model beats the best simple momentum baseline on full-period excess vs SPY: True (6.65% vs -4.44%).
- Final 5D model beats sentiment-only on full-period excess vs SPY: True (6.65% vs -62.20%).
- Final 5D model beats historical-rating-only on full-period excess vs SPY: False (6.65% vs 7.08%).
- Final 5D model beats the equal-weight universe on full-period excess vs SPY: True (6.65% vs -7.16%).
- QQQ not available in the current universe/pricing panel.

## Results

| holding_period_days | label                                     | display_name                    | category    | full_period_total_return | full_period_excess_return_vs_spy | sharpe_ratio | max_drawdown | average_turnover | average_holdings | windows_beating_spy |
| ------------------- | ----------------------------------------- | ------------------------------- | ----------- | ------------------------ | -------------------------------- | ------------ | ------------ | ---------------- | ---------------- | ------------------- |
| 5                   | top_10_historical_rating_score            | Top 10 Historical Rating Score  | baseline    | 0.9476                   | 0.0708                           | 1.3829       | -0.1773      | 0.052            | 9.6667           | 2.0                 |
| 5                   | final_quant_5d_no_snapshot_no_sma_filter  | Final Quant 5D - No SMA Filter  | final_model | 0.9433                   | 0.0665                           | 1.3841       | -0.1738      | 0.7277           | 9.26             | 2.0                 |
| 5                   | spy                                       | SPY                             | benchmark   | 0.8768                   | 0.0                              | 1.5579       | -0.1688      | 0.0              | 1.0              | 0.0                 |
| 5                   | top_10_relative_strength_63d              | Top 10 Relative Strength 63D    | baseline    | 0.8324                   | -0.0444                          | 1.1699       | -0.1996      | 0.4293           | 9.6667           | 1.0                 |
| 5                   | equal_weight_universe                     | Equal Weight Universe           | baseline    | 0.8052                   | -0.0716                          | 1.525        | -0.1589      | 0.0067           | 58.0             | 1.0                 |
| 5                   | top_10_technical_momentum                 | Top 10 Technical Momentum       | baseline    | 0.7927                   | -0.0841                          | 1.234        | -0.2078      | 0.6653           | 9.6667           | 1.0                 |
| 5                   | random_10_universe                        | Random 10 Universe              | baseline    | 0.785                    | -0.0917                          | 1.3389       | -0.1657      | 0.0067           | 9.6667           | 1.33                |
| 5                   | top_10_net_upgrade_score_30d              | Top 10 Net Upgrade Score 30D    | baseline    | 0.7471                   | -0.1296                          | 1.2152       | -0.1885      | 0.644            | 9.6667           | 1.0                 |
| 5                   | top_10_relative_strength_21d              | Top 10 Relative Strength 21D    | baseline    | 0.6745                   | -0.2023                          | 1.0791       | -0.2087      | 0.752            | 9.6667           | 2.0                 |
| 5                   | top_10_sentiment_7d                       | Top 10 Sentiment 7D             | baseline    | 0.2548                   | -0.622                           | 0.5139       | -0.275       | 1.2893           | 9.6667           | 1.0                 |
| 21                  | top_10_sentiment_7d                       | Top 10 Sentiment 7D             | baseline    | 1.2531                   | 0.3911                           | 1.4881       | -0.2431      | 1.2914           | 9.7143           | 3.0                 |
| 21                  | final_quant_21d_no_snapshot_sector_capped | Final Quant 21D - Sector Capped | final_model | 0.9097                   | 0.0477                           | 1.2749       | -0.2264      | 1.3917           | 9.2857           | 2.0                 |
| 21                  | spy                                       | SPY                             | benchmark   | 0.862                    | 0.0                              | 1.4144       | -0.1787      | 0.0              | 1.0              | 0.0                 |
| 21                  | top_10_historical_rating_score            | Top 10 Historical Rating Score  | baseline    | 0.8491                   | -0.0129                          | 1.2679       | -0.1707      | 0.2229           | 9.7143           | 2.0                 |
| 21                  | top_10_relative_strength_21d              | Top 10 Relative Strength 21D    | baseline    | 0.8225                   | -0.0395                          | 1.0611       | -0.2487      | 1.4971           | 9.7143           | 2.0                 |
| 21                  | equal_weight_universe                     | Equal Weight Universe           | baseline    | 0.7568                   | -0.1052                          | 1.3384       | -0.1645      | 0.0286           | 58.2857          | 1.0                 |
| 21                  | random_10_universe                        | Random 10 Universe              | baseline    | 0.7399                   | -0.1221                          | 1.2135       | -0.169       | 0.0286           | 9.7143           | 1.32                |
| 21                  | top_10_technical_momentum                 | Top 10 Technical Momentum       | baseline    | 0.6167                   | -0.2453                          | 0.9003       | -0.2265      | 1.3371           | 9.7143           | 2.0                 |
| 21                  | top_10_net_upgrade_score_30d              | Top 10 Net Upgrade Score 30D    | baseline    | 0.6034                   | -0.2587                          | 0.9454       | -0.1936      | 1.4171           | 9.7143           | 1.0                 |
| 21                  | top_10_relative_strength_63d              | Top 10 Relative Strength 63D    | baseline    | 0.5472                   | -0.3148                          | 0.8075       | -0.262       | 0.92             | 9.7143           | 1.0                 |