# Weight Search 5D No Snapshot

## Baseline
- Baseline strategy: `final_quant_5d_no_snapshot_no_sma_filter`.
- Walk-forward average excess vs SPY: 5.44%.
- Worst window excess vs SPY: -6.11%.
- Max drawdown: -17.38%.
- Average turnover: 0.7277.

## Selection Rules
- Candidate must beat SPY in at least 2 of 3 walk-forward windows.
- Candidate ranking prioritizes walk-forward average excess, then worst-window excess, then drawdown, then turnover.
- Promotion requires a strictly better walk-forward average than the current incumbent tuned model.
- Promotion also requires beating the incumbent in every walk-forward test window and no worse than -3.00 percentage points of drawdown delta vs incumbent.

## Incumbent
- Incumbent reference strategy: `current tuned model`.
- Incumbent walk-forward average excess vs SPY: 7.91%.
- Incumbent 2024 H1 excess vs SPY: 14.64%.
- Incumbent 2024 H2 excess vs SPY: -4.15%.
- Incumbent 2025 excess vs SPY: 13.22%.

## Selected Candidate
- Candidate id: `candidate_0144`.
- Promoted to `final_quant_5d_weight_tuned_no_snapshot`: False.
- Walk-forward average excess vs SPY: 8.53%.
- Walk-forward delta vs baseline: 3.10%.
- Walk-forward delta vs incumbent: 0.63%.
- Beats incumbent in all walk-forward windows: False.
- Worst window excess vs SPY: -4.62%.
- Max drawdown: -18.69%.
- Drawdown delta vs incumbent: -1.09%.
- Average turnover: 0.6868.
- Turnover vs baseline ratio: 0.94x.

### Selected Weights
- `weight_historical_rating_score` = 0.1099
- `weight_historical_positive_rating_ratio` = 0.1819
- `weight_historical_negative_rating_ratio` = -0.1556
- `weight_net_upgrade_score_30d` = 0.1482
- `weight_downgrade_count_30d` = -0.1156
- `weight_relative_strength_21d` = 0.1398
- `weight_relevance_weighted_sentiment_7d` = 0.0375
- `weight_sentiment_change_7d_vs_30d` = 0.0491
- `weight_volatility_21d` = -0.0050
- `weight_breakout_63d` = 0.0437
- `weight_negative_news_flag` = -0.0072
- `weight_recent_downgrade_flag` = -0.0064

### Baseline Normalized Weights
- `weight_historical_rating_score` = 0.1923
- `weight_historical_positive_rating_ratio` = 0.1154
- `weight_historical_negative_rating_ratio` = -0.1154
- `weight_net_upgrade_score_30d` = 0.1538
- `weight_downgrade_count_30d` = -0.1154
- `weight_relative_strength_21d` = 0.1154
- `weight_relevance_weighted_sentiment_7d` = 0.0769
- `weight_sentiment_change_7d_vs_30d` = 0.0385
- `weight_volatility_21d` = -0.0385
- `weight_breakout_63d` = 0.0385
- `weight_negative_news_flag` = 0.0000
- `weight_recent_downgrade_flag` = 0.0000

## Top Candidates

| candidate_id   | walk_forward_average_excess_vs_spy | 2024_h1_excess_return_vs_spy | 2024_h2_excess_return_vs_spy | 2025_excess_return_vs_spy | worst_window_excess_vs_spy | windows_beating_spy | max_drawdown | average_turnover | walk_forward_delta_vs_baseline | walk_forward_delta_vs_incumbent | beats_incumbent_in_all_windows |
| -------------- | ---------------------------------- | ---------------------------- | ---------------------------- | ------------------------- | -------------------------- | ------------------- | ------------ | ---------------- | ------------------------------ | ------------------------------- | ------------------------------ |
| candidate_0144 | 0.0853                             | 0.1967                       | -0.0462                      | 0.1055                    | -0.0462                    | 2                   | -0.1869      | 0.6868           | 0.031                          | 0.0063                          | False                          |
| candidate_0048 | 0.0827                             | 0.1578                       | -0.0421                      | 0.1322                    | -0.0421                    | 2                   | -0.176       | 0.7221           | 0.0283                         | 0.0036                          | False                          |
| candidate_0033 | 0.0791                             | 0.1467                       | -0.0415                      | 0.1322                    | -0.0415                    | 2                   | -0.176       | 0.721            | 0.0248                         | 0.0001                          | False                          |
| candidate_0046 | 0.0791                             | 0.1464                       | -0.0415                      | 0.1322                    | -0.0415                    | 2                   | -0.176       | 0.7223           | 0.0247                         | 0.0                             | False                          |
| candidate_0001 | 0.0791                             | 0.1464                       | -0.0415                      | 0.1322                    | -0.0415                    | 2                   | -0.176       | 0.7236           | 0.0247                         | 0.0                             | False                          |
| candidate_0045 | 0.0791                             | 0.1464                       | -0.0415                      | 0.1322                    | -0.0415                    | 2                   | -0.176       | 0.7236           | 0.0247                         | 0.0                             | False                          |
| candidate_0047 | 0.0791                             | 0.1464                       | -0.0415                      | 0.1322                    | -0.0415                    | 2                   | -0.176       | 0.7236           | 0.0247                         | 0.0                             | False                          |
| candidate_0034 | 0.0781                             | 0.1469                       | -0.0448                      | 0.1322                    | -0.0448                    | 2                   | -0.176       | 0.721            | 0.0237                         | -0.0009                         | False                          |
| candidate_0044 | 0.0763                             | 0.1358                       | -0.0435                      | 0.1367                    | -0.0435                    | 2                   | -0.1764      | 0.7396           | 0.022                          | -0.0027                         | False                          |
| candidate_0077 | 0.076                              | 0.2278                       | -0.0358                      | 0.036                     | -0.0358                    | 2                   | -0.1918      | 0.7291           | 0.0216                         | -0.0031                         | False                          |