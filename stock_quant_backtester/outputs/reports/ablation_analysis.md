# Ablation Analysis

- Base model tested: `final_quant_5d_no_snapshot_no_sma_filter`.
- Ranking is based on the full 2023-2025 window, with walk-forward windows included to avoid optimizing on 2025 alone.

## Findings
- Best simplified variant: `remove_recent_downgrade_filter` with excess vs SPY 8.94% and 2/3 walk-forward windows beating SPY.
- Full model excess vs SPY: 6.65%; best simplified delta: 2.29%.
- Signal groups that clearly help when kept in the model: historical rating score core (-38.93% hit when removed), relative strength (-24.98% hit when removed), grade-event features (-15.01% hit when removed), sentiment (-8.71% hit when removed), volatility penalty (-7.99% hit when removed).
- Components or filters that improved results when removed: recent downgrade filter (2.29% improvement when removed), negative news filter (-0.36% change when removed).
- Full model appears over-blended or over-filtered: True.

## Results

| label                                         | full_period_total_return | full_period_excess_return_vs_spy | sharpe_ratio | max_drawdown | windows_beating_spy | delta_excess_vs_full | delta_sharpe_vs_full | impact_label |
| --------------------------------------------- | ------------------------ | -------------------------------- | ------------ | ------------ | ------------------- | -------------------- | -------------------- | ------------ |
| remove_recent_downgrade_filter                | 0.9662                   | 0.0894                           | 1.4029       | -0.2018      | 2                   | 0.0229               | 0.0188               | helps        |
| full_model                                    | 0.9433                   | 0.0665                           | 1.3841       | -0.1738      | 2                   | 0.0                  | 0.0                  | mixed        |
| remove_negative_news_filter                   | 0.9397                   | 0.0629                           | 1.3815       | -0.1738      | 2                   | -0.0036              | -0.0026              | mixed        |
| only_historical_ratings_and_events            | 0.9308                   | 0.0541                           | 1.2916       | -0.1831      | 1                   | -0.0125              | -0.0925              | hurts        |
| remove_breakout                               | 0.911                    | 0.0342                           | 1.3457       | -0.1879      | 2                   | -0.0323              | -0.0384              | hurts        |
| only_historical_ratings_and_relative_strength | 0.8807                   | 0.0039                           | 1.2628       | -0.1784      | 2                   | -0.0626              | -0.1213              | hurts        |
| remove_volatility_penalty                     | 0.8634                   | -0.0133                          | 1.2547       | -0.1905      | 2                   | -0.0799              | -0.1294              | hurts        |
| remove_sentiment                              | 0.8561                   | -0.0206                          | 1.2656       | -0.185       | 2                   | -0.0871              | -0.1186              | hurts        |
| remove_grade_events                           | 0.7932                   | -0.0835                          | 1.2525       | -0.1887      | 1                   | -0.1501              | -0.1316              | hurts        |
| remove_relative_strength                      | 0.6935                   | -0.1832                          | 1.1261       | -0.1958      | 1                   | -0.2498              | -0.258               | hurts        |
| remove_historical_rating_score                | 0.554                    | -0.3228                          | 0.9803       | -0.1983      | 0                   | -0.3893              | -0.4038              | hurts        |
| only_technical_and_sentiment                  | 0.4734                   | -0.4033                          | 0.8932       | -0.1684      | 1                   | -0.4699              | -0.4909              | hurts        |