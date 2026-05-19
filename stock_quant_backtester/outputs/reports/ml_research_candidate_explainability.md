# ML Research Candidate Explainability

- This is a frozen ML research candidate.
- 2026 data was not used for training, tuning, or model selection.
- Strict leakage timing audits passed, but ML can still overfit.
- Back-tested performance is hypothetical unless actually paper-tracked live.
- Snapshot analyst target fields are excluded.
- Historical rating-count features use dated FMP grades-historical records available on or before each rebalance date.
- News sentiment depends on Alpha Vantage coverage and classification.
- ML models may overfit and require extended forward validation.
- This is research/paper trading only, not financial advice.

- Strategy: `ml_ranker_5d_no_snapshot`
- Model type: `hist_gradient_boosting_regression`
- Importance source: `permutation_importance`

## Top 20 Features

| feature_name                          | importance_mean | importance_std | importance_source      | feature_group            | abs_importance_mean |
| ------------------------------------- | --------------- | -------------- | ---------------------- | ------------------------ | ------------------- |
| distance_to_63d_high                  | -1.2e-05        | 1e-06          | permutation_importance | technical_momentum       | 1.2e-05             |
| relative_strength_63d                 | 1e-05           | 2e-06          | permutation_importance | technical_momentum       | 1e-05               |
| relative_strength_21d                 | 1e-05           | 3e-06          | permutation_importance | technical_momentum       | 1e-05               |
| historical_positive_rating_ratio      | -9e-06          | 3e-06          | permutation_importance | historical_rating_counts | 9e-06               |
| beta_to_spy_63d                       | 9e-06           | 3e-06          | permutation_importance | volatility_risk          | 9e-06               |
| percent_tickers_positive_sentiment_7d | 7e-06           | 1e-06          | permutation_importance | market_sentiment         | 7e-06               |
| relevance_weighted_sentiment_30d      | 7e-06           | 2e-06          | permutation_importance | stock_sentiment          | 7e-06               |
| market_risk_score                     | -7e-06          | 2e-06          | permutation_importance | market_regime            | 7e-06               |
| downgrade_count_30d                   | 5e-06           | 1e-06          | permutation_importance | historical_grade_events  | 5e-06               |
| net_upgrade_score_30d                 | 4e-06           | 2e-06          | permutation_importance | historical_grade_events  | 4e-06               |
| volatility_21d                        | -4e-06          | 2e-06          | permutation_importance | volatility_risk          | 4e-06               |
| sentiment_change_7d_vs_30d            | 4e-06           | 1e-06          | permutation_importance | stock_sentiment          | 4e-06               |
| historical_rating_score_change_30d    | -3e-06          | 1e-06          | permutation_importance | historical_rating_counts | 3e-06               |
| spy_return_21d                        | -3e-06          | 2e-06          | permutation_importance | market_regime            | 3e-06               |
| relevance_weighted_sentiment_7d       | 2e-06           | 1e-06          | permutation_importance | stock_sentiment          | 2e-06               |
| historical_rating_score               | 2e-06           | 2e-06          | permutation_importance | historical_rating_counts | 2e-06               |
| historical_negative_rating_ratio      | 2e-06           | 1e-06          | permutation_importance | historical_rating_counts | 2e-06               |
| market_sentiment_30d                  | -2e-06          | 2e-06          | permutation_importance | market_sentiment         | 2e-06               |
| spy_drawdown_from_63d_high            | -2e-06          | 2e-06          | permutation_importance | market_regime            | 2e-06               |
| market_sentiment_7d                   | 1e-06           | 1e-06          | permutation_importance | market_sentiment         | 1e-06               |

## Feature Groups

| feature_group            | total_abs_importance | average_abs_importance | feature_count |
| ------------------------ | -------------------- | ---------------------- | ------------- |
| technical_momentum       | 3.1e-05              | 8e-06                  | 4             |
| historical_rating_counts | 1.7e-05              | 4e-06                  | 4             |
| volatility_risk          | 1.3e-05              | 7e-06                  | 2             |
| stock_sentiment          | 1.3e-05              | 3e-06                  | 4             |
| market_regime            | 1.3e-05              | 2e-06                  | 6             |
| market_sentiment         | 1.1e-05              | 2e-06                  | 5             |
| historical_grade_events  | 1e-05                | 3e-06                  | 3             |

## Readout

- Top feature group: `technical_momentum`
- Market sentiment/regime features matter: true
- Stock sentiment matters: true
- Ratings/events matter: true
- Technicals dominate: true
- Model appears too dependent on one feature group: false