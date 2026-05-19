# Final Strategy Recommendation

- Back-tested performance is hypothetical.
- Snapshot analyst target models are excluded from the main historically safer ranking.
- Historical rating-count features use dated FMP grades-historical records available on or before each rebalance date.
- News sentiment depends on Alpha Vantage coverage and classification.
- Long/short is experimental and currently not recommended.
- Regime filters were tested and are not recommended for the main model based on current results.
- This is research/paper trading only, not financial advice.

## Current Baseline
- Current baseline strategy: `final_quant_5d_no_snapshot_no_sma_filter`.
- Walk-forward average excess vs SPY: 5.44%.
- 2025 excess vs SPY: 9.10%.
- Windows beating SPY: 2/3.
- Max drawdown: -17.38%.

## Side-By-Side

| strategy_name                                         | display_name                                | walk_forward_average_excess_vs_spy | windows_beating_spy | 2025_excess_return_vs_spy | max_drawdown | average_percent_invested |
| ----------------------------------------------------- | ------------------------------------------- | ---------------------------------- | ------------------- | ------------------------- | ------------ | ------------------------ |
| final_quant_5d_no_snapshot_no_sma_filter              | Final Quant 5D - No SMA Filter              | 0.0544                             | 2                   | 0.091                     | -0.1738      | 0.9667                   |
| final_quant_5d_weight_tuned_no_snapshot               | Final Quant 5D - Weight Tuned No Snapshot   | 0.0791                             | 2                   | 0.1322                    | -0.176       | 1.0                      |
| final_quant_5d_selective_no_snapshot                  | Final Quant 5D Selective - No Snapshot      | 0.0492                             | 2                   | 0.0153                    | -0.2018      | 0.9213                   |
| historical_rating_score_selective_5d                  | Historical Rating Score Selective 5D        | 0.028                              | 3                   | 0.0317                    | -0.1944      | 0.9667                   |
| final_quant_5d_no_recent_downgrade_filter_no_snapshot | Final Quant 5D - No Recent Downgrade Filter | 0.0532                             | 2                   | 0.0276                    | -0.2018      | 0.9667                   |

## Selective Model
- Best selective row: threshold=0.5, top_n=10, allow_cash=True.
- Selective model becomes recommended: False.
- Selective walk-forward average excess vs SPY: 4.92%.
- Selective average percent invested: 92.13%.
- Selective average holdings: 9.21.

## Weight Search
- Weight-tuned model promoted: True.
- Weight search file present: True.
- Tuned walk-forward average excess vs SPY: 7.91%.
- Tuned 2025 excess vs SPY: 13.22%.
- Tuned max drawdown: -17.60%.

## Historical Rating Baseline
- Best historical-rating baseline/challenger: `historical_rating_score_selective_5d`.
- Historical-rating walk-forward average excess vs SPY: 2.80%.
- Complex current baseline beats the simple historical-rating baseline on both walk-forward and 2025 test: True.
- Simple historical-rating-only baseline from the earlier baseline report: 7.08% full-period excess vs SPY.

## Score Spread Diagnostics
- Average selected minus SPY spread: 0.11%.
- Average selected minus non-selected spread: 0.17%.
- Average top-decile minus bottom-decile spread: 0.06%.

## Recommendation
- Recommended strategy: `final_quant_5d_weight_tuned_no_snapshot`.
- Recommended threshold: none.
- Recommended allow_cash setting: False.
- Threshold/selective strategy tested and not promoted: True.
- Long/short tested and not recommended: True.
- Regime filters tested and not recommended: True.
- Standalone no-recent-downgrade hard-filter variant tested and promoted: False.
- Removing the recent downgrade hard filter changed full-period excess by 2.29%.
- Edge is still paper trading only: True.