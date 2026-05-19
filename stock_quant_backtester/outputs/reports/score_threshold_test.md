# Score Threshold Test

- Best selective configuration: `score_gt_0.50` with `top_n=10` and `allow_cash=True`.
- Best selective excess vs SPY: 15.26%.
- Best selective Sharpe: 1.4406.
- Best selective max drawdown: -17.96%.
- Average percent invested for best selective setup: 96.67%.
- Average number of holdings for best selective setup: 8.61.
- Forced buying hurts in at least one like-for-like threshold comparison: True.
- Best cash-vs-forced excess-return improvement from allowing cash: 22.32%.

## Results

| top_n | threshold_type    | allow_cash | full_period_excess_return_vs_spy | sharpe_ratio | max_drawdown | average_percent_invested | average_number_of_holdings | windows_beating_spy |
| ----- | ----------------- | ---------- | -------------------------------- | ------------ | ------------ | ------------------------ | -------------------------- | ------------------- |
| 10    | score_gt_0.50     | True       | 0.1526                           | 1.4406       | -0.1796      | 0.9667                   | 8.6138                     | 2                   |
| 10    | top_percentile_20 | True       | 0.1512                           | 1.452        | -0.1738      | 0.9667                   | 9.4069                     | 2                   |
| 10    | top_percentile_15 | True       | 0.143                            | 1.4065       | -0.2095      | 0.9667                   | 7.7034                     | 2                   |
| 15    | top_percentile_15 | True       | 0.143                            | 1.4065       | -0.2095      | 0.9667                   | 7.7034                     | 2                   |
| 20    | top_percentile_15 | True       | 0.143                            | 1.4065       | -0.2095      | 0.9667                   | 7.7034                     | 2                   |
| 15    | top_percentile_20 | True       | 0.0969                           | 1.4297       | -0.1809      | 0.9667                   | 9.9172                     | 2                   |
| 20    | top_percentile_20 | True       | 0.0969                           | 1.4297       | -0.1809      | 0.9667                   | 9.9172                     | 2                   |
| 15    | score_gt_0.50     | True       | 0.0714                           | 1.3782       | -0.1918      | 0.9667                   | 9.3655                     | 2                   |
| 10    | none              | False      | 0.0665                           | 1.3841       | -0.1738      | 0.9667                   | 9.5793                     | 2                   |
| 10    | score_gt_0        | False      | 0.0665                           | 1.3841       | -0.1738      | 0.9667                   | 9.5793                     | 2                   |
| 10    | score_gt_0        | True       | 0.0665                           | 1.3841       | -0.1738      | 0.9667                   | 9.5793                     | 2                   |
| 10    | score_gt_0.25     | False      | 0.0665                           | 1.3841       | -0.1738      | 0.9667                   | 9.5793                     | 2                   |
| 10    | score_gt_0.25     | True       | 0.0665                           | 1.3841       | -0.1738      | 0.9667                   | 9.5793                     | 2                   |
| 10    | score_gt_0.50     | False      | 0.0665                           | 1.3841       | -0.1738      | 0.9667                   | 9.5793                     | 2                   |
| 10    | top_percentile_10 | False      | 0.0665                           | 1.3841       | -0.1738      | 0.9667                   | 9.5793                     | 2                   |
| 10    | top_percentile_15 | False      | 0.0665                           | 1.3841       | -0.1738      | 0.9667                   | 9.5793                     | 2                   |
| 10    | top_percentile_20 | False      | 0.0665                           | 1.3841       | -0.1738      | 0.9667                   | 9.5793                     | 2                   |
| 20    | score_gt_0.50     | True       | 0.0656                           | 1.3725       | -0.1918      | 0.9667                   | 9.3862                     | 2                   |
| 20    | score_gt_0.25     | True       | -0.0401                          | 1.4473       | -0.1705      | 0.9667                   | 16.1034                    | 2                   |
| 15    | none              | False      | -0.0759                          | 1.3353       | -0.1664      | 0.9667                   | 13.6759                    | 2                   |
| 15    | score_gt_0        | False      | -0.0759                          | 1.3353       | -0.1664      | 0.9667                   | 13.6759                    | 2                   |
| 15    | score_gt_0        | True       | -0.0759                          | 1.3353       | -0.1664      | 0.9667                   | 13.6759                    | 2                   |
| 15    | score_gt_0.25     | False      | -0.0759                          | 1.3353       | -0.1664      | 0.9667                   | 13.6759                    | 2                   |
| 15    | score_gt_0.50     | False      | -0.0759                          | 1.3353       | -0.1664      | 0.9667                   | 13.6759                    | 2                   |
| 15    | top_percentile_10 | False      | -0.0759                          | 1.3353       | -0.1664      | 0.9667                   | 13.6759                    | 2                   |
| 15    | top_percentile_15 | False      | -0.0759                          | 1.3353       | -0.1664      | 0.9667                   | 13.6759                    | 2                   |
| 15    | top_percentile_20 | False      | -0.0759                          | 1.3353       | -0.1664      | 0.9667                   | 13.6759                    | 2                   |
| 15    | score_gt_0.25     | True       | -0.0797                          | 1.3299       | -0.1664      | 0.9667                   | 13.6345                    | 2                   |
| 20    | none              | False      | -0.0802                          | 1.4397       | -0.1693      | 0.9667                   | 17.2345                    | 2                   |
| 20    | score_gt_0        | False      | -0.0802                          | 1.4397       | -0.1693      | 0.9667                   | 17.2345                    | 2                   |
| 20    | score_gt_0.25     | False      | -0.0802                          | 1.4397       | -0.1693      | 0.9667                   | 17.2345                    | 2                   |
| 20    | score_gt_0.50     | False      | -0.0802                          | 1.4397       | -0.1693      | 0.9667                   | 17.2345                    | 2                   |
| 20    | top_percentile_10 | False      | -0.0802                          | 1.4397       | -0.1693      | 0.9667                   | 17.2345                    | 2                   |
| 20    | top_percentile_15 | False      | -0.0802                          | 1.4397       | -0.1693      | 0.9667                   | 17.2345                    | 2                   |
| 20    | top_percentile_20 | False      | -0.0802                          | 1.4397       | -0.1693      | 0.9667                   | 17.2345                    | 2                   |
| 20    | score_gt_0        | True       | -0.0807                          | 1.4391       | -0.1693      | 0.9667                   | 17.2276                    | 2                   |
| 5     | top_percentile_10 | True       | -0.2512                          | 0.9149       | -0.2074      | 0.9667                   | 4.9655                     | 1                   |
| 5     | score_gt_0.50     | True       | -0.2804                          | 0.8892       | -0.2074      | 0.9667                   | 4.9586                     | 1                   |
| 5     | none              | False      | -0.2846                          | 0.8857       | -0.2074      | 0.9667                   | 4.9931                     | 1                   |
| 5     | score_gt_0        | False      | -0.2846                          | 0.8857       | -0.2074      | 0.9667                   | 4.9931                     | 1                   |
| 5     | score_gt_0        | True       | -0.2846                          | 0.8857       | -0.2074      | 0.9667                   | 4.9931                     | 1                   |
| 5     | score_gt_0.25     | False      | -0.2846                          | 0.8857       | -0.2074      | 0.9667                   | 4.9931                     | 1                   |
| 5     | score_gt_0.25     | True       | -0.2846                          | 0.8857       | -0.2074      | 0.9667                   | 4.9931                     | 1                   |
| 5     | score_gt_0.50     | False      | -0.2846                          | 0.8857       | -0.2074      | 0.9667                   | 4.9931                     | 1                   |
| 5     | top_percentile_10 | False      | -0.2846                          | 0.8857       | -0.2074      | 0.9667                   | 4.9931                     | 1                   |
| 5     | top_percentile_15 | False      | -0.2846                          | 0.8857       | -0.2074      | 0.9667                   | 4.9931                     | 1                   |
| 5     | top_percentile_15 | True       | -0.2846                          | 0.8857       | -0.2074      | 0.9667                   | 4.9931                     | 1                   |
| 5     | top_percentile_20 | False      | -0.2846                          | 0.8857       | -0.2074      | 0.9667                   | 4.9931                     | 1                   |
| 5     | top_percentile_20 | True       | -0.2846                          | 0.8857       | -0.2074      | 0.9667                   | 4.9931                     | 1                   |
| 10    | top_percentile_10 | True       | -0.3339                          | 0.8515       | -0.2455      | 0.9667                   | 5.4897                     | 1                   |
| 15    | top_percentile_10 | True       | -0.3339                          | 0.8515       | -0.2455      | 0.9667                   | 5.4897                     | 1                   |
| 20    | top_percentile_10 | True       | -0.3339                          | 0.8515       | -0.2455      | 0.9667                   | 5.4897                     | 1                   |