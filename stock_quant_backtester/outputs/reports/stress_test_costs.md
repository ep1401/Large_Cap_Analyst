# Stress Test Costs

- Back-tested performance is hypothetical.
- Snapshot analyst target models are excluded from the main historically safer ranking.
- Historical rating-count features use dated FMP grades-historical records available on or before each rebalance date.
- News sentiment depends on Alpha Vantage coverage and classification.
- Long/short is experimental and currently not recommended.
- Regime filters were tested and are not recommended for the main model based on current results.
- This is research/paper trading only, not financial advice.

- Recommended config under test: `final_quant_5d_weight_tuned_no_snapshot`, top_n=10, equal_weight, no threshold, no regime filter, long-only.
- Break-even cost where full-period excess vs SPY falls to zero or below: 20 bps.
- Beats SPY at 20 bps: False.
- Beats SPY at 30 bps: False.
- Beats SPY at 50 bps: False.

| total_cost_bps | full_period_excess_return_vs_spy | walk_forward_average_excess_vs_spy | 2025_excess_return_vs_spy | max_drawdown | average_turnover | beats_spy_full_period |
| -------------- | -------------------------------- | ---------------------------------- | ------------------------- | ------------ | ---------------- | --------------------- |
| 0              | 0.3958                           | 0.1022                             | 0.1762                    | -0.1701      | 0.672            | True                  |
| 5              | 0.2846                           | 0.0888                             | 0.1577                    | -0.1731      | 0.672            | True                  |
| 10             | 0.1789                           | 0.0755                             | 0.1395                    | -0.176       | 0.672            | True                  |
| 20             | -0.0175                          | 0.0495                             | 0.1037                    | -0.1819      | 0.672            | False                 |
| 30             | -0.1952                          | 0.024                              | 0.0689                    | -0.1877      | 0.672            | False                 |
| 50             | -0.5016                          | -0.0254                            | 0.002                     | -0.2241      | 0.672            | False                 |
| 75             | -0.8078                          | -0.0841                            | -0.0767                   | -0.2741      | 0.672            | False                 |
| 100            | -1.0463                          | -0.1397                            | -0.1503                   | -0.3723      | 0.672            | False                 |