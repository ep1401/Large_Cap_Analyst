# ML Forward Cost Sensitivity

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
- Break-even cost estimate: >100 bps
- Beats SPY at 20 bps: true
- Beats SPY at 30 bps: true
- Beats SPY at 50 bps: true

| total_cost_bps | ml_return | spy_return | excess_vs_spy | max_drawdown | turnover | rebalance_periods |
| -------------- | --------- | ---------- | ------------- | ------------ | -------- | ----------------- |
| 0.0            | 0.200958  | 0.084162   | 0.116795      | -0.03967     | 1.0      | 6.0               |
| 5.0            | 0.197462  | 0.084162   | 0.1133        | -0.04017     | 1.0      | 6.0               |
| 10.0           | 0.193976  | 0.084162   | 0.109813      | -0.04067     | 1.0      | 6.0               |
| 20.0           | 0.187027  | 0.084162   | 0.102865      | -0.04167     | 1.0      | 6.0               |
| 30.0           | 0.180113  | 0.084162   | 0.095951      | -0.04267     | 1.0      | 6.0               |
| 50.0           | 0.166384  | 0.084162   | 0.082222      | -0.04467     | 1.0      | 6.0               |
| 75.0           | 0.149411  | 0.084162   | 0.065249      | -0.04717     | 1.0      | 6.0               |
| 100.0          | 0.132644  | 0.084162   | 0.048482      | -0.04967     | 1.0      | 6.0               |