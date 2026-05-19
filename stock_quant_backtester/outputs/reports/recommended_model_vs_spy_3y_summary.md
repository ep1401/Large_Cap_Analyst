# Recommended Model vs SPY Summary

- Back-tested performance is hypothetical.
- SPY line is direct buy-and-hold from adjusted close.
- Snapshot analyst target fields are excluded.
- Historical rating-count features use dated FMP grades-historical records available on or before each rebalance date.
- News sentiment depends on Alpha Vantage coverage and classification.
- This is research/paper trading only, not financial advice.

- Strategy name: `final_quant_5d_weight_tuned_low_turnover_no_snapshot`
- Display name: Final Quant 5D - Weight Tuned Low Turnover No Snapshot
- Base score model: `final_quant_5d_weight_tuned_no_snapshot`
- Execution mode: `low_turnover_hold_band`
- Date range requested: 2023-01-01 to 2026-01-01
- Feature panel actual range used: 2023-01-03 to 2025-12-31
- Feature panel covers full requested range: False
- Feature panel path: `/Users/ethanpuckett/Large_Cap_Analyst/stock_quant_backtester/data/final/features_panel_2023-01-01_2026-01-01.csv`
- Enter rank: 10
- Hold rank: 20
- Top N: 10
- Max holding days: 21
- Rebalance frequency: 15 trading days
- Cost assumption: 20 bps total
- Regime filter: none
- Long/short: false
- Snapshot fields allowed: false
- SPY start adjusted close: 396.8541
- SPY end adjusted close: 679.7591
- Direct SPY buy-and-hold return on plotted range: 71.29%
- SPY plotted final value: $17,128.69

## Metrics

- Total return for model: 118.78%
- Total return for SPY: 71.29%
- Excess return vs SPY: 56.77%
- Annualized return: 127.52%
- Annualized volatility: 28.46%
- Sharpe: 3.053
- Max drawdown: -13.79%
- SPY max drawdown: -11.67%
- Average turnover: 0.779167
- Average holdings: 10.00
- Number of rebalance periods: 48
- Walk-forward average excess vs SPY: 6.62%
- 2025 excess vs SPY: 20.24%
- Benchmark validation difference: 0.0447%

## Outputs

- Equity curve: `outputs/charts/recommended_model_vs_spy_3y_equity_curve.png`
- Drawdown chart: `outputs/charts/recommended_model_vs_spy_3y_drawdown.png`
- Returns table: `outputs/tables/recommended_model_vs_spy_3y_returns.csv`
- Benchmark validation: `outputs/tables/recommended_model_vs_spy_3y_benchmark_validation.csv`