# Best Model vs SPY Summary

- Strategy name: `final_quant_5d_no_snapshot_no_sma_filter`
- Display name: Final Quant 5D - No SMA Filter
- Date range: 2023-01-03 to 2025-12-22
- Feature panel: `/Users/ethanpuckett/Large_Cap_Analyst/stock_quant_backtester/data/final/features_panel_2023-01-01_2026-01-01.csv`
- Holding period: 5 trading days
- Top N: 10
- Position sizing: equal_weight
- Cost assumption: 10 bps total
- Regime filter: none
- Long/short: false

## Metrics

- Total return for model: 97.25%
- Total return for SPY: 87.68%
- Excess return vs SPY: 9.57%
- Annualized return: 25.64%
- Sharpe: 1.402
- Max drawdown: -17.38%
- Average turnover: 0.670667
- Average selected count: 9.67
- Number of rebalance periods: 150
- 2025 test-period excess vs SPY: 9.16%
- Walk-forward windows beating SPY: 2

## Outputs

- Equity curve: `outputs/charts/best_model_vs_spy_3y_equity_curve.png`
- Drawdown chart: `outputs/charts/best_model_vs_spy_3y_drawdown.png`
- Returns table: `outputs/tables/best_model_vs_spy_3y_returns.csv`

## Caveats

- Back-tested performance is hypothetical.
- Snapshot analyst target models are excluded.
- Historical rating-count features use dated FMP grades-historical records available on or before each rebalance date.
- News sentiment depends on Alpha Vantage coverage and classification.
- Long/short is experimental and currently not recommended.
- This is research/paper trading only, not financial advice.