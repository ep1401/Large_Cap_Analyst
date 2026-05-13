# Large-Cap Analyst Momentum and Breakout Backtester

This project backtests a weekly large-cap U.S. stock selection strategy that combines analyst price-target strength and technical breakout behavior, with optional sentiment support that is currently disabled in the default workflow. It is designed for historical research and comparison against `SPY`, not for live trading.

This project is a historical research backtest, not financial advice and not a live trading system.

The largest methodological risk is point-in-time analyst data. If the analyst API returns only current consensus data, then it cannot be used for a valid historical backtest. The code therefore supports running the backtest without analyst filters, and the README clearly marks whether analyst data is point-in-time.

The initial universe uses a static large-cap stock list, which introduces survivorship bias. A production-grade backtest should use historical index constituents or historical Fortune 500 membership.

Transaction costs are simplified. Slippage, bid-ask spreads, taxes, borrow costs for shorts, and market impact are not fully modeled.

News sentiment availability may vary by ticker and date, and the Alpha Vantage sentiment leg is currently paused in the default workflow.

## Strategy Overview

Each Monday, or the first trading day of the week:

1. Load the current universe snapshot.
2. Build features using only data available by that rebalance date.
3. Apply minimal hard filters.
4. Rank candidates cross-sectionally.
5. Buy the top `N` names equally weighted.
6. Hold for `5`, `21`, or `63` trading days depending on the chosen test.
7. Compare strategy returns against `SPY`.

The project can run in technical-only mode if analyst or sentiment data is unavailable.

## APIs Required

- `EODHD` for daily OHLCV and benchmark price history
- `Financial Modeling Prep` for analyst targets and coverage
- `Alpha Vantage` for news sentiment is optional and currently not part of the default pipeline

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

```bash
cp .env.example .env
# add API keys
```

## Project Layout

```text
stock_quant_backtester/
├── data/
├── outputs/
├── src/
├── scripts/
└── notebooks/
```

## Run Order

```bash
python scripts/01_fetch_prices.py
python scripts/02_fetch_analyst_data.py
python scripts/04_build_features.py
python scripts/05_run_backtest.py
python scripts/07_grid_search.py
python scripts/06_generate_report.py
python scripts/08_validate_backtest.py
```

## Script Details

### `scripts/01_fetch_prices.py`

Fetches EODHD daily prices for the static large-cap universe and `SPY`, saves per-ticker raw CSV files, and writes `data/processed/prices_all.csv`.

### `scripts/02_fetch_analyst_data.py`

Fetches current analyst target snapshots from FMP and writes processed features to `data/processed/analyst_features.csv`.

Warning: Analyst data may not be point-in-time unless historical target summary fields are available from the API plan. Current-only analyst data should not be used for a true historical backtest.

Supported modes:

- `research_current_snapshot`
- `historical_backtest_without_analyst`

### `scripts/04_build_features.py`

Builds `data/final/features_panel.csv` with technical, analyst, optional sentiment, regime, liquidity, and forward-return evaluation columns.

### `scripts/05_run_backtest.py`

Runs multiple strategy variants for a chosen holding period:

- `full_model`
- `technical_only`
- `analyst_only`

Outputs:

- `outputs/tables/weekly_portfolio_returns.csv`
- `outputs/tables/weekly_holdings.csv`
- `outputs/tables/trades.csv`
- `outputs/tables/strategy_comparison.csv`

### `scripts/06_generate_report.py`

Builds `outputs/reports/backtest_summary.md` with settings, results, limitations, and chart references.

### `scripts/07_grid_search.py`

Runs parameter combinations across strategy, `top_n`, holding period, analyst threshold, and regime settings. Saves `outputs/tables/grid_search_results.csv` and `outputs/reports/grid_search_summary.md`.

### `scripts/08_validate_backtest.py`

Runs basic correctness checks on holdings, benchmark exclusion, turnover-based costs, scoring inputs, and rebalance counts for `5`, `21`, and `63` trading-day schedules.

## Outputs

Primary expected outputs:

- `data/final/features_panel.csv`
- `outputs/tables/weekly_portfolio_returns.csv`
- `outputs/tables/weekly_holdings.csv`
- `outputs/tables/strategy_comparison.csv`
- `outputs/tables/grid_search_results.csv`
- `outputs/charts/equity_curve_vs_spy.png`
- `outputs/charts/drawdown_vs_spy.png`
- `outputs/charts/weekly_excess_returns.png`
- `outputs/charts/qualifying_stocks_per_week.png`
- `outputs/reports/backtest_summary.md`
- `outputs/reports/grid_search_summary.md`

## Notes On Interpretation

- Technical features are point-in-time safe if built from historical prices only.
- Analyst features should be treated as research-only unless a true historical target history feed is available.
- Sentiment coverage can be sparse and skewed toward heavily covered names. The current default run does not include sentiment data.
- The backtester now supports 5, 21, and 63 trading-day holding periods, uses non-overlapping rebalances for those horizons, and uses turnover-based transaction cost modeling.

## Rate Limits

The fetchers throttle requests explicitly using environment-configured provider limits:

- `EODHD_CALLS_PER_MINUTE=1000`
- `FMP_CALLS_PER_MINUTE=300`

The default implementation spaces requests to stay under those ceilings during sequential fetch runs.

## Model Improvement Notes

- The full model is now ranking-first rather than strict filter-first.
- Analyst target data is long-horizon, so the project tests `5`, `21`, and `63` trading-day holding periods.
- The benchmark ticker is excluded from candidate portfolios.
- Transaction costs are turnover-based.
- The SPY 200-day moving average regime filter is optional.
- Analyst snapshot data is not a valid point-in-time backtest.
- `holding_period_days` controls both the forward-return horizon and the rebalance frequency.
- `21`-day and `63`-day returns are not compounded weekly in the corrected engine.
- Annualization uses `252 / holding_period_days`, not a fixed weekly constant.
