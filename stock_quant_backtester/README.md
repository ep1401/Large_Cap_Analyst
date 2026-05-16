# Large-Cap Analyst Momentum and Breakout Backtester

This project backtests a weekly large-cap U.S. stock selection strategy that combines analyst price-target strength, technical breakout behavior, and optional news sentiment features. It is designed for historical research and comparison against `SPY`, not for live trading.

This project is a historical research backtest, not financial advice and not a live trading system.

The largest methodological risk is point-in-time analyst data. If the analyst API returns only current consensus data, then it cannot be used for a valid historical backtest. The code therefore supports running the backtest without analyst filters, and the README clearly marks whether analyst data is point-in-time.

The initial universe uses a static large-cap stock list, which introduces survivorship bias. A production-grade backtest should use historical index constituents or historical Fortune 500 membership.

Transaction costs are simplified. Slippage, bid-ask spreads, taxes, borrow costs for shorts, and market impact are not fully modeled.

News sentiment availability may vary by ticker and date, and the default fast sentiment workflow now uses a cache-first Alpha Vantage news window plus optional local rescoring.

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
- `Alpha Vantage` for ticker-level news sentiment coverage

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
python scripts/16_fetch_fmp_historical_grades.py
python scripts/00_cache_status.py
python scripts/12_fetch_alpha_vantage_news.py --dry-run
python scripts/12_fetch_alpha_vantage_news.py
python scripts/13_build_news_sentiment.py
python scripts/04_build_features.py
python scripts/05_run_backtest.py
python scripts/14_compare_sentiment_models.py
python scripts/15_run_fast_sentiment_backtest.py
python scripts/17_compare_historical_analyst_models.py
python scripts/18_validate_historical_ratings.py
python scripts/20_run_final_quant_model_1y.py
python scripts/30_run_full_3y_rebuild.py --dry-run
python scripts/99_clean_outputs.py --list
python scripts/07_grid_search.py
python scripts/06_generate_report.py
python scripts/08_validate_backtest.py
python scripts/09_compare_model_improvements.py
python scripts/10_walk_forward_search.py
python scripts/11_ml_rank_model.py
```

## Script Details

### `scripts/01_fetch_prices.py`

Fetches EODHD daily prices for the static large-cap universe and `SPY`, saves per-ticker raw CSV caches under `data/raw/prices/eodhd/`, and writes `data/processed/prices_all.csv`.

### `scripts/02_fetch_analyst_data.py`

Fetches current analyst target snapshots from FMP, caches raw endpoint responses under `data/raw/analyst/fmp/`, and writes processed features to `data/processed/analyst_features.csv`.

Warning: Analyst data may not be point-in-time unless historical target summary fields are available from the API plan. Current-only analyst data should not be used for a true historical backtest.

Supported modes:

- `snapshot_current`
- `none`

### `scripts/04_build_features.py`

Builds `data/final/features_panel.csv` with technical features, snapshot analyst target fields, optional sentiment, point-in-time `grades-historical` rating-count features, optional point-in-time grade-event rolling features, regime columns, liquidity columns, and forward-return evaluation columns. It also writes `data/final/features_panel_sentiment_1y.csv` for the default fast sentiment window.

### `scripts/00_cache_status.py`

Prints a cache summary for EODHD, FMP, Alpha Vantage, and the main processed files so you can see what is already reusable before starting a rerun.

### `scripts/12_fetch_alpha_vantage_news.py`

Fetches ticker-month Alpha Vantage news sentiment only for the configured sentiment window by default, caches raw JSON under `data/raw/news/alpha_vantage/`, and writes normalized CSV outputs:

- `data/processed/stock_news_alpha_vantage.csv`
- `data/processed/stock_news.csv`

### `scripts/13_build_news_sentiment.py`

Builds article-level and daily sentiment outputs from `data/processed/stock_news.csv`. By default it uses provider sentiment from Alpha Vantage when available, which makes reruns fast. If `--rescore-with-finbert` or `--prefer-finbert` is passed, it uses `ProsusAI/finbert` when locally available and falls back to a simple lexicon-based scorer if not.

Outputs:

- `data/processed/news_sentiment_articles.csv`
- `data/processed/news_sentiment_daily.csv`

### `scripts/14_compare_sentiment_models.py`

Runs sentiment-aware strategy comparisons, writes:

- `outputs/tables/sentiment_model_comparison.csv`
- `outputs/tables/sentiment_diagnostics.csv`
- `outputs/reports/sentiment_model_comparison.md`

### `scripts/15_run_fast_sentiment_backtest.py`

Runs the quick 1-year sentiment workflow end to end:

1. checks cache status
2. fetches only missing Alpha Vantage ticker-month files
3. rebuilds sentiment CSVs if needed
4. rebuilds features
5. runs the 1-year sentiment comparison with `_1y` outputs

### `scripts/16_fetch_fmp_historical_grades.py`

Fetches both FMP historical analyst datasets, caches raw payloads under `data/raw/analyst/fmp_historical_grades/`, and writes:

- `data/processed/historical_analyst_rating_counts.csv` from `grades-historical`
- `data/processed/historical_analyst_grade_events.csv` from `grades`

### `scripts/17_compare_historical_analyst_models.py`

Compares technical, snapshot-analyst, historical grade-event, and historical rating-count strategies. Writes:

- `outputs/tables/historical_analyst_model_comparison.csv`
- `outputs/tables/historical_rating_count_diagnostics.csv`
- `outputs/reports/historical_analyst_model_comparison.md`

### `scripts/18_validate_historical_ratings.py`

Samples ticker/date rows from the feature panel and confirms the `grades-historical` merge only uses records with `rating_date <= feature_date`, validates safe missing-data fills, and checks analyst data-mode labeling.

### `scripts/20_run_final_quant_model_1y.py`

Runs the finalized one-year quant comparison across snapshot analyst, technical, sentiment-aware, and hybrid final-model variants. It saves dated comparison tables, a final report, and dedicated final-model charts without overwriting generic backtest outputs.

### `scripts/30_run_full_3y_rebuild.py`

Runs the full 3-year rebuild workflow end to end. It prints cache/runtime estimates, optionally clears outputs, optionally clears caches only with `--clear-cache --yes`, fetches all required API data, rebuilds processed datasets and features, validates historical ratings, runs full-period backtests for `5`, `21`, and `63` day horizons, and writes dated full-run reports that separate historically safer models from snapshot/exploratory analyst models.

### `scripts/99_clean_outputs.py`

Lists or deletes generated outputs under `outputs/tables/`, `outputs/reports/`, and `outputs/charts/` while preserving raw API caches and feature data by default.

### `scripts/05_run_backtest.py`

Runs multiple strategy variants for a chosen holding period:

- `full_model`
- `strict_checklist_model`
- `technical_only`
- `technical_momentum_model`
- `analyst_snapshot_model`

It can also optionally run a condition-based exit backtest for checklist-style strategies.

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

### `scripts/09_compare_model_improvements.py`

Builds a side-by-side comparison of rule-based improvements, benchmark baselines, simple momentum baselines, random-selection baselines, and condition-based variants. Saves `outputs/tables/model_improvement_comparison.csv` and `outputs/reports/model_improvement_comparison.md`.

### `scripts/10_walk_forward_search.py`

Runs a development-period parameter search and then reports out-of-sample test performance without retuning on the test window. Saves `outputs/tables/walk_forward_search_results.csv` and `outputs/reports/walk_forward_search_summary.md`.

### `scripts/11_ml_rank_model.py`

Runs an optional ML ranking experiment on the 2023 to 2024 training window and evaluates it on the 2025 test window. This does not replace the rule-based models and keeps the analyst snapshot caveat explicit. Saves `outputs/tables/ml_model_results.csv` and `outputs/reports/ml_model_summary.md`.

## Outputs

Primary expected outputs:

- `data/final/features_panel.csv`
- `data/final/features_panel_sentiment_1y.csv`
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
- `outputs/reports/model_improvement_comparison.md`
- `outputs/reports/walk_forward_search_summary.md`
- `outputs/reports/ml_model_summary.md`
- `outputs/reports/sentiment_model_comparison.md`
- `outputs/reports/sentiment_model_comparison_1y.md`
- `outputs/reports/final_quant_model_1y_report.md`

## Notes On Interpretation

- Technical features are point-in-time safe if built from historical prices only.
- Analyst features should be treated as research-only unless a true historical target history feed is available.
- Sentiment coverage can be sparse and skewed toward heavily covered names.
- The backtester now supports 5, 21, and 63 trading-day holding periods, uses non-overlapping rebalances for those horizons, and uses turnover-based transaction cost modeling.

## API Caching and Fast Sentiment Runs

- API responses are cached under `data/raw/`.
- Processed CSV files are saved under `data/processed/`.
- Feature panels are saved under `data/final/`.
- The default `START_DATE` and `END_DATE` now focus on the most recent one-year window for faster reruns.
- By default, sentiment fetching only uses a 1-year window.
- To fetch the full historical period, pass explicit `--start-date` and `--end-date`.
- To avoid API calls, rerun scripts without `--force`.
- To force a refetch, pass `--force`.
- `scripts/00_cache_status.py` shows what is already cached.
- `scripts/15_run_fast_sentiment_backtest.py` runs a quick 1-year sentiment test.
- `scripts/20_run_final_quant_model_1y.py` runs the final one-year strategy comparison.
- `scripts/99_clean_outputs.py --list` shows what old generated outputs would be removed.
- `scripts/99_clean_outputs.py --outputs-only --yes` clears old generated outputs while keeping caches.

## News Sentiment Pipeline

- Alpha Vantage news is used as the default source for ticker-level news sentiment.
- Provider sentiment from Alpha Vantage is reused by default for fast reruns.
- FinBERT is used locally when explicitly requested or when provider sentiment is unavailable.
- If FinBERT is unavailable, fallback sentiment is used.
- Sentiment features are aggregated daily and rolled over 7-day and 30-day windows.
- Sentiment should be judged by out-of-sample test performance, not full-period performance.

News sentiment is based on available Alpha Vantage news coverage and locally generated model scores. Missing articles, source coverage differences, publication timing, and model classification errors may affect historical accuracy.

## Rate Limits

The fetchers throttle requests explicitly using environment-configured provider limits:

- `EODHD_CALLS_PER_MINUTE=1000`
- `FMP_CALLS_PER_MINUTE=300`
- `ALPHA_VANTAGE_REQUESTS_PER_MINUTE=60`

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

## Historical Analyst Grades

Snapshot analyst target data is not point-in-time and remains exploratory in this project.

### Using FMP grades-historical for Historical Analyst Ratings

- `grades-historical` gives dated historical rating-count snapshots.
- `grades` gives individual analyst action events.
- `price-target-summary` and `price-target-consensus` are snapshot-style unless point-in-time target history is available.
- Historically valid analyst models should prefer `grades-historical` and dated grade events.
- Snapshot target-upside models remain exploratory.

Historically valid analyst signals:

- `grades-historical` rating-count snapshots
- dated grade action events from `grades`

Exploratory analyst signals:

- `price-target-consensus`
- `price-target-summary`
- `consensus_upside`
- `low_target_upside`
- `last_month_target_upside`
- `last_quarter_target_upside`
- `last_year_target_upside`

Historical rating-count features are built from dated FMP grades-historical records and use only the latest record available on or before each rebalance date.

If your FMP plan exposes dated grade events, the project can also build rolling historical grade-event features from them. These event features only use grade actions available on or before each rebalance date, which makes them better suited for historical analyst claims than current snapshot price-target consensus.

Historical grade features are not the same thing as true historical price-target consensus. They capture dated analyst rating actions such as upgrades, downgrades, and maintained ratings. If the FMP historical grade endpoint is blocked by your plan or unavailable for a ticker, the historical-grade strategies are skipped with a clear message.

Important caveat: analyst-driven snapshot results use FMP data as a current snapshot merged across historical dates unless true point-in-time analyst history is provided. These results should be treated as research exploration, not a valid historical analyst-signal backtest.

## Final One-Year Quant Model

The `final_quant_model_1y` family is the project’s best-effort honest one-year workflow. It blends:

- FMP snapshot analyst target consensus and price target summary features
- EODHD technical and momentum features
- Alpha Vantage sentiment features
- FMP historical analyst-grade event overlays when available

The best-performing one-year result may still be exploratory if it relies on snapshot analyst targets. The final report therefore labels analyst data mode explicitly and states whether the top result is historically cleaner or still snapshot-driven.
