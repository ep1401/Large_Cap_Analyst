from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mpl-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter

matplotlib.use("Agg")

sys.path.append(str(PROJECT_ROOT))

from src.build_features import build_feature_panel
from src.build_news_sentiment import build_news_sentiment_outputs
from src.config import Config
from src.fetch_alpha_vantage_news import (
    build_monthly_windows,
    fetch_alpha_vantage_news_cache,
    normalize_alpha_vantage_news_cache,
)
from src.fetch_fmp_historical_grades import build_historical_grade_datasets
from src.fetch_prices import fetch_eodhd_prices
from src.metrics import calculate_performance_metrics
from src.no_snapshot_research import dataframe_to_markdown, summarize_backtest
from src.promoted_weights import assert_promoted_final_5d_tuned_weights_available
from src.recommended_strategy import (
    config_path,
    load_recommended_strategy_config,
    precompute_recommended_low_turnover_panels,
    run_low_turnover_recommended_backtest,
    top_signal_reasons,
)
from src.scoring import NO_SNAPSHOT_STRATEGIES, SNAPSHOT_FIELD_COLUMNS, strategy_display_name, strategy_score_fields
from src.universe import get_tickers
from src.utils import LOGGER, RateLimiter, load_dataframe, save_dataframe


FORWARD_START = pd.Timestamp("2026-01-01")
EXPECTED_STRATEGY = "final_quant_5d_weight_tuned_low_turnover_no_snapshot"
BASE_SCORE_MODEL = "final_quant_5d_weight_tuned_no_snapshot"
EXECUTION_MODE = "low_turnover_hold_band"
GRADES_FULL_HISTORY_START = "2022-01-01"


def _compute_drawdown(values: pd.Series) -> pd.Series:
    return values / values.cummax() - 1.0


def _save_plot(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def _make_equity_curve_plot(returns_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 7), facecolor="white")
    ax.set_facecolor("white")
    ax.plot(returns_df["date"], returns_df["model_value"], linewidth=2.2, label="Recommended Low-Turnover Model", color="#0f766e")
    ax.plot(returns_df["date"], returns_df["spy_value_direct"], linewidth=2.0, label="SPY Buy & Hold", color="#1f2937")
    ax.set_title("Forward 2026: Recommended Low-Turnover Model vs SPY Buy & Hold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of $10,000")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"${value:,.0f}"))
    ax.grid(True, alpha=0.25)
    ax.legend()
    _save_plot(fig, output_path)


def _make_drawdown_plot(returns_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 7), facecolor="white")
    ax.set_facecolor("white")
    ax.plot(returns_df["date"], returns_df["model_drawdown"], linewidth=2.2, label="Recommended Low-Turnover Model", color="#b91c1c")
    ax.plot(returns_df["date"], returns_df["spy_drawdown"], linewidth=2.0, label="SPY Buy & Hold", color="#1f2937")
    ax.set_title("Forward 2026 Drawdown: Recommended Low-Turnover Model vs SPY Buy & Hold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown %")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0%}"))
    ax.grid(True, alpha=0.25)
    ax.legend()
    _save_plot(fig, output_path)


def _validate_recommended_strategy(config: Config):
    recommended = load_recommended_strategy_config(config.project_root)
    if recommended.strategy_name != EXPECTED_STRATEGY:
        raise ValueError(f"recommended_strategy.yaml must be locked to {EXPECTED_STRATEGY}; found {recommended.strategy_name}")
    if recommended.regime_filter != "none":
        raise ValueError("Recommended strategy must have regime_filter set to none.")
    if recommended.long_short:
        raise ValueError("Recommended strategy must be long-only.")
    if recommended.strategy_name not in NO_SNAPSHOT_STRATEGIES:
        raise ValueError("Recommended strategy is not registered as a no-snapshot strategy.")
    offending = sorted(strategy_score_fields(recommended.strategy_name) & SNAPSHOT_FIELD_COLUMNS)
    if offending:
        raise ValueError(f"Recommended strategy uses snapshot fields: {', '.join(offending)}")
    assert_promoted_final_5d_tuned_weights_available(config.project_root)
    return recommended


def _merge_dedup(existing: pd.DataFrame, new: pd.DataFrame, keys: list[str], sort_cols: list[str]) -> pd.DataFrame:
    if existing.empty:
        merged = new.copy()
    elif new.empty:
        merged = existing.copy()
    else:
        merged = pd.concat([existing, new], ignore_index=True)
    merged = merged.sort_values(sort_cols).drop_duplicates(subset=keys, keep="last").reset_index(drop=True)
    return merged


def _update_prices(config: Config, tickers: list[str], start_date: str, end_date: str, force_refresh: bool) -> pd.DataFrame:
    raw_prices_dir = config.raw_dir / "prices" / "eodhd"
    raw_prices_dir.mkdir(parents=True, exist_ok=True)
    rate_limiter = RateLimiter(calls_per_minute=config.eodhd_calls_per_minute)
    new_frames: list[pd.DataFrame] = []
    start_ts = pd.Timestamp(start_date)

    for ticker in tickers:
        cache_path = raw_prices_dir / f"{ticker}_{start_date}_{end_date}.csv"
        use_cache = False
        if config.cache_enabled and not force_refresh and cache_path.exists():
            cached = load_dataframe(cache_path, parse_dates=["date"])
            if not cached.empty and pd.Timestamp(cached["date"].max()) >= start_ts:
                new_frames.append(cached.copy())
                use_cache = True
        if use_cache:
            continue
        fetched = fetch_eodhd_prices(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            api_key=config.eodhd_api_key,
            rate_limiter=rate_limiter,
        )
        save_dataframe(cache_path, fetched)
        new_frames.append(fetched)

    if not new_frames:
        raise ValueError("No forward price data was fetched or loaded from valid 2026-window caches.")
    new_prices = pd.concat(new_frames, ignore_index=True).sort_values(["ticker", "date"]).reset_index(drop=True)
    existing_path = config.processed_dir / "prices_all.csv"
    existing = load_dataframe(existing_path, parse_dates=["date"]) if existing_path.exists() else pd.DataFrame(columns=new_prices.columns)
    merged = _merge_dedup(existing, new_prices, keys=["date", "ticker"], sort_cols=["ticker", "date"])
    save_dataframe(existing_path, merged)
    return merged


def _update_news_and_sentiment(config: Config, tickers: list[str], start_date: str, end_date: str, force_refresh: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    windows = build_monthly_windows(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        raw_news_dir=config.raw_dir / "news" / "alpha_vantage",
    )
    failed_windows: list[str] = []
    for window in windows:
        try:
            fetch_alpha_vantage_news_cache(
                [window],
                api_key=config.alpha_vantage_api_key,
                cache_enabled=config.cache_enabled,
                force=force_refresh,
                limit=1000,
                requests_per_minute=config.alpha_vantage_requests_per_minute,
            )
        except Exception as exc:  # noqa: BLE001
            failed_windows.append(f"{window.ticker}:{window.month_label}")
            LOGGER.warning("Skipping Alpha Vantage window %s %s after error: %s", window.ticker, window.month_label, exc)
    if failed_windows:
        LOGGER.warning("Alpha Vantage windows skipped during forward refresh: %s", ", ".join(failed_windows))
    temp_alpha_path = config.processed_dir / "stock_news_alpha_vantage_2026_forward.csv"
    temp_news_path = config.processed_dir / "stock_news_2026_forward.csv"
    new_news = normalize_alpha_vantage_news_cache(
        windows,
        processed_output_path=temp_alpha_path,
        combined_output_path=temp_news_path,
    )
    existing_news_path = config.processed_dir / "stock_news.csv"
    existing_news = load_dataframe(existing_news_path, parse_dates=["published_date", "date"]) if existing_news_path.exists() else pd.DataFrame(columns=new_news.columns)
    merged_news = _merge_dedup(
        existing_news,
        new_news,
        keys=["ticker", "published_date", "url"],
        sort_cols=["ticker", "published_date", "url"],
    )
    save_dataframe(existing_news_path, merged_news)

    articles_df, daily_df = build_news_sentiment_outputs(
        news_input_path=existing_news_path,
        articles_output_path=config.processed_dir / "news_sentiment_articles.csv",
        daily_output_path=config.processed_dir / "news_sentiment_daily.csv",
        force=True,
    )
    return merged_news, daily_df


def _update_historical_grades(config: Config, tickers: list[str], end_date: str, force_refresh: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    existing_counts_path = config.processed_dir / "historical_analyst_rating_counts.csv"
    existing_events_path = config.processed_dir / "historical_analyst_grade_events.csv"
    stale_counts = True
    stale_events = True
    if existing_counts_path.exists():
        existing_counts = load_dataframe(existing_counts_path, parse_dates=["date"])
        if not existing_counts.empty:
            stale_counts = pd.Timestamp(existing_counts["date"].max()) < pd.Timestamp(end_date)
    if existing_events_path.exists():
        existing_events = load_dataframe(existing_events_path, parse_dates=["date"])
        if not existing_events.empty:
            stale_events = pd.Timestamp(existing_events["date"].max()) < pd.Timestamp(end_date)
    should_refresh = force_refresh or stale_counts or stale_events
    return build_historical_grade_datasets(
        tickers=tickers,
        api_key=config.fmp_api_key,
        raw_output_dir=config.raw_dir / "analyst" / "fmp_historical_grades",
        rating_counts_output_path=existing_counts_path,
        grade_events_output_path=existing_events_path,
        start_date=GRADES_FULL_HISTORY_START,
        end_date=(pd.Timestamp(end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        calls_per_minute=config.fmp_calls_per_minute,
        force=should_refresh,
        cache_enabled=config.cache_enabled,
        limit=1000,
    )


def _build_forward_feature_panel(config: Config, end_date_exclusive: str) -> pd.DataFrame:
    return build_feature_panel(
        prices_path=config.processed_dir / "prices_all.csv",
        universe_path=config.universe_path,
        analyst_path=config.processed_dir / "analyst_features.csv",
        sentiment_path=config.processed_dir / "news_sentiment_daily.csv",
        historical_rating_counts_path=config.processed_dir / "historical_analyst_rating_counts.csv",
        historical_grade_events_path=config.processed_dir / "historical_analyst_grade_events.csv",
        historical_rating_count_features_output_path=config.processed_dir / "historical_rating_count_features.csv",
        historical_grade_features_output_path=config.processed_dir / "historical_grade_features.csv",
        output_path=config.final_dir / "features_panel_2026_forward.csv",
        start_date=FORWARD_START.strftime("%Y-%m-%d"),
        end_date=end_date_exclusive,
        benchmark_ticker=config.benchmark,
        use_current_snapshot_analyst=False,
    )


def _build_direct_spy_series(features_forward: pd.DataFrame, benchmark: str, decision_dates: pd.Series, initial_capital: float) -> tuple[pd.DataFrame, float, float, float, float]:
    spy_daily = (
        features_forward.loc[features_forward["ticker"] == benchmark, ["date", "adjusted_close"]]
        .dropna(subset=["adjusted_close"])
        .drop_duplicates(subset=["date"])
        .sort_values("date")
        .rename(columns={"adjusted_close": "spy_adjusted_close"})
        .reset_index(drop=True)
    )
    if spy_daily.empty:
        raise ValueError("No SPY adjusted_close data found in the forward feature panel.")
    forward_returns = pd.DataFrame({"date": pd.to_datetime(decision_dates).sort_values().reset_index(drop=True)})
    forward_returns = pd.merge_asof(forward_returns, spy_daily, on="date", direction="backward")
    if forward_returns["spy_adjusted_close"].isna().any():
        raise ValueError("Unable to align SPY adjusted_close values to forward model dates.")
    start_close = float(forward_returns["spy_adjusted_close"].iloc[0])
    end_close = float(forward_returns["spy_adjusted_close"].iloc[-1])
    direct_return = end_close / start_close - 1.0
    forward_returns["spy_value_direct"] = initial_capital * forward_returns["spy_adjusted_close"] / start_close
    return forward_returns, start_close, end_close, direct_return, float(forward_returns["spy_value_direct"].iloc[-1])


def _update_paper_trading_state(config: Config, latest_date: pd.Timestamp, recommendations_df: pd.DataFrame, simulated_model_value: float, simulated_spy_value: float) -> Path:
    paper_dir = config.data_dir / "paper_trading"
    paper_dir.mkdir(parents=True, exist_ok=True)
    performance_path = paper_dir / "performance_history.csv"
    initial_live_value = config.initial_capital
    note = "initialized_from_forward_test_no_prior_live_tracking"
    row = pd.DataFrame(
        [
            {
                "date": latest_date,
                "source": "initialized_from_forward_test",
                "live_model_value": initial_live_value,
                "live_spy_value": initial_live_value,
                "simulated_model_value": simulated_model_value,
                "simulated_spy_value": simulated_spy_value,
                "holdings_count": len(recommendations_df),
                "note": note,
            }
        ]
    )
    if performance_path.exists():
        existing = load_dataframe(performance_path, parse_dates=["date"])
        merged = _merge_dedup(existing, row, keys=["date", "source"], sort_cols=["date", "source"])
    else:
        merged = row
    save_dataframe(performance_path, merged)
    return performance_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-refresh", action="store_true")
    args = parser.parse_args()

    config = Config.from_env()
    recommended = _validate_recommended_strategy(config)
    config_text_before = config_path(config.project_root).read_text(encoding="utf-8")
    config_copy_path = config.reports_dir / "recommended_strategy_used_2026_forward.yaml"
    config_copy_path.write_text(config_text_before, encoding="utf-8")

    if recommended.strategy_name != EXPECTED_STRATEGY:
        raise ValueError("Forward test is locked to the promoted low-turnover model.")

    tickers = get_tickers(config.universe_path)
    if config.benchmark not in tickers:
        tickers = [*tickers, config.benchmark]

    requested_end = pd.Timestamp.today().normalize().strftime("%Y-%m-%d")
    prices = _update_prices(config, tickers, FORWARD_START.strftime("%Y-%m-%d"), requested_end, args.force_refresh)
    latest_available_price_date = pd.Timestamp(prices.loc[prices["date"] >= FORWARD_START, "date"].max())
    if pd.isna(latest_available_price_date):
        raise ValueError("No 2026+ prices are available after refresh.")
    actual_end_exclusive = (latest_available_price_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    _update_news_and_sentiment(config, tickers=[ticker for ticker in tickers if ticker != config.benchmark], start_date=FORWARD_START.strftime("%Y-%m-%d"), end_date=actual_end_exclusive, force_refresh=args.force_refresh)
    _update_historical_grades(config, tickers=[ticker for ticker in tickers if ticker != config.benchmark], end_date=latest_available_price_date.strftime("%Y-%m-%d"), force_refresh=args.force_refresh)

    features_forward = _build_forward_feature_panel(config, end_date_exclusive=actual_end_exclusive)
    if features_forward.empty:
        raise ValueError("Forward 2026 feature panel is empty.")

    first_trading_date = pd.Timestamp(features_forward["date"].min())
    panels = precompute_recommended_low_turnover_panels(features_forward, config, recommended)
    weekly, holdings, actions = run_low_turnover_recommended_backtest(
        panels=panels,
        top_n=recommended.top_n,
        cost_bps=float(recommended.total_cost_bps),
        enter_rank=recommended.enter_rank or recommended.top_n,
        hold_rank=recommended.hold_rank or max(recommended.top_n, 20),
        max_holding_days=recommended.max_holding_days or 20,
        rebalance_frequency_days=recommended.rebalance_frequency_days or recommended.holding_period_days,
        strategy_name=recommended.strategy_name,
        max_turnover_per_rebalance=recommended.max_turnover_per_rebalance,
    )
    if weekly.empty:
        raise ValueError("Forward 2026 low-turnover backtest returned no rows.")

    spy_plot_df, spy_start_close, spy_end_close, direct_spy_return, direct_spy_final_value = _build_direct_spy_series(
        features_forward,
        config.benchmark,
        weekly["date"],
        config.initial_capital,
    )

    returns_df = weekly.loc[:, ["date", "net_return", "portfolio_value", "spy_return", "turnover", "transaction_cost", "selected_count", "exposure"]].copy()
    returns_df = returns_df.rename(
        columns={
            "net_return": "model_period_return",
            "portfolio_value": "model_value",
            "transaction_cost": "trading_cost",
            "spy_return": "plotted_spy_period_return",
        }
    ).sort_values("date")
    returns_df = returns_df.merge(spy_plot_df, on="date", how="left")
    returns_df["model_drawdown"] = _compute_drawdown(returns_df["model_value"])
    returns_df["spy_drawdown"] = _compute_drawdown(returns_df["spy_value_direct"])
    returns_df = returns_df[
        [
            "date",
            "model_period_return",
            "model_value",
            "spy_adjusted_close",
            "spy_value_direct",
            "plotted_spy_period_return",
            "model_drawdown",
            "spy_drawdown",
            "turnover",
            "trading_cost",
            "selected_count",
            "exposure",
        ]
    ]
    save_dataframe(config.tables_dir / "forward_2026_model_vs_spy_returns.csv", returns_df)

    plotted_spy_return = float(returns_df["spy_value_direct"].iloc[-1] / config.initial_capital - 1.0)
    compounded_spy_return = float((1.0 + weekly["spy_return"]).cumprod().iloc[-1] - 1.0)
    benchmark_validation = pd.DataFrame(
        [
            {
                "start_date": pd.Timestamp(returns_df["date"].iloc[0]),
                "end_date": pd.Timestamp(returns_df["date"].iloc[-1]),
                "direct_spy_return": direct_spy_return,
                "plotted_spy_return": plotted_spy_return,
                "absolute_difference": abs(plotted_spy_return - direct_spy_return),
                "direct_spy_final_value": direct_spy_final_value,
                "plotted_spy_final_value": float(returns_df["spy_value_direct"].iloc[-1]),
            }
        ]
    )
    save_dataframe(config.tables_dir / "forward_2026_benchmark_validation.csv", benchmark_validation)
    benchmark_diff = float(benchmark_validation["absolute_difference"].iloc[0])
    if benchmark_diff > 0.005:
        raise ValueError("Direct SPY buy-and-hold and plotted SPY benchmark differ by more than 0.5%.")

    _make_equity_curve_plot(
        returns_df.rename(columns={"spy_value_direct": "spy_value_direct"}),
        config.charts_dir / "forward_2026_model_vs_spy_equity_curve.png",
    )
    _make_drawdown_plot(returns_df, config.charts_dir / "forward_2026_model_vs_spy_drawdown.png")

    metrics = calculate_performance_metrics(
        weekly.assign(spy_value=returns_df["spy_value_direct"].to_numpy(), excess_return=weekly["net_return"] - weekly["spy_return"]),
        holding_period_days=recommended.holding_period_days,
    )
    summary = summarize_backtest(weekly, recommended.holding_period_days, recommended.strategy_name)
    walk_forward_average_excess = float(
        pd.Series(
            [
                summary.get("2024_h1_excess_return_vs_spy", float("nan")),
                summary.get("2024_h2_excess_return_vs_spy", float("nan")),
                summary.get("2025_excess_return_vs_spy", float("nan")),
            ]
        ).mean()
    )

    latest_decision_date = pd.Timestamp(holdings["date"].max())
    latest_holdings = holdings.loc[holdings["date"] == latest_decision_date].copy()
    latest_actions = actions.loc[actions["date"] == latest_decision_date].copy() if not actions.empty else pd.DataFrame(columns=["date", "ticker", "action", "reason"])
    latest_panel_candidates = [panel for panel_date, panel, _, _ in panels if pd.Timestamp(panel_date) == latest_decision_date]
    if not latest_panel_candidates:
        raise ValueError("Could not locate latest panel for forward current recommendations.")
    latest_panel = latest_panel_candidates[-1].copy()
    latest_panel["top_signal_reasons"] = latest_panel.apply(top_signal_reasons, axis=1)
    latest_holdings = latest_holdings.merge(
        latest_panel[
            [
                column
                for column in [
                    "ticker",
                    "sector",
                    "historical_rating_score",
                    "net_upgrade_score_30d",
                    "downgrade_count_30d",
                    "relevance_weighted_sentiment_7d",
                    "negative_news_ratio_7d",
                    "relative_strength_21d",
                    "volatility_21d",
                    "top_signal_reasons",
                ]
                if column in latest_panel.columns
            ]
        ],
        on="ticker",
        how="left",
    )
    latest_holdings = latest_holdings.rename(columns={"weight": "target_weight"})
    latest_holdings["reason_for_action"] = latest_holdings["reason"]
    current_cols = [
        "date",
        "ticker",
        "action",
        "reason_for_action",
        "rank",
        "score",
        "target_weight",
        "historical_rating_score",
        "net_upgrade_score_30d",
        "downgrade_count_30d",
        "relevance_weighted_sentiment_7d",
        "negative_news_ratio_7d",
        "relative_strength_21d",
        "volatility_21d",
        "top_signal_reasons",
    ]
    current_forward_df = latest_holdings[[column for column in current_cols if column in latest_holdings.columns]].copy()
    save_dataframe(config.tables_dir / "current_recommendations_2026_forward.csv", current_forward_df)

    current_report_lines = [
        "# Current Recommendations 2026 Forward",
        "",
        "- This is a forward/out-of-sample test using the frozen model configuration.",
        "- No 2026 data was used to tune the model.",
        "- Back-tested performance is hypothetical unless trades were actually paper-tracked live.",
        "- Snapshot analyst target fields are excluded.",
        "- Historical rating-count features use dated FMP grades-historical records available on or before each rebalance date.",
        "- News sentiment depends on Alpha Vantage coverage and classification.",
        "- This is research/paper trading only, not financial advice.",
        "",
        f"- Latest feature date: {pd.Timestamp(features_forward['date'].max()).date()}",
        f"- Latest decision date: {latest_decision_date.date()}",
        "",
        dataframe_to_markdown(current_forward_df.round(4)),
    ]
    (config.reports_dir / "current_recommendations_2026_forward.md").write_text("\n".join(current_report_lines), encoding="utf-8")

    performance_path = _update_paper_trading_state(
        config,
        latest_date=latest_decision_date,
        recommendations_df=current_forward_df,
        simulated_model_value=float(returns_df["model_value"].iloc[-1]),
        simulated_spy_value=float(returns_df["spy_value_direct"].iloc[-1]),
    )

    total_estimated_trading_cost = float(returns_df["trading_cost"].sum())
    spy_max_drawdown = float(returns_df["spy_drawdown"].min())
    report_lines = [
        "# Forward 2026 Model vs SPY Summary",
        "",
        "- This is a forward/out-of-sample test using the frozen model configuration.",
        "- No 2026 data was used to tune the model.",
        "- Back-tested performance is hypothetical unless trades were actually paper-tracked live.",
        "- Snapshot analyst target fields are excluded.",
        "- Historical rating-count features use dated FMP grades-historical records available on or before each rebalance date.",
        "- News sentiment depends on Alpha Vantage coverage and classification.",
        "- This is research/paper trading only, not financial advice.",
        "",
        f"- Strategy name: `{recommended.strategy_name}`",
        f"- Display name: {strategy_display_name(recommended.strategy_name)}",
        f"- Base score model: `{BASE_SCORE_MODEL}`",
        f"- Execution mode: `{EXECUTION_MODE}`",
        f"- Forward test start date requested: {FORWARD_START.date()}",
        f"- Forward test start date used: {first_trading_date.date()}",
        f"- Forward test end date: {pd.Timestamp(returns_df['date'].max()).date()}",
        f"- Feature panel latest date: {pd.Timestamp(features_forward['date'].max()).date()}",
        f"- Enter rank: {recommended.enter_rank}",
        f"- Hold rank: {recommended.hold_rank}",
        f"- Top N: {recommended.top_n}",
        f"- Max holding days: {recommended.max_holding_days}",
        f"- Rebalance frequency: {recommended.rebalance_frequency_days} trading days",
        f"- Cost assumption: {recommended.total_cost_bps:.0f} bps total",
        f"- Position sizing: {recommended.position_sizing}",
        f"- Allow cash: {str(recommended.allow_cash).lower()}",
        f"- Long/short: {str(recommended.long_short).lower()}",
        f"- Regime filter: {recommended.regime_filter}",
        "",
        "## Metrics",
        "",
        f"- Number of rebalance periods: {int(metrics['number_of_rebalance_periods'])}",
        f"- Model total return: {metrics['total_return']:.2%}",
        f"- SPY total return: {direct_spy_return:.2%}",
        f"- Excess return vs SPY: {metrics['total_return'] - direct_spy_return:.2%}",
        f"- Annualized return: {metrics['annualized_return']:.2%}",
        f"- Annualized volatility: {metrics['annualized_volatility']:.2%}",
        f"- Sharpe: {metrics['sharpe_ratio']:.3f}",
        f"- Max drawdown: {metrics['max_drawdown']:.2%}",
        f"- SPY max drawdown: {spy_max_drawdown:.2%}",
        f"- Average turnover: {metrics['average_turnover']:.6f}",
        f"- Average holdings: {metrics['average_selected_count']:.2f}",
        f"- Estimated trading costs: {total_estimated_trading_cost:.4f}",
        f"- Percent periods invested: {metrics['average_percent_invested']:.2%}",
        "",
        "## Benchmark Validation",
        "",
        f"- SPY start adjusted close: {spy_start_close:.4f}",
        f"- SPY end adjusted close: {spy_end_close:.4f}",
        f"- Direct SPY buy-and-hold return: {direct_spy_return:.2%}",
        f"- Plotted SPY final value: ${direct_spy_final_value:,.2f}",
        f"- Benchmark validation difference: {benchmark_diff:.4%}",
        "",
        "## Latest Actions",
        "",
        f"- Latest buys: {', '.join(sorted(latest_actions.loc[latest_actions['action'] == 'BUY', 'ticker'].tolist())) or 'none'}",
        f"- Latest sells: {', '.join(sorted(latest_actions.loc[latest_actions['action'] == 'SELL', 'ticker'].tolist())) or 'none'}",
        f"- Latest holds: {', '.join(sorted(latest_holdings.loc[latest_holdings['action'] == 'HOLD', 'ticker'].tolist())) or 'none'}",
        "",
        "## Current Holdings",
        "",
        dataframe_to_markdown(current_forward_df.round(4)),
        "",
        "## Outputs",
        "",
        "- Feature panel: `data/final/features_panel_2026_forward.csv`",
        "- Forward returns table: `outputs/tables/forward_2026_model_vs_spy_returns.csv`",
        "- Forward benchmark validation: `outputs/tables/forward_2026_benchmark_validation.csv`",
        "- Current recommendations table: `outputs/tables/current_recommendations_2026_forward.csv`",
        "- Equity curve: `outputs/charts/forward_2026_model_vs_spy_equity_curve.png`",
        "- Drawdown chart: `outputs/charts/forward_2026_model_vs_spy_drawdown.png`",
        f"- Frozen config copy: `{config_copy_path.relative_to(config.project_root)}`",
        f"- Paper trading performance history: `{performance_path.relative_to(config.project_root)}`",
    ]
    (config.reports_dir / "forward_2026_model_vs_spy_summary.md").write_text("\n".join(report_lines), encoding="utf-8")

    config_text_after = config_path(config.project_root).read_text(encoding="utf-8")
    if config_text_after != config_text_before:
        raise ValueError("recommended_strategy.yaml changed during the forward test run.")

    print(f"Saved {config.final_dir / 'features_panel_2026_forward.csv'}")
    print(f"Saved {config.tables_dir / 'forward_2026_model_vs_spy_returns.csv'}")
    print(f"Saved {config.tables_dir / 'forward_2026_benchmark_validation.csv'}")
    print(f"Saved {config.tables_dir / 'current_recommendations_2026_forward.csv'}")
    print(f"Saved {config.charts_dir / 'forward_2026_model_vs_spy_equity_curve.png'}")
    print(f"Saved {config.charts_dir / 'forward_2026_model_vs_spy_drawdown.png'}")
    print(f"Saved {config.reports_dir / 'forward_2026_model_vs_spy_summary.md'}")
    print(f"Saved {config.reports_dir / 'current_recommendations_2026_forward.md'}")
    print(f"Saved {config_copy_path}")


if __name__ == "__main__":
    main()
