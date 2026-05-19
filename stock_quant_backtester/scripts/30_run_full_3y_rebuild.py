from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.backtest import _build_validation_row, run_weekly_backtest, select_rebalance_dates
from src.build_features import build_feature_panel
from src.build_news_sentiment import build_news_sentiment_outputs
from src.config import Config
from src.fetch_alpha_vantage_news import (
    build_monthly_windows,
    fetch_alpha_vantage_news_cache,
    normalize_alpha_vantage_news_cache,
    summarize_request_plan,
)
from src.fetch_analyst_data import build_analyst_snapshot
from src.fetch_fmp_historical_grades import build_historical_grade_datasets
from src.fetch_prices import fetch_and_save_prices
from src.metrics import calculate_performance_metrics
from src.scoring import (
    strategy_analyst_data_mode,
    strategy_display_name,
    strategy_historical_validity_group,
    strategy_uses_historical_grade_events,
    strategy_uses_historical_ratings,
    strategy_uses_sentiment,
    strategy_uses_snapshot_fields,
)
from src.universe import get_tickers
from src.utils import load_dataframe, save_dataframe


SNAPSHOT_ANALYST_CAVEAT = (
    "Important caveat: analyst-driven snapshot results use FMP data as a current snapshot merged across historical dates unless true point-in-time analyst target history is provided. These results should be treated as research exploration, not a valid historical analyst-signal backtest."
)
BACKTEST_PERFORMANCE_CAVEAT = "Back-tested performance is hypothetical and does not reflect actual live performance."
HISTORICAL_RATING_NOTE = (
    "Historical rating-count features are built from dated FMP grades-historical records and use only the latest record available on or before each rebalance date."
)
SENTIMENT_CAVEAT = (
    "News sentiment results depend on Alpha Vantage coverage, ticker relevance scoring, publication timestamps, and provider sentiment classification."
)
BACKTEST_CAVEAT = "This is a historical research backtest, not financial advice and not a live trading system."
LONG_SHORT_CAVEAT = "Long/short is experimental and currently not recommended."
DEV_END = pd.Timestamp("2024-12-31")
TEST_START = pd.Timestamp("2025-01-01")
VALIDATION_TOLERANCE = 0.02
HISTORICALLY_SAFER_MODELS = [
    "technical_only",
    "technical_momentum_model",
    "sentiment_only",
    "technical_sentiment_model",
    "historical_rating_counts_model",
    "historical_rating_counts_plus_sentiment",
    "historical_rating_counts_plus_events",
    "historical_rating_counts_plus_events_sentiment",
    "final_quant_model_1y_no_snapshot",
]
SNAPSHOT_EXPLORATORY_MODELS = [
    "analyst_snapshot_model",
    "full_model",
    "final_quant_model_1y",
]
FULL_BACKTEST_STRATEGIES = [
    "SPY",
    "technical_only",
    "technical_momentum_model",
    "sentiment_only",
    "technical_sentiment_model",
    "historical_rating_counts_model",
    "historical_rating_counts_plus_sentiment",
    "historical_rating_counts_plus_events",
    "historical_rating_counts_plus_events_sentiment",
    "final_quant_model_1y_no_snapshot",
    "analyst_snapshot_model",
    "full_model",
    "final_quant_model_1y",
]


def _dataframe_to_markdown(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    widths = [max(len(str(header)), *(len(str(value)) for value in df[header].tolist())) for header in headers]
    header_line = "| " + " | ".join(str(header).ljust(widths[idx]) for idx, header in enumerate(headers)) + " |"
    separator = "| " + " | ".join("-" * widths[idx] for idx in range(len(headers))) + " |"
    rows = []
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(str(row[header]).ljust(widths[idx]) for idx, header in enumerate(headers)) + " |")
    return "\n".join([header_line, separator, *rows])


def _slice_period(df: pd.DataFrame, start: pd.Timestamp | None = None, end: pd.Timestamp | None = None) -> pd.DataFrame:
    out = df.copy()
    if start is not None:
        out = out.loc[out["date"] >= start]
    if end is not None:
        out = out.loc[out["date"] <= end]
    return out


def _safe_metrics(frame: pd.DataFrame, holding_period_days: int) -> dict[str, float]:
    if frame.empty:
        return {
            "total_return": float("nan"),
            "excess_total_return": float("nan"),
            "annualized_return": float("nan"),
            "annualized_volatility": float("nan"),
            "sharpe_ratio": float("nan"),
            "max_drawdown": float("nan"),
            "average_selected_count": float("nan"),
            "average_turnover": float("nan"),
            "number_of_rebalance_periods": 0,
            "spy_total_return": float("nan"),
            "win_rate": float("nan"),
            "weeks_beating_spy": float("nan"),
        }
    return calculate_performance_metrics(frame, holding_period_days=holding_period_days)


def _comparison_row(strategy_name: str, weekly: pd.DataFrame, *, holding_period_days: int, top_n: int) -> dict:
    full = _safe_metrics(weekly, holding_period_days)
    dev = _safe_metrics(_slice_period(weekly, end=DEV_END), holding_period_days)
    test = _safe_metrics(_slice_period(weekly, start=TEST_START), holding_period_days)
    return {
        "strategy_name": strategy_name,
        "display_name": strategy_display_name(strategy_name),
        "historical_validity_group": strategy_historical_validity_group(strategy_name),
        "analyst_data_mode": strategy_analyst_data_mode(strategy_name),
        "uses_snapshot_fields": strategy_uses_snapshot_fields(strategy_name),
        "uses_sentiment": strategy_uses_sentiment(strategy_name),
        "uses_historical_ratings": strategy_uses_historical_ratings(strategy_name),
        "uses_historical_grade_events": strategy_uses_historical_grade_events(strategy_name),
        "holding_period_days": holding_period_days,
        "top_n": top_n,
        "full_period_total_return": full["total_return"],
        "development_period_total_return": dev["total_return"],
        "test_period_total_return": test["total_return"],
        "full_period_excess_return_vs_spy": full["excess_total_return"],
        "test_period_excess_return_vs_spy": test["excess_total_return"],
        "annualized_return": full["annualized_return"],
        "annualized_volatility": full["annualized_volatility"],
        "sharpe_ratio": full["sharpe_ratio"],
        "test_sharpe_ratio": test["sharpe_ratio"],
        "max_drawdown": full["max_drawdown"],
        "average_holdings": full["average_selected_count"],
        "average_turnover": full["average_turnover"],
        "number_of_rebalance_periods": full["number_of_rebalance_periods"],
    }


def _build_spy_weekly(features: pd.DataFrame, benchmark: str, holding_period_days: int, initial_capital: float) -> pd.DataFrame:
    future_map = {
        5: "future_5d_spy_return",
        21: "future_21d_spy_return",
        63: "future_63d_spy_return",
    }
    future_spy_return_column = future_map[holding_period_days]
    spy_value = initial_capital
    rows = []
    benchmark_dates = select_rebalance_dates(features, holding_period_days=holding_period_days, benchmark=benchmark)
    for date in benchmark_dates:
        spy_return = float(
            features.loc[(features["ticker"] == benchmark) & (features["date"] == date), future_spy_return_column].iloc[0]
        )
        spy_value *= 1 + spy_return
        rows.append(
            {
                "date": pd.to_datetime(date),
                "strategy_name": "SPY",
                "holding_period_days": holding_period_days,
                "selected_count": 1,
                "qualified_count": 1,
                "gross_return": spy_return,
                "turnover": 0.0,
                "transaction_cost": 0.0,
                "net_return": spy_return,
                "spy_return": spy_return,
                "excess_return": 0.0,
                "portfolio_value": spy_value,
                "spy_value": spy_value,
                "exposure": 1.0,
                "regime_allowed": True,
            }
        )
    return pd.DataFrame(rows)


def _estimate_price_calls(
    tickers: list[str],
    benchmark: str,
    raw_prices_dir: Path,
    start_date: str,
    end_date: str,
    force_refresh: bool,
) -> dict[str, int]:
    expected_tickers = tickers if benchmark in tickers else tickers + [benchmark]
    cached = 0
    suffix = f"{start_date}_{end_date}.csv"
    for ticker in expected_tickers:
        cache_path = raw_prices_dir / f"{ticker}_{suffix}"
        legacy_path = raw_prices_dir.parent / f"{ticker}.csv"
        if not force_refresh and (cache_path.exists() or legacy_path.exists()):
            cached += 1
    total = len(expected_tickers)
    return {"total": total, "cached": cached, "missing": total - cached}


def _estimate_fmp_snapshot_calls(tickers: list[str], raw_output_dir: Path, force_refresh: bool) -> dict[str, int]:
    endpoint_suffixes = [
        "_price_target_consensus.json",
        "_price_target_summary.json",
        "_ratings_snapshot.json",
        "_historical_ratings.json",
    ]
    cached = 0
    total = len(tickers) * len(endpoint_suffixes)
    for ticker in tickers:
        for suffix in endpoint_suffixes:
            if not force_refresh and (raw_output_dir / f"{ticker}{suffix}").exists():
                cached += 1
    return {"total": total, "cached": cached, "missing": total - cached}


def _estimate_historical_grade_calls(tickers: list[str], raw_output_dir: Path, force_refresh: bool) -> dict[str, int]:
    endpoint_suffixes = ["_grades_historical.json", "_grades_events.json"]
    cached = 0
    total = len(tickers) * len(endpoint_suffixes)
    for ticker in tickers:
        for suffix in endpoint_suffixes:
            if not force_refresh and (raw_output_dir / f"{ticker}{suffix}").exists():
                cached += 1
    return {"total": total, "cached": cached, "missing": total - cached}


def _clear_cache_dirs(config: Config) -> None:
    for path in [config.raw_dir, config.processed_dir, config.final_dir]:
        if path.exists():
            shutil.rmtree(path)
    config.ensure_directories()


def _run_subprocess(cmd: list[str], project_root: Path, dry_run: bool, capture_output: bool = False) -> subprocess.CompletedProcess[str] | None:
    print(f"$ {' '.join(cmd)}")
    if dry_run:
        return None
    return subprocess.run(
        cmd,
        cwd=project_root,
        text=True,
        capture_output=capture_output,
        check=False,
    )


def _copy_if_exists(source: Path, target: Path) -> None:
    if source.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _best_rows(comparison_df: pd.DataFrame) -> pd.DataFrame:
    if comparison_df.empty:
        return comparison_df
    ranked = comparison_df.sort_values(
        ["test_period_excess_return_vs_spy", "test_sharpe_ratio", "max_drawdown"],
        ascending=[False, False, False],
    )
    return ranked.groupby("strategy_name", as_index=False).head(1).reset_index(drop=True)


def _resolve_window_file(base_path: Path, start_date: str, end_date: str) -> Path:
    dated = base_path.with_name(f"{base_path.stem}_{start_date}_{end_date}{base_path.suffix}")
    return dated if dated.exists() else base_path


def _validate_feature_panel_window(
    features: pd.DataFrame,
    *,
    start_date: str,
    end_date: str,
    sentiment_start_date: str,
    sentiment_end_date: str,
    allow_partial_sentiment: bool,
) -> None:
    if features.empty:
        raise SystemExit("Feature panel is empty for the requested full rebuild window.")

    min_date = pd.Timestamp(features["date"].min()).normalize()
    max_date = pd.Timestamp(features["date"].max()).normalize()
    expected_start = pd.Timestamp(start_date)
    expected_end_last = pd.Timestamp(end_date) - pd.Timedelta(days=1)

    print(f"Feature panel window: {min_date.date()} to {max_date.date()}")
    print(f"Development period: 2023-01-01 to 2024-12-31")
    print(f"Test period: 2025-01-01 to 2025-12-31")

    allowed_start_slippage = expected_start + pd.Timedelta(days=7)
    if min_date > allowed_start_slippage:
        raise SystemExit(
            f"Feature panel starts at {min_date.date()} during a full rebuild; expected start no later than {allowed_start_slippage.date()}."
        )
    if max_date < expected_end_last:
        raise SystemExit(
            f"Feature panel ends at {max_date.date()} during a full rebuild; expected through at least {expected_end_last.date()}."
        )

    sentiment_mask = features["article_count_30d"].fillna(0).gt(0) if "article_count_30d" in features.columns else pd.Series(False, index=features.index)
    sentiment_dates = features.loc[sentiment_mask, "date"]
    sentiment_min = pd.Timestamp(sentiment_dates.min()).normalize() if not sentiment_dates.empty else None
    sentiment_max = pd.Timestamp(sentiment_dates.max()).normalize() if not sentiment_dates.empty else None
    print(f"Sentiment window: {sentiment_min.date() if sentiment_min is not None else 'n/a'} to {sentiment_max.date() if sentiment_max is not None else 'n/a'}")

    expected_sentiment_start = pd.Timestamp(sentiment_start_date)
    expected_sentiment_end_last = pd.Timestamp(sentiment_end_date) - pd.Timedelta(days=1)
    allowed_sentiment_start_slippage = expected_sentiment_start + pd.Timedelta(days=7)
    if (
        sentiment_min is None
        or sentiment_max is None
        or sentiment_min > allowed_sentiment_start_slippage
        or sentiment_max < expected_sentiment_end_last
    ):
        message = (
            "Sentiment data does not cover the requested full sentiment window. "
            f"Expected {sentiment_start_date} to {sentiment_end_date}, got "
            f"{sentiment_min.date() if sentiment_min is not None else 'n/a'} to {sentiment_max.date() if sentiment_max is not None else 'n/a'}."
        )
        if allow_partial_sentiment:
            print(f"WARNING: {message}")
        else:
            raise SystemExit(message)


def _build_backtest_report(
    comparison_df: pd.DataFrame,
    validation_row: pd.Series,
    *,
    holding_period_days: int,
    date_label: str,
    output_path: Path,
) -> None:
    leaders = comparison_df.sort_values(
        ["test_period_excess_return_vs_spy", "test_sharpe_ratio", "max_drawdown"],
        ascending=[False, False, False],
    ).head(15)
    lines = [
        f"# Full 3Y Backtest h{holding_period_days}",
        "",
        f"- Window: {date_label}",
        f"- Holding period days: {holding_period_days}",
        f"- {HISTORICAL_RATING_NOTE}",
        f"- {SNAPSHOT_ANALYST_CAVEAT}",
        f"- {BACKTEST_PERFORMANCE_CAVEAT}",
        f"- {SENTIMENT_CAVEAT}",
        f"- {BACKTEST_CAVEAT}",
        "",
        "## Benchmark Validation",
        f"- Compounded SPY return from backtest: {validation_row['compounded_spy_return_from_backtest']:.4f}",
        f"- Direct SPY buy-and-hold return: {validation_row['direct_spy_buy_hold_return']:.4f}",
        f"- Absolute difference: {validation_row['absolute_difference']:.6f}",
        "",
        "## Test Period Leaders",
        "",
        _dataframe_to_markdown(leaders.round(6)),
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _build_sentiment_report(
    comparison_df: pd.DataFrame,
    diagnostics_df: pd.DataFrame,
    features: pd.DataFrame,
    benchmark: str,
    selected_holdings_sentiment_coverage: float,
    output_path: Path,
) -> None:
    best_sentiment = _best_rows(comparison_df.loc[comparison_df["strategy_name"].isin(["sentiment_only", "technical_sentiment_model", "historical_rating_counts_plus_sentiment", "historical_rating_counts_plus_events_sentiment", "final_quant_model_1y_no_snapshot"])])
    best_technical = _best_rows(comparison_df.loc[comparison_df["strategy_name"].isin(["technical_only", "technical_momentum_model", "historical_rating_counts_model", "historical_rating_counts_plus_events"])])
    sentiment_helped = False
    if not best_sentiment.empty and not best_technical.empty:
        sentiment_helped = any(
            [
                float(best_sentiment.iloc[0]["test_period_excess_return_vs_spy"]) > float(best_technical.iloc[0]["test_period_excess_return_vs_spy"]),
                float(best_sentiment.iloc[0]["test_sharpe_ratio"]) > float(best_technical.iloc[0]["test_sharpe_ratio"]),
                float(best_sentiment.iloc[0]["max_drawdown"]) > float(best_technical.iloc[0]["max_drawdown"]),
            ]
        )
    candidate_features = features.loc[features["ticker"] != benchmark].copy()
    lines = [
        "# Full 3Y Sentiment Coverage Report",
        "",
        f"- {SNAPSHOT_ANALYST_CAVEAT}",
        f"- {HISTORICAL_RATING_NOTE}",
        f"- {BACKTEST_PERFORMANCE_CAVEAT}",
        f"- {SENTIMENT_CAVEAT}",
        f"- {BACKTEST_CAVEAT}",
        "",
        "## Coverage",
        f"- Percent of universe with sentiment coverage by period: {diagnostics_df['coverage_pct_7d'].mean():.2%}",
        f"- Average article_count_7d: {diagnostics_df['avg_article_count_7d'].mean():.2f}",
        f"- Average article_count_30d: {candidate_features['article_count_30d'].mean():.2f}",
        f"- Percent of selected holdings with sentiment data: {selected_holdings_sentiment_coverage:.2%}",
        f"- Sentiment coverage is sparse: {diagnostics_df['coverage_pct_7d'].mean() < 0.15}",
        f"- Sentiment improves returns, Sharpe, or drawdown: {sentiment_helped}",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _build_historical_coverage_outputs(
    features: pd.DataFrame,
    benchmark: str,
    diagnostics_output_path: Path,
    report_output_path: Path,
) -> pd.DataFrame:
    candidates = features.loc[features["ticker"] != benchmark].copy()
    rows = []
    for date, day in candidates.groupby("date"):
        rows.append(
            {
                "date": pd.to_datetime(date),
                "percent_with_historical_rating_counts": float(day["historical_rating_count_data_available"].fillna(False).mean()),
                "percent_with_total_ratings_ge_1": float((day["historical_total_ratings"].fillna(0) >= 1).mean()),
                "percent_with_total_ratings_ge_5": float((day["historical_total_ratings"].fillna(0) >= 5).mean()),
                "percent_with_total_ratings_ge_10": float((day["historical_total_ratings"].fillna(0) >= 10).mean()),
                "avg_days_since_historical_rating_update": float(day["days_since_historical_rating_update"].mean()),
            }
        )
    diagnostics_df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    by_ticker = (
        candidates.groupby("ticker")
        .agg(
            percent_with_data=("historical_rating_count_data_available", lambda s: float(s.fillna(False).mean())),
            avg_total_ratings=("historical_total_ratings", "mean"),
            avg_days_since_update=("days_since_historical_rating_update", "mean"),
        )
        .reset_index()
        .sort_values(["percent_with_data", "avg_total_ratings"], ascending=[False, False])
    )
    save_dataframe(diagnostics_output_path, diagnostics_df)

    lines = [
        "# Full 3Y Historical Analyst Coverage Report",
        "",
        f"- {SNAPSHOT_ANALYST_CAVEAT}",
        f"- {HISTORICAL_RATING_NOTE}",
        f"- {BACKTEST_PERFORMANCE_CAVEAT}",
        f"- {SENTIMENT_CAVEAT}",
        f"- {BACKTEST_CAVEAT}",
        "",
        "## Coverage Summary",
        f"- Percent of universe with historical rating-count data: {diagnostics_df['percent_with_historical_rating_counts'].mean():.2%}",
        f"- Percent with total ratings >= 1: {diagnostics_df['percent_with_total_ratings_ge_1'].mean():.2%}",
        f"- Percent with total ratings >= 5: {diagnostics_df['percent_with_total_ratings_ge_5'].mean():.2%}",
        f"- Percent with total ratings >= 10: {diagnostics_df['percent_with_total_ratings_ge_10'].mean():.2%}",
        f"- Average days since last rating update: {diagnostics_df['avg_days_since_historical_rating_update'].mean():.2f}",
        f"- Historical analyst coverage is sufficient for backtesting: {diagnostics_df['percent_with_total_ratings_ge_5'].mean() >= 0.20}",
        "",
        "## Coverage By Ticker",
        "",
        _dataframe_to_markdown(by_ticker.head(25).round(4)),
    ]
    report_output_path.write_text("\n".join(lines), encoding="utf-8")
    return diagnostics_df


def _build_final_family_report(
    comparison_df: pd.DataFrame,
    output_csv: Path,
    output_report: Path,
    *,
    include_snapshot_models: bool,
    start_date: str,
    end_date: str,
    sentiment_start_date: str,
    sentiment_end_date: str,
) -> pd.DataFrame:
    best_rows = _best_rows(comparison_df)
    save_dataframe(output_csv, best_rows)
    safer = best_rows.loc[best_rows["strategy_name"].isin(HISTORICALLY_SAFER_MODELS)]
    snapshot = best_rows.loc[best_rows["strategy_name"].isin(SNAPSHOT_EXPLORATORY_MODELS)] if include_snapshot_models else pd.DataFrame()
    best_overall = best_rows.sort_values(
        ["test_period_excess_return_vs_spy", "test_sharpe_ratio", "max_drawdown"],
        ascending=[False, False, False],
    ).iloc[0]
    best_sharpe = best_rows.sort_values("test_sharpe_ratio", ascending=False).iloc[0]
    best_drawdown = best_rows.sort_values("max_drawdown", ascending=False).iloc[0]
    lines = [
        "# Full 3Y Final Model Report",
        "",
        f"- Backtest window: {start_date} to {end_date}",
        f"- Sentiment window: {sentiment_start_date} to {sentiment_end_date}",
        f"- Feature panel window: {start_date} to {end_date}",
        f"- Development period: 2023-01-01 to 2024-12-31",
        f"- Test period: 2025-01-01 to 2025-12-31",
        f"- {SNAPSHOT_ANALYST_CAVEAT}",
        f"- {HISTORICAL_RATING_NOTE}",
        f"- {BACKTEST_PERFORMANCE_CAVEAT}",
        f"- {SENTIMENT_CAVEAT}",
        f"- {BACKTEST_CAVEAT}",
        f"- {LONG_SHORT_CAVEAT}",
        "",
        "## Winners",
        f"- Best historically safer model: {safer.iloc[0]['display_name'] if not safer.empty else 'n/a'}",
        f"- Best snapshot/exploratory model: {snapshot.iloc[0]['display_name'] if not snapshot.empty else 'n/a'}",
        f"- Best overall model: {best_overall['display_name']}",
        f"- Best model by Sharpe: {best_sharpe['display_name']}",
        f"- Best model by drawdown: {best_drawdown['display_name']}",
        f"- Best model by test-period excess return vs SPY: {best_overall['display_name']}",
        "",
        "## Historically Safer Models",
        "",
        _dataframe_to_markdown(safer.round(6)) if not safer.empty else "No historically safer model rows available.",
    ]
    if include_snapshot_models:
        lines.extend(
            [
                "",
                "## Snapshot / Exploratory Models",
                "",
                _dataframe_to_markdown(snapshot.round(6)) if not snapshot.empty else "No snapshot model rows available.",
            ]
        )
    output_report.write_text("\n".join(lines), encoding="utf-8")
    return best_rows


def _build_master_report(
    *,
    config_lines: list[str],
    api_lines: list[str],
    benchmark_df: pd.DataFrame,
    sentiment_report_df: pd.DataFrame,
    historical_diag_df: pd.DataFrame,
    combined_comparison_df: pd.DataFrame,
    final_best_rows: pd.DataFrame,
    horizon_comparison_df: pd.DataFrame,
    horizon_walk_forward_df: pd.DataFrame,
    include_snapshot_models: bool,
    start_date: str,
    end_date: str,
    sentiment_start_date: str,
    sentiment_end_date: str,
    output_path: Path,
) -> None:
    safer = final_best_rows.loc[final_best_rows["strategy_name"].isin(HISTORICALLY_SAFER_MODELS)]
    snapshot = final_best_rows.loc[final_best_rows["strategy_name"].isin(SNAPSHOT_EXPLORATORY_MODELS)] if include_snapshot_models else pd.DataFrame()
    best_overall = final_best_rows.sort_values(
        ["test_period_excess_return_vs_spy", "test_sharpe_ratio", "max_drawdown"],
        ascending=[False, False, False],
    ).iloc[0]
    best_safer = safer.iloc[0] if not safer.empty else None
    best_snapshot = snapshot.iloc[0] if not snapshot.empty else None
    sentiment_helped = False
    if "historical_rating_counts_plus_sentiment" in set(final_best_rows["strategy_name"]) and "historical_rating_counts_model" in set(final_best_rows["strategy_name"]):
        plus = final_best_rows.loc[final_best_rows["strategy_name"] == "historical_rating_counts_plus_sentiment"].iloc[0]
        base = final_best_rows.loc[final_best_rows["strategy_name"] == "historical_rating_counts_model"].iloc[0]
        sentiment_helped = plus["test_period_excess_return_vs_spy"] > base["test_period_excess_return_vs_spy"]
    historical_helped = False
    if best_safer is not None and "technical_only" in set(final_best_rows["strategy_name"]):
        technical = final_best_rows.loc[final_best_rows["strategy_name"] == "technical_only"].iloc[0]
        historical_helped = best_safer["test_period_excess_return_vs_spy"] > technical["test_period_excess_return_vs_spy"]
    historical_grade_events_helped = False
    if "historical_rating_counts_plus_events" in set(final_best_rows["strategy_name"]) and "historical_rating_counts_model" in set(final_best_rows["strategy_name"]):
        events = final_best_rows.loc[final_best_rows["strategy_name"] == "historical_rating_counts_plus_events"].iloc[0]
        base = final_best_rows.loc[final_best_rows["strategy_name"] == "historical_rating_counts_model"].iloc[0]
        historical_grade_events_helped = events["test_period_excess_return_vs_spy"] > base["test_period_excess_return_vs_spy"]

    horizon_summary_rows = []
    if not horizon_comparison_df.empty:
        for horizon in [5, 21, 63]:
            horizon_slice = horizon_comparison_df.loc[horizon_comparison_df["holding_period_days"] == horizon].copy()
            horizon_slice = horizon_slice.sort_values(
                ["test_period_excess_return_vs_spy", "test_sharpe_ratio", "max_drawdown"],
                ascending=[False, False, False],
            )
            best_horizon = horizon_slice.iloc[0] if not horizon_slice.empty else None
            previous_blend_slice = combined_comparison_df.loc[
                (combined_comparison_df["strategy_name"] == "final_quant_model_1y_no_snapshot")
                & (combined_comparison_df["holding_period_days"] == horizon)
            ]
            previous_blend = previous_blend_slice.iloc[0] if not previous_blend_slice.empty else None
            walk_slice = horizon_walk_forward_df.loc[horizon_walk_forward_df["holding_period_days"] == horizon].copy()
            walk_slice = walk_slice.loc[
                walk_slice["strategy_name"] == (best_horizon["strategy_name"] if best_horizon is not None else "")
            ]
            windows_beating_spy = int(walk_slice["beat_spy"].sum()) if not walk_slice.empty else 0
            avg_walk_excess = float(walk_slice["test_excess_return_vs_spy"].mean()) if not walk_slice.empty else float("nan")
            horizon_summary_rows.append(
                {
                    "holding_period_days": horizon,
                    "best_model": best_horizon["display_name"] if best_horizon is not None else "n/a",
                    "beat_spy_on_2025_test": bool(best_horizon is not None and best_horizon["test_period_excess_return_vs_spy"] > 0),
                    "test_period_excess_return_vs_spy": float(best_horizon["test_period_excess_return_vs_spy"]) if best_horizon is not None else float("nan"),
                    "beat_previous_final_blend": bool(
                        best_horizon is not None
                        and previous_blend is not None
                        and best_horizon["test_period_excess_return_vs_spy"] > previous_blend["test_period_excess_return_vs_spy"]
                    ),
                    "walk_forward_windows_beating_spy": windows_beating_spy,
                    "walk_forward_average_excess_return_vs_spy": avg_walk_excess,
                    "performance_concentrated_in_one_period": bool(windows_beating_spy <= 1),
                }
            )

    lines = [
        "# Full 3Y Master Report",
        "",
        "## Run Configuration",
        f"- Backtest window: {start_date} to {end_date}",
        f"- Sentiment window: {sentiment_start_date} to {sentiment_end_date}",
        f"- Feature panel window: {start_date} to {end_date}",
        f"- Development period: 2023-01-01 to 2024-12-31",
        f"- Test period: 2025-01-01 to 2025-12-31",
        *config_lines,
        "",
        "## API / Cache Summary",
        *api_lines,
        "",
        "## Benchmark Validation",
        "",
        _dataframe_to_markdown(benchmark_df.round(6)),
        "",
        "## Sentiment Coverage",
        f"- Average universe sentiment coverage: {sentiment_report_df['coverage_pct_7d'].mean():.2%}",
        f"- Average article_count_7d: {sentiment_report_df['avg_article_count_7d'].mean():.2f}",
        "",
        "## Historical Analyst Coverage",
        f"- Average universe historical rating-count coverage: {historical_diag_df['percent_with_historical_rating_counts'].mean():.2%}",
        f"- Average coverage with total ratings >= 5: {historical_diag_df['percent_with_total_ratings_ge_5'].mean():.2%}",
        "",
        "## Best Historically Safer Models",
        "",
        _dataframe_to_markdown(safer.head(10).round(6)) if not safer.empty else "No historically safer rows available.",
        "",
        "## Best Snapshot / Exploratory Models",
        "",
        _dataframe_to_markdown(snapshot.head(10).round(6)) if not snapshot.empty else "No snapshot rows available.",
        "",
        "## Best Overall Models",
        f"- Best overall model: {best_overall['display_name']}",
        f"- Best test-period excess return vs SPY: {best_overall['test_period_excess_return_vs_spy']:.2%}",
        f"- Best test-period Sharpe: {best_overall['test_sharpe_ratio']:.2f}",
        f"- Best max drawdown: {best_overall['max_drawdown']:.2%}",
        "",
        "## Horizon-Specific No-Snapshot Models",
        "",
        _dataframe_to_markdown(pd.DataFrame(horizon_summary_rows).round(6)) if horizon_summary_rows else "No horizon-specific model rows available.",
        "",
        "## Test Period Results",
        f"- Whether any model beat SPY on test period: {best_overall['test_period_excess_return_vs_spy'] > 0}",
        f"- Whether any historically safer model beat SPY on test period: {bool(best_safer is not None and best_safer['test_period_excess_return_vs_spy'] > 0)}",
        f"- Whether sentiment helped: {sentiment_helped}",
        f"- Whether historical ratings helped: {historical_helped}",
        f"- Whether historical grade events helped: {historical_grade_events_helped}",
        f"- Whether snapshot analyst models outperformed but are caveated: {bool(best_snapshot is not None and best_snapshot['test_period_excess_return_vs_spy'] > (best_safer['test_period_excess_return_vs_spy'] if best_safer is not None else float('-inf')))}",
        "",
        "## Model Caveats",
        f"- {SNAPSHOT_ANALYST_CAVEAT}",
        f"- {HISTORICAL_RATING_NOTE}",
        f"- {BACKTEST_PERFORMANCE_CAVEAT}",
        f"- {SENTIMENT_CAVEAT}",
        f"- {BACKTEST_CAVEAT}",
        f"- {LONG_SHORT_CAVEAT}",
        "",
        "## Suggested Next Improvements",
        "- Add true point-in-time historical target-price datasets if available.",
        "- Expand walk-forward validation beyond one fixed dev/test split.",
        "- Add richer sector and turnover controls for historically safer models.",
        "- Measure coverage stability by ticker and by sector before trusting sparse analyst signals.",
    ]
    if include_snapshot_models:
        lines[lines.index("## Best Snapshot / Exploratory Models") + 2] = _dataframe_to_markdown(snapshot.head(10).round(6)) if not snapshot.empty else "No snapshot rows available."
    else:
        lines[lines.index("## Best Snapshot / Exploratory Models") + 2] = "Snapshot / exploratory models were excluded by default for this run."
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--sentiment-start-date", default=None)
    parser.add_argument("--sentiment-end-date", default=None)
    parser.add_argument("--force-refresh", action="store_true")
    parser.add_argument("--clear-cache", action="store_true")
    parser.add_argument("--clear-outputs", action="store_true")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--tickers", default=None)
    parser.add_argument("--requests-per-minute", type=int, default=None)
    parser.add_argument("--include-ml", action="store_true")
    parser.add_argument("--include-snapshot-models", action="store_true")
    parser.add_argument("--allow-partial-sentiment", action="store_true")
    args = parser.parse_args()

    config = Config.from_env()
    active_start_date = args.start_date or config.full_backtest_start_date
    active_end_date = args.end_date or config.full_backtest_end_date
    active_sentiment_start_date = args.sentiment_start_date or config.full_sentiment_start_date
    active_sentiment_end_date = args.sentiment_end_date or config.full_sentiment_end_date
    force_refresh = args.force_refresh or config.full_run_force_refresh
    clear_cache = args.clear_cache or config.full_run_clear_cache
    clear_outputs = args.clear_outputs or config.full_run_clear_outputs
    requests_per_minute = args.requests_per_minute or config.alpha_vantage_requests_per_minute
    tickers = (
        [ticker.strip().upper() for ticker in args.tickers.split(",") if ticker.strip()]
        if args.tickers
        else get_tickers(config.universe_path)
    )
    date_label = f"{active_start_date}_{active_end_date}"
    fetch_start_date = (pd.Timestamp(active_start_date) - pd.Timedelta(days=config.historical_analyst_lookback_days)).strftime("%Y-%m-%d")

    print("Full 3Y Rebuild Configuration")
    config_lines = [
        f"- Universe tickers: {len(tickers)}",
        f"- Price window: {active_start_date} to {active_end_date}",
        f"- Sentiment window: {active_sentiment_start_date} to {active_sentiment_end_date}",
        f"- Feature panel window: {active_start_date} to {active_end_date}",
        f"- Backtest window: {active_start_date} to {active_end_date}",
        f"- Development period: 2023-01-01 to 2024-12-31",
        f"- Test period: 2025-01-01 to 2025-12-31",
        f"- Historical analyst fetch window with lookback: {fetch_start_date} to {active_end_date}",
        f"- Force refresh: {force_refresh}",
        f"- Clear cache: {clear_cache}",
        f"- Clear outputs: {clear_outputs}",
        f"- Alpha Vantage requests per minute: {requests_per_minute}",
        f"- Include ML: {args.include_ml}",
        f"- Include snapshot models: {args.include_snapshot_models}",
        f"- Allow partial sentiment: {args.allow_partial_sentiment}",
    ]
    print("\n".join(config_lines))

    windows = build_monthly_windows(
        tickers=tickers,
        start_date=active_sentiment_start_date,
        end_date=active_sentiment_end_date,
        raw_news_dir=config.raw_dir / "news" / "alpha_vantage",
    )
    av_plan = summarize_request_plan(
        windows,
        force=force_refresh,
        cache_enabled=config.cache_enabled,
        requests_per_minute=requests_per_minute,
    )
    price_plan = _estimate_price_calls(
        tickers,
        config.benchmark,
        config.raw_dir / "prices" / "eodhd",
        active_start_date,
        active_end_date,
        force_refresh,
    )
    snapshot_plan = _estimate_fmp_snapshot_calls(tickers, config.raw_dir / "analyst" / "fmp", force_refresh)
    historical_plan = _estimate_historical_grade_calls(tickers, config.raw_dir / "analyst" / "fmp_historical_grades", force_refresh)
    estimated_total_runtime_minutes = (
        av_plan["missing_requests"] / max(requests_per_minute, 1)
        + snapshot_plan["missing"] / max(config.fmp_calls_per_minute, 1)
        + historical_plan["missing"] / max(config.fmp_calls_per_minute, 1)
        + price_plan["missing"] / max(config.eodhd_calls_per_minute, 1)
    )
    api_lines = [
        f"- Number of tickers: {len(tickers)}",
        f"- Price date range: {active_start_date} to {active_end_date}",
        f"- Sentiment date range: {active_sentiment_start_date} to {active_sentiment_end_date}",
        f"- Alpha Vantage ticker-month requests expected: {av_plan['total_possible_requests']}",
        f"- Alpha Vantage already cached: {av_plan['cached_requests']}",
        f"- Alpha Vantage missing: {av_plan['missing_requests']}",
        f"- Estimated Alpha Vantage runtime (minutes): {av_plan['estimated_runtime_minutes']:.2f}",
        f"- Expected FMP analyst snapshot calls: {snapshot_plan['missing']} missing of {snapshot_plan['total']}",
        f"- Expected FMP historical grade calls: {historical_plan['missing']} missing of {historical_plan['total']}",
        f"- Expected EODHD price calls: {price_plan['missing']} missing of {price_plan['total']}",
        f"- Estimated total runtime (minutes): {estimated_total_runtime_minutes:.2f}",
    ]
    print("\n".join(api_lines))

    if args.dry_run:
        return

    if clear_cache and not args.yes:
        raise SystemExit("Refusing to clear caches without both --clear-cache and --yes.")

    if clear_outputs:
        result = _run_subprocess(
            [sys.executable, "scripts/99_clean_outputs.py", "--outputs-only", "--yes"],
            config.project_root,
            dry_run=False,
            capture_output=True,
        )
        if result is not None:
            print(result.stdout)
            if result.returncode != 0:
                raise SystemExit(result.stderr or "Failed to clear outputs.")

    if clear_cache:
        _clear_cache_dirs(config)

    print("Step 5: Fetching EODHD prices")
    print(f"Price window: {active_start_date} to {active_end_date}")
    fetch_and_save_prices(
        tickers=tickers + ([config.benchmark] if config.benchmark not in tickers else []),
        start_date=active_start_date,
        end_date=active_end_date,
        api_key=config.eodhd_api_key,
        raw_prices_dir=config.raw_dir / "prices" / "eodhd",
        combined_output_path=config.processed_dir / "prices_all.csv",
        calls_per_minute=config.eodhd_calls_per_minute,
        force=force_refresh,
        cache_enabled=config.cache_enabled,
    )

    print("Step 6: Fetching FMP analyst snapshot data")
    build_analyst_snapshot(
        tickers=tickers,
        api_key=config.fmp_api_key,
        raw_output_dir=config.raw_dir / "analyst" / "fmp",
        processed_output_path=config.processed_dir / "analyst_features.csv",
        prices_path=config.processed_dir / "prices_all.csv",
        calls_per_minute=config.fmp_calls_per_minute,
        force=force_refresh,
        cache_enabled=config.cache_enabled,
    )

    print("Step 7: Fetching FMP historical grades and grades-historical")
    build_historical_grade_datasets(
        tickers=tickers,
        api_key=config.fmp_api_key,
        raw_output_dir=config.raw_dir / "analyst" / "fmp_historical_grades",
        rating_counts_output_path=config.processed_dir / "historical_analyst_rating_counts.csv",
        grade_events_output_path=config.processed_dir / "historical_analyst_grade_events.csv",
        start_date=fetch_start_date,
        end_date=active_end_date,
        calls_per_minute=config.fmp_calls_per_minute,
        force=force_refresh,
        cache_enabled=config.cache_enabled,
        limit=1000,
    )

    print("Step 8: Fetching Alpha Vantage sentiment cache")
    print(f"Sentiment window: {active_sentiment_start_date} to {active_sentiment_end_date}")
    fetch_alpha_vantage_news_cache(
        windows,
        api_key=config.alpha_vantage_api_key,
        cache_enabled=config.cache_enabled,
        force=force_refresh,
        limit=1000,
        requests_per_minute=requests_per_minute,
    )
    normalize_alpha_vantage_news_cache(
        windows,
        processed_output_path=config.processed_dir / "stock_news_alpha_vantage.csv",
        combined_output_path=config.processed_dir / "stock_news.csv",
    )

    print("Step 9: Building daily news sentiment")
    build_news_sentiment_outputs(
        news_input_path=config.processed_dir / "stock_news.csv",
        articles_output_path=config.processed_dir / "news_sentiment_articles.csv",
        daily_output_path=config.processed_dir / "news_sentiment_daily.csv",
        start_date=active_sentiment_start_date,
        end_date=active_sentiment_end_date,
        force=force_refresh,
        rescore_with_finbert=False,
        prefer_finbert=False,
    )
    sentiment_daily_window_path = config.processed_dir / f"news_sentiment_daily_{active_sentiment_start_date}_{active_sentiment_end_date}.csv"
    sentiment_articles_window_path = config.processed_dir / f"news_sentiment_articles_{active_sentiment_start_date}_{active_sentiment_end_date}.csv"
    _copy_if_exists(config.processed_dir / "news_sentiment_daily.csv", sentiment_daily_window_path)
    _copy_if_exists(config.processed_dir / "news_sentiment_articles.csv", sentiment_articles_window_path)

    print("Step 10: Building feature panel")
    print(f"Feature panel window: {active_start_date} to {active_end_date}")
    features = build_feature_panel(
        prices_path=config.processed_dir / "prices_all.csv",
        universe_path=config.universe_path,
        analyst_path=config.processed_dir / "analyst_features.csv",
        sentiment_path=_resolve_window_file(config.processed_dir / "news_sentiment_daily.csv", active_sentiment_start_date, active_sentiment_end_date),
        historical_rating_counts_path=config.processed_dir / "historical_analyst_rating_counts.csv",
        historical_grade_events_path=config.processed_dir / "historical_analyst_grade_events.csv",
        historical_rating_count_features_output_path=config.processed_dir / "historical_rating_count_features.csv",
        historical_grade_features_output_path=config.processed_dir / "historical_grade_features.csv",
        output_path=config.final_dir / "features_panel.csv",
        start_date=active_start_date,
        end_date=active_end_date,
        benchmark_ticker=config.benchmark,
        use_current_snapshot_analyst=True,
    )
    dated_features_path = config.final_dir / f"features_panel_{date_label}.csv"
    save_dataframe(dated_features_path, features)
    _validate_feature_panel_window(
        features,
        start_date=active_start_date,
        end_date=active_end_date,
        sentiment_start_date=active_sentiment_start_date,
        sentiment_end_date=active_sentiment_end_date,
        allow_partial_sentiment=args.allow_partial_sentiment,
    )

    print("Step 11: Validating historical ratings")
    validation_proc = _run_subprocess(
        [sys.executable, "scripts/18_validate_historical_ratings.py"],
        config.project_root,
        dry_run=False,
        capture_output=True,
    )
    historical_validation_report = config.reports_dir / "full_3y_historical_rating_validation.md"
    historical_validation_text = ""
    if validation_proc is not None:
        historical_validation_text = (validation_proc.stdout or "") + ("\n" + validation_proc.stderr if validation_proc.stderr else "")
        historical_validation_report.write_text(f"# Full 3Y Historical Rating Validation\n\n```\n{historical_validation_text.strip()}\n```\n", encoding="utf-8")
        if validation_proc.returncode != 0:
            raise SystemExit("Historical rating validation failed.")
    no_snapshot_proc = _run_subprocess(
        [sys.executable, "scripts/19_validate_no_snapshot_models.py"],
        config.project_root,
        dry_run=False,
        capture_output=True,
    )
    if no_snapshot_proc is not None and no_snapshot_proc.returncode != 0:
        raise SystemExit(no_snapshot_proc.stderr or no_snapshot_proc.stdout or "No-snapshot validation failed.")

    print("Step 12: Running fixed-horizon backtests")
    print(f"Backtest window: {active_start_date} to {active_end_date}")
    benchmark_rows = []
    all_comparison_rows = []
    full_backtest_strategies = list(HISTORICALLY_SAFER_MODELS)
    if args.include_snapshot_models:
        full_backtest_strategies.extend(SNAPSHOT_EXPLORATORY_MODELS)
    full_backtest_strategies = ["SPY", *full_backtest_strategies]
    for holding_period_days in [5, 21, 63]:
        rows = []
        for strategy_name in full_backtest_strategies:
            if strategy_name == "SPY":
                weekly = _build_spy_weekly(features, config.benchmark, holding_period_days, config.initial_capital)
            else:
                weekly, _, _ = run_weekly_backtest(
                    features=features,
                    holding_period_days=holding_period_days,
                    benchmark=config.benchmark,
                    top_n=config.top_n,
                    initial_capital=config.initial_capital,
                    transaction_cost_bps=config.transaction_cost_bps,
                    use_regime_filter=False,
                    regime_exposure=0.0,
                    use_analyst_filters=strategy_name in {"analyst_snapshot_model", "full_model", "final_quant_model_1y"},
                    analyst_count_threshold=config.analyst_count_threshold,
                    min_avg_dollar_volume=config.min_avg_dollar_volume,
                    strategy_name=strategy_name,
                    min_historical_rating_count=5,
                )
            rows.append(_comparison_row(strategy_name, weekly, holding_period_days=holding_period_days, top_n=config.top_n))
            if strategy_name == "SPY":
                validation = _build_validation_row(
                    features=features,
                    weekly_returns=weekly,
                    benchmark=config.benchmark,
                    holding_period_days=holding_period_days,
                )
                validation_row = validation.iloc[0]
                benchmark_rows.append(
                    {
                        "holding_period_days": holding_period_days,
                        **validation_row.to_dict(),
                    }
                )
                if float(validation_row["absolute_difference"]) > VALIDATION_TOLERANCE:
                    raise SystemExit(
                        f"Benchmark validation failed for holding period {holding_period_days}: absolute difference {float(validation_row['absolute_difference']):.6f}"
                    )
                report_path = config.reports_dir / f"full_3y_backtest_h{holding_period_days}_{date_label}.md"
        comparison_df = pd.DataFrame(rows).sort_values(
            ["test_period_excess_return_vs_spy", "test_sharpe_ratio", "max_drawdown"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
        save_dataframe(config.tables_dir / f"full_3y_strategy_comparison_h{holding_period_days}_{date_label}.csv", comparison_df)
        all_comparison_rows.append(comparison_df)
        _build_backtest_report(
            comparison_df,
            pd.Series(benchmark_rows[-1]),
            holding_period_days=holding_period_days,
            date_label=date_label,
            output_path=config.reports_dir / f"full_3y_backtest_h{holding_period_days}_{date_label}.md",
        )

    benchmark_df = pd.DataFrame(benchmark_rows)
    save_dataframe(config.tables_dir / "full_3y_benchmark_validation.csv", benchmark_df)

    print("Step 13: Running sentiment model comparison")
    sentiment_suffix = f"full_3y_{date_label}"
    sentiment_proc = _run_subprocess(
        [
            sys.executable,
            "scripts/14_compare_sentiment_models.py",
            "--features-path",
            str(dated_features_path),
            "--start-date",
            active_start_date,
            "--end-date",
            active_end_date,
            "--output-suffix",
            sentiment_suffix,
        ],
        config.project_root,
        dry_run=False,
        capture_output=True,
    )
    if sentiment_proc is not None and sentiment_proc.returncode != 0:
        raise SystemExit(sentiment_proc.stderr or "Sentiment comparison failed.")
    sentiment_diag_source = config.tables_dir / f"sentiment_diagnostics_{sentiment_suffix}.csv"
    sentiment_cmp_source = config.tables_dir / f"sentiment_model_comparison_{sentiment_suffix}.csv"
    if not sentiment_cmp_source.exists():
        sentiment_cmp_source = config.tables_dir / f"sentiment_model_comparison{('_' + sentiment_suffix) if sentiment_suffix else ''}.csv"
    if not sentiment_diag_source.exists():
        sentiment_diag_source = config.tables_dir / f"sentiment_diagnostics{('_' + sentiment_suffix) if sentiment_suffix else ''}.csv"
    _copy_if_exists(sentiment_diag_source, config.tables_dir / "full_3y_sentiment_diagnostics.csv")
    sentiment_diagnostics = load_dataframe(config.tables_dir / "full_3y_sentiment_diagnostics.csv", parse_dates=["date"])
    if not sentiment_cmp_source.exists():
        sentiment_cmp_source = config.tables_dir / "sentiment_model_comparison.csv"
    sentiment_comparison = load_dataframe(sentiment_cmp_source)
    _, sentiment_holdings, _ = run_weekly_backtest(
        features=features,
        holding_period_days=21,
        benchmark=config.benchmark,
        top_n=config.top_n,
        initial_capital=config.initial_capital,
        transaction_cost_bps=config.transaction_cost_bps,
        use_regime_filter=False,
        regime_exposure=0.0,
        use_analyst_filters=False,
        analyst_count_threshold=config.analyst_count_threshold,
        min_avg_dollar_volume=config.min_avg_dollar_volume,
        strategy_name="technical_sentiment_model",
    )
    selected_holdings_sentiment_coverage = float(sentiment_holdings["article_count_7d"].fillna(0).gt(0).mean()) if not sentiment_holdings.empty else 0.0
    _build_sentiment_report(
        sentiment_comparison,
        sentiment_diagnostics,
        features,
        config.benchmark,
        selected_holdings_sentiment_coverage,
        config.reports_dir / "full_3y_sentiment_coverage_report.md",
    )

    print("Step 14: Running historical analyst model comparison")
    historical_proc = _run_subprocess(
        [
            sys.executable,
            "scripts/17_compare_historical_analyst_models.py",
            "--features-path",
            str(dated_features_path),
            "--start-date",
            active_start_date,
            "--end-date",
            active_end_date,
        ],
        config.project_root,
        dry_run=False,
        capture_output=True,
    )
    if historical_proc is not None and historical_proc.returncode != 0:
        raise SystemExit(historical_proc.stderr or "Historical analyst comparison failed.")
    _copy_if_exists(config.tables_dir / "historical_rating_count_diagnostics.csv", config.tables_dir / "full_3y_historical_rating_count_diagnostics.csv")
    historical_coverage_df = _build_historical_coverage_outputs(
        features,
        config.benchmark,
        config.tables_dir / "full_3y_historical_rating_count_diagnostics.csv",
        config.reports_dir / "full_3y_historical_analyst_coverage_report.md",
    )

    print("Step 15: Building final full-run model comparison")
    combined_comparison = pd.concat(all_comparison_rows, ignore_index=True)
    final_best_rows = _build_final_family_report(
        combined_comparison,
        config.tables_dir / f"full_3y_final_model_comparison_{date_label}.csv",
        config.reports_dir / f"full_3y_final_model_report_{date_label}.md",
        include_snapshot_models=args.include_snapshot_models,
        start_date=active_start_date,
        end_date=active_end_date,
        sentiment_start_date=active_sentiment_start_date,
        sentiment_end_date=active_sentiment_end_date,
    )

    if args.include_ml:
        print("Step 16: Running experimental ML ranking")
        ml_proc = _run_subprocess([sys.executable, "scripts/11_ml_rank_model.py"], config.project_root, dry_run=False, capture_output=True)
        if ml_proc is not None and ml_proc.returncode != 0:
            raise SystemExit(ml_proc.stderr or "ML ranking experiment failed.")
        _copy_if_exists(config.tables_dir / "ml_model_results.csv", config.tables_dir / "full_3y_ml_model_results.csv")
        _copy_if_exists(config.reports_dir / "ml_model_summary.md", config.reports_dir / "full_3y_ml_model_summary.md")

    print("Step 17: Running horizon-specific no-snapshot comparisons")
    horizon_proc = _run_subprocess(
        [
            sys.executable,
            "scripts/31_compare_horizon_specific_models.py",
            "--features-path",
            str(dated_features_path),
        ],
        config.project_root,
        dry_run=False,
        capture_output=True,
    )
    if horizon_proc is not None and horizon_proc.returncode != 0:
        raise SystemExit(horizon_proc.stderr or "Horizon-specific comparison failed.")

    print("Step 18: Running horizon-specific walk-forward checks")
    walk_forward_proc = _run_subprocess(
        [
            sys.executable,
            "scripts/32_walk_forward_horizon_models.py",
            "--features-path",
            str(dated_features_path),
        ],
        config.project_root,
        dry_run=False,
        capture_output=True,
    )
    if walk_forward_proc is not None and walk_forward_proc.returncode != 0:
        raise SystemExit(walk_forward_proc.stderr or "Horizon-specific walk-forward failed.")

    print("Step 19: Building master report")
    horizon_comparison_df = load_dataframe(config.tables_dir / "horizon_specific_model_comparison.csv")
    horizon_walk_forward_df = load_dataframe(config.tables_dir / "horizon_specific_walk_forward_results.csv")
    _build_master_report(
        config_lines=config_lines,
        api_lines=api_lines,
        benchmark_df=benchmark_df,
        sentiment_report_df=sentiment_diagnostics,
        historical_diag_df=historical_coverage_df,
        combined_comparison_df=combined_comparison,
        final_best_rows=final_best_rows,
        horizon_comparison_df=horizon_comparison_df,
        horizon_walk_forward_df=horizon_walk_forward_df,
        include_snapshot_models=args.include_snapshot_models,
        start_date=active_start_date,
        end_date=active_end_date,
        sentiment_start_date=active_sentiment_start_date,
        sentiment_end_date=active_sentiment_end_date,
        output_path=config.reports_dir / f"full_3y_master_report_{date_label}.md",
    )
    print("Full 3Y rebuild complete.")


if __name__ == "__main__":
    main()
