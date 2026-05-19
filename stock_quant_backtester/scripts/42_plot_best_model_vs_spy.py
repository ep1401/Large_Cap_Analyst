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

from src.backtest import run_weekly_backtest
from src.config import Config
from src.metrics import calculate_performance_metrics
from src.scoring import NO_SNAPSHOT_STRATEGIES, SNAPSHOT_FIELD_COLUMNS, strategy_display_name, strategy_score_fields
from src.utils import load_dataframe, save_dataframe


BACKTEST_CAVEAT = "Back-tested performance is hypothetical."
SNAPSHOT_CAVEAT = "Snapshot analyst target models are excluded."
HISTORICAL_NOTE = "Historical rating-count features use dated FMP grades-historical records available on or before each rebalance date."
SENTIMENT_CAVEAT = "News sentiment depends on Alpha Vantage coverage and classification."
LONG_SHORT_CAVEAT = "Long/short is experimental and currently not recommended."
RESEARCH_CAVEAT = "This is research/paper trading only, not financial advice."

TARGET_STRATEGY = "final_quant_5d_no_snapshot_no_sma_filter"
TARGET_HOLDING_PERIOD_DAYS = 5
TARGET_TOP_N = 10
TARGET_POSITION_SIZING = "equal_weight"
TARGET_TOTAL_COST_BPS = 10
TARGET_BENCHMARK = "SPY"
TARGET_START = pd.Timestamp("2023-01-01")
TARGET_END = pd.Timestamp("2026-01-01")


def _resolve_features_path(config: Config) -> Path:
    preferred = config.final_dir / "features_panel_2023-01-01_2026-01-01.csv"
    fallback = config.final_dir / "features_panel.csv"
    if preferred.exists():
        return preferred
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Could not find feature panel at {preferred} or {fallback}")


def _compute_drawdown(values: pd.Series) -> pd.Series:
    return values / values.cummax() - 1


def _validate_no_snapshot_strategy(strategy_name: str) -> None:
    if strategy_name not in NO_SNAPSHOT_STRATEGIES:
        raise ValueError(f"{strategy_name} is not registered as a no-snapshot strategy.")
    offending = sorted(strategy_score_fields(strategy_name) & SNAPSHOT_FIELD_COLUMNS)
    if offending:
        raise ValueError(f"{strategy_name} uses snapshot fields: {', '.join(offending)}")


def _load_optional_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return load_dataframe(path)


def _save_plot(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def _make_equity_curve_plot(returns_df: pd.DataFrame, output_path: Path, display_name: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 7), facecolor="white")
    ax.set_facecolor("white")
    ax.plot(returns_df["date"], returns_df["model_value"], linewidth=2.2, label=display_name, color="#0f766e")
    ax.plot(returns_df["date"], returns_df["spy_value"], linewidth=2.0, label="SPY", color="#1f2937")
    ax.set_title("Best No-Snapshot Model vs SPY, 2023-2026")
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of $10,000")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"${value:,.0f}"))
    ax.grid(True, alpha=0.25)
    ax.legend()
    _save_plot(fig, output_path)


def _make_drawdown_plot(returns_df: pd.DataFrame, output_path: Path, display_name: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 7), facecolor="white")
    ax.set_facecolor("white")
    ax.plot(returns_df["date"], returns_df["model_drawdown"], linewidth=2.2, label=display_name, color="#b91c1c")
    ax.plot(returns_df["date"], returns_df["spy_drawdown"], linewidth=2.0, label="SPY", color="#1f2937")
    ax.set_title("Drawdown: Best No-Snapshot Model vs SPY")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown %")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0%}"))
    ax.grid(True, alpha=0.25)
    ax.legend()
    _save_plot(fig, output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-path", default=None)
    args = parser.parse_args()

    config = Config.from_env()
    features_path = Path(args.features_path) if args.features_path else _resolve_features_path(config)

    if TARGET_STRATEGY != "final_quant_5d_no_snapshot_no_sma_filter":
        raise ValueError("This script is locked to the current best long-only no-snapshot baseline strategy.")

    _validate_no_snapshot_strategy(TARGET_STRATEGY)

    features = load_dataframe(features_path, parse_dates=["date"])
    features["date"] = pd.to_datetime(features["date"])
    actual_start = pd.Timestamp(features["date"].min())
    actual_end = pd.Timestamp(features["date"].max())
    if actual_start > TARGET_START or actual_end < TARGET_END:
        print(
            "Warning: feature panel does not fully cover 2023-01-01 to 2026-01-01. "
            f"Using available range {actual_start.date()} to {actual_end.date()}."
        )

    weekly, holdings, _ = run_weekly_backtest(
        features=features,
        holding_period_days=TARGET_HOLDING_PERIOD_DAYS,
        benchmark=TARGET_BENCHMARK,
        top_n=TARGET_TOP_N,
        initial_capital=config.initial_capital,
        transaction_cost_bps=TARGET_TOTAL_COST_BPS,
        use_regime_filter=False,
        regime_exposure=0.0,
        use_analyst_filters=False,
        analyst_count_threshold=config.analyst_count_threshold,
        min_avg_dollar_volume=config.min_avg_dollar_volume,
        strategy_name=TARGET_STRATEGY,
        position_sizing=TARGET_POSITION_SIZING,
        max_names_per_sector=None,
        use_inverse_vol_weighting=False,
        min_historical_rating_count=5,
    )

    if weekly.empty:
        raise ValueError("Backtest returned no rows for the requested baseline strategy.")
    if holdings["ticker"].eq(TARGET_BENCHMARK).any():
        raise ValueError("Benchmark ticker found in holdings.")
    if not weekly["strategy_name"].eq(TARGET_STRATEGY).all():
        raise ValueError("Weekly results contain an unexpected strategy name.")
    if weekly["use_regime_filter"].any():
        raise ValueError("Regime filter was enabled unexpectedly.")
    if not weekly["holding_period_days"].eq(TARGET_HOLDING_PERIOD_DAYS).all():
        raise ValueError("Unexpected holding period in weekly results.")

    returns_df = weekly.loc[
        :,
        [
            "date",
            "strategy_name",
            "net_return",
            "spy_return",
            "portfolio_value",
            "spy_value",
            "turnover",
            "transaction_cost",
            "selected_count",
        ],
    ].copy()
    returns_df = returns_df.rename(
        columns={
            "net_return": "model_return",
            "portfolio_value": "model_value",
            "transaction_cost": "trading_cost",
        }
    ).sort_values("date")
    returns_df["model_drawdown"] = _compute_drawdown(returns_df["model_value"])
    returns_df["spy_drawdown"] = _compute_drawdown(returns_df["spy_value"])
    returns_df["excess_return"] = returns_df["model_return"] - returns_df["spy_return"]

    save_dataframe(config.tables_dir / "best_model_vs_spy_3y_returns.csv", returns_df)

    display_name = strategy_display_name(TARGET_STRATEGY)
    _make_equity_curve_plot(returns_df, config.charts_dir / "best_model_vs_spy_3y_equity_curve.png", display_name)
    _make_drawdown_plot(returns_df, config.charts_dir / "best_model_vs_spy_3y_drawdown.png", display_name)

    metrics = calculate_performance_metrics(
        weekly.rename(columns={"net_return": "net_return", "spy_return": "spy_return"}),
        holding_period_days=TARGET_HOLDING_PERIOD_DAYS,
    )
    test_2025 = weekly.loc[(weekly["date"] >= pd.Timestamp("2025-01-01")) & (weekly["date"] <= pd.Timestamp("2025-12-31"))].copy()
    test_2025_excess = (
        calculate_performance_metrics(test_2025, holding_period_days=TARGET_HOLDING_PERIOD_DAYS)["excess_total_return"]
        if not test_2025.empty
        else float("nan")
    )

    walk_forward_df = _load_optional_csv(config.tables_dir / "horizon_specific_walk_forward_results.csv")
    walk_forward_windows_beating_spy: int | None = None
    if walk_forward_df is not None and not walk_forward_df.empty:
        filtered = walk_forward_df.loc[
            (walk_forward_df["strategy_name"] == TARGET_STRATEGY)
            & (walk_forward_df["holding_period_days"] == TARGET_HOLDING_PERIOD_DAYS)
        ].copy()
        if not filtered.empty and "beat_spy" in filtered.columns:
            walk_forward_windows_beating_spy = int(filtered["beat_spy"].fillna(False).astype(bool).sum())

    report_lines = [
        "# Best Model vs SPY Summary",
        "",
        f"- Strategy name: `{TARGET_STRATEGY}`",
        f"- Display name: {display_name}",
        f"- Date range: {returns_df['date'].min().date()} to {returns_df['date'].max().date()}",
        f"- Feature panel: `{features_path}`",
        f"- Holding period: {TARGET_HOLDING_PERIOD_DAYS} trading days",
        f"- Top N: {TARGET_TOP_N}",
        f"- Position sizing: {TARGET_POSITION_SIZING}",
        f"- Cost assumption: {TARGET_TOTAL_COST_BPS} bps total",
        f"- Regime filter: none",
        f"- Long/short: false",
        "",
        "## Metrics",
        "",
        f"- Total return for model: {metrics['total_return']:.2%}",
        f"- Total return for SPY: {metrics['spy_total_return']:.2%}",
        f"- Excess return vs SPY: {metrics['excess_total_return']:.2%}",
        f"- Annualized return: {metrics['annualized_return']:.2%}",
        f"- Sharpe: {metrics['sharpe_ratio']:.3f}",
        f"- Max drawdown: {metrics['max_drawdown']:.2%}",
        f"- Average turnover: {metrics['average_turnover']:.6f}",
        f"- Average selected count: {metrics['average_selected_count']:.2f}",
        f"- Number of rebalance periods: {int(metrics['number_of_rebalance_periods'])}",
        f"- 2025 test-period excess vs SPY: {test_2025_excess:.2%}" if pd.notna(test_2025_excess) else "- 2025 test-period excess vs SPY: n/a",
        (
            f"- Walk-forward windows beating SPY: {walk_forward_windows_beating_spy}"
            if walk_forward_windows_beating_spy is not None
            else "- Walk-forward windows beating SPY: n/a"
        ),
        "",
        "## Outputs",
        "",
        f"- Equity curve: `outputs/charts/best_model_vs_spy_3y_equity_curve.png`",
        f"- Drawdown chart: `outputs/charts/best_model_vs_spy_3y_drawdown.png`",
        f"- Returns table: `outputs/tables/best_model_vs_spy_3y_returns.csv`",
        "",
        "## Caveats",
        "",
        f"- {BACKTEST_CAVEAT}",
        f"- {SNAPSHOT_CAVEAT}",
        f"- {HISTORICAL_NOTE}",
        f"- {SENTIMENT_CAVEAT}",
        f"- {LONG_SHORT_CAVEAT}",
        f"- {RESEARCH_CAVEAT}",
    ]
    (config.reports_dir / "best_model_vs_spy_3y_summary.md").write_text("\n".join(report_lines), encoding="utf-8")

    print(f"Saved returns table to {config.tables_dir / 'best_model_vs_spy_3y_returns.csv'}")
    print(f"Saved equity curve to {config.charts_dir / 'best_model_vs_spy_3y_equity_curve.png'}")
    print(f"Saved drawdown chart to {config.charts_dir / 'best_model_vs_spy_3y_drawdown.png'}")
    print(f"Saved summary report to {config.reports_dir / 'best_model_vs_spy_3y_summary.md'}")


if __name__ == "__main__":
    main()
