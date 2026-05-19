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

from src.metrics import calculate_performance_metrics
from src.no_snapshot_research import summarize_backtest
from src.recommended_strategy import (
    caveat_lines,
    load_recommended_strategy_config,
    precompute_recommended_low_turnover_panels,
    run_low_turnover_recommended_backtest,
)
from src.config import Config
from src.scoring import NO_SNAPSHOT_STRATEGIES, SNAPSHOT_FIELD_COLUMNS, strategy_display_name, strategy_score_fields
from src.utils import load_dataframe, save_dataframe


TARGET_START = pd.Timestamp("2023-01-01")
TARGET_END = pd.Timestamp("2026-01-01")
EXPECTED_STRATEGY = "final_quant_5d_weight_tuned_low_turnover_no_snapshot"
BASE_SCORE_MODEL = "final_quant_5d_weight_tuned_no_snapshot"
EXECUTION_MODE = "low_turnover_hold_band"


def _resolve_features_path(config: Config, override: str | None) -> Path:
    if override:
        return Path(override)
    preferred = config.final_dir / "features_panel_2023-01-01_2026-01-01.csv"
    fallback = config.final_dir / "features_panel.csv"
    if preferred.exists():
        return preferred
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Could not find feature panel at {preferred} or {fallback}")


def _compute_drawdown(values: pd.Series) -> pd.Series:
    return values / values.cummax() - 1.0


def _validate_recommended_strategy(config: Config) -> tuple[object, bool]:
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
    return recommended, False


def _save_plot(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def _make_equity_curve_plot(returns_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 7), facecolor="white")
    ax.set_facecolor("white")
    ax.plot(
        returns_df["date"],
        returns_df["model_value"],
        linewidth=2.2,
        label="Recommended Low-Turnover Model",
        color="#0f766e",
    )
    ax.plot(returns_df["date"], returns_df["spy_value_direct"], linewidth=2.0, label="SPY Buy & Hold", color="#1f2937")
    ax.set_title("Recommended Low-Turnover Model vs SPY Buy & Hold (2023–2026)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of $10,000")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"${value:,.0f}"))
    ax.grid(True, alpha=0.25)
    ax.legend()
    _save_plot(fig, output_path)


def _make_drawdown_plot(returns_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 7), facecolor="white")
    ax.set_facecolor("white")
    ax.plot(
        returns_df["date"],
        returns_df["model_drawdown"],
        linewidth=2.2,
        label="Recommended Low-Turnover Model",
        color="#b91c1c",
    )
    ax.plot(returns_df["date"], returns_df["spy_drawdown"], linewidth=2.0, label="SPY Buy & Hold", color="#1f2937")
    ax.set_title("Drawdown: Recommended Low-Turnover Model vs SPY")
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
    recommended, snapshot_fields_allowed = _validate_recommended_strategy(config)
    features_path = _resolve_features_path(config, args.features_path)

    features = load_dataframe(features_path, parse_dates=["date"])
    features["date"] = pd.to_datetime(features["date"])
    features = features.loc[(features["date"] >= TARGET_START) & (features["date"] <= TARGET_END)].copy()
    if features.empty:
        raise ValueError("Filtered feature panel is empty for the requested 2023-01-01 to 2026-01-01 range.")

    actual_start = pd.Timestamp(features["date"].min())
    actual_end = pd.Timestamp(features["date"].max())
    covers_target_range = actual_start <= TARGET_START and actual_end >= TARGET_END

    spy_daily = (
        features.loc[features["ticker"] == config.benchmark, ["date", "adjusted_close"]]
        .dropna(subset=["adjusted_close"])
        .drop_duplicates(subset=["date"])
        .sort_values("date")
        .rename(columns={"adjusted_close": "spy_adjusted_close"})
        .reset_index(drop=True)
    )
    if spy_daily.empty:
        raise ValueError("No SPY adjusted_close data found in the feature panel.")

    panels = precompute_recommended_low_turnover_panels(features, config, recommended)
    weekly, holdings, _ = run_low_turnover_recommended_backtest(
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
        raise ValueError("Backtest returned no rows for the promoted recommended strategy.")
    if not holdings.empty and holdings["ticker"].eq(config.benchmark).any():
        raise ValueError("Benchmark ticker found in model holdings.")

    returns_df = weekly.loc[
        :,
        [
            "date",
            "net_return",
            "spy_return",
            "portfolio_value",
            "spy_value",
            "turnover",
            "transaction_cost",
            "selected_count",
            "exposure",
        ],
    ].copy().sort_values("date")
    returns_df = returns_df.rename(
        columns={
            "net_return": "model_period_return",
            "portfolio_value": "model_value",
            "transaction_cost": "trading_cost",
            "spy_return": "spy_period_return_if_used",
        }
    )
    returns_df = pd.merge_asof(
        returns_df.sort_values("date"),
        spy_daily.sort_values("date"),
        on="date",
        direction="backward",
    )
    if returns_df["spy_adjusted_close"].isna().any():
        raise ValueError("Unable to align SPY adjusted_close values to model equity dates.")

    first_spy_close = float(spy_daily["spy_adjusted_close"].iloc[0])
    last_spy_close = float(spy_daily["spy_adjusted_close"].iloc[-1])
    direct_spy_total_return = last_spy_close / first_spy_close - 1.0
    returns_df["spy_value_direct"] = 10000.0 * returns_df["spy_adjusted_close"] / first_spy_close
    returns_df["model_drawdown"] = _compute_drawdown(returns_df["model_value"])
    returns_df["spy_drawdown"] = _compute_drawdown(returns_df["spy_value_direct"])
    returns_df = returns_df[
        [
            "date",
            "model_period_return",
            "model_value",
            "spy_adjusted_close",
            "spy_value_direct",
            "spy_period_return_if_used",
            "model_drawdown",
            "spy_drawdown",
            "turnover",
            "trading_cost",
            "selected_count",
            "exposure",
        ]
    ]
    save_dataframe(config.tables_dir / "recommended_model_vs_spy_3y_returns.csv", returns_df)

    first_model_date = pd.Timestamp(returns_df["date"].iloc[0])
    last_model_date = pd.Timestamp(returns_df["date"].iloc[-1])
    spy_start_row = spy_daily.loc[spy_daily["date"] <= first_model_date].iloc[-1]
    spy_end_row = spy_daily.loc[spy_daily["date"] <= last_model_date].iloc[-1]
    direct_spy_return_on_model_dates = float(spy_end_row["spy_adjusted_close"] / spy_start_row["spy_adjusted_close"] - 1.0)
    compounded_spy_final_value = float(10000.0 * (1.0 + weekly["spy_return"]).cumprod().iloc[-1])
    direct_spy_final_value_on_model_dates = float(10000.0 * spy_end_row["spy_adjusted_close"] / spy_start_row["spy_adjusted_close"])
    benchmark_validation = pd.DataFrame(
        [
            {
                "start_date": first_model_date,
                "end_date": last_model_date,
                "direct_spy_return": direct_spy_return_on_model_dates,
                "compounded_period_spy_return": compounded_spy_final_value / 10000.0 - 1.0,
                "absolute_difference": abs(compounded_spy_final_value / direct_spy_final_value_on_model_dates - 1.0),
                "direct_spy_final_value": direct_spy_final_value_on_model_dates,
                "compounded_spy_final_value": compounded_spy_final_value,
            }
        ]
    )
    save_dataframe(config.tables_dir / "recommended_model_vs_spy_3y_benchmark_validation.csv", benchmark_validation)
    benchmark_validation_diff = float(benchmark_validation["absolute_difference"].iloc[0])
    if benchmark_validation_diff > 0.005:
        print("WARNING: Period-compounded SPY return does not match direct SPY buy-and-hold. Plot uses direct SPY adjusted-close benchmark.")

    _make_equity_curve_plot(returns_df, config.charts_dir / "recommended_model_vs_spy_3y_equity_curve.png")
    _make_drawdown_plot(returns_df, config.charts_dir / "recommended_model_vs_spy_3y_drawdown.png")

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
    window_2025 = weekly.loc[(weekly["date"] >= pd.Timestamp("2025-01-01")) & (weekly["date"] <= pd.Timestamp("2025-12-31"))].copy()
    excess_2025 = (
        calculate_performance_metrics(window_2025, holding_period_days=recommended.holding_period_days)["excess_total_return"]
        if not window_2025.empty
        else float("nan")
    )
    spy_max_drawdown = float(returns_df["spy_drawdown"].min())

    report_lines = [
        "# Recommended Model vs SPY Summary",
        "",
        "- Back-tested performance is hypothetical.",
        "- SPY line is direct buy-and-hold from adjusted close.",
        "- Snapshot analyst target fields are excluded.",
        "- Historical rating-count features use dated FMP grades-historical records available on or before each rebalance date.",
        "- News sentiment depends on Alpha Vantage coverage and classification.",
        "- This is research/paper trading only, not financial advice.",
        "",
        f"- Strategy name: `{recommended.strategy_name}`",
        f"- Display name: {strategy_display_name(recommended.strategy_name)}",
        f"- Base score model: `{BASE_SCORE_MODEL}`",
        f"- Execution mode: `{EXECUTION_MODE}`",
        f"- Date range requested: {TARGET_START.date()} to {TARGET_END.date()}",
        f"- Feature panel actual range used: {actual_start.date()} to {actual_end.date()}",
        f"- Feature panel covers full requested range: {covers_target_range}",
        f"- Feature panel path: `{features_path}`",
        f"- Enter rank: {recommended.enter_rank}",
        f"- Hold rank: {recommended.hold_rank}",
        f"- Top N: {recommended.top_n}",
        f"- Max holding days: {recommended.max_holding_days}",
        f"- Rebalance frequency: {recommended.rebalance_frequency_days} trading days",
        f"- Cost assumption: {recommended.total_cost_bps:.0f} bps total",
        f"- Regime filter: {recommended.regime_filter}",
        f"- Long/short: {str(recommended.long_short).lower()}",
        f"- Snapshot fields allowed: {str(snapshot_fields_allowed).lower()}",
        f"- SPY start adjusted close: {float(spy_start_row['spy_adjusted_close']):.4f}",
        f"- SPY end adjusted close: {float(spy_end_row['spy_adjusted_close']):.4f}",
        f"- Direct SPY buy-and-hold return on plotted range: {direct_spy_return_on_model_dates:.2%}",
        f"- SPY plotted final value: ${direct_spy_final_value_on_model_dates:,.2f}",
        "",
        "## Metrics",
        "",
        f"- Total return for model: {metrics['total_return']:.2%}",
        f"- Total return for SPY: {direct_spy_return_on_model_dates:.2%}",
        f"- Excess return vs SPY: {metrics['excess_total_return']:.2%}",
        f"- Annualized return: {metrics['annualized_return']:.2%}",
        f"- Annualized volatility: {metrics['annualized_volatility']:.2%}",
        f"- Sharpe: {metrics['sharpe_ratio']:.3f}",
        f"- Max drawdown: {metrics['max_drawdown']:.2%}",
        f"- SPY max drawdown: {spy_max_drawdown:.2%}",
        f"- Average turnover: {metrics['average_turnover']:.6f}",
        f"- Average holdings: {metrics['average_selected_count']:.2f}",
        f"- Number of rebalance periods: {int(metrics['number_of_rebalance_periods'])}",
        f"- Walk-forward average excess vs SPY: {walk_forward_average_excess:.2%}",
        f"- 2025 excess vs SPY: {excess_2025:.2%}" if pd.notna(excess_2025) else "- 2025 excess vs SPY: n/a",
        f"- Benchmark validation difference: {benchmark_validation_diff:.4%}",
        "",
        "## Outputs",
        "",
        "- Equity curve: `outputs/charts/recommended_model_vs_spy_3y_equity_curve.png`",
        "- Drawdown chart: `outputs/charts/recommended_model_vs_spy_3y_drawdown.png`",
        "- Returns table: `outputs/tables/recommended_model_vs_spy_3y_returns.csv`",
        "- Benchmark validation: `outputs/tables/recommended_model_vs_spy_3y_benchmark_validation.csv`",
    ]
    (config.reports_dir / "recommended_model_vs_spy_3y_summary.md").write_text("\n".join(report_lines), encoding="utf-8")

    print(f"Saved returns table to {config.tables_dir / 'recommended_model_vs_spy_3y_returns.csv'}")
    print(f"Saved benchmark validation to {config.tables_dir / 'recommended_model_vs_spy_3y_benchmark_validation.csv'}")
    print(f"Saved equity curve to {config.charts_dir / 'recommended_model_vs_spy_3y_equity_curve.png'}")
    print(f"Saved drawdown chart to {config.charts_dir / 'recommended_model_vs_spy_3y_drawdown.png'}")
    print(f"Saved summary report to {config.reports_dir / 'recommended_model_vs_spy_3y_summary.md'}")
    print(f"model final value: {float(returns_df['model_value'].iloc[-1]):.2f}")
    print(f"SPY final value: {direct_spy_final_value_on_model_dates:.2f}")
    print(f"model total return: {metrics['total_return']:.2%}")
    print(f"SPY total return: {direct_spy_return_on_model_dates:.2%}")
    print(f"excess return: {metrics['excess_total_return']:.2%}")
    print(f"model max drawdown: {metrics['max_drawdown']:.2%}")
    print(f"SPY max drawdown: {spy_max_drawdown:.2%}")
    print(f"benchmark validation difference: {benchmark_validation_diff:.4%}")


if __name__ == "__main__":
    main()
