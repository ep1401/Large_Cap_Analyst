from __future__ import annotations

import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mpl-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter

matplotlib.use("Agg")

sys.path.append(str(PROJECT_ROOT))

from src.ml_candidate_monitoring import (
    compute_drawdown,
    ensure_market_features,
    load_frozen_ml_context,
    ml_report_caveat_lines,
    run_frozen_ml_backtest_over_features,
    run_frozen_ml_forward,
)
from src.no_snapshot_research import dataframe_to_markdown
from src.scoring import SNAPSHOT_FIELD_COLUMNS
from src.utils import load_dataframe, save_dataframe


INITIAL_CAPITAL = 10000.0
VALIDATION_START = pd.Timestamp("2025-01-01")
VALIDATION_END = pd.Timestamp("2025-12-31")
FORWARD_BOUNDARY = pd.Timestamp("2026-01-01")


def _plot_caveat_lines() -> list[str]:
    return [
        "This is a frozen ML research candidate.",
        "2025 was used as validation/model-selection period.",
        "2026 forward data was not used for training, tuning, or model selection.",
        "Back-tested performance is hypothetical unless actually paper-tracked live.",
        "Snapshot analyst target fields are excluded.",
        "Historical rating-count features use dated FMP grades-historical records available on or before each rebalance date.",
        "News sentiment depends on Alpha Vantage coverage and classification.",
        "ML models may overfit and require extended forward validation.",
        "This is research/paper trading only, not financial advice.",
    ]


def _save_plot(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def _series_stats(returns: pd.Series, values: pd.Series, drawdown: pd.Series, turnover: pd.Series, holdings: pd.Series, rebalance_frequency_days: int) -> dict[str, float]:
    periods_per_year = 252.0 / rebalance_frequency_days
    numeric_returns = pd.to_numeric(returns, errors="coerce").fillna(0.0)
    total_return = float(values.iloc[-1] / INITIAL_CAPITAL - 1.0)
    if len(numeric_returns) == 0:
        ann_return = float("nan")
        ann_vol = float("nan")
        sharpe = float("nan")
    else:
        ann_return = float((1.0 + total_return) ** (periods_per_year / len(numeric_returns)) - 1.0)
        ann_vol = float(numeric_returns.std(ddof=0) * np.sqrt(periods_per_year))
        sharpe = float((numeric_returns.mean() / numeric_returns.std(ddof=0)) * np.sqrt(periods_per_year)) if numeric_returns.std(ddof=0) > 0 else float("nan")
    return {
        "total_return": total_return,
        "annualized_return": ann_return,
        "annualized_volatility": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": float(pd.to_numeric(drawdown, errors="coerce").min()),
        "average_turnover": float(pd.to_numeric(turnover, errors="coerce").mean()),
        "average_holdings": float(pd.to_numeric(holdings, errors="coerce").mean()),
        "rebalance_periods": int(len(numeric_returns)),
    }


def _validate_frozen_ml(candidate, artifact: dict[str, object]) -> None:
    if str(artifact.get("train_end_date")) > "2024-12-31":
        raise ValueError("Frozen ML artifact training window extends beyond 2024-12-31.")
    if str(artifact.get("validation_end_date")) > "2025-12-31":
        raise ValueError("Frozen ML artifact validation/model-selection window extends beyond 2025-12-31.")
    feature_names = list(artifact.get("feature_names", []))
    offending_snapshot = sorted(set(feature_names) & SNAPSHOT_FIELD_COLUMNS)
    if offending_snapshot:
        raise ValueError(f"Frozen ML feature matrix includes snapshot fields: {', '.join(offending_snapshot)}")
    offending_future = sorted(feature for feature in feature_names if feature.startswith("future_"))
    if offending_future:
        raise ValueError(f"Frozen ML feature matrix includes future-return columns: {', '.join(offending_future)}")
    if candidate.long_short:
        raise ValueError("Frozen ML candidate must remain long-only.")
    if candidate.use_regime_filter:
        raise ValueError("Frozen ML candidate must keep regime filter off.")
    if candidate.snapshot_fields_allowed:
        raise ValueError("Frozen ML candidate must keep snapshot fields disabled.")


def _load_validation_features(runtime) -> pd.DataFrame:
    preferred = runtime.final_dir / "features_panel_2023-01-01_2026-01-01.csv"
    fallback = runtime.final_dir / "features_panel.csv"
    path = preferred if preferred.exists() else fallback
    features = load_dataframe(path, parse_dates=["date"])
    features = ensure_market_features(runtime, features)
    features["date"] = pd.to_datetime(features["date"])
    return features


def _first_available_benchmark_date(features: pd.DataFrame, benchmark: str, start_date: pd.Timestamp) -> pd.Timestamp:
    spy_dates = (
        features.loc[
            (features["ticker"] == benchmark)
            & pd.to_datetime(features["date"]).ge(pd.Timestamp(start_date))
            & features["adjusted_close"].notna(),
            "date",
        ]
        .drop_duplicates()
        .sort_values()
    )
    if spy_dates.empty:
        raise ValueError(f"No {benchmark} adjusted_close data available on or after {pd.Timestamp(start_date).date()}.")
    return pd.Timestamp(spy_dates.iloc[0])


def _build_spy_direct_series(features: pd.DataFrame, benchmark: str, decision_dates: pd.Series, initial_capital: float = INITIAL_CAPITAL) -> tuple[pd.DataFrame, dict[str, float]]:
    spy_daily = (
        features.loc[features["ticker"] == benchmark, ["date", "adjusted_close"]]
        .dropna(subset=["adjusted_close"])
        .drop_duplicates(subset=["date"])
        .sort_values("date")
        .rename(columns={"adjusted_close": "spy_adjusted_close"})
        .reset_index(drop=True)
    )
    if spy_daily.empty:
        raise ValueError("No SPY adjusted_close data found for benchmark plotting.")

    out = pd.DataFrame({"date": pd.Series(pd.to_datetime(decision_dates)).sort_values().reset_index(drop=True)})
    out = pd.merge_asof(out, spy_daily, on="date", direction="backward")
    if out["spy_adjusted_close"].isna().any():
        raise ValueError("Unable to align SPY adjusted_close values to model dates.")
    start_close = float(out["spy_adjusted_close"].iloc[0])
    end_close = float(out["spy_adjusted_close"].iloc[-1])
    out["spy_value_direct"] = initial_capital * out["spy_adjusted_close"] / start_close
    out["spy_drawdown"] = compute_drawdown(out["spy_value_direct"])
    metrics = {
        "start_close": start_close,
        "end_close": end_close,
        "direct_return": end_close / start_close - 1.0,
        "final_value": float(out["spy_value_direct"].iloc[-1]),
    }
    return out, metrics


def _build_returns_frame(
    weekly: pd.DataFrame,
    features: pd.DataFrame,
    benchmark: str,
    label: str,
    window_start_date: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    first_trading_date = _first_available_benchmark_date(features, benchmark, window_start_date)
    plot_dates = pd.concat(
        [
            pd.Series([first_trading_date]),
            pd.Series(pd.to_datetime(weekly["period_end_date"])),
        ],
        ignore_index=True,
    ).drop_duplicates().sort_values().reset_index(drop=True)
    spy_df, spy_metrics = _build_spy_direct_series(features, benchmark, plot_dates)
    returns_df = weekly.loc[:, ["period_end_date", "net_return", "portfolio_value", "turnover", "transaction_cost", "selected_count", "exposure"]].copy()
    returns_df = returns_df.rename(
        columns={
            "period_end_date": "date",
            "net_return": "ml_period_return",
            "portfolio_value": "ml_value",
            "transaction_cost": "trading_cost",
        }
    ).sort_values("date")
    anchor_row = pd.DataFrame(
        [
            {
                "date": first_trading_date,
                "ml_period_return": 0.0,
                "ml_value": INITIAL_CAPITAL,
                "turnover": np.nan,
                "trading_cost": 0.0,
                "selected_count": np.nan,
                "exposure": np.nan,
            }
        ]
    )
    returns_df = pd.concat([anchor_row, returns_df], ignore_index=True).sort_values("date").reset_index(drop=True)
    returns_df = returns_df.merge(spy_df, on="date", how="left")
    returns_df["ml_drawdown"] = compute_drawdown(returns_df["ml_value"])
    validation_df = pd.DataFrame(
        [
            {
                "period_name": label,
                "start_date": first_trading_date,
                "end_date": pd.Timestamp(returns_df["date"].iloc[-1]),
                "direct_spy_return": spy_metrics["direct_return"],
                "plotted_spy_return": float(returns_df["spy_value_direct"].iloc[-1] / INITIAL_CAPITAL - 1.0),
                "absolute_difference": abs(float(returns_df["spy_value_direct"].iloc[-1] / INITIAL_CAPITAL - 1.0) - spy_metrics["direct_return"]),
                "direct_spy_final_value": spy_metrics["final_value"],
                "plotted_spy_final_value": float(returns_df["spy_value_direct"].iloc[-1]),
            }
        ]
    )
    if float(validation_df["absolute_difference"].iloc[0]) > 0.005:
        raise ValueError(f"{label}: plotted SPY differs from direct SPY buy-and-hold by more than 0.5%.")
    return returns_df, validation_df


def _make_two_line_equity_plot(df: pd.DataFrame, title: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 7), facecolor="white")
    ax.set_facecolor("white")
    ax.plot(df["date"], df["ml_value"], linewidth=2.2, label="Frozen ML Ranker", color="#7c3aed")
    ax.plot(df["date"], df["spy_value_direct"], linewidth=2.0, label="SPY Buy & Hold", color="#1f2937")
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of $10,000")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"${value:,.0f}"))
    ax.grid(True, alpha=0.25)
    ax.legend()
    _save_plot(fig, output_path)


def _make_two_line_drawdown_plot(df: pd.DataFrame, title: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 7), facecolor="white")
    ax.set_facecolor("white")
    ax.plot(df["date"], df["ml_drawdown"], linewidth=2.2, label="Frozen ML Ranker", color="#b91c1c")
    ax.plot(df["date"], df["spy_drawdown"], linewidth=2.0, label="SPY Buy & Hold", color="#1f2937")
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown %")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0%}"))
    ax.grid(True, alpha=0.25)
    ax.legend()
    _save_plot(fig, output_path)


def _make_combined_equity_plot(df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 7), facecolor="white")
    ax.set_facecolor("white")
    ax.plot(df["date"], df["ml_value"], linewidth=2.2, label="Frozen ML Ranker", color="#7c3aed")
    ax.plot(df["date"], df["spy_value_direct"], linewidth=2.0, label="SPY Buy & Hold", color="#1f2937")
    ax.axvline(FORWARD_BOUNDARY, color="#6b7280", linestyle="--", linewidth=1.2)
    y_top = max(float(df["ml_value"].max()), float(df["spy_value_direct"].max()))
    ax.text(pd.Timestamp("2025-06-15"), y_top * 0.97, "2025 validation", color="#374151", fontsize=10)
    ax.text(pd.Timestamp("2026-02-15"), y_top * 0.97, "2026 frozen forward", color="#374151", fontsize=10)
    ax.set_title("Frozen ML Ranker vs SPY Buy & Hold (2025 Validation + 2026 Forward)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of $10,000")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"${value:,.0f}"))
    ax.grid(True, alpha=0.25)
    ax.legend()
    _save_plot(fig, output_path)


def _make_combined_drawdown_plot(df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 7), facecolor="white")
    ax.set_facecolor("white")
    ax.plot(df["date"], df["ml_drawdown"], linewidth=2.2, label="Frozen ML Ranker", color="#b91c1c")
    ax.plot(df["date"], df["spy_drawdown"], linewidth=2.0, label="SPY Buy & Hold", color="#1f2937")
    ax.axvline(FORWARD_BOUNDARY, color="#6b7280", linestyle="--", linewidth=1.2)
    ax.set_title("Drawdown: Frozen ML Ranker vs SPY Buy & Hold (2025 Validation + 2026 Forward)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown %")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0%}"))
    ax.grid(True, alpha=0.25)
    ax.legend()
    _save_plot(fig, output_path)


def _make_forward_three_line_plot(ml_df: pd.DataFrame, rule_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 7), facecolor="white")
    ax.set_facecolor("white")
    ax.plot(ml_df["date"], ml_df["ml_value"], linewidth=2.2, label="Frozen ML Ranker", color="#7c3aed")
    ax.plot(rule_df["date"], rule_df["rule_value"], linewidth=2.0, label="Current Rule-Based Model", color="#0f766e")
    ax.plot(ml_df["date"], ml_df["spy_value_direct"], linewidth=2.0, label="SPY Buy & Hold", color="#1f2937")
    ax.set_title("Frozen ML Ranker vs Rule-Based Model vs SPY — 2026 Forward")
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of $10,000")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"${value:,.0f}"))
    ax.grid(True, alpha=0.25)
    ax.legend()
    _save_plot(fig, output_path)


def _latest_forward_actions(holdings: pd.DataFrame) -> tuple[list[str], list[str], list[str], list[str]]:
    latest_date = pd.Timestamp(holdings["date"].max())
    current = holdings.loc[holdings["date"] == latest_date].copy()
    decision_dates = sorted(pd.to_datetime(holdings["date"]).drop_duplicates())
    previous_date = pd.Timestamp(decision_dates[-2]) if len(decision_dates) >= 2 else pd.NaT
    previous = set(holdings.loc[holdings["date"] == previous_date, "ticker"].tolist()) if pd.notna(previous_date) else set()
    current_set = set(current["ticker"].tolist())
    buys = sorted(current_set - previous)
    sells = sorted(previous - current_set)
    holds = sorted(current_set & previous)
    return sorted(current_set), buys, sells, holds


def _write_summary(path: Path, heading: str, metrics: dict[str, float], spy_metrics: dict[str, float], extra_lines: list[str], validation_df: pd.DataFrame) -> None:
    lines = [
        heading,
        "",
        *[f"- {line}" for line in _plot_caveat_lines()],
        "",
        *extra_lines,
        "",
        "## Metrics",
        "",
        f"- ML total return: {metrics['total_return']:.2%}",
        f"- SPY total return: {spy_metrics['total_return']:.2%}",
        f"- Excess return vs SPY: {metrics['total_return'] - spy_metrics['total_return']:.2%}",
        f"- ML max drawdown: {metrics['max_drawdown']:.2%}",
        f"- SPY max drawdown: {spy_metrics['max_drawdown']:.2%}",
        f"- Sharpe: {metrics['sharpe']:.3f}" if pd.notna(metrics.get("sharpe")) else "- Sharpe: n/a",
        f"- Average turnover: {metrics['average_turnover']:.6f}",
        f"- Average holdings: {metrics['average_holdings']:.2f}",
        f"- Number of rebalance periods: {metrics['rebalance_periods']}",
        "",
        "## Benchmark Validation",
        "",
        dataframe_to_markdown(validation_df.round(6)),
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    runtime, candidate, artifact, forward_features = load_frozen_ml_context()
    _validate_frozen_ml(candidate, artifact)
    validation_features = _load_validation_features(runtime)

    validation_weekly, _, _, _ = run_frozen_ml_backtest_over_features(
        runtime,
        candidate,
        artifact,
        validation_features,
        start_date=VALIDATION_START,
        end_date=VALIDATION_END,
    )
    if validation_weekly.empty:
        raise ValueError("Validation 2025 backtest produced no rows.")
    validation_returns, validation_benchmark = _build_returns_frame(
        validation_weekly,
        validation_features,
        runtime.benchmark,
        "2025_validation",
        VALIDATION_START,
    )
    save_dataframe(runtime.tables_dir / "ml_vs_spy_2025_validation_returns.csv", validation_returns)
    save_dataframe(runtime.tables_dir / "ml_vs_spy_2025_benchmark_validation.csv", validation_benchmark)
    _make_two_line_equity_plot(validation_returns, "Frozen ML Ranker vs SPY — 2025 Validation", runtime.charts_dir / "ml_vs_spy_2025_validation_equity_curve.png")
    _make_two_line_drawdown_plot(validation_returns, "Drawdown: Frozen ML Ranker vs SPY — 2025 Validation", runtime.charts_dir / "ml_vs_spy_2025_validation_drawdown.png")
    validation_metrics = _series_stats(
        validation_weekly["net_return"],
        validation_returns["ml_value"],
        validation_returns["ml_drawdown"],
        validation_weekly["turnover"],
        validation_weekly["selected_count"],
        int(candidate.rebalance_frequency_days),
    )
    validation_spy_metrics = _series_stats(
        validation_returns["spy_value_direct"].pct_change().fillna(0.0),
        validation_returns["spy_value_direct"],
        validation_returns["spy_drawdown"],
        validation_returns["turnover"] * 0.0,
        validation_returns["selected_count"] * 0.0 + 1.0,
        int(candidate.rebalance_frequency_days),
    )
    _write_summary(
        runtime.reports_dir / "ml_vs_spy_2025_validation_summary.md",
        "# Frozen ML Ranker vs SPY — 2025 Validation Summary",
        validation_metrics,
        validation_spy_metrics,
        [
            f"- Date range: {pd.Timestamp(validation_returns['date'].min()).date()} to {pd.Timestamp(validation_returns['date'].max()).date()}",
            f"- Strategy: `{candidate.strategy_name}`",
            f"- Model loaded from disk: `{candidate.model_path}`",
        ],
        validation_benchmark,
    )

    forward_weekly, forward_holdings, _, _ = run_frozen_ml_forward(runtime, candidate, artifact, forward_features)
    if forward_weekly.empty:
        raise ValueError("Forward 2026 ML backtest produced no rows.")
    forward_returns, forward_benchmark = _build_returns_frame(
        forward_weekly,
        forward_features,
        runtime.benchmark,
        "2026_forward",
        pd.Timestamp(candidate.forward_window_start),
    )
    save_dataframe(runtime.tables_dir / "ml_vs_spy_2026_forward_returns.csv", forward_returns)
    save_dataframe(runtime.tables_dir / "ml_vs_spy_2026_benchmark_validation.csv", forward_benchmark)
    _make_two_line_equity_plot(forward_returns, "Frozen ML Ranker vs SPY — 2026 Forward", runtime.charts_dir / "ml_vs_spy_2026_forward_equity_curve.png")
    _make_two_line_drawdown_plot(forward_returns, "Drawdown: Frozen ML Ranker vs SPY — 2026 Forward", runtime.charts_dir / "ml_vs_spy_2026_forward_drawdown.png")
    forward_metrics = _series_stats(
        forward_weekly["net_return"],
        forward_returns["ml_value"],
        forward_returns["ml_drawdown"],
        forward_weekly["turnover"],
        forward_weekly["selected_count"],
        int(candidate.rebalance_frequency_days),
    )
    forward_spy_metrics = _series_stats(
        forward_returns["spy_value_direct"].pct_change().fillna(0.0),
        forward_returns["spy_value_direct"],
        forward_returns["spy_drawdown"],
        forward_returns["turnover"] * 0.0,
        forward_returns["selected_count"] * 0.0 + 1.0,
        int(candidate.rebalance_frequency_days),
    )
    current_holdings, latest_buys, latest_sells, latest_holds = _latest_forward_actions(forward_holdings)
    _write_summary(
        runtime.reports_dir / "ml_vs_spy_2026_forward_summary.md",
        "# Frozen ML Ranker vs SPY — 2026 Forward Summary",
        forward_metrics,
        forward_spy_metrics,
        [
            f"- Forward start date: {pd.Timestamp(forward_returns['date'].min()).date()}",
            f"- Latest available date: {pd.Timestamp(forward_returns['date'].max()).date()}",
            f"- Strategy: `{candidate.strategy_name}`",
            f"- Model loaded from disk: `{candidate.model_path}`",
            f"- Trading costs: {float(pd.to_numeric(forward_returns['trading_cost'], errors='coerce').sum()):.4f}",
            f"- Current ML holdings: {', '.join(current_holdings) or 'none'}",
            f"- Latest buys: {', '.join(latest_buys) or 'none'}",
            f"- Latest sells: {', '.join(latest_sells) or 'none'}",
            f"- Latest holds: {', '.join(latest_holds) or 'none'}",
        ],
        forward_benchmark,
    )

    combined = pd.concat(
        [
            validation_returns.assign(period_name="2025_validation"),
            forward_returns.assign(period_name="2026_forward"),
        ],
        ignore_index=True,
    ).sort_values("date").reset_index(drop=True)
    combined["ml_value"] = INITIAL_CAPITAL * (1.0 + combined["ml_period_return"]).cumprod()
    combined["ml_drawdown"] = compute_drawdown(combined["ml_value"])
    combined_features = pd.concat([validation_features, forward_features], ignore_index=True).drop_duplicates(subset=["date", "ticker"], keep="last")
    combined_spy, combined_spy_info = _build_spy_direct_series(combined_features, runtime.benchmark, combined["date"])
    combined = combined.drop(columns=["spy_adjusted_close", "spy_value_direct", "spy_drawdown"], errors="ignore").merge(combined_spy, on="date", how="left")
    combined_validation = pd.DataFrame(
        [
            {
                "period_name": "2025_2026_combined",
                "start_date": pd.Timestamp(combined["date"].iloc[0]),
                "end_date": pd.Timestamp(combined["date"].iloc[-1]),
                "direct_spy_return": combined_spy_info["direct_return"],
                "plotted_spy_return": float(combined["spy_value_direct"].iloc[-1] / INITIAL_CAPITAL - 1.0),
                "absolute_difference": abs(float(combined["spy_value_direct"].iloc[-1] / INITIAL_CAPITAL - 1.0) - combined_spy_info["direct_return"]),
                "direct_spy_final_value": combined_spy_info["final_value"],
                "plotted_spy_final_value": float(combined["spy_value_direct"].iloc[-1]),
            }
        ]
    )
    if float(combined_validation["absolute_difference"].iloc[0]) > 0.005:
        raise ValueError("Combined period plotted SPY differs from direct SPY buy-and-hold by more than 0.5%.")
    save_dataframe(runtime.tables_dir / "ml_vs_spy_2025_2026_combined_returns.csv", combined)
    save_dataframe(runtime.tables_dir / "ml_vs_spy_combined_benchmark_validation.csv", combined_validation)
    _make_combined_equity_plot(combined, runtime.charts_dir / "ml_vs_spy_2025_2026_combined_equity_curve.png")
    _make_combined_drawdown_plot(combined, runtime.charts_dir / "ml_vs_spy_2025_2026_combined_drawdown.png")
    combined_turnover = pd.concat([validation_weekly["turnover"], forward_weekly["turnover"]], ignore_index=True)
    combined_selected_count = pd.concat([validation_weekly["selected_count"], forward_weekly["selected_count"]], ignore_index=True)
    combined_period_returns = pd.concat([validation_weekly["net_return"], forward_weekly["net_return"]], ignore_index=True)
    combined_metrics = _series_stats(
        combined_period_returns,
        combined["ml_value"],
        combined["ml_drawdown"],
        combined_turnover,
        combined_selected_count,
        int(candidate.rebalance_frequency_days),
    )
    combined_spy_metrics = _series_stats(
        combined["spy_value_direct"].pct_change().fillna(0.0),
        combined["spy_value_direct"],
        combined["spy_drawdown"],
        combined["turnover"] * 0.0,
        combined["selected_count"] * 0.0 + 1.0,
        int(candidate.rebalance_frequency_days),
    )
    _write_summary(
        runtime.reports_dir / "ml_vs_spy_2025_2026_combined_summary.md",
        "# Frozen ML Ranker vs SPY — 2025 Validation + 2026 Forward Summary",
        combined_metrics,
        combined_spy_metrics,
        [
            f"- Date range: {pd.Timestamp(combined['date'].min()).date()} to {pd.Timestamp(combined['date'].max()).date()}",
            "- Vertical split at 2026-01-01 marks the start of frozen forward monitoring.",
            "- 2026 was not used for training, tuning, or model selection.",
        ],
        combined_validation,
    )

    rule_returns_path = runtime.tables_dir / "forward_2026_model_vs_spy_returns.csv"
    if rule_returns_path.exists():
        rule_df = load_dataframe(rule_returns_path, parse_dates=["date"]).rename(columns={"model_value": "rule_value"})
        if not rule_df.empty:
            merged_forward = forward_returns.merge(rule_df[["date", "rule_value"]], on="date", how="left")
            _make_forward_three_line_plot(merged_forward, merged_forward, runtime.charts_dir / "ml_vs_rule_vs_spy_2026_forward_equity_curve.png")

    print(f"Saved {runtime.charts_dir / 'ml_vs_spy_2025_validation_equity_curve.png'}")
    print(f"Saved {runtime.charts_dir / 'ml_vs_spy_2025_validation_drawdown.png'}")
    print(f"Saved {runtime.tables_dir / 'ml_vs_spy_2025_validation_returns.csv'}")
    print(f"Saved {runtime.reports_dir / 'ml_vs_spy_2025_validation_summary.md'}")
    print(f"Saved {runtime.charts_dir / 'ml_vs_spy_2026_forward_equity_curve.png'}")
    print(f"Saved {runtime.charts_dir / 'ml_vs_spy_2026_forward_drawdown.png'}")
    print(f"Saved {runtime.tables_dir / 'ml_vs_spy_2026_forward_returns.csv'}")
    print(f"Saved {runtime.reports_dir / 'ml_vs_spy_2026_forward_summary.md'}")
    print(f"Saved {runtime.charts_dir / 'ml_vs_spy_2025_2026_combined_equity_curve.png'}")
    print(f"Saved {runtime.charts_dir / 'ml_vs_spy_2025_2026_combined_drawdown.png'}")
    print(f"Saved {runtime.tables_dir / 'ml_vs_spy_2025_2026_combined_returns.csv'}")
    print(f"Saved {runtime.reports_dir / 'ml_vs_spy_2025_2026_combined_summary.md'}")


if __name__ == "__main__":
    main()
