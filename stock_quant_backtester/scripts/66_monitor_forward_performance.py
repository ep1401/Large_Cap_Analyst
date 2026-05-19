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

from src.config import Config
from src.no_snapshot_research import WALK_FORWARD_WINDOWS, dataframe_to_markdown, fmt_pct
from src.recommended_strategy import (
    load_recommended_strategy_config,
    precompute_recommended_low_turnover_panels,
)
from src.scoring import NO_SNAPSHOT_STRATEGIES, SNAPSHOT_FIELD_COLUMNS, strategy_score_fields
from src.utils import load_dataframe, save_dataframe


EXPECTED_STRATEGY = "final_quant_5d_weight_tuned_low_turnover_no_snapshot"
FORWARD_START = pd.Timestamp("2026-01-01")
INITIAL_CAPITAL = 10000.0


def _monitor_caveats() -> list[str]:
    return [
        "This is a frozen forward/out-of-sample monitoring report.",
        "No 2026 data should be used to retune the model.",
        "Current forward sample is short.",
        "Back-tested performance is hypothetical unless trades were actually paper-tracked live.",
        "This is research/paper trading only, not financial advice.",
    ]


def _validate_recommended_strategy(runtime: Config):
    recommended = load_recommended_strategy_config(runtime.project_root)
    if recommended.strategy_name != EXPECTED_STRATEGY:
        raise ValueError(f"recommended_strategy.yaml must remain locked to {EXPECTED_STRATEGY}; found {recommended.strategy_name}")
    if recommended.strategy_name not in NO_SNAPSHOT_STRATEGIES:
        raise ValueError("Recommended strategy must remain a no-snapshot strategy.")
    if recommended.long_short:
        raise ValueError("Recommended strategy must remain long-only.")
    if recommended.regime_filter != "none":
        raise ValueError("Recommended strategy must keep regime_filter=none.")
    offending = sorted(strategy_score_fields(recommended.strategy_name) & SNAPSHOT_FIELD_COLUMNS)
    if offending:
        raise ValueError(f"Recommended strategy uses snapshot fields: {', '.join(offending)}")
    return recommended


def _compute_drawdown(values: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    return values / values.cummax() - 1.0


def _save_plot(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def _make_equity_curve_plot(monitor_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 7), facecolor="white")
    ax.set_facecolor("white")
    ax.plot(monitor_df["date"], monitor_df["model_value"], linewidth=2.2, label="Frozen Recommended Model", color="#0f766e")
    ax.plot(monitor_df["date"], monitor_df["spy_value"], linewidth=2.0, label="SPY Buy & Hold", color="#1f2937")
    ax.set_title("Frozen Forward Performance: Model vs SPY")
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of $10,000")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"${value:,.0f}"))
    ax.grid(True, alpha=0.25)
    ax.legend()
    _save_plot(fig, output_path)


def _make_excess_return_plot(monitor_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 7), facecolor="white")
    ax.set_facecolor("white")
    ax.plot(monitor_df["date"], monitor_df["cumulative_excess_return"], linewidth=2.2, color="#7c3aed")
    ax.axhline(0.0, color="#1f2937", linewidth=1.0, alpha=0.6)
    ax.set_title("Frozen Forward Performance: Cumulative Excess Return vs SPY")
    ax.set_xlabel("Date")
    ax.set_ylabel("Excess Return")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0%}"))
    ax.grid(True, alpha=0.25)
    _save_plot(fig, output_path)


def _make_drawdown_plot(monitor_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 7), facecolor="white")
    ax.set_facecolor("white")
    ax.plot(monitor_df["date"], monitor_df["model_drawdown"], linewidth=2.2, label="Frozen Recommended Model", color="#b91c1c")
    ax.plot(monitor_df["date"], monitor_df["spy_drawdown"], linewidth=2.0, label="SPY Buy & Hold", color="#1f2937")
    ax.set_title("Frozen Forward Performance: Drawdown")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown %")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0%}"))
    ax.grid(True, alpha=0.25)
    ax.legend()
    _save_plot(fig, output_path)


def _build_monitor_table(returns_df: pd.DataFrame) -> pd.DataFrame:
    monitor_df = returns_df.copy()
    monitor_df["model_cumulative_return"] = monitor_df["model_value"] / INITIAL_CAPITAL - 1.0
    monitor_df["spy_cumulative_return"] = monitor_df["spy_value_direct"] / INITIAL_CAPITAL - 1.0
    monitor_df["cumulative_excess_return"] = monitor_df["model_cumulative_return"] - monitor_df["spy_cumulative_return"]
    monitor_df = monitor_df.rename(
        columns={
            "model_period_return": "model_return",
            "spy_value_direct": "spy_value",
            "plotted_spy_period_return": "spy_return",
            "trading_cost": "estimated_trading_cost",
        }
    )
    return monitor_df


def _checkpoint_metrics(monitor_df: pd.DataFrame, start_date: pd.Timestamp) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    latest_date = pd.Timestamp(monitor_df["date"].max())
    for months in (3, 6, 9, 12):
        cutoff = start_date + pd.DateOffset(months=months)
        sliced = monitor_df.loc[monitor_df["date"] <= cutoff].copy()
        reached = latest_date >= cutoff and not sliced.empty
        if sliced.empty:
            rows.append(
                {
                    "checkpoint": f"{months}_months",
                    "checkpoint_date": cutoff.normalize(),
                    "reached": False,
                    "model_return": np.nan,
                    "spy_return": np.nan,
                    "excess_return": np.nan,
                    "model_drawdown": np.nan,
                    "periods_beating_spy": np.nan,
                }
            )
            continue
        rows.append(
            {
                "checkpoint": f"{months}_months",
                "checkpoint_date": cutoff.normalize(),
                "reached": reached,
                "model_return": float(sliced["model_value"].iloc[-1] / INITIAL_CAPITAL - 1.0),
                "spy_return": float(sliced["spy_value"].iloc[-1] / INITIAL_CAPITAL - 1.0),
                "excess_return": float(sliced["cumulative_excess_return"].iloc[-1]),
                "model_drawdown": float(sliced["model_drawdown"].min()),
                "periods_beating_spy": int((pd.to_numeric(sliced["model_return"], errors="coerce") > pd.to_numeric(sliced["spy_return"], errors="coerce")).sum()),
            }
        )
    return pd.DataFrame(rows)


def _window_drawdown_range(weekly_df: pd.DataFrame) -> tuple[float, float]:
    window_drawdowns: list[float] = []
    for _, start, end in WALK_FORWARD_WINDOWS:
        sliced = weekly_df.loc[(weekly_df["date"] >= pd.Timestamp(start)) & (weekly_df["date"] <= pd.Timestamp(end))].copy()
        if sliced.empty:
            continue
        window_values = pd.to_numeric(sliced["portfolio_value"], errors="coerce")
        if window_values.empty:
            continue
        window_drawdowns.append(float(_compute_drawdown(window_values).min()))
    if not window_drawdowns:
        return float("nan"), float("nan")
    return float(min(window_drawdowns)), float(max(window_drawdowns))


def _build_historical_expectation(runtime: Config, recommended) -> dict[str, float]:
    historical_path = runtime.final_dir / "features_panel_2023-01-01_2026-01-01.csv"
    if not historical_path.exists():
        historical_path = runtime.final_dir / "features_panel.csv"
    if not historical_path.exists():
        return {
            "historical_average_excess_per_period": float("nan"),
            "historical_drawdown_min": float("nan"),
            "historical_drawdown_max": float("nan"),
        }
    historical_features = load_dataframe(historical_path, parse_dates=["date"])
    historical_features = historical_features.loc[historical_features["date"] < FORWARD_START].copy()
    if historical_features.empty:
        return {
            "historical_average_excess_per_period": float("nan"),
            "historical_drawdown_min": float("nan"),
            "historical_drawdown_max": float("nan"),
        }
    panels = precompute_recommended_low_turnover_panels(historical_features, runtime, recommended)
    historical_weekly, _, _ = _simulate_with_attribution(
        panels=panels,
        top_n=recommended.top_n,
        cost_bps=float(recommended.total_cost_bps),
        enter_rank=int(recommended.enter_rank or recommended.top_n),
        hold_rank=int(recommended.hold_rank or recommended.top_n),
        max_holding_days=int(recommended.max_holding_days or recommended.holding_period_days),
        rebalance_frequency_days=int(recommended.rebalance_frequency_days or recommended.holding_period_days),
        strategy_name=recommended.strategy_name,
        max_turnover_per_rebalance=recommended.max_turnover_per_rebalance,
    )
    if historical_weekly.empty:
        return {
            "historical_average_excess_per_period": float("nan"),
            "historical_drawdown_min": float("nan"),
            "historical_drawdown_max": float("nan"),
        }
    dd_min, dd_max = _window_drawdown_range(historical_weekly)
    return {
        "historical_average_excess_per_period": float(pd.to_numeric(historical_weekly["excess_return"], errors="coerce").mean()),
        "historical_drawdown_min": dd_min,
        "historical_drawdown_max": dd_max,
    }


def _simulate_with_attribution(
    panels: list[tuple[pd.Timestamp, pd.DataFrame, float, pd.DataFrame]],
    top_n: int,
    cost_bps: float,
    enter_rank: int,
    hold_rank: int,
    max_holding_days: int,
    rebalance_frequency_days: int,
    strategy_name: str,
    max_turnover_per_rebalance: float | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    holdings: dict[str, dict[str, object]] = {}
    weekly_rows: list[dict[str, object]] = []
    holding_rows: list[dict[str, object]] = []
    attribution_rows: list[dict[str, object]] = []
    portfolio_value = INITIAL_CAPITAL
    spy_value = INITIAL_CAPITAL

    if len(panels) < 2:
        return pd.DataFrame(weekly_rows), pd.DataFrame(holding_rows), pd.DataFrame(attribution_rows)

    for idx in range(len(panels) - 1):
        rebalance_date, panel, spy_price, _ = panels[idx]
        prior_weights = {ticker: float(meta.get("weight", 0.0)) for ticker, meta in holdings.items()}
        ranked = panel.sort_values("score", ascending=False).reset_index(drop=True).copy()
        ranked["rank"] = np.arange(1, len(ranked) + 1, dtype=int)
        ranked_by_ticker = ranked.set_index("ticker", drop=False)
        next_panel_by_ticker = panels[idx + 1][3].set_index("ticker", drop=False)
        next_spy_price = float(panels[idx + 1][2])

        forced_sells: list[tuple[str, str]] = []
        discretionary_sells: list[tuple[str, int]] = []
        for ticker, meta in list(holdings.items()):
            if ticker not in ranked_by_ticker.index:
                forced_sells.append((ticker, "price_data_missing"))
                continue
            row = ranked_by_ticker.loc[ticker]
            rank = int(row["rank"])
            if bool(row["strong_negative_news_flag"]):
                forced_sells.append((ticker, "strong_negative_news_flag"))
            elif bool(row["recent_downgrade_flag_30d"]):
                forced_sells.append((ticker, "recent_downgrade_flag_30d"))
            elif int(meta["holding_days"]) >= max_holding_days:
                forced_sells.append((ticker, "max_holding_days"))
            elif rank > hold_rank:
                discretionary_sells.append((ticker, rank))

        for ticker, _ in forced_sells:
            holdings.pop(ticker, None)

        current_after_forced = list(holdings.keys())
        desired_buys = [
            ticker
            for ticker in ranked.loc[ranked["rank"] <= enter_rank, "ticker"].tolist()
            if ticker not in current_after_forced
        ]
        turnover_used = len(forced_sells) * (1.0 / top_n)
        turnover_budget = float("inf") if max_turnover_per_rebalance is None else max(0.0, float(max_turnover_per_rebalance) - turnover_used)
        allowed_sells = len(discretionary_sells)
        allowed_buys = len(desired_buys)
        if max_turnover_per_rebalance is not None:
            unit = 1.0 / top_n
            max_steps = int(turnover_budget / unit + 1e-9)
            allowed_sells = min(len(discretionary_sells), max_steps)
            remaining_steps = max(0, max_steps - allowed_sells)
            allowed_buys = min(len(desired_buys), remaining_steps)

        discretionary_sells = sorted(discretionary_sells, key=lambda item: item[1], reverse=True)
        for ticker, _ in discretionary_sells[:allowed_sells]:
            holdings.pop(ticker, None)

        current_tickers = list(holdings.keys())
        open_slots = max(0, top_n - len(current_tickers))
        desired_buys = [
            ticker
            for ticker in ranked.loc[ranked["rank"] <= enter_rank, "ticker"].tolist()
            if ticker not in current_tickers
        ]
        executed_buys = desired_buys[: min(open_slots, allowed_buys)]
        for ticker in executed_buys:
            holdings[ticker] = {
                "holding_days": 0,
                "entry_date": rebalance_date,
                "action": "BUY",
            }

        selected_tickers = [ticker for ticker in ranked["ticker"].tolist() if ticker in holdings][:top_n]
        for ticker in list(holdings):
            if ticker not in selected_tickers:
                holdings.pop(ticker, None)

        selected = ranked.loc[ranked["ticker"].isin(selected_tickers)].copy().sort_values("rank")
        selected_count = len(selected)
        if selected_count > 0:
            weight = 1.0 / selected_count
            selected["weight"] = weight
            new_weights = dict(zip(selected["ticker"], selected["weight"]))
        else:
            selected["weight"] = pd.Series(dtype=float)
            new_weights = {}

        turnover = sum(abs(new_weights.get(ticker, 0.0) - prior_weights.get(ticker, 0.0)) for ticker in set(new_weights) | set(prior_weights))
        transaction_cost = turnover * cost_bps / 10000.0
        spy_return = next_spy_price / float(spy_price) - 1 if next_spy_price is not None and spy_price else 0.0

        gross_return = 0.0
        if selected_count:
            for _, row in selected.iterrows():
                if row["ticker"] in next_panel_by_ticker.index:
                    next_row = next_panel_by_ticker.loc[row["ticker"]]
                    current_price = float(row["adjusted_close"])
                    next_price = float(next_row["adjusted_close"])
                    realized_return = next_price / current_price - 1 if current_price else 0.0
                else:
                    realized_return = float(row["future_return_used"])
                contribution = float(row["weight"]) * realized_return
                excess_contribution = float(row["weight"]) * (realized_return - spy_return)
                gross_return += contribution
                attribution_rows.append(
                    {
                        "date": rebalance_date,
                        "ticker": row["ticker"],
                        "weight": float(row["weight"]),
                        "realized_return_while_held": realized_return,
                        "period_spy_return": spy_return,
                        "total_contribution": contribution,
                        "contribution_to_excess_return": excess_contribution,
                    }
                )

        net_return = gross_return - transaction_cost
        portfolio_value *= 1 + net_return
        spy_value *= 1 + spy_return

        for ticker in list(holdings):
            row = ranked_by_ticker.loc[ticker]
            holdings[ticker]["holding_days"] = int(holdings[ticker].get("holding_days", 0) + rebalance_frequency_days)
            holdings[ticker]["weight"] = float(new_weights.get(ticker, 0.0))
            holdings[ticker]["last_rank"] = int(row["rank"])
            holdings[ticker]["last_score"] = float(row["score"])
            holding_rows.append(
                {
                    "date": rebalance_date,
                    "strategy_name": strategy_name,
                    "ticker": ticker,
                    "rank": int(row["rank"]),
                    "score": float(row["score"]),
                    "weight": float(new_weights.get(ticker, 0.0)),
                    "holding_days": int(holdings[ticker]["holding_days"]),
                }
            )

        weekly_rows.append(
            {
                "date": rebalance_date,
                "strategy_name": strategy_name,
                "holding_period_days": 5,
                "top_n": top_n,
                "selected_count": selected_count,
                "gross_return": gross_return,
                "turnover": turnover,
                "transaction_cost": transaction_cost,
                "net_return": net_return,
                "spy_return": spy_return,
                "excess_return": net_return - spy_return,
                "portfolio_value": portfolio_value,
                "spy_value": spy_value,
                "exposure": float(selected["weight"].sum()) if selected_count else 0.0,
            }
        )

    return pd.DataFrame(weekly_rows), pd.DataFrame(holding_rows), pd.DataFrame(attribution_rows)


def _summarize_attribution(attribution_df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    if attribution_df.empty:
        return pd.DataFrame(
            columns=[
                "ticker",
                "total_contribution",
                "average_weight",
                "number_of_periods_held",
                "average_return_while_held",
                "contribution_to_excess_return",
            ]
        ), "Attribution unavailable because no forward holdings were present."
    summary = (
        attribution_df.groupby("ticker", as_index=False)
        .agg(
            total_contribution=("total_contribution", "sum"),
            average_weight=("weight", "mean"),
            number_of_periods_held=("date", "count"),
            average_return_while_held=("realized_return_while_held", "mean"),
            contribution_to_excess_return=("contribution_to_excess_return", "sum"),
        )
        .sort_values("contribution_to_excess_return", ascending=False)
        .reset_index(drop=True)
    )

    negative_total = float(summary.loc[summary["contribution_to_excess_return"] < 0, "contribution_to_excess_return"].abs().sum())
    bottom_five = summary.sort_values("contribution_to_excess_return").head(5)
    bottom_five_share = (
        float(bottom_five["contribution_to_excess_return"].abs().sum() / negative_total)
        if negative_total > 0
        else float("nan")
    )
    if pd.notna(bottom_five_share) and bottom_five_share >= 0.65:
        narrative = "Underperformance was concentrated in a few bad names rather than spread across the whole book."
    else:
        narrative = "Underperformance looked broad-based rather than coming from only a few names."
    return summary, narrative


def main() -> None:
    runtime = Config.from_env()
    recommended = _validate_recommended_strategy(runtime)

    features_path = runtime.final_dir / "features_panel_2026_forward.csv"
    forward_returns_path = runtime.tables_dir / "forward_2026_model_vs_spy_returns.csv"
    current_recs_path = runtime.tables_dir / "current_recommendations_2026_forward.csv"
    paper_history_path = runtime.project_root / "data" / "paper_trading" / "performance_history.csv"

    if not features_path.exists():
        raise FileNotFoundError(f"Missing forward feature panel: {features_path}")
    if not forward_returns_path.exists():
        raise FileNotFoundError(f"Missing forward returns table: {forward_returns_path}")

    features_forward = load_dataframe(features_path, parse_dates=["date"])
    returns_df = load_dataframe(forward_returns_path, parse_dates=["date"])
    current_recs = load_dataframe(current_recs_path, parse_dates=["date"]) if current_recs_path.exists() else pd.DataFrame()
    paper_history = load_dataframe(paper_history_path, parse_dates=["date"]) if paper_history_path.exists() else pd.DataFrame()

    monitor_df = _build_monitor_table(returns_df)
    save_dataframe(runtime.tables_dir / "forward_performance_monitor.csv", monitor_df)

    start_date = pd.Timestamp(monitor_df["date"].min())
    latest_date = pd.Timestamp(monitor_df["date"].max())
    checkpoint_df = _checkpoint_metrics(monitor_df, start_date)

    panels = precompute_recommended_low_turnover_panels(features_forward, runtime, recommended)
    weekly_df, holding_history_df, attribution_period_df = _simulate_with_attribution(
        panels=panels,
        top_n=int(recommended.top_n),
        cost_bps=float(recommended.total_cost_bps),
        enter_rank=int(recommended.enter_rank or recommended.top_n),
        hold_rank=int(recommended.hold_rank or recommended.top_n),
        max_holding_days=int(recommended.max_holding_days or recommended.holding_period_days),
        rebalance_frequency_days=int(recommended.rebalance_frequency_days or recommended.holding_period_days),
        strategy_name=recommended.strategy_name,
        max_turnover_per_rebalance=recommended.max_turnover_per_rebalance,
    )
    attribution_summary_df, attribution_narrative = _summarize_attribution(attribution_period_df)
    save_dataframe(runtime.tables_dir / "forward_underperformance_attribution.csv", attribution_summary_df)

    _make_equity_curve_plot(monitor_df, runtime.charts_dir / "forward_performance_equity_curve.png")
    _make_excess_return_plot(monitor_df, runtime.charts_dir / "forward_performance_excess_return.png")
    _make_drawdown_plot(monitor_df, runtime.charts_dir / "forward_performance_drawdown.png")

    top_contributors = attribution_summary_df.head(5).copy()
    bottom_contributors = attribution_summary_df.sort_values("contribution_to_excess_return").head(5).copy()
    historical_expectation = _build_historical_expectation(runtime, recommended)
    historical_avg_excess = historical_expectation["historical_average_excess_per_period"]
    forward_avg_excess = float(pd.to_numeric(weekly_df["excess_return"], errors="coerce").mean()) if not weekly_df.empty else float("nan")
    current_model_drawdown = float(monitor_df["model_drawdown"].min()) if not monitor_df.empty else float("nan")

    latest_buys = current_recs.loc[current_recs["action"] == "BUY", "ticker"].tolist() if not current_recs.empty else []
    latest_sells = []
    latest_holds = current_recs.loc[current_recs["action"] == "HOLD", "ticker"].tolist() if not current_recs.empty else []
    if not holding_history_df.empty:
        decision_dates = sorted(pd.to_datetime(holding_history_df["date"]).drop_duplicates())
        latest_decision_date = pd.Timestamp(decision_dates[-1])
        previous_decision_date = pd.Timestamp(decision_dates[-2]) if len(decision_dates) >= 2 else pd.NaT
        previous_holdings = (
            set(holding_history_df.loc[holding_history_df["date"] == previous_decision_date, "ticker"].tolist())
            if pd.notna(previous_decision_date)
            else set()
        )
        current_holdings = set(current_recs["ticker"].tolist()) if not current_recs.empty else set()
        latest_sells = sorted(previous_holdings - current_holdings)
    else:
        latest_decision_date = pd.NaT

    months_observed = (latest_date - start_date).days / 30.44 if pd.notna(latest_date) and pd.notna(start_date) else 0.0
    sample_note = "Forward sample is still short and should not be used for retuning." if months_observed < 6 else "Forward sample now spans at least six months, but the model should still remain frozen until 12 months are available."

    report_lines = [
        "# Forward Performance Monitor",
        "",
        *[f"- {line}" for line in _monitor_caveats()],
        "",
        "- Strategy name: `final_quant_5d_weight_tuned_low_turnover_no_snapshot`",
        "- Base score model: `final_quant_5d_weight_tuned_no_snapshot`",
        f"- Forward start date: {start_date.date()}",
        f"- Latest date: {latest_date.date()}",
        f"- Latest feature date: {pd.Timestamp(features_forward['date'].max()).date()}",
        f"- Sample note: {sample_note}",
        "",
        "## Current Metrics",
        "",
        f"- Model return: {fmt_pct(float(monitor_df['model_cumulative_return'].iloc[-1]))}",
        f"- SPY return: {fmt_pct(float(monitor_df['spy_cumulative_return'].iloc[-1]))}",
        f"- Excess return: {fmt_pct(float(monitor_df['cumulative_excess_return'].iloc[-1]))}",
        f"- Model max drawdown: {fmt_pct(float(monitor_df['model_drawdown'].min()))}",
        f"- SPY max drawdown: {fmt_pct(float(monitor_df['spy_drawdown'].min()))}",
        f"- Number of rebalance periods: {int(len(monitor_df))}",
        f"- Average turnover: {float(pd.to_numeric(monitor_df['turnover'], errors='coerce').mean()):.4f}",
        f"- Estimated trading costs: {float(pd.to_numeric(monitor_df['estimated_trading_cost'], errors='coerce').sum()):.4f}",
        f"- Average holdings: {float(pd.to_numeric(monitor_df['selected_count'], errors='coerce').mean()):.2f}",
        "",
        "## Current Book",
        "",
        f"- Current holdings: {', '.join(current_recs['ticker'].tolist()) if not current_recs.empty else 'n/a'}",
        f"- Latest buys: {', '.join(latest_buys) if latest_buys else 'none'}",
        f"- Latest sells: {', '.join(latest_sells) if latest_sells else 'none'}",
        f"- Latest holds: {', '.join(latest_holds) if latest_holds else 'none'}",
        "",
        "## Checkpoints",
        "",
        dataframe_to_markdown(
            checkpoint_df.assign(
                model_return=checkpoint_df["model_return"].map(fmt_pct),
                spy_return=checkpoint_df["spy_return"].map(fmt_pct),
                excess_return=checkpoint_df["excess_return"].map(fmt_pct),
                model_drawdown=checkpoint_df["model_drawdown"].map(fmt_pct),
            )
        ),
        "",
        "## Paper Trading Decision Rules",
        "",
        "- Do not retune until at least 12 months of forward data exist.",
        "- If the model is still behind SPY after 12 months and has worse drawdown, mark the strategy as a failed forward test.",
        "- If the model beats SPY after 12 months with similar or better drawdown, continue paper trading.",
        "- If the model beats SPY by more than 5 percentage points after 12 months with acceptable drawdown and realistic costs, consider deeper live-trading due diligence, not immediate real-money deployment.",
        "",
        "## Underperformance Attribution",
        "",
        f"- Attribution read: {attribution_narrative}",
        "- Top 5 contributors:",
        dataframe_to_markdown(
            top_contributors.assign(
                total_contribution=top_contributors["total_contribution"].map(fmt_pct),
                average_weight=top_contributors["average_weight"].map(fmt_pct),
                average_return_while_held=top_contributors["average_return_while_held"].map(fmt_pct),
                contribution_to_excess_return=top_contributors["contribution_to_excess_return"].map(fmt_pct),
            )
        ),
        "",
        "- Bottom 5 contributors:",
        dataframe_to_markdown(
            bottom_contributors.assign(
                total_contribution=bottom_contributors["total_contribution"].map(fmt_pct),
                average_weight=bottom_contributors["average_weight"].map(fmt_pct),
                average_return_while_held=bottom_contributors["average_return_while_held"].map(fmt_pct),
                contribution_to_excess_return=bottom_contributors["contribution_to_excess_return"].map(fmt_pct),
            )
        ),
        "",
        "## Historical Expectation Comparison",
        "",
        f"- Historical average excess per rebalance period: {fmt_pct(historical_avg_excess)}",
        f"- 2026 excess per rebalance period so far: {fmt_pct(forward_avg_excess)}",
        f"- Historical drawdown range across walk-forward windows: {fmt_pct(historical_expectation['historical_drawdown_min'])} to {fmt_pct(historical_expectation['historical_drawdown_max'])}",
        f"- 2026 drawdown so far: {fmt_pct(current_model_drawdown)}",
        "",
        "## Paper Tracking State",
        "",
        f"- Paper-trading history file present: {not paper_history.empty}",
    ]

    if not paper_history.empty:
        latest_live = paper_history.sort_values("date").iloc[-1]
        report_lines.extend(
            [
                f"- Latest paper-tracking date: {pd.Timestamp(latest_live['date']).date()}",
                f"- Live model value: ${float(latest_live['live_model_value']):,.2f}",
                f"- Live SPY value: ${float(latest_live['live_spy_value']):,.2f}",
                f"- Note: {latest_live['note']}",
            ]
        )

    report_lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- Monitor table: `outputs/tables/forward_performance_monitor.csv`",
            "- Attribution table: `outputs/tables/forward_underperformance_attribution.csv`",
            "- Equity curve: `outputs/charts/forward_performance_equity_curve.png`",
            "- Excess return chart: `outputs/charts/forward_performance_excess_return.png`",
            "- Drawdown chart: `outputs/charts/forward_performance_drawdown.png`",
        ]
    )

    (runtime.reports_dir / "forward_performance_monitor.md").write_text("\n".join(report_lines), encoding="utf-8")

    print(f"Saved {runtime.tables_dir / 'forward_performance_monitor.csv'}")
    print(f"Saved {runtime.tables_dir / 'forward_underperformance_attribution.csv'}")
    print(f"Saved {runtime.charts_dir / 'forward_performance_equity_curve.png'}")
    print(f"Saved {runtime.charts_dir / 'forward_performance_excess_return.png'}")
    print(f"Saved {runtime.charts_dir / 'forward_performance_drawdown.png'}")
    print(f"Saved {runtime.reports_dir / 'forward_performance_monitor.md'}")


if __name__ == "__main__":
    main()
