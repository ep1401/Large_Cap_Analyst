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
from src.no_snapshot_research import dataframe_to_markdown
from src.recommended_strategy import load_recommended_strategy_config
from src.research_models import (
    load_ml_artifact,
    load_ml_research_candidate_config,
    precompute_research_panels,
)
from src.scoring import NO_SNAPSHOT_STRATEGIES, SNAPSHOT_FIELD_COLUMNS, strategy_display_name, strategy_score_fields
from src.utils import load_dataframe, save_dataframe


INITIAL_CAPITAL = 10000.0
FORWARD_START = pd.Timestamp("2026-01-01")
EXPECTED_RULE_BASED = "final_quant_5d_weight_tuned_low_turnover_no_snapshot"


def _report_caveats() -> list[str]:
    return [
        "This is a research candidate workflow.",
        "2026 forward data was not used for ML training or model selection.",
        "Back-tested performance is hypothetical unless actually paper-tracked live.",
        "Snapshot analyst target fields are excluded.",
        "Historical rating-count features use dated FMP grades-historical records available on or before each rebalance date.",
        "News sentiment depends on Alpha Vantage coverage and classification.",
        "ML models may overfit and require future forward validation.",
        "This is research/paper trading only, not financial advice.",
    ]


def _compute_drawdown(values: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    return values / values.cummax() - 1.0


def _save_plot(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def _make_equity_curve_plot(plot_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 7), facecolor="white")
    ax.set_facecolor("white")
    ax.plot(plot_df["date"], plot_df["spy_value"], linewidth=2.0, label="SPY Buy & Hold", color="#1f2937")
    ax.plot(plot_df["date"], plot_df["rule_model_value"], linewidth=2.2, label="Current Rule-Based Low-Turnover Model", color="#0f766e")
    ax.plot(plot_df["date"], plot_df["ml_model_value"], linewidth=2.2, label="Frozen ML Ranker", color="#7c3aed")
    ax.set_title("Frozen ML Ranker vs Rule-Based Model vs SPY (2026 Forward)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of $10,000")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"${value:,.0f}"))
    ax.grid(True, alpha=0.25)
    ax.legend()
    _save_plot(fig, output_path)


def _make_drawdown_plot(plot_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 7), facecolor="white")
    ax.set_facecolor("white")
    ax.plot(plot_df["date"], plot_df["spy_drawdown"], linewidth=2.0, label="SPY Buy & Hold", color="#1f2937")
    ax.plot(plot_df["date"], plot_df["rule_drawdown"], linewidth=2.2, label="Current Rule-Based Low-Turnover Model", color="#b45309")
    ax.plot(plot_df["date"], plot_df["ml_drawdown"], linewidth=2.2, label="Frozen ML Ranker", color="#7c3aed")
    ax.set_title("Drawdown: Frozen ML Ranker vs Rule-Based Model vs SPY")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown %")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0%}"))
    ax.grid(True, alpha=0.25)
    ax.legend()
    _save_plot(fig, output_path)


def _build_direct_spy_series(features_forward: pd.DataFrame, benchmark: str, decision_dates: pd.Series) -> pd.DataFrame:
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
    plot_df = pd.DataFrame({"date": pd.to_datetime(decision_dates).sort_values().reset_index(drop=True)})
    plot_df = pd.merge_asof(plot_df, spy_daily, on="date", direction="backward")
    if plot_df["spy_adjusted_close"].isna().any():
        raise ValueError("Unable to align SPY adjusted_close values to ML forward dates.")
    start_close = float(plot_df["spy_adjusted_close"].iloc[0])
    plot_df["spy_value"] = INITIAL_CAPITAL * plot_df["spy_adjusted_close"] / start_close
    plot_df["spy_drawdown"] = _compute_drawdown(plot_df["spy_value"])
    return plot_df


def _ensure_market_features(runtime: Config, features_forward: pd.DataFrame) -> pd.DataFrame:
    merged = features_forward.copy()
    sentiment_path = runtime.processed_dir / "market_sentiment_daily.csv"
    regime_path = runtime.processed_dir / "market_regime_daily.csv"

    if sentiment_path.exists():
        sentiment = load_dataframe(sentiment_path, parse_dates=["date"])
        sentiment_cols = [
            "market_sentiment_7d",
            "market_sentiment_30d",
            "market_sentiment_change_7d_vs_30d",
            "market_negative_news_ratio_7d",
            "percent_tickers_positive_sentiment_7d",
            "percent_tickers_negative_sentiment_7d",
            "sentiment_dispersion_7d",
        ]
        missing = [column for column in sentiment_cols if column not in merged.columns]
        if missing:
            merged = merged.merge(sentiment[["date", *[column for column in missing if column in sentiment.columns]]], on="date", how="left")

    if regime_path.exists():
        regime = load_dataframe(regime_path, parse_dates=["date"])
        regime_cols = [
            "market_risk_score",
            "market_regime_label",
            "normalized_market_risk_score",
            "spy_return_21d",
            "spy_volatility_21d",
            "spy_drawdown_from_63d_high",
            "spy_above_sma_50",
            "spy_above_sma_200",
        ]
        missing = [column for column in regime_cols if column not in merged.columns]
        if missing:
            merged = merged.merge(regime[["date", *[column for column in missing if column in regime.columns]]], on="date", how="left")

    if "spy_above_sma_50" not in merged.columns and "above_sma_50" in merged.columns:
        benchmark_rows = (
            merged.loc[merged["ticker"] == runtime.benchmark, ["date", "above_sma_50"]]
            .drop_duplicates(subset=["date"])
            .rename(columns={"above_sma_50": "spy_above_sma_50"})
        )
        merged = merged.merge(benchmark_rows, on="date", how="left")
    return merged


def _validate_rule_based_config(runtime: Config) -> None:
    recommended = load_recommended_strategy_config(runtime.project_root)
    if recommended.strategy_name != EXPECTED_RULE_BASED:
        raise ValueError(f"recommended_strategy.yaml must remain locked to {EXPECTED_RULE_BASED}; found {recommended.strategy_name}")


def _summarize_attribution(attribution_df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    if attribution_df.empty:
        empty = pd.DataFrame(
            columns=[
                "ticker",
                "total_contribution",
                "average_weight",
                "periods_held",
                "average_return_while_held",
                "contribution_to_excess_return",
            ]
        )
        return empty, "Attribution unavailable because the frozen ML backtest produced no holdings."

    summary = (
        attribution_df.groupby("ticker", as_index=False)
        .agg(
            total_contribution=("total_contribution", "sum"),
            average_weight=("weight", "mean"),
            periods_held=("date", "count"),
            average_return_while_held=("realized_return_while_held", "mean"),
            contribution_to_excess_return=("contribution_to_excess_return", "sum"),
        )
        .sort_values("contribution_to_excess_return", ascending=False)
        .reset_index(drop=True)
    )
    negative_total = float(summary.loc[summary["contribution_to_excess_return"] < 0, "contribution_to_excess_return"].abs().sum())
    bottom_five = summary.sort_values("contribution_to_excess_return").head(5)
    bottom_five_share = float(bottom_five["contribution_to_excess_return"].abs().sum() / negative_total) if negative_total > 0 else float("nan")
    if pd.notna(bottom_five_share) and bottom_five_share >= 0.65:
        narrative = "The contribution profile was fairly concentrated, with a few weaker ML selections driving most of the drag."
    else:
        narrative = "The contribution profile was broad-based rather than dominated by only a few names."
    return summary, narrative


def _simulate_ml_forward_with_attribution(
    panels: list[tuple[pd.Timestamp, pd.DataFrame, float, pd.DataFrame]],
    *,
    top_n: int,
    cost_bps: float,
    enter_rank: int,
    hold_rank: int,
    max_holding_days: int,
    rebalance_frequency_days: int,
    strategy_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    holdings: dict[str, dict[str, object]] = {}
    weekly_rows: list[dict[str, object]] = []
    holding_rows: list[dict[str, object]] = []
    action_rows: list[dict[str, object]] = []
    attribution_rows: list[dict[str, object]] = []
    portfolio_value = INITIAL_CAPITAL
    spy_value = INITIAL_CAPITAL

    if len(panels) < 2:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

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

        for ticker, reason in forced_sells:
            action_rows.append({"date": rebalance_date, "ticker": ticker, "action": "SELL", "reason": reason})
            holdings.pop(ticker, None)

        for ticker, rank in sorted(discretionary_sells, key=lambda item: item[1], reverse=True):
            action_rows.append({"date": rebalance_date, "ticker": ticker, "action": "SELL", "reason": f"rank>{hold_rank}:{rank}"})
            holdings.pop(ticker, None)

        current_tickers = list(holdings.keys())
        desired_buys = [
            ticker
            for ticker in ranked.loc[ranked["rank"] <= enter_rank, "ticker"].tolist()
            if ticker not in current_tickers
        ]
        open_slots = max(0, top_n - len(current_tickers))
        for ticker in desired_buys[:open_slots]:
            holdings[ticker] = {"holding_days": 0, "entry_date": rebalance_date}
            action_rows.append({"date": rebalance_date, "ticker": ticker, "action": "BUY", "reason": f"enter_rank<={enter_rank}"})

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
        spy_return = next_spy_price / float(spy_price) - 1 if spy_price else 0.0

        gross_return = 0.0
        if selected_count:
            for _, row in selected.iterrows():
                if row["ticker"] in next_panel_by_ticker.index:
                    next_row = next_panel_by_ticker.loc[row["ticker"]]
                    current_price = float(row["adjusted_close"])
                    next_price = float(next_row["adjusted_close"])
                    realized_return = next_price / current_price - 1 if current_price else 0.0
                else:
                    realized_return = float(row.get("future_return_used", 0.0))
                contribution = float(row["weight"]) * realized_return
                gross_return += contribution
                attribution_rows.append(
                    {
                        "date": rebalance_date,
                        "ticker": row["ticker"],
                        "weight": float(row["weight"]),
                        "realized_return_while_held": realized_return,
                        "total_contribution": contribution,
                        "contribution_to_excess_return": float(row["weight"]) * (realized_return - spy_return),
                    }
                )

        net_return = gross_return - transaction_cost
        portfolio_value *= 1 + net_return
        spy_value *= 1 + spy_return

        buy_tickers = {
            action["ticker"]
            for action in action_rows
            if action["date"] == rebalance_date and action["action"] == "BUY"
        }
        for ticker in list(holdings):
            row = ranked_by_ticker.loc[ticker]
            action_type = "BUY" if ticker in buy_tickers else "HOLD"
            holdings[ticker]["holding_days"] = int(holdings[ticker].get("holding_days", 0) + rebalance_frequency_days)
            holdings[ticker]["weight"] = float(new_weights.get(ticker, 0.0))
            holding_rows.append(
                {
                    "date": rebalance_date,
                    "ticker": ticker,
                    "action": action_type,
                    "reason": "new_buy" if action_type == "BUY" else "kept_within_hold_band",
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
                "average_holding_days": float(np.mean([meta["holding_days"] for meta in holdings.values()])) if holdings else 0.0,
            }
        )

    return (
        pd.DataFrame(weekly_rows),
        pd.DataFrame(holding_rows),
        pd.DataFrame(action_rows),
        pd.DataFrame(attribution_rows),
    )


def _validate_candidate(runtime: Config, artifact: dict[str, object]) -> None:
    candidate = load_ml_research_candidate_config(runtime.project_root)
    if candidate.long_short:
        raise ValueError("Frozen ML candidate must remain long-only.")
    if candidate.use_regime_filter:
        raise ValueError("Frozen ML candidate must keep regime filter off.")
    if candidate.snapshot_fields_allowed:
        raise ValueError("Frozen ML candidate must keep snapshot fields disabled.")
    if candidate.strategy_name not in NO_SNAPSHOT_STRATEGIES:
        raise ValueError("Frozen ML candidate is not registered as a no-snapshot strategy.")
    offending = sorted(strategy_score_fields(candidate.strategy_name) & SNAPSHOT_FIELD_COLUMNS)
    if offending:
        raise ValueError(f"Frozen ML candidate uses snapshot fields: {', '.join(offending)}")
    if str(artifact.get("train_end_date")) > "2024-12-31":
        raise ValueError("ML artifact training window extends into 2025/2026.")
    if str(artifact.get("validation_end_date")) > "2025-12-31":
        raise ValueError("ML artifact validation/model-selection window extends into 2026.")


def main() -> None:
    runtime = Config.from_env()
    candidate = load_ml_research_candidate_config(runtime.project_root)
    artifact = load_ml_artifact(runtime.project_root / candidate.model_path)
    _validate_candidate(runtime, artifact)
    _validate_rule_based_config(runtime)

    features_path = runtime.final_dir / "features_panel_2026_forward.csv"
    if not features_path.exists():
        raise FileNotFoundError(f"Missing 2026 forward feature panel: {features_path}")
    features_forward = load_dataframe(features_path, parse_dates=["date"])
    if features_forward.empty:
        raise ValueError("2026 forward feature panel is empty.")
    features_forward = _ensure_market_features(runtime, features_forward)

    features_forward = features_forward.loc[features_forward["date"] >= pd.Timestamp(candidate.forward_window_start)].copy()
    if features_forward.empty:
        raise ValueError("No forward rows remain after applying the frozen ML forward window start.")

    estimator = artifact["estimator"]
    feature_names = list(artifact["feature_names"])
    prediction_rows = features_forward.loc[features_forward["ticker"] != runtime.benchmark, ["date", "ticker", *feature_names]].copy()
    prediction_rows["predicted_score"] = estimator.predict(prediction_rows[feature_names])
    prediction_df = prediction_rows.loc[:, ["date", "ticker", "predicted_score"]].copy()

    ml_panels = precompute_research_panels(
        features_forward,
        runtime,
        scoring_mode="ml_prediction",
        start_date=candidate.forward_window_start,
        end_date=None,
        prediction_df=prediction_df,
        rebalance_frequency_days=int(candidate.rebalance_frequency_days),
        holding_period_days=5,
    )
    ml_weekly, ml_holdings, ml_actions, ml_attribution = _simulate_ml_forward_with_attribution(
        ml_panels,
        top_n=int(candidate.top_n),
        cost_bps=float(candidate.total_cost_bps),
        enter_rank=int(candidate.enter_rank),
        hold_rank=int(candidate.hold_rank),
        max_holding_days=int(candidate.max_holding_days),
        rebalance_frequency_days=int(candidate.rebalance_frequency_days),
        strategy_name=candidate.strategy_name,
    )
    if ml_weekly.empty:
        raise ValueError("Frozen ML 2026 forward backtest returned no rows.")

    rule_returns_path = runtime.tables_dir / "forward_2026_model_vs_spy_returns.csv"
    if not rule_returns_path.exists():
        raise FileNotFoundError(f"Missing rule-based forward returns table: {rule_returns_path}")
    rule_returns = load_dataframe(rule_returns_path, parse_dates=["date"]).sort_values("date")
    rule_current_path = runtime.tables_dir / "current_recommendations_2026_forward.csv"
    rule_current = load_dataframe(rule_current_path, parse_dates=["date"]) if rule_current_path.exists() else pd.DataFrame()

    ml_plot_df = (
        ml_weekly.loc[:, ["date", "net_return", "portfolio_value", "turnover", "transaction_cost", "selected_count", "exposure"]]
        .rename(
            columns={
                "net_return": "ml_period_return",
                "portfolio_value": "ml_model_value",
                "transaction_cost": "ml_trading_cost",
            }
        )
        .sort_values("date")
        .reset_index(drop=True)
    )
    spy_df = _build_direct_spy_series(features_forward, runtime.benchmark, ml_plot_df["date"])
    ml_plot_df = ml_plot_df.merge(spy_df, on="date", how="left")
    ml_plot_df["ml_drawdown"] = _compute_drawdown(ml_plot_df["ml_model_value"])

    rule_plot_df = rule_returns.loc[:, ["date", "model_value", "model_drawdown"]].rename(
        columns={
            "model_value": "rule_model_value",
            "model_drawdown": "rule_drawdown",
        }
    )
    plot_df = ml_plot_df.merge(rule_plot_df, on="date", how="left")
    plot_df["spy_value"] = plot_df["spy_value"]
    plot_df["rule_drawdown"] = plot_df["rule_drawdown"].fillna(_compute_drawdown(plot_df["rule_model_value"]))
    plot_df["ml_excess_vs_spy"] = plot_df["ml_model_value"] / INITIAL_CAPITAL - 1.0 - (plot_df["spy_value"] / INITIAL_CAPITAL - 1.0)
    plot_df["rule_excess_vs_spy"] = plot_df["rule_model_value"] / INITIAL_CAPITAL - 1.0 - (plot_df["spy_value"] / INITIAL_CAPITAL - 1.0)

    _make_equity_curve_plot(plot_df, runtime.charts_dir / "ml_2026_forward_equity_curve.png")
    _make_drawdown_plot(plot_df, runtime.charts_dir / "ml_2026_forward_drawdown.png")

    ml_total_return = float(plot_df["ml_model_value"].iloc[-1] / INITIAL_CAPITAL - 1.0)
    rule_total_return = float(plot_df["rule_model_value"].iloc[-1] / INITIAL_CAPITAL - 1.0)
    spy_total_return = float(plot_df["spy_value"].iloc[-1] / INITIAL_CAPITAL - 1.0)

    comparison_df = pd.DataFrame(
        [
            {
                "series_name": "Frozen ML Ranker",
                "strategy_name": candidate.strategy_name,
                "model_type": candidate.model_type,
                "total_return": ml_total_return,
                "excess_vs_spy": ml_total_return - spy_total_return,
                "excess_vs_rule_based": ml_total_return - rule_total_return,
                "max_drawdown": float(plot_df["ml_drawdown"].min()),
                "average_turnover": float(pd.to_numeric(ml_weekly["turnover"], errors="coerce").mean()),
                "estimated_trading_costs": float(pd.to_numeric(ml_weekly["transaction_cost"], errors="coerce").sum()),
                "average_holdings": float(pd.to_numeric(ml_weekly["selected_count"], errors="coerce").mean()),
                "rebalance_periods": int(len(ml_weekly)),
            },
            {
                "series_name": "Current Rule-Based Low-Turnover Model",
                "strategy_name": EXPECTED_RULE_BASED,
                "model_type": "rule_based",
                "total_return": rule_total_return,
                "excess_vs_spy": rule_total_return - spy_total_return,
                "excess_vs_rule_based": 0.0,
                "max_drawdown": float(plot_df["rule_drawdown"].min()),
                "average_turnover": float(pd.to_numeric(rule_returns["turnover"], errors="coerce").mean()),
                "estimated_trading_costs": float(pd.to_numeric(rule_returns["trading_cost"], errors="coerce").sum()),
                "average_holdings": float(pd.to_numeric(rule_returns["selected_count"], errors="coerce").mean()),
                "rebalance_periods": int(len(rule_returns)),
            },
            {
                "series_name": "SPY Buy & Hold",
                "strategy_name": runtime.benchmark,
                "model_type": "benchmark",
                "total_return": spy_total_return,
                "excess_vs_spy": 0.0,
                "excess_vs_rule_based": spy_total_return - rule_total_return,
                "max_drawdown": float(plot_df["spy_drawdown"].min()),
                "average_turnover": 0.0,
                "estimated_trading_costs": 0.0,
                "average_holdings": 1.0,
                "rebalance_periods": int(len(plot_df)),
            },
        ]
    )
    save_dataframe(runtime.tables_dir / "ml_2026_forward_comparison.csv", comparison_df)

    attribution_summary_df, attribution_narrative = _summarize_attribution(ml_attribution)
    save_dataframe(runtime.tables_dir / "ml_2026_forward_attribution.csv", attribution_summary_df)

    latest_ml_date = pd.Timestamp(ml_holdings["date"].max())
    latest_ml_holdings = ml_holdings.loc[ml_holdings["date"] == latest_ml_date].copy()
    ml_decision_dates = sorted(pd.to_datetime(ml_holdings["date"]).drop_duplicates())
    previous_ml_date = pd.Timestamp(ml_decision_dates[-2]) if len(ml_decision_dates) >= 2 else pd.NaT
    previous_ml_holdings = (
        set(ml_holdings.loc[ml_holdings["date"] == previous_ml_date, "ticker"].tolist())
        if pd.notna(previous_ml_date)
        else set()
    )
    current_ml_holdings = set(latest_ml_holdings["ticker"].tolist())
    latest_ml_buys = sorted(current_ml_holdings - previous_ml_holdings)
    latest_ml_sells = sorted(previous_ml_holdings - current_ml_holdings)
    latest_ml_holds = sorted(current_ml_holdings & previous_ml_holdings)

    rule_holdings_set = set(rule_current["ticker"].tolist()) if not rule_current.empty else set()
    ml_holdings_set = set(latest_ml_holdings["ticker"].tolist())
    overlap = sorted(ml_holdings_set & rule_holdings_set)
    ml_only = sorted(ml_holdings_set - rule_holdings_set)
    rule_only = sorted(rule_holdings_set - ml_holdings_set)
    overlap_ratio = float(len(overlap) / max(1, len(ml_holdings_set | rule_holdings_set)))
    materially_different = overlap_ratio < 0.60

    report_lines = [
        "# ML 2026 Forward Summary",
        "",
        *[f"- {line}" for line in _report_caveats()],
        "",
        f"- Forward start date: {pd.Timestamp(plot_df['date'].min()).date()}",
        f"- Latest available date: {pd.Timestamp(plot_df['date'].max()).date()}",
        f"- Frozen candidate strategy: `{candidate.strategy_name}`",
        f"- Frozen candidate model type: `{candidate.model_type}`",
        f"- Model path loaded from disk: `{candidate.model_path}`",
        f"- Execution mode: `{candidate.execution_mode}`",
        f"- Enter rank: {candidate.enter_rank}",
        f"- Hold rank: {candidate.hold_rank}",
        f"- Top N: {candidate.top_n}",
        f"- Max holding days: {candidate.max_holding_days}",
        f"- Rebalance frequency: {candidate.rebalance_frequency_days} trading days",
        f"- Cost assumption: {candidate.total_cost_bps:.0f} bps",
        "",
        "## Metrics",
        "",
        f"- ML return: {ml_total_return:.2%}",
        f"- Rule-based model return: {rule_total_return:.2%}",
        f"- SPY return: {spy_total_return:.2%}",
        f"- ML excess vs SPY: {ml_total_return - spy_total_return:.2%}",
        f"- Rule-based excess vs SPY: {rule_total_return - spy_total_return:.2%}",
        f"- ML excess vs rule-based: {ml_total_return - rule_total_return:.2%}",
        f"- ML max drawdown: {float(plot_df['ml_drawdown'].min()):.2%}",
        f"- Rule-based max drawdown: {float(plot_df['rule_drawdown'].min()):.2%}",
        f"- SPY max drawdown: {float(plot_df['spy_drawdown'].min()):.2%}",
        f"- ML average turnover: {float(pd.to_numeric(ml_weekly['turnover'], errors='coerce').mean()):.6f}",
        f"- Rule-based average turnover: {float(pd.to_numeric(rule_returns['turnover'], errors='coerce').mean()):.6f}",
        f"- ML estimated trading costs: {float(pd.to_numeric(ml_weekly['transaction_cost'], errors='coerce').sum()):.4f}",
        f"- Rule-based estimated trading costs: {float(pd.to_numeric(rule_returns['trading_cost'], errors='coerce').sum()):.4f}",
        f"- ML average holdings: {float(pd.to_numeric(ml_weekly['selected_count'], errors='coerce').mean()):.2f}",
        f"- Rule-based average holdings: {float(pd.to_numeric(rule_returns['selected_count'], errors='coerce').mean()):.2f}",
        f"- Rebalance periods: {len(ml_weekly)}",
        "",
        "## Latest ML Actions",
        "",
        f"- Current holdings: {', '.join(sorted(ml_holdings_set)) or 'none'}",
        f"- Latest buys: {', '.join(latest_ml_buys) or 'none'}",
        f"- Latest sells: {', '.join(latest_ml_sells) or 'none'}",
        f"- Latest holds: {', '.join(latest_ml_holds) or 'none'}",
        "",
        "## Attribution",
        "",
        f"- Top 5 ML contributors: {', '.join(attribution_summary_df.head(5)['ticker'].tolist()) or 'none'}",
        f"- Bottom 5 ML detractors: {', '.join(attribution_summary_df.sort_values('contribution_to_excess_return').head(5)['ticker'].tolist()) or 'none'}",
        f"- Attribution read: {attribution_narrative}",
        "",
        "## Overlap Vs Rule-Based",
        "",
        f"- Overlap tickers: {', '.join(overlap) or 'none'}",
        f"- ML-only tickers: {', '.join(ml_only) or 'none'}",
        f"- Rule-only tickers: {', '.join(rule_only) or 'none'}",
        f"- Holdings overlap ratio: {overlap_ratio:.2%}",
        f"- ML selecting materially different names: {str(materially_different).lower()}",
        "",
        "## Research Candidate Forward Status",
        "",
        "- If ML beats SPY and rule-based model over 6+ months with acceptable drawdown, continue monitoring.",
        "- If ML underperforms SPY, keep as research only.",
        "- If ML beats SPY for 12 months and has acceptable drawdown/cost robustness, consider promotion later.",
        "- Do not promote based on fewer than 6 months.",
    ]
    (runtime.reports_dir / "ml_2026_forward_summary.md").write_text("\n".join(report_lines), encoding="utf-8")

    research_candidate_lines = [
        "# Research Candidate Summary",
        "",
        *[f"- {line}" for line in _report_caveats()],
        "",
        "## Frozen Production/Paper Model",
        "",
        f"- Current frozen model: `{EXPECTED_RULE_BASED}`",
        f"- 2026 forward return so far: {rule_total_return:.2%}",
        f"- 2026 forward excess vs SPY: {rule_total_return - spy_total_return:.2%}",
        "",
        "## Best 2025 ML Validation Candidate",
        "",
        f"- Model: `{candidate.model_type}`",
        f"- Strategy: `{candidate.strategy_name}`",
        "- 2026 forward data was not used for ML training or model selection.",
        "",
        "## 2026 Forward Comparison",
        "",
        f"- ML return: {ml_total_return:.2%}",
        f"- Rule-based return: {rule_total_return:.2%}",
        f"- SPY return: {spy_total_return:.2%}",
        f"- ML excess vs SPY: {ml_total_return - spy_total_return:.2%}",
        f"- ML excess vs rule-based: {ml_total_return - rule_total_return:.2%}",
        f"- ML max drawdown: {float(plot_df['ml_drawdown'].min()):.2%}",
        f"- Rule-based max drawdown: {float(plot_df['rule_drawdown'].min()):.2%}",
        "",
        "## Forward Status",
        "",
        f"- ML should remain research candidate: {'true' if len(ml_weekly) < 8 or ml_total_return <= spy_total_return else 'false'}",
        f"- Move to extended paper monitoring: {'true' if len(ml_weekly) >= 8 and ml_total_return > spy_total_return else 'false'}",
        "- recommended_strategy.yaml remains unchanged pending more forward evidence.",
    ]
    (runtime.reports_dir / "research_candidate_summary.md").write_text("\n".join(research_candidate_lines), encoding="utf-8")

    print(f"ML final value: ${plot_df['ml_model_value'].iloc[-1]:,.2f}")
    print(f"Rule-based final value: ${plot_df['rule_model_value'].iloc[-1]:,.2f}")
    print(f"SPY final value: ${plot_df['spy_value'].iloc[-1]:,.2f}")
    print(f"ML total return: {ml_total_return:.2%}")
    print(f"Rule-based total return: {rule_total_return:.2%}")
    print(f"SPY total return: {spy_total_return:.2%}")
    print(f"ML excess vs SPY: {ml_total_return - spy_total_return:.2%}")
    print(f"ML excess vs rule-based: {ml_total_return - rule_total_return:.2%}")
    print(f"ML max drawdown: {float(plot_df['ml_drawdown'].min()):.2%}")
    print(f"Rule-based max drawdown: {float(plot_df['rule_drawdown'].min()):.2%}")
    print(f"SPY max drawdown: {float(plot_df['spy_drawdown'].min()):.2%}")
    print(f"Saved {runtime.tables_dir / 'ml_2026_forward_comparison.csv'}")
    print(f"Saved {runtime.tables_dir / 'ml_2026_forward_attribution.csv'}")
    print(f"Saved {runtime.charts_dir / 'ml_2026_forward_equity_curve.png'}")
    print(f"Saved {runtime.charts_dir / 'ml_2026_forward_drawdown.png'}")
    print(f"Saved {runtime.reports_dir / 'ml_2026_forward_summary.md'}")
    print(f"Saved {runtime.reports_dir / 'research_candidate_summary.md'}")


if __name__ == "__main__":
    main()
