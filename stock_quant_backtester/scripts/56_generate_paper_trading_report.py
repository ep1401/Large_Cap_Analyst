from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.no_snapshot_research import dataframe_to_markdown
from src.recommended_strategy import (
    caveat_lines,
    latest_recommended_holdings,
    load_runtime_and_recommended,
    precompute_recommended_low_turnover_panels,
    run_low_turnover_recommended_backtest,
    suggested_rebalance_date,
    top_signal_reasons,
)
from src.utils import save_dataframe


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-path", default=None)
    args = parser.parse_args()

    runtime, recommended, features = load_runtime_and_recommended(args.features_path)
    latest_feature_date = features["date"].max()
    report_lines = ["# Paper Trading Report", "", *[f"- {line}" for line in caveat_lines()], ""]

    if recommended.strategy_name == "final_quant_5d_weight_tuned_low_turnover_no_snapshot":
        rebalance_frequency_days = recommended.rebalance_frequency_days or recommended.holding_period_days
        panels = precompute_recommended_low_turnover_panels(features, runtime, recommended)
        weekly_df, holdings_df, actions_df = run_low_turnover_recommended_backtest(
            panels=panels,
            top_n=recommended.top_n,
            cost_bps=float(recommended.total_cost_bps),
            enter_rank=recommended.enter_rank or recommended.top_n,
            hold_rank=recommended.hold_rank or max(recommended.top_n, 20),
            max_holding_days=recommended.max_holding_days or 20,
            rebalance_frequency_days=rebalance_frequency_days,
            strategy_name=recommended.strategy_name,
            max_turnover_per_rebalance=recommended.max_turnover_per_rebalance,
        )
        if holdings_df.empty:
            raise SystemExit("No recommended holdings could be produced from the latest low-turnover feature panel.")
        selected_date = pd.Timestamp(holdings_df["date"].max())
        latest_holdings = holdings_df.loc[holdings_df["date"] == selected_date].copy()
        latest_panel_candidates = [panel for panel_date, panel, _, _ in panels if pd.Timestamp(panel_date) == selected_date]
        if not latest_panel_candidates:
            raise SystemExit("Could not locate the latest decision-date panel for the low-turnover paper trading report.")
        latest_panel = latest_panel_candidates[-1].copy()
        latest_panel["top_signal_reasons"] = latest_panel.apply(top_signal_reasons, axis=1)
        latest_panel["sentiment_7d"] = latest_panel.get("relevance_weighted_sentiment_7d")
        recommendations_df = latest_holdings.merge(
            latest_panel[
                [
                    column
                    for column in [
                        "ticker",
                        "sector",
                        "top_signal_reasons",
                        "historical_rating_score",
                        "net_upgrade_score_30d",
                        "downgrade_count_30d",
                        "sentiment_7d",
                        "negative_news_ratio_7d",
                        "relative_strength_21d",
                        "volatility_21d",
                    ]
                    if column in latest_panel.columns
                ]
            ],
            on="ticker",
            how="left",
        )
        recommendations_df["strategy_name"] = recommended.strategy_name
        recommendations_df["holding_period_days"] = recommended.holding_period_days
        recommendations_df["position_sizing"] = recommended.position_sizing
        recommendations_df["total_cost_bps"] = recommended.total_cost_bps
        recommendations_df["min_score_threshold"] = recommended.threshold
        recommendations_df["allow_cash"] = recommended.allow_cash
        recommendations_df["cash_weight"] = 1.0 - recommendations_df["weight"].sum()
        latest_actions = actions_df.loc[actions_df["date"] == selected_date].copy() if not actions_df.empty else actions_df
        latest_weekly = weekly_df.loc[weekly_df["date"] == selected_date].iloc[0]
        sells_df = latest_actions.loc[latest_actions["action"] == "SELL", ["ticker", "reason"]].copy()
        buys_df = latest_actions.loc[latest_actions["action"] == "BUY", ["ticker", "reason"]].copy()
        report_lines.extend(
            [
                f"- Latest feature date in panel: {latest_feature_date.date()}.",
                f"- Signal selection date used for recommendations: {selected_date.date()}.",
                f"- Suggested rebalance date: {suggested_rebalance_date(selected_date, rebalance_frequency_days).date()}.",
                f"- Strategy under paper trading: `{recommended.strategy_name}`.",
                f"- Estimated turnover at latest rebalance: {float(latest_weekly['turnover']):.4f}.",
                f"- Estimated trading cost at latest rebalance: {float(latest_weekly['transaction_cost']):.4f}.",
                "",
            ]
        )
        if not sells_df.empty:
            report_lines.extend(["## Sells", "", dataframe_to_markdown(sells_df)])
            report_lines.append("")
        if not buys_df.empty:
            report_lines.extend(["## New Buys", "", dataframe_to_markdown(buys_df)])
            report_lines.append("")
    else:
        recommendations_df, selected_date = latest_recommended_holdings(features, runtime, recommended)
        if recommendations_df.empty:
            raise SystemExit("No recommended holdings could be produced from the latest feature panel.")
        recommendations_df = recommendations_df.copy()
        recommendations_df["top_signal_reasons"] = recommendations_df.apply(top_signal_reasons, axis=1)
        recommendations_df["sentiment_7d"] = recommendations_df.get("relevance_weighted_sentiment_7d")
        report_lines.extend(
            [
                f"- Latest feature date in panel: {latest_feature_date.date()}.",
                f"- Signal selection date used for recommendations: {selected_date.date()}.",
                f"- Suggested rebalance date: {suggested_rebalance_date(selected_date, recommended.holding_period_days).date()}.",
                f"- Strategy under paper trading: `{recommended.strategy_name}`.",
                "",
            ]
        )

    keep_cols = [
        "date",
        "ticker",
        "sector",
        "action",
        "reason",
        "rank",
        "score",
        "weight",
        "strategy_name",
        "holding_period_days",
        "position_sizing",
        "total_cost_bps",
        "min_score_threshold",
        "allow_cash",
        "cash_weight",
        "top_signal_reasons",
        "historical_rating_score",
        "net_upgrade_score_30d",
        "downgrade_count_30d",
        "sentiment_7d",
        "negative_news_ratio_7d",
        "relative_strength_21d",
        "volatility_21d",
        "holding_days",
        "rebalance_frequency_days",
        "enter_rank",
        "hold_rank",
        "max_holding_days",
    ]
    output_df = recommendations_df[[column for column in keep_cols if column in recommendations_df.columns]].copy()
    save_dataframe(runtime.tables_dir / "current_recommendations_final_strategy.csv", output_df)
    report_lines.extend(["## Current Holdings", "", dataframe_to_markdown(output_df.round(4))])
    (runtime.reports_dir / "paper_trading_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Saved {runtime.tables_dir / 'current_recommendations_final_strategy.csv'}")
    print(f"Saved {runtime.reports_dir / 'paper_trading_report.md'}")


if __name__ == "__main__":
    main()
