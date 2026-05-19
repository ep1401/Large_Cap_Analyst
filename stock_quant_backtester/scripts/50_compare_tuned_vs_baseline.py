from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.backtest import run_weekly_backtest
from src.config import Config
from src.no_snapshot_research import dataframe_to_markdown, fmt_pct, get_best_5d_config, load_features, summarize_backtest
from src.scoring import strategy_display_name
from src.utils import load_dataframe, save_dataframe


BASELINE_STRATEGY = "final_quant_5d_no_snapshot_no_sma_filter"
TUNED_STRATEGY = "final_quant_5d_weight_tuned_no_snapshot"
COMPARISON_STRATEGIES = [
    BASELINE_STRATEGY,
    TUNED_STRATEGY,
    "historical_rating_counts_plus_events",
    "historical_rating_score_only_5d",
]


def _run_summary(
    features: pd.DataFrame,
    config: Config,
    strategy_name: str,
    max_names_per_sector: int | None,
) -> dict[str, object]:
    weekly, holdings, _ = run_weekly_backtest(
        features=features,
        holding_period_days=5,
        benchmark=config.benchmark,
        top_n=10,
        initial_capital=config.initial_capital,
        transaction_cost_bps=10,
        use_regime_filter=False,
        use_analyst_filters=False,
        analyst_count_threshold=config.analyst_count_threshold,
        min_avg_dollar_volume=config.min_avg_dollar_volume,
        strategy_name=strategy_name,
        max_names_per_sector=max_names_per_sector,
        position_sizing="equal_weight",
        min_historical_rating_count=5,
        allow_cash=False,
    )
    summary = summarize_backtest(weekly, 5, strategy_name)
    summary["strategy_name"] = strategy_name
    summary["display_name"] = strategy_display_name(strategy_name)
    summary["walk_forward_average_excess_vs_spy"] = float(
        pd.Series(
            [
                summary["2024_h1_excess_return_vs_spy"],
                summary["2024_h2_excess_return_vs_spy"],
                summary["2025_excess_return_vs_spy"],
            ]
        ).mean()
    )
    summary["top_n"] = 10
    summary["total_cost_bps"] = 10.0
    summary["allow_cash"] = False
    summary["min_score_threshold"] = pd.NA
    summary["latest_rebalance_date"] = holdings["date"].max() if not holdings.empty else pd.NaT
    summary["latest_top10"] = ", ".join(holdings.loc[holdings["date"] == holdings["date"].max(), "ticker"].head(10).tolist()) if not holdings.empty else ""
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-path", default=None)
    args = parser.parse_args()

    config = Config.from_env()
    features = load_features(config, args.features_path)
    best_config = get_best_5d_config(config)
    weight_search_path = config.tables_dir / "weight_search_5d_no_snapshot.csv"
    weight_search_df = load_dataframe(weight_search_path) if weight_search_path.exists() else pd.DataFrame()
    tuned_created = bool(weight_search_df["promoted"].fillna(False).astype(bool).any()) if not weight_search_df.empty else False

    rows: list[dict[str, object]] = []
    for strategy_name in COMPARISON_STRATEGIES:
        if strategy_name == TUNED_STRATEGY and not tuned_created:
            continue
        sector_cap = best_config["max_names_per_sector"] if "final_quant_5d" in strategy_name else None
        rows.append(_run_summary(features, config, strategy_name, sector_cap))

    comparison_df = pd.DataFrame(rows)
    if comparison_df.empty:
        raise SystemExit("No comparison rows were generated.")
    baseline_walk_forward = float(
        comparison_df.loc[comparison_df["strategy_name"] == BASELINE_STRATEGY, "walk_forward_average_excess_vs_spy"].iloc[0]
    )
    comparison_df["delta_vs_baseline_walk_forward"] = comparison_df["walk_forward_average_excess_vs_spy"] - baseline_walk_forward
    comparison_df = comparison_df.sort_values(
        ["walk_forward_average_excess_vs_spy", "windows_beating_spy", "max_drawdown"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    save_dataframe(config.tables_dir / "tuned_vs_baseline_comparison.csv", comparison_df)

    report_cols = [
        "strategy_name",
        "display_name",
        "walk_forward_average_excess_vs_spy",
        "2024_h1_excess_return_vs_spy",
        "2024_h2_excess_return_vs_spy",
        "2025_excess_return_vs_spy",
        "windows_beating_spy",
        "max_drawdown",
        "average_turnover",
        "delta_vs_baseline_walk_forward",
    ]
    report_lines = [
        "# Tuned Vs Baseline Comparison",
        "",
        f"- Tuned model created: {tuned_created}.",
        f"- Baseline strategy: `{BASELINE_STRATEGY}`.",
        f"- Tuned strategy included: {TUNED_STRATEGY if tuned_created else 'not created'}.",
        "",
        dataframe_to_markdown(comparison_df[report_cols].round(4)),
        "",
        "## Recommendation Check",
        f"- Best walk-forward average excess vs SPY: {fmt_pct(float(comparison_df.iloc[0]['walk_forward_average_excess_vs_spy']))}.",
        f"- Baseline walk-forward average excess vs SPY: {fmt_pct(baseline_walk_forward)}.",
        f"- Current top row: `{comparison_df.iloc[0]['strategy_name']}`.",
    ]
    (config.reports_dir / "tuned_vs_baseline_comparison.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Saved {config.tables_dir / 'tuned_vs_baseline_comparison.csv'}")
    print(f"Saved {config.reports_dir / 'tuned_vs_baseline_comparison.md'}")


if __name__ == "__main__":
    main()
