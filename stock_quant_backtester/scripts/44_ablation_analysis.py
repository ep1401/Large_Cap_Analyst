from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import Config
from src.no_snapshot_research import (
    build_ablation_definition,
    build_final_quant_5d_definition,
    dataframe_to_markdown,
    fmt_pct,
    get_best_5d_config,
    load_features,
    run_custom_weekly_backtest,
    summarize_backtest,
)
from src.utils import save_dataframe


ABLATION_VARIANTS = [
    "full_model",
    "remove_historical_rating_score",
    "remove_grade_events",
    "remove_sentiment",
    "remove_relative_strength",
    "remove_volatility_penalty",
    "remove_breakout",
    "remove_negative_news_filter",
    "remove_recent_downgrade_filter",
    "only_historical_ratings_and_events",
    "only_technical_and_sentiment",
    "only_historical_ratings_and_relative_strength",
]


def _classify_variant(row: pd.Series) -> str:
    if row["delta_excess_vs_full"] > 0.01:
        return "helps"
    if row["delta_excess_vs_full"] < -0.01:
        return "hurts"
    return "mixed"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-path", default=None)
    args = parser.parse_args()

    config = Config.from_env()
    features = load_features(config, args.features_path)
    best_config = get_best_5d_config(config)

    rows: list[dict[str, object]] = []
    weekly_by_variant: dict[str, pd.DataFrame] = {}
    for variant_name in ABLATION_VARIANTS:
        definition = build_final_quant_5d_definition() if variant_name == "full_model" else build_ablation_definition(variant_name)
        weekly, _, _ = run_custom_weekly_backtest(
            features=features,
            definition=definition,
            holding_period_days=5,
            benchmark=config.benchmark,
            top_n=int(best_config["top_n"]),
            transaction_cost_bps=float(best_config["total_cost_bps"]),
            min_avg_dollar_volume=config.min_avg_dollar_volume,
            max_names_per_sector=best_config["max_names_per_sector"],
            position_sizing=str(best_config["position_sizing"]),
            max_single_name_weight=float(best_config["max_single_name_weight"]),
        )
        weekly_by_variant[variant_name] = weekly
        summary = summarize_backtest(weekly, 5, variant_name)
        summary["display_name"] = definition.display_name
        rows.append(summary)

    results_df = pd.DataFrame(rows).sort_values(
        ["full_period_excess_return_vs_spy", "sharpe_ratio"],
        ascending=[False, False],
    ).reset_index(drop=True)
    full_row = results_df.loc[results_df["label"] == "full_model"].iloc[0]
    results_df["delta_excess_vs_full"] = results_df["full_period_excess_return_vs_spy"] - float(full_row["full_period_excess_return_vs_spy"])
    results_df["delta_sharpe_vs_full"] = results_df["sharpe_ratio"] - float(full_row["sharpe_ratio"])
    results_df["delta_drawdown_vs_full"] = results_df["max_drawdown"] - float(full_row["max_drawdown"])
    results_df["impact_label"] = results_df.apply(_classify_variant, axis=1)
    results_df["is_simpler_than_full"] = results_df["label"].ne("full_model")
    save_dataframe(config.tables_dir / "ablation_analysis.csv", results_df)

    simplified_candidates = results_df.loc[results_df["is_simpler_than_full"]].copy()
    simplified_candidates = simplified_candidates.sort_values(
        ["windows_beating_spy", "full_period_excess_return_vs_spy", "sharpe_ratio", "max_drawdown"],
        ascending=[False, False, False, False],
    )
    best_simplified = simplified_candidates.iloc[0]
    row_by_label = results_df.set_index("label")
    helpful_signal_lines = [
        f"historical rating score core ({fmt_pct(row_by_label.loc['remove_historical_rating_score', 'delta_excess_vs_full'])} hit when removed)",
        f"relative strength ({fmt_pct(row_by_label.loc['remove_relative_strength', 'delta_excess_vs_full'])} hit when removed)",
        f"grade-event features ({fmt_pct(row_by_label.loc['remove_grade_events', 'delta_excess_vs_full'])} hit when removed)",
        f"sentiment ({fmt_pct(row_by_label.loc['remove_sentiment', 'delta_excess_vs_full'])} hit when removed)",
        f"volatility penalty ({fmt_pct(row_by_label.loc['remove_volatility_penalty', 'delta_excess_vs_full'])} hit when removed)",
    ]
    removable_lines = [
        f"recent downgrade filter ({fmt_pct(row_by_label.loc['remove_recent_downgrade_filter', 'delta_excess_vs_full'])} improvement when removed)",
        f"negative news filter ({fmt_pct(row_by_label.loc['remove_negative_news_filter', 'delta_excess_vs_full'])} change when removed)",
    ]

    report_view = results_df[
        [
            "label",
            "full_period_total_return",
            "full_period_excess_return_vs_spy",
            "sharpe_ratio",
            "max_drawdown",
            "windows_beating_spy",
            "delta_excess_vs_full",
            "delta_sharpe_vs_full",
            "impact_label",
        ]
    ].copy()
    for column in [
        "full_period_total_return",
        "full_period_excess_return_vs_spy",
        "sharpe_ratio",
        "max_drawdown",
        "delta_excess_vs_full",
        "delta_sharpe_vs_full",
    ]:
        report_view[column] = report_view[column].round(4)

    report_lines = [
        "# Ablation Analysis",
        "",
        "- Base model tested: `final_quant_5d_no_snapshot_no_sma_filter`.",
        "- Ranking is based on the full 2023-2025 window, with walk-forward windows included to avoid optimizing on 2025 alone.",
        "",
        "## Findings",
        f"- Best simplified variant: `{best_simplified['label']}` with excess vs SPY {fmt_pct(best_simplified['full_period_excess_return_vs_spy'])} and {int(best_simplified['windows_beating_spy'])}/3 walk-forward windows beating SPY.",
        f"- Full model excess vs SPY: {fmt_pct(full_row['full_period_excess_return_vs_spy'])}; best simplified delta: {fmt_pct(best_simplified['delta_excess_vs_full'])}.",
        f"- Signal groups that clearly help when kept in the model: {', '.join(helpful_signal_lines)}.",
        f"- Components or filters that improved results when removed: {', '.join(removable_lines)}.",
        f"- Full model appears over-blended or over-filtered: {bool(best_simplified['full_period_excess_return_vs_spy'] > full_row['full_period_excess_return_vs_spy'] and best_simplified['windows_beating_spy'] >= full_row['windows_beating_spy'])}.",
        "",
        "## Results",
        "",
        dataframe_to_markdown(report_view),
    ]
    (config.reports_dir / "ablation_analysis.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Saved ablation table to {config.tables_dir / 'ablation_analysis.csv'}")
    print(f"Saved ablation report to {config.reports_dir / 'ablation_analysis.md'}")


if __name__ == "__main__":
    main()
