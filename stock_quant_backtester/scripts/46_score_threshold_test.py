from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import Config
from src.no_snapshot_research import (
    build_final_quant_5d_definition,
    dataframe_to_markdown,
    fmt_pct,
    get_best_5d_config,
    load_features,
    run_custom_weekly_backtest,
    summarize_backtest,
)
from src.utils import save_dataframe


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-path", default=None)
    args = parser.parse_args()

    config = Config.from_env()
    features = load_features(config, args.features_path)
    best_config = get_best_5d_config(config)
    definition = build_final_quant_5d_definition()

    rows: list[dict[str, object]] = []
    threshold_specs = [
        {"threshold_type": "none", "score_threshold": None, "top_percentile": None},
        {"threshold_type": "score_gt_0", "score_threshold": 0.0, "top_percentile": None},
        {"threshold_type": "score_gt_0.25", "score_threshold": 0.25, "top_percentile": None},
        {"threshold_type": "score_gt_0.50", "score_threshold": 0.50, "top_percentile": None},
        {"threshold_type": "top_percentile_10", "score_threshold": None, "top_percentile": 0.10},
        {"threshold_type": "top_percentile_15", "score_threshold": None, "top_percentile": 0.15},
        {"threshold_type": "top_percentile_20", "score_threshold": None, "top_percentile": 0.20},
    ]

    for top_n in [5, 10, 15, 20]:
        for spec in threshold_specs:
            allow_cash_options = [False] if spec["threshold_type"] == "none" else [False, True]
            for allow_cash in allow_cash_options:
                weekly, holdings, _ = run_custom_weekly_backtest(
                    features=features,
                    definition=definition,
                    holding_period_days=5,
                    benchmark=config.benchmark,
                    top_n=top_n,
                    transaction_cost_bps=float(best_config["total_cost_bps"]),
                    min_avg_dollar_volume=config.min_avg_dollar_volume,
                    max_names_per_sector=best_config["max_names_per_sector"],
                    position_sizing=str(best_config["position_sizing"]),
                    max_single_name_weight=float(best_config["max_single_name_weight"]),
                    score_threshold=spec["score_threshold"],
                    top_percentile=spec["top_percentile"],
                    allow_cash_if_threshold_unmet=allow_cash,
                )
                summary = summarize_backtest(weekly, 5, f"{spec['threshold_type']}_top_{top_n}_{'cash' if allow_cash else 'forced'}")
                summary["top_n"] = top_n
                summary["threshold_type"] = spec["threshold_type"]
                summary["score_threshold"] = spec["score_threshold"]
                summary["top_percentile"] = spec["top_percentile"]
                summary["allow_cash"] = allow_cash
                summary["average_percent_invested"] = float(weekly["exposure"].mean()) if not weekly.empty else 0.0
                summary["average_number_of_holdings"] = float(holdings.groupby("date")["ticker"].nunique().mean()) if not holdings.empty else 0.0
                summary["selected_periods_with_cash"] = int((weekly["exposure"] < 0.9999).sum()) if not weekly.empty else 0
                rows.append(summary)

    results_df = pd.DataFrame(rows).sort_values(
        ["full_period_excess_return_vs_spy", "sharpe_ratio", "max_drawdown"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    save_dataframe(config.tables_dir / "score_threshold_test.csv", results_df)

    threshold_only = results_df.loc[results_df["threshold_type"] != "none"].copy()
    best_selective = threshold_only.sort_values(
        ["windows_beating_spy", "full_period_excess_return_vs_spy", "sharpe_ratio", "max_drawdown"],
        ascending=[False, False, False, False],
    ).iloc[0]
    comparable = threshold_only.pivot_table(
        index=["top_n", "threshold_type"],
        columns="allow_cash",
        values=["full_period_excess_return_vs_spy", "sharpe_ratio"],
    )
    forced_buying_hurts = False
    best_pair_improvement = float("nan")
    if not comparable.empty and ("full_period_excess_return_vs_spy", True) in comparable.columns and ("full_period_excess_return_vs_spy", False) in comparable.columns:
        improvement = comparable[("full_period_excess_return_vs_spy", True)] - comparable[("full_period_excess_return_vs_spy", False)]
        best_pair_improvement = float(improvement.max()) if not improvement.dropna().empty else float("nan")
        forced_buying_hurts = bool(pd.notna(best_pair_improvement) and best_pair_improvement > 0.01)

    report_view = results_df[
        [
            "top_n",
            "threshold_type",
            "allow_cash",
            "full_period_excess_return_vs_spy",
            "sharpe_ratio",
            "max_drawdown",
            "average_percent_invested",
            "average_number_of_holdings",
            "windows_beating_spy",
        ]
    ].copy()
    for column in [
        "full_period_excess_return_vs_spy",
        "sharpe_ratio",
        "max_drawdown",
        "average_percent_invested",
        "average_number_of_holdings",
    ]:
        report_view[column] = report_view[column].round(4)

    report_lines = [
        "# Score Threshold Test",
        "",
        f"- Best selective configuration: `{best_selective['threshold_type']}` with `top_n={int(best_selective['top_n'])}` and `allow_cash={bool(best_selective['allow_cash'])}`.",
        f"- Best selective excess vs SPY: {fmt_pct(best_selective['full_period_excess_return_vs_spy'])}.",
        f"- Best selective Sharpe: {best_selective['sharpe_ratio']:.4f}.",
        f"- Best selective max drawdown: {fmt_pct(best_selective['max_drawdown'])}.",
        f"- Average percent invested for best selective setup: {fmt_pct(best_selective['average_percent_invested'])}.",
        f"- Average number of holdings for best selective setup: {best_selective['average_number_of_holdings']:.2f}.",
        f"- Forced buying hurts in at least one like-for-like threshold comparison: {forced_buying_hurts}.",
        f"- Best cash-vs-forced excess-return improvement from allowing cash: {fmt_pct(best_pair_improvement)}.",
        "",
        "## Results",
        "",
        dataframe_to_markdown(report_view),
    ]
    (config.reports_dir / "score_threshold_test.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Saved threshold test table to {config.tables_dir / 'score_threshold_test.csv'}")
    print(f"Saved threshold test report to {config.reports_dir / 'score_threshold_test.md'}")


if __name__ == "__main__":
    main()
