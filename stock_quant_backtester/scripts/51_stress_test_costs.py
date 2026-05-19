from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.no_snapshot_research import dataframe_to_markdown, fmt_pct, summarize_backtest
from src.recommended_strategy import caveat_lines, load_runtime_and_recommended, run_recommended_backtest
from src.utils import save_dataframe


COST_LEVELS = [0, 5, 10, 20, 30, 50, 75, 100]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-path", default=None)
    args = parser.parse_args()

    runtime, recommended, features = load_runtime_and_recommended(args.features_path)
    rows: list[dict[str, object]] = []
    for cost_bps in COST_LEVELS:
        weekly, _, _ = run_recommended_backtest(
            features=features,
            runtime=runtime,
            recommended=recommended,
            total_cost_bps=cost_bps,
        )
        summary = summarize_backtest(weekly, recommended.holding_period_days, f"cost_{cost_bps}")
        summary["strategy_name"] = recommended.strategy_name
        summary["total_cost_bps"] = cost_bps
        summary["walk_forward_average_excess_vs_spy"] = float(
            pd.Series(
                [
                    summary["2024_h1_excess_return_vs_spy"],
                    summary["2024_h2_excess_return_vs_spy"],
                    summary["2025_excess_return_vs_spy"],
                ]
            ).mean()
        )
        rows.append(summary)

    results_df = pd.DataFrame(rows).sort_values("total_cost_bps").reset_index(drop=True)
    results_df["beats_spy_full_period"] = results_df["full_period_excess_return_vs_spy"] > 0
    break_even_cost = results_df.loc[results_df["full_period_excess_return_vs_spy"] <= 0, "total_cost_bps"].min()
    break_even_cost = None if pd.isna(break_even_cost) else int(break_even_cost)
    save_dataframe(runtime.tables_dir / "stress_test_costs.csv", results_df)

    focus = results_df.loc[results_df["total_cost_bps"].isin([20, 30, 50]), ["total_cost_bps", "beats_spy_full_period"]]
    report_lines = [
        "# Stress Test Costs",
        "",
        *[f"- {line}" for line in caveat_lines()],
        "",
        f"- Recommended config under test: `{recommended.strategy_name}`, top_n={recommended.top_n}, equal_weight, no threshold, no regime filter, long-only.",
        f"- Break-even cost where full-period excess vs SPY falls to zero or below: {break_even_cost if break_even_cost is not None else 'not reached through 100 bps'} bps.",
        f"- Beats SPY at 20 bps: {bool(focus.loc[focus['total_cost_bps'] == 20, 'beats_spy_full_period'].iloc[0])}.",
        f"- Beats SPY at 30 bps: {bool(focus.loc[focus['total_cost_bps'] == 30, 'beats_spy_full_period'].iloc[0])}.",
        f"- Beats SPY at 50 bps: {bool(focus.loc[focus['total_cost_bps'] == 50, 'beats_spy_full_period'].iloc[0])}.",
        "",
        dataframe_to_markdown(
            results_df[
                [
                    "total_cost_bps",
                    "full_period_excess_return_vs_spy",
                    "walk_forward_average_excess_vs_spy",
                    "2025_excess_return_vs_spy",
                    "max_drawdown",
                    "average_turnover",
                    "beats_spy_full_period",
                ]
            ].round(4)
        ),
    ]
    (runtime.reports_dir / "stress_test_costs.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Saved {runtime.tables_dir / 'stress_test_costs.csv'}")
    print(f"Saved {runtime.reports_dir / 'stress_test_costs.md'}")


if __name__ == "__main__":
    main()
