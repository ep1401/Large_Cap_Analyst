from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.no_snapshot_research import dataframe_to_markdown, summarize_backtest
from src.recommended_strategy import (
    caveat_lines,
    load_runtime_and_recommended,
    precompute_recommended_low_turnover_panels,
    run_low_turnover_recommended_backtest,
)
from src.utils import save_dataframe


TURNOVER_CAPS = [None, 0.25, 0.50, 0.75]
COST_LEVELS = [10, 20, 30]


def _summary(weekly: pd.DataFrame, label: str) -> dict[str, object]:
    summary = summarize_backtest(weekly, 5, label)
    summary["walk_forward_average_excess_vs_spy"] = float(
        pd.Series(
            [
                summary["2024_h1_excess_return_vs_spy"],
                summary["2024_h2_excess_return_vs_spy"],
                summary["2025_excess_return_vs_spy"],
            ]
        ).mean()
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-path", default=None)
    args = parser.parse_args()

    runtime, recommended, features = load_runtime_and_recommended(args.features_path)
    panels = precompute_recommended_low_turnover_panels(features, runtime, recommended)
    rows: list[dict[str, object]] = []

    for cap, cost_bps in product(TURNOVER_CAPS, COST_LEVELS):
        base_weekly, _, _ = run_low_turnover_recommended_backtest(
            panels=panels,
            top_n=10,
            cost_bps=cost_bps,
            enter_rank=10,
            hold_rank=10,
            max_holding_days=5,
            rebalance_frequency_days=5,
            strategy_name="base_partial",
            max_turnover_per_rebalance=cap,
        )
        base_summary = _summary(base_weekly, f"base_partial_{cap}_{cost_bps}")
        base_summary.update({"variant_type": "base", "max_turnover_per_rebalance": cap, "total_cost_bps": cost_bps})
        rows.append(base_summary)

        lt_weekly, _, _ = run_low_turnover_recommended_backtest(
            panels=panels,
            top_n=10,
            cost_bps=cost_bps,
            enter_rank=10,
            hold_rank=20,
            max_holding_days=20,
            rebalance_frequency_days=5,
            strategy_name="low_turnover_partial",
            max_turnover_per_rebalance=cap,
        )
        lt_summary = _summary(lt_weekly, f"lt_partial_{cap}_{cost_bps}")
        lt_summary.update({"variant_type": "low_turnover_hold_band", "max_turnover_per_rebalance": cap, "total_cost_bps": cost_bps})
        rows.append(lt_summary)

    results_df = pd.DataFrame(rows).sort_values(
        ["total_cost_bps", "walk_forward_average_excess_vs_spy", "average_turnover"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    save_dataframe(runtime.tables_dir / "partial_rebalance_test.csv", results_df)

    report_lines = [
        "# Partial Rebalance Test",
        "",
        *[f"- {line}" for line in caveat_lines()],
        "",
        dataframe_to_markdown(
            results_df[
                [
                    "variant_type",
                    "max_turnover_per_rebalance",
                    "total_cost_bps",
                    "walk_forward_average_excess_vs_spy",
                    "2025_excess_return_vs_spy",
                    "max_drawdown",
                    "average_turnover",
                ]
            ].round(4)
        ),
    ]
    (runtime.reports_dir / "partial_rebalance_test.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Saved {runtime.tables_dir / 'partial_rebalance_test.csv'}")
    print(f"Saved {runtime.reports_dir / 'partial_rebalance_test.md'}")


if __name__ == "__main__":
    main()
