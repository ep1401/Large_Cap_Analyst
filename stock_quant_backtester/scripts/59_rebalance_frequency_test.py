from __future__ import annotations

import argparse
from dataclasses import replace
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


FREQUENCIES = [5, 10, 15, 21]
MAX_HOLDS = [5, 10, 15, 21, 30]
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
    rows: list[dict[str, object]] = []

    for frequency_days, max_holding_days, cost_bps in product(FREQUENCIES, MAX_HOLDS, COST_LEVELS):
        frequency_recommended = replace(recommended, rebalance_frequency_days=frequency_days)
        panels = precompute_recommended_low_turnover_panels(features, runtime, frequency_recommended)
        fixed_weekly, _, _ = run_low_turnover_recommended_backtest(
            panels=panels,
            top_n=10,
            cost_bps=cost_bps,
            enter_rank=10,
            hold_rank=10,
            max_holding_days=max_holding_days,
            rebalance_frequency_days=frequency_days,
            strategy_name=f"fixed_{frequency_days}",
        )
        fixed_summary = _summary(fixed_weekly, f"fixed_{frequency_days}_{max_holding_days}_{cost_bps}")
        fixed_summary.update(
            {
                "variant_type": "fixed",
                "rebalance_frequency_days": frequency_days,
                "max_holding_days": max_holding_days,
                "total_cost_bps": cost_bps,
            }
        )
        rows.append(fixed_summary)

        hold_band_weekly, _, _ = run_low_turnover_recommended_backtest(
            panels=panels,
            top_n=10,
            cost_bps=cost_bps,
            enter_rank=10,
            hold_rank=20,
            max_holding_days=max_holding_days,
            rebalance_frequency_days=frequency_days,
            strategy_name=f"hold_band_{frequency_days}",
        )
        hold_band_summary = _summary(hold_band_weekly, f"hold_band_{frequency_days}_{max_holding_days}_{cost_bps}")
        hold_band_summary.update(
            {
                "variant_type": "low_turnover_hold_band",
                "rebalance_frequency_days": frequency_days,
                "max_holding_days": max_holding_days,
                "total_cost_bps": cost_bps,
            }
        )
        rows.append(hold_band_summary)

    results_df = pd.DataFrame(rows).sort_values(
        ["total_cost_bps", "walk_forward_average_excess_vs_spy", "average_turnover"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    save_dataframe(runtime.tables_dir / "rebalance_frequency_test.csv", results_df)

    report_lines = [
        "# Rebalance Frequency Test",
        "",
        *[f"- {line}" for line in caveat_lines()],
        "",
        dataframe_to_markdown(
            results_df[
                [
                    "variant_type",
                    "rebalance_frequency_days",
                    "max_holding_days",
                    "total_cost_bps",
                    "walk_forward_average_excess_vs_spy",
                    "2025_excess_return_vs_spy",
                    "max_drawdown",
                    "average_turnover",
                ]
            ]
            .head(30)
            .round(4)
        ),
    ]
    (runtime.reports_dir / "rebalance_frequency_test.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Saved {runtime.tables_dir / 'rebalance_frequency_test.csv'}")
    print(f"Saved {runtime.reports_dir / 'rebalance_frequency_test.md'}")


if __name__ == "__main__":
    main()
