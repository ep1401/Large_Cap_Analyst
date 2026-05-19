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
    run_recommended_backtest,
)
from src.utils import save_dataframe


ENTER_RANKS = [5, 10]
HOLD_RANKS = [15, 20, 25, 30]
MAX_HOLDING_DAYS = [10, 20, 30]
COST_LEVELS = [10, 20, 30, 50]


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
    summary["beats_spy_at_cost"] = bool(summary["full_period_excess_return_vs_spy"] > 0)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-path", default=None)
    args = parser.parse_args()

    runtime, recommended, features = load_runtime_and_recommended(args.features_path)
    panels = precompute_recommended_low_turnover_panels(features, runtime, recommended)
    rows: list[dict[str, object]] = []

    for cost_bps in COST_LEVELS:
        base_weekly, _, _ = run_recommended_backtest(features, runtime, recommended, total_cost_bps=cost_bps)
        base = _summary(base_weekly, f"base_{cost_bps}")
        base.update(
            {
                "variant_name": "base_model",
                "enter_rank": pd.NA,
                "hold_rank": pd.NA,
                "max_holding_days": pd.NA,
                "total_cost_bps": cost_bps,
                "average_holding_days": 5.0,
            }
        )
        rows.append(base)

    for enter_rank, hold_rank, max_holding_days, cost_bps in product(ENTER_RANKS, HOLD_RANKS, MAX_HOLDING_DAYS, COST_LEVELS):
        weekly, holdings, _ = run_low_turnover_recommended_backtest(
            panels=panels,
            top_n=10,
            cost_bps=cost_bps,
            enter_rank=enter_rank,
            hold_rank=hold_rank,
            max_holding_days=max_holding_days,
            rebalance_frequency_days=5,
            strategy_name="final_quant_5d_weight_tuned_low_turnover_no_snapshot",
        )
        summary = _summary(weekly, f"lt_{enter_rank}_{hold_rank}_{max_holding_days}_{cost_bps}")
        summary.update(
            {
                "variant_name": "low_turnover_hold_band",
                "enter_rank": enter_rank,
                "hold_rank": hold_rank,
                "max_holding_days": max_holding_days,
                "total_cost_bps": cost_bps,
                "average_holding_days": float(holdings["holding_days"].mean()) if not holdings.empty else 0.0,
            }
        )
        rows.append(summary)

    results_df = pd.DataFrame(rows)
    cost_break_even = []
    group_cols = ["variant_name", "enter_rank", "hold_rank", "max_holding_days"]
    for keys, group in results_df.groupby(group_cols, dropna=False):
        surviving = group.loc[group["full_period_excess_return_vs_spy"] > 0, "total_cost_bps"]
        break_even = float(surviving.max()) if not surviving.empty else float("nan")
        cost_break_even.append((*keys, break_even))
    break_even_df = pd.DataFrame(cost_break_even, columns=[*group_cols, "break_even_cost_bps"])
    results_df = results_df.merge(break_even_df, on=group_cols, how="left")
    results_df["beats_spy_at_20_bps"] = results_df.apply(
        lambda row: bool(
            results_df.loc[
                (results_df["variant_name"] == row["variant_name"])
                & (results_df["enter_rank"].fillna(-1) == pd.Series([row["enter_rank"]]).fillna(-1).iloc[0])
                & (results_df["hold_rank"].fillna(-1) == pd.Series([row["hold_rank"]]).fillna(-1).iloc[0])
                & (results_df["max_holding_days"].fillna(-1) == pd.Series([row["max_holding_days"]]).fillna(-1).iloc[0])
                & (results_df["total_cost_bps"] == 20),
                "full_period_excess_return_vs_spy",
            ].iloc[0]
            > 0
        ),
        axis=1,
    )
    results_df["beats_spy_at_30_bps"] = results_df.apply(
        lambda row: bool(
            results_df.loc[
                (results_df["variant_name"] == row["variant_name"])
                & (results_df["enter_rank"].fillna(-1) == pd.Series([row["enter_rank"]]).fillna(-1).iloc[0])
                & (results_df["hold_rank"].fillna(-1) == pd.Series([row["hold_rank"]]).fillna(-1).iloc[0])
                & (results_df["max_holding_days"].fillna(-1) == pd.Series([row["max_holding_days"]]).fillna(-1).iloc[0])
                & (results_df["total_cost_bps"] == 30),
                "full_period_excess_return_vs_spy",
            ].iloc[0]
            > 0
        ),
        axis=1,
    )
    results_df = results_df.sort_values(
        ["beats_spy_at_20_bps", "walk_forward_average_excess_vs_spy", "windows_beating_spy", "average_turnover", "max_drawdown"],
        ascending=[False, False, False, True, False],
    ).reset_index(drop=True)
    save_dataframe(runtime.tables_dir / "low_turnover_hold_band_test.csv", results_df)

    report_lines = [
        "# Low Turnover Hold Band Test",
        "",
        *[f"- {line}" for line in caveat_lines()],
        "",
        dataframe_to_markdown(
            results_df[
                [
                    "variant_name",
                    "enter_rank",
                    "hold_rank",
                    "max_holding_days",
                    "total_cost_bps",
                    "walk_forward_average_excess_vs_spy",
                    "2025_excess_return_vs_spy",
                    "max_drawdown",
                    "average_turnover",
                    "average_holding_days",
                    "break_even_cost_bps",
                    "beats_spy_at_20_bps",
                    "beats_spy_at_30_bps",
                ]
            ]
            .head(25)
            .round(4)
        ),
    ]
    (runtime.reports_dir / "low_turnover_hold_band_test.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Saved {runtime.tables_dir / 'low_turnover_hold_band_test.csv'}")
    print(f"Saved {runtime.reports_dir / 'low_turnover_hold_band_test.md'}")


if __name__ == "__main__":
    main()
