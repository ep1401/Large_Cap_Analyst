from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.ml_candidate_monitoring import (
    load_frozen_ml_context,
    ml_report_caveat_lines,
    run_frozen_ml_forward,
    summarize_backtest_frame,
)
from src.no_snapshot_research import dataframe_to_markdown
from src.utils import save_dataframe


COST_LEVELS = [0, 5, 10, 20, 30, 50, 75, 100]


def _break_even_cost(df: pd.DataFrame) -> float | None:
    ordered = df.sort_values("total_cost_bps").reset_index(drop=True)
    for idx in range(1, len(ordered)):
        prev_excess = float(ordered.loc[idx - 1, "excess_vs_spy"])
        curr_excess = float(ordered.loc[idx, "excess_vs_spy"])
        if prev_excess > 0 >= curr_excess:
            prev_cost = float(ordered.loc[idx - 1, "total_cost_bps"])
            curr_cost = float(ordered.loc[idx, "total_cost_bps"])
            slope = (curr_excess - prev_excess) / (curr_cost - prev_cost)
            return prev_cost - prev_excess / slope if slope != 0 else curr_cost
    if float(ordered["excess_vs_spy"].iloc[-1]) > 0:
        return None
    return float(ordered.loc[ordered["excess_vs_spy"] <= 0, "total_cost_bps"].iloc[0])


def main() -> None:
    runtime, candidate, artifact, features_forward = load_frozen_ml_context()
    rows: list[dict[str, object]] = []
    for cost_bps in COST_LEVELS:
        weekly, _, _, _ = run_frozen_ml_forward(runtime, candidate, artifact, features_forward, cost_bps=float(cost_bps))
        metrics = summarize_backtest_frame(weekly)
        rows.append(
            {
                "total_cost_bps": cost_bps,
                "ml_return": metrics["total_return"],
                "spy_return": metrics["spy_return"],
                "excess_vs_spy": metrics["excess_vs_spy"],
                "max_drawdown": metrics["max_drawdown"],
                "turnover": metrics["average_turnover"],
                "rebalance_periods": metrics["rebalance_periods"],
            }
        )
    cost_df = pd.DataFrame(rows)
    save_dataframe(runtime.tables_dir / "ml_forward_cost_sensitivity.csv", cost_df)
    break_even = _break_even_cost(cost_df)

    report_lines = [
        "# ML Forward Cost Sensitivity",
        "",
        *[f"- {line}" for line in ml_report_caveat_lines()],
        "",
        f"- Strategy: `{candidate.strategy_name}`",
        f"- Break-even cost estimate: {'>100 bps' if break_even is None else f'{break_even:.2f} bps'}",
        f"- Beats SPY at 20 bps: {str(float(cost_df.loc[cost_df['total_cost_bps'] == 20, 'excess_vs_spy'].iloc[0]) > 0).lower()}",
        f"- Beats SPY at 30 bps: {str(float(cost_df.loc[cost_df['total_cost_bps'] == 30, 'excess_vs_spy'].iloc[0]) > 0).lower()}",
        f"- Beats SPY at 50 bps: {str(float(cost_df.loc[cost_df['total_cost_bps'] == 50, 'excess_vs_spy'].iloc[0]) > 0).lower()}",
        "",
        dataframe_to_markdown(cost_df.round(6)),
    ]
    (runtime.reports_dir / "ml_forward_cost_sensitivity.md").write_text("\n".join(report_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
