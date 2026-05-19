from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.no_snapshot_research import dataframe_to_markdown, safe_metrics, slice_period
from src.recommended_strategy import caveat_lines, load_runtime_and_recommended, run_recommended_backtest
from src.utils import save_dataframe


WINDOWS = [
    ("2023 H1", "2023-01-01", "2023-06-30"),
    ("2023 H2", "2023-07-01", "2023-12-31"),
    ("2024 H1", "2024-01-01", "2024-06-30"),
    ("2024 H2", "2024-07-01", "2024-12-31"),
    ("2025 H1", "2025-01-01", "2025-06-30"),
    ("2025 H2", "2025-07-01", "2025-12-31"),
    ("Full 2023", "2023-01-01", "2023-12-31"),
    ("Full 2024", "2024-01-01", "2024-12-31"),
    ("Full 2025", "2025-01-01", "2025-12-31"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-path", default=None)
    args = parser.parse_args()

    runtime, recommended, features = load_runtime_and_recommended(args.features_path)
    weekly, _, _ = run_recommended_backtest(features=features, runtime=runtime, recommended=recommended)
    rows: list[dict[str, object]] = []
    for label, start, end in WINDOWS:
        period = slice_period(weekly, start, end)
        metrics = safe_metrics(period, recommended.holding_period_days)
        rows.append(
            {
                "window_label": label,
                "model_return": metrics["total_return"],
                "spy_return": metrics["spy_total_return"],
                "excess_return": metrics["excess_total_return"],
                "max_drawdown": metrics["max_drawdown"],
                "number_of_rebalance_periods": metrics["number_of_rebalance_periods"],
            }
        )
    results_df = pd.DataFrame(rows)
    half_year = results_df.loc[results_df["window_label"].str.contains("H"), "excess_return"]
    concentration_flag = bool((half_year > 0).sum() <= 2 or half_year.max() > half_year.mean() + 0.10)
    save_dataframe(runtime.tables_dir / "calendar_robustness.csv", results_df)

    report_lines = [
        "# Calendar Robustness",
        "",
        *[f"- {line}" for line in caveat_lines()],
        "",
        f"- Edge concentrated in one half-year/year: {concentration_flag}.",
        "",
        dataframe_to_markdown(results_df.round(4)),
    ]
    (runtime.reports_dir / "calendar_robustness.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Saved {runtime.tables_dir / 'calendar_robustness.csv'}")
    print(f"Saved {runtime.reports_dir / 'calendar_robustness.md'}")


if __name__ == "__main__":
    main()
