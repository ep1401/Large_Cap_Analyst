from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.no_snapshot_research import dataframe_to_markdown, summarize_backtest
from src.recommended_strategy import caveat_lines, load_runtime_and_recommended, run_recommended_backtest
from src.utils import save_dataframe


TOP_N_VALUES = [3, 5, 8, 10, 15, 20, 25]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-path", default=None)
    args = parser.parse_args()

    runtime, recommended, features = load_runtime_and_recommended(args.features_path)
    rows: list[dict[str, object]] = []
    for top_n in TOP_N_VALUES:
        weekly, _, _ = run_recommended_backtest(features=features, runtime=runtime, recommended=recommended, top_n=top_n)
        summary = summarize_backtest(weekly, recommended.holding_period_days, f"top_n_{top_n}")
        summary["top_n"] = top_n
        summary["walk_forward_average_excess_vs_spy"] = float(
            pd.Series(
                [
                    summary["2024_h1_excess_return_vs_spy"],
                    summary["2024_h2_excess_return_vs_spy"],
                    summary["2025_excess_return_vs_spy"],
                ]
            ).mean()
        )
        summary["concentration_risk"] = 1.0 / top_n
        rows.append(summary)

    results_df = pd.DataFrame(rows).sort_values("top_n").reset_index(drop=True)
    best_row = results_df.sort_values(
        ["walk_forward_average_excess_vs_spy", "max_drawdown", "average_turnover"],
        ascending=[False, False, True],
    ).iloc[0]
    top10_row = results_df.loc[results_df["top_n"] == 10].iloc[0]
    stable_top10 = bool(top10_row["walk_forward_average_excess_vs_spy"] >= best_row["walk_forward_average_excess_vs_spy"] - 0.01)
    save_dataframe(runtime.tables_dir / "stress_test_holdings.csv", results_df)

    report_lines = [
        "# Stress Test Holdings",
        "",
        *[f"- {line}" for line in caveat_lines()],
        "",
        f"- Best top_n by walk-forward average excess vs SPY: {int(best_row['top_n'])}.",
        f"- top_n=10 stable within 1 percentage point of the best walk-forward average: {stable_top10}.",
        f"- top_n=10 concentration risk proxy (1/top_n): {top10_row['concentration_risk']:.2%}.",
        "",
        dataframe_to_markdown(
            results_df[
                [
                    "top_n",
                    "walk_forward_average_excess_vs_spy",
                    "2025_excess_return_vs_spy",
                    "max_drawdown",
                    "average_turnover",
                    "average_holdings",
                    "concentration_risk",
                ]
            ].round(4)
        ),
    ]
    (runtime.reports_dir / "stress_test_holdings.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Saved {runtime.tables_dir / 'stress_test_holdings.csv'}")
    print(f"Saved {runtime.reports_dir / 'stress_test_holdings.md'}")


if __name__ == "__main__":
    main()
