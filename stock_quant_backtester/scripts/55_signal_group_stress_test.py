from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.no_snapshot_research import (
    FINAL_5D_WEIGHT_COMPONENT_ORDER,
    build_weight_tuned_final_quant_5d_definition,
    dataframe_to_markdown,
    normalize_final_5d_weights,
    summarize_backtest,
)
from src.recommended_strategy import SIGNAL_GROUP_COMPONENTS, caveat_lines, load_promoted_tuned_weights, load_runtime_and_recommended, run_recommended_backtest
from src.utils import save_dataframe


def _variant_weights(base: dict[str, float], group_name: str, scale: float) -> dict[str, float]:
    weights = dict(base)
    for component in SIGNAL_GROUP_COMPONENTS[group_name]:
        weights[component] = weights.get(component, 0.0) * scale
    return normalize_final_5d_weights(weights)


VARIANT_GROUP_MAP = {
    "ratings": "historical_rating_counts",
    "events": "historical_grade_events",
    "technical": "technical_momentum",
    "sentiment": "sentiment",
    "risk_penalty": "risk_penalties",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-path", default=None)
    args = parser.parse_args()

    runtime, recommended, features = load_runtime_and_recommended(args.features_path)
    base_weights = load_promoted_tuned_weights(runtime.project_root)
    variants = [("full_incumbent", base_weights)]
    for variant_prefix, group_name in VARIANT_GROUP_MAP.items():
        variants.append((f"{variant_prefix}_weight_half", _variant_weights(base_weights, group_name, 0.5)))
        variants.append((f"{variant_prefix}_removed", _variant_weights(base_weights, group_name, 0.0)))

    rows: list[dict[str, object]] = []
    for variant_name, weights in variants:
        definition = build_weight_tuned_final_quant_5d_definition(weights)
        weekly, _, _ = run_recommended_backtest(
            features=features,
            runtime=runtime,
            recommended=recommended,
            definition_override=definition,
        )
        summary = summarize_backtest(weekly, recommended.holding_period_days, variant_name)
        summary["variant_name"] = variant_name
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

    results_df = pd.DataFrame(rows)
    full_row = results_df.loc[results_df["variant_name"] == "full_incumbent"].iloc[0]
    results_df["delta_vs_full_walk_forward"] = results_df["walk_forward_average_excess_vs_spy"] - float(
        full_row["walk_forward_average_excess_vs_spy"]
    )
    save_dataframe(runtime.tables_dir / "signal_group_stress_test.csv", results_df)

    essential = results_df.loc[
        results_df["variant_name"].str.endswith("_removed") & (results_df["delta_vs_full_walk_forward"] <= -0.02),
        "variant_name",
    ].tolist()
    removable = results_df.loc[
        results_df["variant_name"].str.endswith("_removed") & (results_df["delta_vs_full_walk_forward"] >= -0.005),
        "variant_name",
    ].tolist()
    report_lines = [
        "# Signal Group Stress Test",
        "",
        *[f"- {line}" for line in caveat_lines()],
        "",
        f"- Essential signal groups (removed variant loses at least 2 percentage points of walk-forward average excess): {', '.join(essential) if essential else 'none flagged'}.",
        f"- Removable or low-impact signal groups (removed variant loses less than 0.5 percentage points): {', '.join(removable) if removable else 'none flagged'}.",
        "",
        dataframe_to_markdown(
            results_df[
                [
                    "variant_name",
                    "walk_forward_average_excess_vs_spy",
                    "2025_excess_return_vs_spy",
                    "max_drawdown",
                    "average_turnover",
                    "delta_vs_full_walk_forward",
                ]
            ].round(4)
        ),
    ]
    (runtime.reports_dir / "signal_group_stress_test.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Saved {runtime.tables_dir / 'signal_group_stress_test.csv'}")
    print(f"Saved {runtime.reports_dir / 'signal_group_stress_test.md'}")


if __name__ == "__main__":
    main()
