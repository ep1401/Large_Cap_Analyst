from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import Config
from src.recommended_strategy import caveat_lines, default_recommended_strategy_config, save_recommended_strategy_config
from src.utils import load_dataframe


def _load_optional(path: Path) -> pd.DataFrame:
    return load_dataframe(path) if path.exists() else pd.DataFrame()


def main() -> None:
    runtime = Config.from_env()
    costs = _load_optional(runtime.tables_dir / "stress_test_costs.csv")
    holdings = _load_optional(runtime.tables_dir / "stress_test_holdings.csv")
    calendar = _load_optional(runtime.tables_dir / "calendar_robustness.csv")
    degradation = _load_optional(runtime.tables_dir / "signal_degradation_test.csv")
    groups = _load_optional(runtime.tables_dir / "signal_group_stress_test.csv")
    low_turnover = _load_optional(runtime.tables_dir / "low_turnover_hold_band_test.csv")
    rebalance = _load_optional(runtime.tables_dir / "rebalance_frequency_test.csv")
    partial = _load_optional(runtime.tables_dir / "partial_rebalance_test.csv")
    recommendation_path = runtime.reports_dir / "final_strategy_recommendation.md"
    recommendation_text = recommendation_path.read_text(encoding="utf-8") if recommendation_path.exists() else ""

    baseline_better = "Recommended strategy: `final_quant_5d_weight_tuned_no_snapshot`." in recommendation_text
    survives_30 = bool(costs.loc[costs["total_cost_bps"] == 30, "full_period_excess_return_vs_spy"].iloc[0] > 0) if not costs.empty else False
    top10_row = holdings.loc[holdings["top_n"] == 10].iloc[0] if not holdings.empty and (holdings["top_n"] == 10).any() else None
    best_holdings_row = holdings.sort_values("walk_forward_average_excess_vs_spy", ascending=False).iloc[0] if not holdings.empty else None
    top10_stable = bool(
        top10_row is not None
        and best_holdings_row is not None
        and float(top10_row["walk_forward_average_excess_vs_spy"]) >= float(best_holdings_row["walk_forward_average_excess_vs_spy"]) - 0.01
    )
    half_year = calendar.loc[calendar["window_label"].str.contains("H"), "excess_return"] if not calendar.empty else pd.Series(dtype=float)
    concentrated = bool(not half_year.empty and ((half_year > 0).sum() <= 2 or half_year.max() > half_year.mean() + 0.10))
    aggregate_degradation = (
        degradation.groupby("noise_std", as_index=False)["full_period_excess_return_vs_spy"].mean() if not degradation.empty else pd.DataFrame()
    )
    fragile = bool(
        not degradation.empty
        and float((degradation.loc[degradation["noise_std"] == 0.05, "full_period_excess_return_vs_spy"] > 0).mean()) < 0.70
    )
    essential_groups = []
    if not groups.empty:
        full = float(groups.loc[groups["variant_name"] == "full_incumbent", "walk_forward_average_excess_vs_spy"].iloc[0])
        removed = groups.loc[groups["variant_name"].str.endswith("_removed")].copy()
        essential_groups = removed.loc[removed["walk_forward_average_excess_vs_spy"] <= full - 0.02, "variant_name"].tolist()

    candidate_frames = []
    if not low_turnover.empty:
        candidate_frames.append(low_turnover.assign(source_table="low_turnover_hold_band_test"))
    if not rebalance.empty:
        rebalance = rebalance.copy()
        rebalance["enter_rank"] = rebalance["variant_type"].map({"fixed": 10, "low_turnover_hold_band": 10})
        rebalance["hold_rank"] = rebalance["variant_type"].map({"fixed": 10, "low_turnover_hold_band": 20})
        rebalance["max_turnover_per_rebalance"] = pd.NA
        candidate_frames.append(rebalance.assign(source_table="rebalance_frequency_test"))
    if not partial.empty:
        partial = partial.copy()
        partial["enter_rank"] = partial["variant_type"].map({"base": 10, "low_turnover_hold_band": 10})
        partial["hold_rank"] = partial["variant_type"].map({"base": 10, "low_turnover_hold_band": 20})
        partial["max_holding_days"] = partial["variant_type"].map({"base": 5, "low_turnover_hold_band": 20})
        partial["rebalance_frequency_days"] = 5
        candidate_frames.append(partial.assign(source_table="partial_rebalance_test"))
    candidate_df = pd.concat(candidate_frames, ignore_index=True) if candidate_frames else pd.DataFrame()
    promoted_candidate = None
    any_20_survivor = False
    any_30_survivor = False
    if not candidate_df.empty and "total_cost_bps" in candidate_df.columns:
        surviving_20 = candidate_df.loc[
            (candidate_df["total_cost_bps"] == 20)
            & (candidate_df["full_period_excess_return_vs_spy"] > 0)
            & (candidate_df["max_drawdown"] >= -0.25)
        ].copy()
        any_20_survivor = not surviving_20.empty
        any_30_survivor = bool(
            (
                (candidate_df["total_cost_bps"] == 30)
                & (candidate_df["full_period_excess_return_vs_spy"] > 0)
                & (candidate_df["max_drawdown"] >= -0.25)
            ).any()
        )
        if not surviving_20.empty:
            promoted_candidate = surviving_20.sort_values(
                ["walk_forward_average_excess_vs_spy", "average_turnover", "max_drawdown"],
                ascending=[False, True, False],
            ).iloc[0]

    if promoted_candidate is not None:
        updated = default_recommended_strategy_config()
        updated.strategy_name = "final_quant_5d_weight_tuned_low_turnover_no_snapshot"
        updated.enter_rank = int(promoted_candidate.get("enter_rank", 10)) if pd.notna(promoted_candidate.get("enter_rank", pd.NA)) else 10
        updated.hold_rank = int(promoted_candidate.get("hold_rank", 20)) if pd.notna(promoted_candidate.get("hold_rank", pd.NA)) else 20
        updated.max_holding_days = int(promoted_candidate.get("max_holding_days", 20)) if pd.notna(promoted_candidate.get("max_holding_days", pd.NA)) else 20
        updated.rebalance_frequency_days = int(promoted_candidate.get("rebalance_frequency_days", 5)) if pd.notna(promoted_candidate.get("rebalance_frequency_days", pd.NA)) else 5
        updated.total_cost_bps = float(promoted_candidate["total_cost_bps"])
        updated.max_turnover_per_rebalance = None if pd.isna(promoted_candidate.get("max_turnover_per_rebalance", pd.NA)) else float(promoted_candidate["max_turnover_per_rebalance"])
        save_recommended_strategy_config(updated, runtime.project_root)

    ready_for_paper = bool(baseline_better and top10_stable and not fragile and promoted_candidate is not None)
    report_lines = [
        "# Final Robustness Summary",
        "",
        *[f"- {line}" for line in caveat_lines()],
        "",
        f"- Is the tuned model materially better than the old baseline? {baseline_better}.",
        f"- Does it survive higher cost assumptions? {survives_30}.",
        f"- Did cost fragility improve? {promoted_candidate is not None}.",
        f"- Does any tested variant beat SPY at 20 bps? {any_20_survivor}.",
        f"- Does any tested variant beat SPY at 30 bps? {any_30_survivor}.",
        f"- Is top_n=10 stable? {top10_stable}.",
        f"- Is performance concentrated in one period? {concentrated}.",
        f"- Is it fragile to small scoring noise? {fragile}.",
        f"- Which signal groups actually matter? {', '.join(essential_groups) if essential_groups else 'no group crossed the essential threshold'}.",
        f"- Is it ready for paper trading? {ready_for_paper}.",
        f"- Recommended trading configuration if improved: {promoted_candidate.to_dict() if promoted_candidate is not None else 'none; model remains too cost-sensitive for paper trading'}",
        "- What would make it ready for real-money testing? More live paper-trading history, better capacity/slippage evidence, and stronger robustness across additional out-of-sample windows.",
    ]
    (runtime.reports_dir / "final_robustness_summary.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Saved {runtime.reports_dir / 'final_robustness_summary.md'}")


if __name__ == "__main__":
    main()
