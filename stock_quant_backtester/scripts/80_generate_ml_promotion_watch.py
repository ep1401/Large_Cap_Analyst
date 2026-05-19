from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.ml_candidate_monitoring import (
    load_frozen_ml_context,
    ml_report_caveat_lines,
    months_of_forward_data,
    run_frozen_ml_forward,
    run_frozen_rule_forward,
    summarize_backtest_frame,
)
from src.no_snapshot_research import dataframe_to_markdown
from src.utils import load_dataframe, save_dataframe


def main() -> None:
    runtime, candidate, artifact, features_forward = load_frozen_ml_context()
    ml_weekly, _, _, _ = run_frozen_ml_forward(runtime, candidate, artifact, features_forward, variant_name="normal_features")
    rule_weekly, _, _, _ = run_frozen_rule_forward(runtime, features_forward)
    metrics = summarize_backtest_frame(ml_weekly)
    rule_metrics = summarize_backtest_frame(rule_weekly)
    months_forward = months_of_forward_data(ml_weekly)

    strict_audit = load_dataframe(runtime.tables_dir / "ml_strict_leakage_timing_audit.csv")
    leakage_report = (runtime.reports_dir / "ml_forward_no_leakage_validation.md").read_text(encoding="utf-8")
    preprocessing_report = (runtime.reports_dir / "ml_preprocessing_leakage_audit.md").read_text(encoding="utf-8")
    strict_audit_report = (runtime.reports_dir / "ml_strict_leakage_timing_audit.md").read_text(encoding="utf-8")

    alt_row = strict_audit.loc[strict_audit["variant_name"] == "all_non_price_alt_data_lag_1d"].iloc[0]
    all_row = strict_audit.loc[strict_audit["variant_name"] == "all_features_lag_1d"].iloc[0]

    criteria = {
        "ml_excess_vs_spy_positive": float(metrics["excess_vs_spy"]) > 0,
        "ml_excess_vs_rule_positive": float(metrics["total_return"] - rule_metrics["total_return"]) > 0,
        "drawdown_not_5pts_worse_than_spy": float(metrics["max_drawdown"]) >= float(metrics["spy_max_drawdown"]) - 0.05,
        "beats_spy_in_60pct_periods": float(metrics["percent_periods_beating_spy"]) >= 0.60,
        "beats_spy_at_20bps": float(metrics["excess_vs_spy"]) > 0,
        "strict_leakage_audit_pass": "Audit status: PASS" in strict_audit_report,
        "alt_data_lag_still_positive": float(alt_row["excess_vs_spy"]) > 0,
        "all_features_lag_still_positive": float(all_row["excess_vs_spy"]) > 0,
        "no_preprocessing_leakage": "FAIL" not in preprocessing_report,
        "no_2026_training_or_selection": "FAIL" not in leakage_report,
    }
    all_criteria_pass = all(criteria.values())

    if months_forward < 6:
        status = "RESEARCH_CANDIDATE_MONITORING"
    elif months_forward < 12 and all_criteria_pass:
        status = "EXTENDED_PAPER_MONITORING"
    elif months_forward >= 12 and all_criteria_pass:
        status = "PROMOTION_CANDIDATE"
    elif months_forward >= 12 and (float(metrics["excess_vs_spy"]) <= 0 or float(metrics["max_drawdown"]) < float(metrics["spy_max_drawdown"]) - 0.05):
        status = "FAILED_FORWARD_TEST"
    else:
        status = "RESEARCH_CANDIDATE_MONITORING"

    status_df = pd.DataFrame(
        [
            {
                "run_timestamp": pd.Timestamp.utcnow(),
                "strategy_name": candidate.strategy_name,
                "model_type": candidate.model_type,
                "forward_start_date": metrics["forward_start_date"],
                "latest_date": metrics["latest_date"],
                "months_forward_data": months_forward,
                "status": status,
                "ml_total_return": float(metrics["total_return"]),
                "spy_total_return": float(metrics["spy_return"]),
                "rule_total_return": float(rule_metrics["total_return"]),
                "ml_excess_vs_spy": float(metrics["excess_vs_spy"]),
                "ml_excess_vs_rule_based": float(metrics["total_return"] - rule_metrics["total_return"]),
                "ml_max_drawdown": float(metrics["max_drawdown"]),
                "spy_max_drawdown": float(metrics["spy_max_drawdown"]),
                "periods_beating_spy_pct": float(metrics["percent_periods_beating_spy"]),
                **criteria,
            }
        ]
    )
    save_dataframe(runtime.tables_dir / "ml_promotion_watch_status.csv", status_df)

    history_path = runtime.data_dir / "paper_trading" / "ml_promotion_watch_history.csv"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history = load_dataframe(history_path, parse_dates=["run_timestamp", "forward_start_date", "latest_date"]) if history_path.exists() else pd.DataFrame()
    history = pd.concat([history, status_df], ignore_index=True)
    save_dataframe(history_path, history)

    report_lines = [
        "# ML Promotion Watch",
        "",
        *[f"- {line}" for line in ml_report_caveat_lines()],
        "",
        f"- Strategy name: `{candidate.strategy_name}`",
        f"- Model type: `{candidate.model_type}`",
        f"- Forward months observed: {months_forward:.2f}",
        f"- Current status: `{status}`",
        f"- Forward window: {pd.Timestamp(metrics['forward_start_date']).date()} to {pd.Timestamp(metrics['latest_date']).date()}",
        "",
        "## Current Forward Metrics",
        "",
        f"- ML return: {float(metrics['total_return']):.2%}",
        f"- SPY return: {float(metrics['spy_return']):.2%}",
        f"- Rule-based return: {float(rule_metrics['total_return']):.2%}",
        f"- ML excess vs SPY: {float(metrics['excess_vs_spy']):.2%}",
        f"- ML excess vs rule-based: {float(metrics['total_return'] - rule_metrics['total_return']):.2%}",
        f"- ML max drawdown: {float(metrics['max_drawdown']):.2%}",
        f"- SPY max drawdown: {float(metrics['spy_max_drawdown']):.2%}",
        f"- Periods beating SPY: {metrics['periods_beating_spy']} / {metrics['rebalance_periods']} ({float(metrics['percent_periods_beating_spy']):.2%})",
        "",
        "## Status Criteria",
        "",
        dataframe_to_markdown(status_df.drop(columns=["run_timestamp"]).assign(**{c: status_df[c].astype(str) for c in criteria}).round(6)),
        "",
        "## Watch Rules",
        "",
        "- `<6 months`: `RESEARCH_CANDIDATE_MONITORING`",
        "- `6-12 months` with all criteria passing: `EXTENDED_PAPER_MONITORING`",
        "- `>=12 months` with all criteria passing: `PROMOTION_CANDIDATE`",
        "- `>=12 months` with underperformance or materially worse drawdown: `FAILED_FORWARD_TEST`",
        "- This report does not edit `recommended_strategy.yaml` automatically.",
    ]
    (runtime.reports_dir / "ml_promotion_watch.md").write_text("\n".join(report_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
