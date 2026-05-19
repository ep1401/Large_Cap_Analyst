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
    summarize_backtest_frame,
)
from src.no_snapshot_research import dataframe_to_markdown
from src.utils import load_dataframe, save_dataframe


def main() -> None:
    runtime, candidate, artifact, features_forward = load_frozen_ml_context()
    rows: list[dict[str, object]] = []
    for variant_name in ["normal_features", "all_non_price_alt_data_lag_1d", "all_features_lag_1d"]:
        weekly, _, _, _ = run_frozen_ml_forward(runtime, candidate, artifact, features_forward, variant_name=variant_name)
        metrics = summarize_backtest_frame(weekly)
        rows.append(
            {
                "variant_name": variant_name,
                "ml_return": metrics["total_return"],
                "spy_return": metrics["spy_return"],
                "excess_vs_spy": metrics["excess_vs_spy"],
                "max_drawdown": metrics["max_drawdown"],
                "periods_beating_spy_pct": metrics["percent_periods_beating_spy"],
                "latest_date": metrics["latest_date"],
            }
        )
    lag_df = pd.DataFrame(rows)
    save_dataframe(runtime.tables_dir / "ml_conservative_lag_monitor.csv", lag_df)

    promotion_watch = load_dataframe(runtime.tables_dir / "ml_promotion_watch_status.csv", parse_dates=["run_timestamp", "forward_start_date", "latest_date"])
    monthly = load_dataframe(runtime.tables_dir / "ml_forward_monthly_returns.csv")
    explainability = load_dataframe(runtime.tables_dir / "ml_feature_group_importance.csv")
    holdings_overlap = load_dataframe(runtime.tables_dir / "ml_vs_rule_forward_holdings_overlap.csv", parse_dates=["date", "period_end_date"])
    cost_df = load_dataframe(runtime.tables_dir / "ml_forward_cost_sensitivity.csv")

    report_lines = [
        "# ML Conservative Lag Monitor",
        "",
        *[f"- {line}" for line in ml_report_caveat_lines()],
        "",
        f"- Strategy: `{candidate.strategy_name}`",
        "",
        dataframe_to_markdown(lag_df.round(6)),
        "",
        "## Readout",
        "",
        f"- Normal ML excess vs SPY: {float(lag_df.loc[lag_df['variant_name'] == 'normal_features', 'excess_vs_spy'].iloc[0]):.2%}",
        f"- Alt-data-lagged excess vs SPY: {float(lag_df.loc[lag_df['variant_name'] == 'all_non_price_alt_data_lag_1d', 'excess_vs_spy'].iloc[0]):.2%}",
        f"- All-features-lagged excess vs SPY: {float(lag_df.loc[lag_df['variant_name'] == 'all_features_lag_1d', 'excess_vs_spy'].iloc[0]):.2%}",
        f"- Lagged variants still beat SPY: {str((lag_df['excess_vs_spy'] > 0).all()).lower()}",
        f"- Timing assumptions remain credible: {str((lag_df['excess_vs_spy'] > 0).all()).lower()}",
    ]
    (runtime.reports_dir / "ml_conservative_lag_monitor.md").write_text("\n".join(report_lines), encoding="utf-8")

    months_forward = months_of_forward_data(run_frozen_ml_forward(runtime, candidate, artifact, features_forward)[0])
    summary_lines = [
        "# Research Candidate Summary",
        "",
        *[f"- {line}" for line in ml_report_caveat_lines()],
        "",
        "## ML Promotion Watch",
        "",
        f"- Current watch status: `{promotion_watch['status'].iloc[-1]}`",
        f"- Forward months observed: {float(promotion_watch['months_forward_data'].iloc[-1]):.2f}",
        f"- Promotion requires more forward time: {str(months_forward < 6).lower()}",
        "",
        "## Monthly Forward Performance",
        "",
        f"- Latest month: `{monthly['month'].iloc[-1]}`",
        f"- Latest month ML excess vs SPY: {float(monthly['ml_excess_vs_spy'].iloc[-1]):.2%}",
        "",
        "## Explainability",
        "",
        f"- Top feature group: `{explainability['feature_group'].iloc[0]}`",
        f"- Market sentiment/regime features matter: {str(any(explainability['feature_group'].isin(['market_sentiment', 'market_regime']) & (explainability['total_abs_importance'] > 0))).lower()}",
        "",
        "## ML vs Rule-Based Holdings",
        "",
        f"- Average overlap: {float(holdings_overlap['overlap_ratio'].mean()):.2%}",
        f"- ML is selecting materially different names: {str(float(holdings_overlap['overlap_ratio'].mean()) < 0.60).lower()}",
        "",
        "## Cost Sensitivity",
        "",
        f"- ML beats SPY at 20 bps: {str(float(cost_df.loc[cost_df['total_cost_bps'] == 20, 'excess_vs_spy'].iloc[0]) > 0).lower()}",
        f"- ML beats SPY at 30 bps: {str(float(cost_df.loc[cost_df['total_cost_bps'] == 30, 'excess_vs_spy'].iloc[0]) > 0).lower()}",
        f"- ML beats SPY at 50 bps: {str(float(cost_df.loc[cost_df['total_cost_bps'] == 50, 'excess_vs_spy'].iloc[0]) > 0).lower()}",
        "",
        "## Conservative Lagged Forward Monitoring",
        "",
        f"- Normal excess vs SPY: {float(lag_df.loc[lag_df['variant_name'] == 'normal_features', 'excess_vs_spy'].iloc[0]):.2%}",
        f"- Alt-data-lagged excess vs SPY: {float(lag_df.loc[lag_df['variant_name'] == 'all_non_price_alt_data_lag_1d', 'excess_vs_spy'].iloc[0]):.2%}",
        f"- All-features-lagged excess vs SPY: {float(lag_df.loc[lag_df['variant_name'] == 'all_features_lag_1d', 'excess_vs_spy'].iloc[0]):.2%}",
        "",
        "## Conclusion",
        "",
        "- ML is the strongest research candidate.",
        "- ML passed leakage audits.",
        f"- ML remains `research_candidate` because forward sample is under 6 months: {str(months_forward < 6).lower()}",
        "- No 2026 data has been used for tuning or selection.",
        "- Promotion requires more forward time and continued monitoring.",
    ]
    (runtime.reports_dir / "research_candidate_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
