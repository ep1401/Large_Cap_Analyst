from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.ml_candidate_monitoring import (
    load_frozen_ml_context,
    ml_report_caveat_lines,
    run_frozen_ml_forward,
    run_frozen_rule_forward,
)
from src.no_snapshot_research import dataframe_to_markdown
from src.utils import save_dataframe


def main() -> None:
    runtime, candidate, artifact, features_forward = load_frozen_ml_context()
    ml_weekly, ml_holdings, _, ml_attr = run_frozen_ml_forward(runtime, candidate, artifact, features_forward)
    rule_weekly, rule_holdings, _, rule_attr = run_frozen_rule_forward(runtime, features_forward)

    dates = sorted(set(pd.to_datetime(ml_holdings["date"]).tolist()) | set(pd.to_datetime(rule_holdings["date"]).tolist()))
    overlap_rows: list[dict[str, object]] = []
    bucket_rows: list[dict[str, object]] = []

    for date in dates:
        ml_slice = ml_holdings.loc[ml_holdings["date"] == date].copy()
        rule_slice = rule_holdings.loc[rule_holdings["date"] == date].copy()
        ml_tickers = set(ml_slice["ticker"].tolist())
        rule_tickers = set(rule_slice["ticker"].tolist())
        shared = sorted(ml_tickers & rule_tickers)
        ml_only = sorted(ml_tickers - rule_tickers)
        rule_only = sorted(rule_tickers - ml_tickers)
        union_count = len(ml_tickers | rule_tickers)
        overlap_rows.append(
            {
                "date": date,
                "period_end_date": pd.Timestamp(ml_slice["period_end_date"].iloc[0]) if not ml_slice.empty else pd.Timestamp(rule_slice["period_end_date"].iloc[0]),
                "ml_count": len(ml_tickers),
                "rule_count": len(rule_tickers),
                "overlap_count": len(shared),
                "overlap_ratio": float(len(shared) / max(1, union_count)),
                "shared_tickers": ", ".join(shared),
                "ml_only_tickers": ", ".join(ml_only),
                "rule_only_tickers": ", ".join(rule_only),
            }
        )

        ml_attr_slice = ml_attr.loc[ml_attr["date"] == date].copy()
        rule_attr_slice = rule_attr.loc[rule_attr["date"] == date].copy()
        for bucket_name, tickers, source in [
            ("shared", shared, ml_attr_slice),
            ("ml_only", ml_only, ml_attr_slice),
            ("rule_only", rule_only, rule_attr_slice),
        ]:
            bucket = source.loc[source["ticker"].isin(tickers)].copy()
            bucket_rows.append(
                {
                    "date": date,
                    "period_end_date": pd.Timestamp(ml_slice["period_end_date"].iloc[0]) if not ml_slice.empty else pd.Timestamp(rule_slice["period_end_date"].iloc[0]),
                    "bucket": bucket_name,
                    "ticker_count": len(tickers),
                    "average_return": float(pd.to_numeric(bucket["realized_return_while_held"], errors="coerce").mean()) if not bucket.empty else float("nan"),
                    "total_contribution": float(pd.to_numeric(bucket["total_contribution"], errors="coerce").sum()) if not bucket.empty else 0.0,
                    "contribution_to_excess_return": float(pd.to_numeric(bucket["contribution_to_excess_return"], errors="coerce").sum()) if not bucket.empty else 0.0,
                }
            )

    overlap_df = pd.DataFrame(overlap_rows)
    bucket_df = pd.DataFrame(bucket_rows)
    save_dataframe(runtime.tables_dir / "ml_vs_rule_forward_holdings_overlap.csv", overlap_df)
    save_dataframe(runtime.tables_dir / "ml_only_vs_rule_only_performance.csv", bucket_df)

    aggregate_bucket = (
        bucket_df.groupby("bucket", as_index=False)
        .agg(
            average_ticker_count=("ticker_count", "mean"),
            average_return=("average_return", "mean"),
            total_contribution=("total_contribution", "sum"),
            contribution_to_excess_return=("contribution_to_excess_return", "sum"),
        )
        .sort_values("contribution_to_excess_return", ascending=False)
        .reset_index(drop=True)
    )

    report_lines = [
        "# ML vs Rule-Based Forward Holdings",
        "",
        *[f"- {line}" for line in ml_report_caveat_lines()],
        "",
        f"- Strategy: `{candidate.strategy_name}`",
        f"- Average overlap ratio: {float(overlap_df['overlap_ratio'].mean()):.2%}",
        "",
        "## Overlap By Rebalance Date",
        "",
        dataframe_to_markdown(overlap_df.round(6)),
        "",
        "## ML-Only vs Rule-Only Performance",
        "",
        dataframe_to_markdown(aggregate_bucket.round(6)),
        "",
        "## Readout",
        "",
        f"- Shared tickers ever held: {', '.join(sorted(set(', '.join(overlap_df['shared_tickers']).split(', ')) - {''})) or 'none'}",
        f"- ML-only contribution to excess return: {float(aggregate_bucket.loc[aggregate_bucket['bucket'] == 'ml_only', 'contribution_to_excess_return'].iloc[0]) if 'ml_only' in aggregate_bucket['bucket'].tolist() else 0.0:.2%}",
        f"- Rule-only contribution to excess return: {float(aggregate_bucket.loc[aggregate_bucket['bucket'] == 'rule_only', 'contribution_to_excess_return'].iloc[0]) if 'rule_only' in aggregate_bucket['bucket'].tolist() else 0.0:.2%}",
        f"- ML is genuinely selecting better names: {str((aggregate_bucket.set_index('bucket').get('contribution_to_excess_return', pd.Series())).get('ml_only', 0.0) > (aggregate_bucket.set_index('bucket').get('contribution_to_excess_return', pd.Series())).get('rule_only', 0.0)).lower()}",
    ]
    (runtime.reports_dir / "ml_vs_rule_forward_holdings.md").write_text("\n".join(report_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
