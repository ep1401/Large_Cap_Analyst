from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.backtest import select_rebalance_dates
from src.config import Config
from src.no_snapshot_research import dataframe_to_markdown, fmt_pct, load_features
from src.scoring import apply_filters, get_future_return_columns, get_strategy_filter_params, score_rebalance_date
from src.utils import load_dataframe, save_dataframe


BACKTEST_CAVEAT = "Back-tested performance is hypothetical."
SNAPSHOT_CAVEAT = "Snapshot analyst target models are excluded from the main historically safer ranking."
HISTORICAL_NOTE = "Historical rating-count features use dated FMP grades-historical records available on or before each rebalance date."
SENTIMENT_CAVEAT = "News sentiment depends on Alpha Vantage coverage and classification."
LONG_SHORT_CAVEAT = "Long/short is experimental and currently not recommended."
REGIME_CAVEAT = "Regime filters were tested and are not recommended for the main model based on current results."
RESEARCH_CAVEAT = "This is research/paper trading only, not financial advice."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-path", default=None)
    args = parser.parse_args()

    config = Config.from_env()
    features = load_features(config, args.features_path)
    selective_path = config.tables_dir / "selective_strategy_test.csv"
    if selective_path.exists():
        selective_df = load_dataframe(selective_path)
        selective_best = selective_df.loc[selective_df["strategy_name"] == "final_quant_5d_selective_no_snapshot"].iloc[0]
        threshold = None if pd.isna(selective_best["min_score_threshold"]) else float(selective_best["min_score_threshold"])
        top_n = int(selective_best["top_n"])
    else:
        threshold = 0.50
        top_n = 10

    future_return_column, future_spy_return_column, _ = get_future_return_columns(5)
    params = get_strategy_filter_params(
        strategy_name="final_quant_5d_selective_no_snapshot",
        use_analyst_filters=False,
        analyst_count_threshold=config.analyst_count_threshold,
        min_avg_dollar_volume=config.min_avg_dollar_volume,
        min_historical_rating_count=5,
    )
    rebalance_dates = select_rebalance_dates(features, holding_period_days=5, benchmark=config.benchmark)

    rows: list[dict[str, object]] = []
    for rebalance_date in rebalance_dates:
        day_all = features.loc[features["date"] == rebalance_date].copy()
        day = day_all.loc[day_all["ticker"] != config.benchmark].copy()
        qualified, _ = apply_filters(day, params=params, holding_period_days=5, benchmark=config.benchmark)
        scored = score_rebalance_date(qualified, strategy_name="final_quant_5d_selective_no_snapshot", use_analyst_filters=False).sort_values("score", ascending=False)
        passed = scored.copy()
        if threshold is not None:
            passed = passed.loc[pd.to_numeric(passed["score"], errors="coerce").fillna(-np.inf) > threshold].copy()
        selected = passed.head(top_n).copy()
        non_selected = scored.loc[~scored["ticker"].isin(selected["ticker"])].copy()
        top_decile_count = max(1, int(np.ceil(len(scored) * 0.10))) if not scored.empty else 0
        bottom_decile = scored.tail(top_decile_count).copy() if top_decile_count else scored.iloc[0:0].copy()
        top_decile = scored.head(top_decile_count).copy() if top_decile_count else scored.iloc[0:0].copy()
        spy_slice = day_all.loc[day_all["ticker"] == config.benchmark, future_spy_return_column]
        spy_return = float(spy_slice.iloc[0]) if not spy_slice.empty else 0.0
        selected_return = float(pd.to_numeric(selected[future_return_column], errors="coerce").mean()) if not selected.empty else 0.0
        threshold_return = float(pd.to_numeric(passed[future_return_column], errors="coerce").mean()) if not passed.empty else 0.0
        non_selected_return = float(pd.to_numeric(non_selected[future_return_column], errors="coerce").mean()) if not non_selected.empty else 0.0
        bottom_decile_return = float(pd.to_numeric(bottom_decile[future_return_column], errors="coerce").mean()) if not bottom_decile.empty else 0.0
        top_decile_return = float(pd.to_numeric(top_decile[future_return_column], errors="coerce").mean()) if not top_decile.empty else 0.0
        rows.append(
            {
                "date": rebalance_date,
                "month": str(pd.Timestamp(rebalance_date).to_period("M")),
                "selected_count": len(selected),
                "threshold_pass_count": len(passed),
                "qualified_count": len(scored),
                "selected_return": selected_return,
                "threshold_passing_return": threshold_return,
                "non_selected_return": non_selected_return,
                "bottom_decile_return": bottom_decile_return,
                "top_decile_return": top_decile_return,
                "spy_return": spy_return,
                "selected_minus_spy": selected_return - spy_return,
                "selected_minus_non_selected": selected_return - non_selected_return,
                "top_decile_minus_bottom_decile": top_decile_return - bottom_decile_return,
            }
        )

    diagnostics_df = pd.DataFrame(rows)
    save_dataframe(config.tables_dir / "score_spread_diagnostics.csv", diagnostics_df)

    month_summary = diagnostics_df.groupby("month", as_index=False).agg(
        selected_minus_spy=("selected_minus_spy", "mean"),
        selected_minus_non_selected=("selected_minus_non_selected", "mean"),
        top_decile_minus_bottom_decile=("top_decile_minus_bottom_decile", "mean"),
        periods=("date", "count"),
    ).sort_values("month")
    worked_months = month_summary.sort_values("selected_minus_spy", ascending=False).head(5)
    failed_months = month_summary.sort_values("selected_minus_spy", ascending=True).head(5)
    higher_score_predicts_higher_return = bool(diagnostics_df["top_decile_minus_bottom_decile"].mean() > 0)
    threshold_selects_better_stocks = bool(diagnostics_df["selected_minus_non_selected"].mean() > 0)

    report_lines = [
        "# Score Spread Diagnostics",
        "",
        f"- {BACKTEST_CAVEAT}",
        f"- {SNAPSHOT_CAVEAT}",
        f"- {HISTORICAL_NOTE}",
        f"- {SENTIMENT_CAVEAT}",
        f"- {LONG_SHORT_CAVEAT}",
        f"- {REGIME_CAVEAT}",
        f"- {RESEARCH_CAVEAT}",
        "",
        "## Findings",
        f"- Higher score predicts higher next-5D return: {higher_score_predicts_higher_return}.",
        f"- Threshold is selecting genuinely better stocks: {threshold_selects_better_stocks}.",
        f"- Average selected minus SPY spread: {fmt_pct(diagnostics_df['selected_minus_spy'].mean())}.",
        f"- Average selected minus non-selected spread: {fmt_pct(diagnostics_df['selected_minus_non_selected'].mean())}.",
        f"- Average top-decile minus bottom-decile spread: {fmt_pct(diagnostics_df['top_decile_minus_bottom_decile'].mean())}.",
        "",
        "## Best Months",
        "",
        dataframe_to_markdown(worked_months.round(6)),
        "",
        "## Weakest Months",
        "",
        dataframe_to_markdown(failed_months.round(6)),
        "",
        "## Monthly Summary",
        "",
        dataframe_to_markdown(month_summary.round(6)),
    ]
    (config.reports_dir / "score_spread_diagnostics.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Saved score spread table to {config.tables_dir / 'score_spread_diagnostics.csv'}")
    print(f"Saved score spread report to {config.reports_dir / 'score_spread_diagnostics.md'}")


if __name__ == "__main__":
    main()
