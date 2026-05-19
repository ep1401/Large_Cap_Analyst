from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.backtest import _apply_sector_limit, _build_weights
from src.config import Config
from src.no_snapshot_research import dataframe_to_markdown, fmt_pct, get_best_5d_config, load_features
from src.recommended_strategy import (
    caveat_lines,
    latest_recommended_holdings,
    load_recommended_strategy_config,
    strategy_display_from_config,
)
from src.scoring import apply_filters, get_future_return_columns, get_strategy_filter_params, score_rebalance_date, strategy_display_name
from src.utils import load_dataframe, save_dataframe


def _load_optional(path: Path) -> pd.DataFrame:
    return load_dataframe(path) if path.exists() else pd.DataFrame()


def _pick_latest_recommendations(
    features: pd.DataFrame,
    config: Config,
    strategy_name: str,
    top_n: int,
    min_score_threshold: float | None,
    allow_cash: bool,
    max_names_per_sector: int | None,
    total_cost_bps: float,
) -> pd.DataFrame:
    params = get_strategy_filter_params(
        strategy_name=strategy_name,
        use_analyst_filters=False,
        analyst_count_threshold=config.analyst_count_threshold,
        min_avg_dollar_volume=config.min_avg_dollar_volume,
        min_historical_rating_count=5,
    )
    selected = pd.DataFrame()
    latest_date = pd.Timestamp(features["date"].max())
    future_cols = get_future_return_columns(5)
    for candidate_date in sorted(pd.to_datetime(features["date"]).drop_duplicates(), reverse=True):
        day = features.loc[(features["date"] == candidate_date) & (features["ticker"] != config.benchmark)].copy()
        for column in future_cols:
            if column in day.columns:
                day[column] = day[column].fillna(0.0)
        qualified, _ = apply_filters(day, params=params, holding_period_days=5, benchmark=config.benchmark)
        scored = score_rebalance_date(qualified, strategy_name=strategy_name, use_analyst_filters=False).sort_values("score", ascending=False)
        candidate_selected = scored.copy()
        if min_score_threshold is not None:
            candidate_selected = candidate_selected.loc[
                pd.to_numeric(candidate_selected["score"], errors="coerce").fillna(-np.inf) > float(min_score_threshold)
            ].copy()
        candidate_selected = candidate_selected.head(top_n).copy()
        if not allow_cash and len(candidate_selected) < top_n:
            refill = scored.loc[~scored["ticker"].isin(candidate_selected["ticker"])].head(top_n - len(candidate_selected))
            candidate_selected = pd.concat([candidate_selected, refill], ignore_index=False).sort_values("score", ascending=False).head(top_n).copy()
        candidate_selected = _apply_sector_limit(candidate_selected, max_names_per_sector=max_names_per_sector)
        if not candidate_selected.empty:
            selected = candidate_selected
            latest_date = pd.Timestamp(candidate_date)
            break
    invested_exposure = (len(selected) / top_n) if allow_cash and top_n > 0 else (1.0 if not selected.empty else 0.0)
    selected = _build_weights(
        selected,
        exposure=invested_exposure if not selected.empty else 0.0,
        use_inverse_vol_weighting=False,
        max_single_name_weight=0.15,
        position_sizing="equal_weight",
    )
    selected = selected.copy()
    selected["date"] = latest_date
    selected["strategy_name"] = strategy_name
    selected["holding_period_days"] = 5
    selected["position_sizing"] = "equal_weight"
    selected["total_cost_bps"] = total_cost_bps
    selected["min_score_threshold"] = min_score_threshold
    selected["allow_cash"] = allow_cash
    selected["cash_weight"] = max(0.0, 1.0 - float(selected["weight"].sum())) if not selected.empty else 1.0
    keep_cols = [
        "date",
        "ticker",
        "sector",
        "rank",
        "score",
        "weight",
        "strategy_name",
        "holding_period_days",
        "position_sizing",
        "total_cost_bps",
        "min_score_threshold",
        "allow_cash",
        "cash_weight",
        "historical_rating_score",
        "net_upgrade_score_30d",
        "downgrade_count_30d",
        "relative_strength_21d",
        "relevance_weighted_sentiment_7d",
        "negative_news_ratio_7d",
        "volatility_21d",
    ]
    return selected[[column for column in keep_cols if column in selected.columns]].copy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-path", default=None)
    args = parser.parse_args()

    config = Config.from_env()
    recommended_config = load_recommended_strategy_config(config.project_root)
    features = load_features(config, args.features_path)

    selective_df = _load_optional(config.tables_dir / "selective_strategy_test.csv")
    spread_df = _load_optional(config.tables_dir / "score_spread_diagnostics.csv")
    baseline_df = _load_optional(config.tables_dir / "simple_baseline_comparison.csv")
    ablation_df = _load_optional(config.tables_dir / "ablation_analysis.csv")
    weight_search_df = _load_optional(config.tables_dir / "weight_search_5d_no_snapshot.csv")
    comparison_df = _load_optional(config.tables_dir / "tuned_vs_baseline_comparison.csv")

    if selective_df.empty:
        raise SystemExit("Missing outputs/tables/selective_strategy_test.csv. Run scripts/47_selective_strategy_test.py first.")

    baseline_row = selective_df.loc[
        (selective_df["strategy_name"] == "final_quant_5d_no_snapshot_no_sma_filter")
        & (selective_df["top_n"] == 10)
        & (selective_df["total_cost_bps"] == 10)
    ].iloc[0]
    selective_row = selective_df.loc[selective_df["strategy_name"] == "final_quant_5d_selective_no_snapshot"].iloc[0]
    historical_rows = selective_df.loc[
        selective_df["strategy_name"].isin(["historical_rating_score_only_5d", "historical_rating_score_selective_5d"])
    ].copy()
    historical_best = historical_rows.iloc[0] if not historical_rows.empty else baseline_row
    no_recent_row = selective_df.loc[
        selective_df["strategy_name"] == "final_quant_5d_no_recent_downgrade_filter_no_snapshot"
    ].iloc[0]

    tuned_promoted = bool(weight_search_df["promoted"].fillna(False).astype(bool).any()) if not weight_search_df.empty else False
    tuned_row = comparison_df.loc[comparison_df["strategy_name"] == "final_quant_5d_weight_tuned_no_snapshot"].iloc[0] if tuned_promoted and not comparison_df.empty and (comparison_df["strategy_name"] == "final_quant_5d_weight_tuned_no_snapshot").any() else None
    recommended_row = tuned_row if tuned_row is not None else baseline_row
    recommended_strategy = recommended_config.strategy_name
    recommended_top_n = recommended_config.top_n
    recommended_threshold = recommended_config.threshold
    recommended_allow_cash = recommended_config.allow_cash
    recommended_cost = recommended_config.total_cost_bps

    recommendations_df, _ = latest_recommended_holdings(features, config, recommended_config)
    save_dataframe(config.tables_dir / "current_recommendations_final_strategy.csv", recommendations_df)

    spread_summary = {
        "selected_minus_spy": float(spread_df["selected_minus_spy"].mean()) if not spread_df.empty else float("nan"),
        "selected_minus_non_selected": float(spread_df["selected_minus_non_selected"].mean()) if not spread_df.empty else float("nan"),
        "top_decile_minus_bottom_decile": float(spread_df["top_decile_minus_bottom_decile"].mean()) if not spread_df.empty else float("nan"),
    }

    comparison_anchor = tuned_row if tuned_row is not None else baseline_row
    complex_beats_historical = bool(
        comparison_anchor["walk_forward_average_excess_vs_spy"] > historical_best["walk_forward_average_excess_vs_spy"]
        and comparison_anchor["2025_excess_return_vs_spy"] > historical_best["2025_excess_return_vs_spy"]
    )
    still_paper_trading_only = True

    summary_rows = [
        {
            "strategy_name": baseline_row["strategy_name"],
            "display_name": baseline_row["display_name"],
            "walk_forward_average_excess_vs_spy": baseline_row["walk_forward_average_excess_vs_spy"],
            "windows_beating_spy": baseline_row["windows_beating_spy"],
            "2025_excess_return_vs_spy": baseline_row["2025_excess_return_vs_spy"],
            "max_drawdown": baseline_row["max_drawdown"],
            "average_percent_invested": baseline_row["average_percent_invested"],
        },
        {
            "strategy_name": selective_row["strategy_name"],
            "display_name": selective_row["display_name"],
            "walk_forward_average_excess_vs_spy": selective_row["walk_forward_average_excess_vs_spy"],
            "windows_beating_spy": selective_row["windows_beating_spy"],
            "2025_excess_return_vs_spy": selective_row["2025_excess_return_vs_spy"],
            "max_drawdown": selective_row["max_drawdown"],
            "average_percent_invested": selective_row["average_percent_invested"],
        },
        {
            "strategy_name": historical_best["strategy_name"],
            "display_name": historical_best["display_name"],
            "walk_forward_average_excess_vs_spy": historical_best["walk_forward_average_excess_vs_spy"],
            "windows_beating_spy": historical_best["windows_beating_spy"],
            "2025_excess_return_vs_spy": historical_best["2025_excess_return_vs_spy"],
            "max_drawdown": historical_best["max_drawdown"],
            "average_percent_invested": historical_best["average_percent_invested"],
        },
        {
            "strategy_name": no_recent_row["strategy_name"],
            "display_name": no_recent_row["display_name"],
            "walk_forward_average_excess_vs_spy": no_recent_row["walk_forward_average_excess_vs_spy"],
            "windows_beating_spy": no_recent_row["windows_beating_spy"],
            "2025_excess_return_vs_spy": no_recent_row["2025_excess_return_vs_spy"],
            "max_drawdown": no_recent_row["max_drawdown"],
            "average_percent_invested": no_recent_row["average_percent_invested"],
        },
    ]
    if tuned_row is not None:
        summary_rows.insert(
            1,
            {
                "strategy_name": tuned_row["strategy_name"],
                "display_name": tuned_row["display_name"],
                "walk_forward_average_excess_vs_spy": tuned_row["walk_forward_average_excess_vs_spy"],
                "windows_beating_spy": tuned_row["windows_beating_spy"],
                "2025_excess_return_vs_spy": tuned_row["2025_excess_return_vs_spy"],
                "max_drawdown": tuned_row["max_drawdown"],
                "average_percent_invested": 1.0,
            },
        )
    summary_table = pd.DataFrame(summary_rows).round(4)

    baseline_lines = []
    if not baseline_df.empty:
        baseline_5d = baseline_df.loc[baseline_df["holding_period_days"] == 5]
        if not baseline_5d.empty:
            history_only = baseline_5d.loc[baseline_5d["label"] == "top_10_historical_rating_score"]
            if not history_only.empty:
                baseline_lines.append(
                    f"- Simple historical-rating-only baseline from the earlier baseline report: {fmt_pct(history_only.iloc[0]['full_period_excess_return_vs_spy'])} full-period excess vs SPY."
                )

    ablation_lines = []
    if not ablation_df.empty:
        recent_filter = ablation_df.loc[ablation_df["label"] == "remove_recent_downgrade_filter"]
        if not recent_filter.empty:
            ablation_lines.append(
                f"- Removing the recent downgrade hard filter changed full-period excess by {fmt_pct(recent_filter.iloc[0]['delta_excess_vs_full'])}."
            )

    report_lines = [
        "# Final Strategy Recommendation",
        "",
        *[f"- {line}" for line in caveat_lines()],
        "",
        "## Current Baseline",
        f"- Current baseline strategy: `final_quant_5d_no_snapshot_no_sma_filter`.",
        f"- Walk-forward average excess vs SPY: {fmt_pct(baseline_row['walk_forward_average_excess_vs_spy'])}.",
        f"- 2025 excess vs SPY: {fmt_pct(baseline_row['2025_excess_return_vs_spy'])}.",
        f"- Windows beating SPY: {int(baseline_row['windows_beating_spy'])}/3.",
        f"- Max drawdown: {fmt_pct(baseline_row['max_drawdown'])}.",
        "",
        "## Side-By-Side",
        "",
        dataframe_to_markdown(summary_table),
        "",
        "## Selective Model",
        f"- Best selective row: threshold={selective_row['min_score_threshold']}, top_n={int(selective_row['top_n'])}, allow_cash={bool(selective_row['allow_cash'])}.",
        "- Selective model becomes recommended: False.",
        f"- Selective walk-forward average excess vs SPY: {fmt_pct(selective_row['walk_forward_average_excess_vs_spy'])}.",
        f"- Selective average percent invested: {fmt_pct(selective_row['average_percent_invested'])}.",
        f"- Selective average holdings: {selective_row['average_holdings']:.2f}.",
        "",
        "## Weight Search",
        f"- Weight-tuned model promoted: {tuned_promoted}.",
        f"- Weight search file present: {not weight_search_df.empty}.",
        *(
            [
                f"- Tuned walk-forward average excess vs SPY: {fmt_pct(tuned_row['walk_forward_average_excess_vs_spy'])}.",
                f"- Tuned 2025 excess vs SPY: {fmt_pct(tuned_row['2025_excess_return_vs_spy'])}.",
                f"- Tuned max drawdown: {fmt_pct(tuned_row['max_drawdown'])}.",
            ]
            if tuned_row is not None
            else ["- No tuned model cleared the promotion gate, so the baseline stays in place."]
        ),
        "",
        "## Historical Rating Baseline",
        f"- Best historical-rating baseline/challenger: `{historical_best['strategy_name']}`.",
        f"- Historical-rating walk-forward average excess vs SPY: {fmt_pct(historical_best['walk_forward_average_excess_vs_spy'])}.",
        f"- Complex current baseline beats the simple historical-rating baseline on both walk-forward and 2025 test: {complex_beats_historical}.",
        *baseline_lines,
        "",
        "## Score Spread Diagnostics",
        f"- Average selected minus SPY spread: {fmt_pct(spread_summary['selected_minus_spy'])}.",
        f"- Average selected minus non-selected spread: {fmt_pct(spread_summary['selected_minus_non_selected'])}.",
        f"- Average top-decile minus bottom-decile spread: {fmt_pct(spread_summary['top_decile_minus_bottom_decile'])}.",
        "",
        "## Recommendation",
        f"- Recommended strategy: `{recommended_strategy}`.",
        f"- Recommended display name: {strategy_display_from_config(recommended_config)}.",
        f"- Recommended threshold: {recommended_threshold if recommended_threshold is not None else 'none'}.",
        f"- Recommended allow_cash setting: {recommended_allow_cash}.",
        f"- Threshold/selective strategy tested and not promoted: True.",
        f"- Long/short tested and not recommended: True.",
        f"- Regime filters tested and not recommended: True.",
        f"- Standalone no-recent-downgrade hard-filter variant tested and promoted: {bool(no_recent_row['walk_forward_average_excess_vs_spy'] > baseline_row['walk_forward_average_excess_vs_spy'])}.",
        *ablation_lines,
        f"- Edge is still paper trading only: {still_paper_trading_only}.",
    ]
    (config.reports_dir / "final_strategy_recommendation.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Saved current recommendations to {config.tables_dir / 'current_recommendations_final_strategy.csv'}")
    print(f"Saved final recommendation report to {config.reports_dir / 'final_strategy_recommendation.md'}")


if __name__ == "__main__":
    main()
