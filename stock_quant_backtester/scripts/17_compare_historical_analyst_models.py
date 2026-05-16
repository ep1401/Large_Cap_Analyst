from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.backtest import run_weekly_backtest, select_rebalance_dates
from src.config import Config
from src.metrics import calculate_performance_metrics
from src.scoring import (
    strategy_analyst_data_mode,
    strategy_historical_validity_group,
    strategy_uses_historical_grade_events,
    strategy_uses_historical_ratings,
    strategy_uses_sentiment,
    strategy_uses_snapshot_fields,
)
from src.utils import load_dataframe, save_dataframe


IMPORTANT_SNAPSHOT_CAVEAT = (
    "Important caveat: analyst-driven snapshot results use FMP data as a current snapshot merged across historical dates unless true point-in-time analyst target history is provided. These results should be treated as research exploration, not a valid historical analyst-signal backtest."
)
HISTORICAL_RATING_NOTE = (
    "Historical rating-count features are built from dated FMP grades-historical records and use only the latest record available on or before each rebalance date."
)
HISTORICAL_EVENT_NOTE = (
    "Historically valid analyst signals also include dated grade action events from the FMP grades endpoint, using only events available on or before each rebalance date."
)
DEV_END = pd.Timestamp("2024-12-31")
TEST_START = pd.Timestamp("2025-01-01")


def _slice_period(df: pd.DataFrame, start: pd.Timestamp | None = None, end: pd.Timestamp | None = None) -> pd.DataFrame:
    out = df.copy()
    if start is not None:
        out = out.loc[out["date"] >= start]
    if end is not None:
        out = out.loc[out["date"] <= end]
    return out


def _safe_metrics(frame: pd.DataFrame, holding_period_days: int) -> dict[str, float]:
    if frame.empty:
        return {
            "total_return": float("nan"),
            "excess_total_return": float("nan"),
            "annualized_return": float("nan"),
            "annualized_volatility": float("nan"),
            "sharpe_ratio": float("nan"),
            "max_drawdown": float("nan"),
            "average_selected_count": float("nan"),
            "average_turnover": float("nan"),
            "number_of_rebalance_periods": 0,
        }
    return calculate_performance_metrics(frame, holding_period_days=holding_period_days)


def _metrics_row(
    strategy_name: str,
    weekly: pd.DataFrame,
    *,
    holding_period_days: int,
    top_n: int,
    min_historical_rating_count: int,
    use_regime_filter: bool,
    max_names_per_sector: int | None,
) -> dict:
    full = _safe_metrics(weekly, holding_period_days)
    dev = _safe_metrics(_slice_period(weekly, end=DEV_END), holding_period_days)
    test = _safe_metrics(_slice_period(weekly, start=TEST_START), holding_period_days)
    return {
        "strategy_name": strategy_name,
        "historical_validity_group": strategy_historical_validity_group(strategy_name),
        "analyst_data_mode": strategy_analyst_data_mode(strategy_name),
        "uses_snapshot_fields": strategy_uses_snapshot_fields(strategy_name),
        "uses_sentiment": strategy_uses_sentiment(strategy_name),
        "uses_historical_ratings": strategy_uses_historical_ratings(strategy_name),
        "uses_historical_grade_events": strategy_uses_historical_grade_events(strategy_name),
        "holding_period_days": holding_period_days,
        "top_n": top_n,
        "min_historical_rating_count": min_historical_rating_count,
        "use_regime_filter": use_regime_filter,
        "max_names_per_sector": max_names_per_sector,
        "full_period_total_return": full["total_return"],
        "development_period_total_return": dev["total_return"],
        "test_period_total_return": test["total_return"],
        "full_period_excess_return_vs_spy": full["excess_total_return"],
        "test_period_excess_return_vs_spy": test["excess_total_return"],
        "annualized_return": full["annualized_return"],
        "annualized_volatility": full["annualized_volatility"],
        "sharpe_ratio": full["sharpe_ratio"],
        "test_sharpe_ratio": test["sharpe_ratio"],
        "max_drawdown": full["max_drawdown"],
        "average_holdings": full["average_selected_count"],
        "average_turnover": full["average_turnover"],
        "number_of_rebalance_periods": full["number_of_rebalance_periods"],
    }


def _dataframe_to_markdown(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    widths = [max(len(str(header)), *(len(str(value)) for value in df[header].tolist())) for header in headers]
    header_line = "| " + " | ".join(str(header).ljust(widths[idx]) for idx, header in enumerate(headers)) + " |"
    separator = "| " + " | ".join("-" * widths[idx] for idx in range(len(headers))) + " |"
    body = []
    for _, row in df.iterrows():
        body.append("| " + " | ".join(str(row[header]).ljust(widths[idx]) for idx, header in enumerate(headers)) + " |")
    return "\n".join([header_line, separator, *body])


def _build_spy_weekly(features: pd.DataFrame, config: Config, holding_period_days: int) -> pd.DataFrame:
    future_spy_map = {5: "future_5d_spy_return", 21: "future_21d_spy_return", 63: "future_63d_spy_return"}
    portfolio_value = config.initial_capital
    rows: list[dict] = []
    for date in select_rebalance_dates(features, holding_period_days=holding_period_days, benchmark=config.benchmark):
        spy_return = float(
            features.loc[
                (features["ticker"] == config.benchmark) & (features["date"] == date),
                future_spy_map[holding_period_days],
            ].iloc[0]
        )
        portfolio_value *= 1 + spy_return
        rows.append(
            {
                "date": pd.to_datetime(date),
                "strategy_name": "SPY",
                "selected_count": 1,
                "qualified_count": 1,
                "gross_return": spy_return,
                "turnover": 0.0,
                "transaction_cost": 0.0,
                "net_return": spy_return,
                "spy_return": spy_return,
                "excess_return": 0.0,
                "portfolio_value": portfolio_value,
                "spy_value": portfolio_value,
                "exposure": 1.0,
                "regime_allowed": True,
            }
        )
    return pd.DataFrame(rows)


def _build_rating_count_diagnostics(
    features: pd.DataFrame,
    selected_holdings: pd.DataFrame,
    benchmark: str,
) -> pd.DataFrame:
    selected_daily = (
        selected_holdings.groupby("date")
        .agg(
            selected_avg_historical_rating_score=("historical_rating_score", "mean"),
            selected_avg_historical_positive_rating_ratio=("historical_positive_rating_ratio", "mean"),
            selected_avg_historical_negative_rating_ratio=("historical_negative_rating_ratio", "mean"),
        )
        .reset_index()
    ) if not selected_holdings.empty else pd.DataFrame(columns=["date"])

    rows: list[dict] = []
    for date, day in features.loc[features["ticker"] != benchmark].groupby("date"):
        day = day.copy()
        rows.append(
            {
                "date": pd.to_datetime(date),
                "total_candidates": len(day),
                "candidates_with_historical_rating_counts": int(day["historical_rating_count_data_available"].fillna(False).sum()),
                "candidates_with_total_ratings_ge_1": int((day["historical_total_ratings"].fillna(0) >= 1).sum()),
                "candidates_with_total_ratings_ge_5": int((day["historical_total_ratings"].fillna(0) >= 5).sum()),
                "candidates_with_total_ratings_ge_10": int((day["historical_total_ratings"].fillna(0) >= 10).sum()),
                "avg_historical_rating_score": float(day["historical_rating_score"].mean()),
                "avg_historical_positive_rating_ratio": float(day["historical_positive_rating_ratio"].mean()),
                "avg_historical_negative_rating_ratio": float(day["historical_negative_rating_ratio"].mean()),
                "avg_days_since_historical_rating_update": float(day["days_since_historical_rating_update"].mean()),
            }
        )

    diagnostics = pd.DataFrame(rows).merge(selected_daily, on="date", how="left")
    diagnostics["selected_avg_historical_rating_score"] = diagnostics["selected_avg_historical_rating_score"].fillna(0.0)
    diagnostics["selected_avg_historical_positive_rating_ratio"] = diagnostics[
        "selected_avg_historical_positive_rating_ratio"
    ].fillna(0.0)
    diagnostics["selected_avg_historical_negative_rating_ratio"] = diagnostics[
        "selected_avg_historical_negative_rating_ratio"
    ].fillna(0.0)
    return diagnostics.sort_values("date").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-path", default=None)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    args = parser.parse_args()

    config = Config.from_env()
    features_path = Path(args.features_path) if args.features_path else config.final_dir / "features_panel.csv"
    features = load_dataframe(features_path, parse_dates=["date"])
    if args.start_date:
        features = features.loc[features["date"] >= pd.Timestamp(args.start_date)].copy()
    if args.end_date:
        features = features.loc[features["date"] < pd.Timestamp(args.end_date)].copy()

    has_sentiment = (
        "relevance_weighted_sentiment_7d" in features.columns
        and "negative_news_ratio_7d" in features.columns
        and not features["sentiment_data_mode"].fillna("").eq("missing_news_sentiment").all()
    )
    has_rating_counts = (
        "historical_rating_count_data_available" in features.columns
        and features["historical_rating_count_data_available"].fillna(False).any()
    )
    has_grade_events = (
        "historical_grade_data_available" in features.columns
        and features["historical_grade_data_available"].fillna(False).any()
    )

    strategy_specs = [
        {"strategy_name": "technical_only", "use_analyst_filters": False},
        {"strategy_name": "technical_momentum_model", "use_analyst_filters": False},
        {"strategy_name": "analyst_snapshot_model", "use_analyst_filters": True},
        {"strategy_name": "final_quant_model_1y", "use_analyst_filters": True},
    ]
    skipped_models: list[str] = []

    if has_grade_events:
        strategy_specs.append({"strategy_name": "historical_grades_model", "use_analyst_filters": False})
    else:
        skipped_models.append("historical_grades_model (missing historical grade-event data)")

    if has_rating_counts:
        strategy_specs.append({"strategy_name": "historical_rating_counts_model", "use_analyst_filters": False})
        if has_sentiment:
            strategy_specs.append({"strategy_name": "historical_rating_counts_plus_sentiment", "use_analyst_filters": False})
        else:
            skipped_models.append("historical_rating_counts_plus_sentiment (missing sentiment features)")

        if has_grade_events:
            strategy_specs.append({"strategy_name": "historical_rating_counts_plus_events", "use_analyst_filters": False})
            if has_sentiment:
                strategy_specs.append(
                    {"strategy_name": "historical_rating_counts_plus_events_sentiment", "use_analyst_filters": False}
                )
            else:
                skipped_models.append("historical_rating_counts_plus_events_sentiment (missing sentiment features)")
        else:
            skipped_models.extend(
                [
                    "historical_rating_counts_plus_events (missing historical grade-event data)",
                    "historical_rating_counts_plus_events_sentiment (missing historical grade-event data)",
                ]
            )

        if has_sentiment:
            strategy_specs.append({"strategy_name": "final_quant_model_1y_no_snapshot", "use_analyst_filters": False})
        else:
            skipped_models.append("final_quant_model_1y_no_snapshot (missing sentiment features)")
    else:
        skipped_models.extend(
            [
                "historical_rating_counts_model (missing historical rating-count data)",
                "historical_rating_counts_plus_sentiment (missing historical rating-count data)",
                "historical_rating_counts_plus_events (missing historical rating-count data)",
                "historical_rating_counts_plus_events_sentiment (missing historical rating-count data)",
                "final_quant_model_1y_no_snapshot (missing historical rating-count data)",
            ]
        )

    comparison_rows: list[dict] = []
    selected_for_diagnostics = pd.DataFrame()

    sector_limits = [None, 3] if "sector" in features.columns and features["sector"].notna().any() else [None]
    baseline_diagnostic_key = ("historical_rating_counts_model", 21, 10, 5, False, None)

    for holding_period_days in [5, 21, 63]:
        spy_weekly = _build_spy_weekly(features, config, holding_period_days)
        comparison_rows.append(
            _metrics_row(
                "SPY",
                spy_weekly,
                holding_period_days=holding_period_days,
                top_n=1,
                min_historical_rating_count=0,
                use_regime_filter=False,
                max_names_per_sector=None,
            )
        )

        for top_n in [10, 20]:
            for min_historical_rating_count in [1, 5, 10]:
                for use_regime_filter in [False, True]:
                    for max_names_per_sector in sector_limits:
                        for spec in strategy_specs:
                            strategy_name = spec["strategy_name"]
                            try:
                                weekly, holdings, _ = run_weekly_backtest(
                                    features=features,
                                    holding_period_days=holding_period_days,
                                    benchmark=config.benchmark,
                                    top_n=top_n,
                                    initial_capital=config.initial_capital,
                                    transaction_cost_bps=config.transaction_cost_bps,
                                    use_regime_filter=use_regime_filter,
                                    regime_exposure=0.0,
                                    use_analyst_filters=spec["use_analyst_filters"],
                                    analyst_count_threshold=config.analyst_count_threshold,
                                    min_avg_dollar_volume=config.min_avg_dollar_volume,
                                    strategy_name=strategy_name,
                                    max_names_per_sector=max_names_per_sector,
                                    min_grade_events_90d=1,
                                    min_historical_rating_count=min_historical_rating_count,
                                )
                            except ValueError as exc:
                                skipped_models.append(f"{strategy_name} ({exc})")
                                continue

                            comparison_rows.append(
                                _metrics_row(
                                    strategy_name,
                                    weekly,
                                    holding_period_days=holding_period_days,
                                    top_n=top_n,
                                    min_historical_rating_count=min_historical_rating_count,
                                    use_regime_filter=use_regime_filter,
                                    max_names_per_sector=max_names_per_sector,
                                )
                            )

                            current_key = (
                                strategy_name,
                                holding_period_days,
                                top_n,
                                min_historical_rating_count,
                                use_regime_filter,
                                max_names_per_sector,
                            )
                            if current_key == baseline_diagnostic_key:
                                selected_for_diagnostics = holdings.copy()

    comparison_df = pd.DataFrame(comparison_rows).sort_values(
        ["test_period_excess_return_vs_spy", "test_sharpe_ratio", "max_drawdown"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    save_dataframe(config.tables_dir / "historical_analyst_model_comparison.csv", comparison_df)

    diagnostics_df = _build_rating_count_diagnostics(features, selected_for_diagnostics, config.benchmark)
    save_dataframe(config.tables_dir / "historical_rating_count_diagnostics.csv", diagnostics_df)

    best_by_strategy = {
        row["strategy_name"]: row for _, row in comparison_df.groupby("strategy_name", as_index=False).head(1).iterrows()
    }
    historical_rating_subset = comparison_df.loc[
        comparison_df["strategy_name"].isin(
            [
                "historical_rating_counts_model",
                "historical_rating_counts_plus_sentiment",
                "historical_rating_counts_plus_events",
                "historical_rating_counts_plus_events_sentiment",
            ]
        )
    ]
    historical_best = historical_rating_subset.iloc[0] if not historical_rating_subset.empty else None
    snapshot_best = best_by_strategy.get("analyst_snapshot_model")
    technical_best = best_by_strategy.get("technical_only")

    candidate_rows = features.loc[features["ticker"] != config.benchmark].copy()
    coverage_pct_any = float(candidate_rows["historical_rating_count_data_available"].fillna(False).mean()) if has_rating_counts else 0.0
    coverage_pct_ge_5 = float((candidate_rows["historical_total_ratings"].fillna(0) >= 5).mean()) if has_rating_counts else 0.0
    coverage_pct_ge_10 = float((candidate_rows["historical_total_ratings"].fillna(0) >= 10).mean()) if has_rating_counts else 0.0
    avg_days_since_update = float(candidate_rows["days_since_historical_rating_update"].mean()) if has_rating_counts else float("nan")

    lines = [
        "# Historical Analyst Model Comparison",
        "",
        f"- Benchmark: {config.benchmark}",
        f"- Feature panel: {features_path.name}",
        f"- {HISTORICAL_RATING_NOTE}",
        f"- {HISTORICAL_EVENT_NOTE}",
        f"- {IMPORTANT_SNAPSHOT_CAVEAT}",
        "",
        "## Coverage Summary",
        f"- Percent of universe with historical rating-count data: {coverage_pct_any:.2%}",
        f"- Percent with at least 5 ratings: {coverage_pct_ge_5:.2%}",
        f"- Percent with at least 10 ratings: {coverage_pct_ge_10:.2%}",
        f"- Average days since last rating update: {avg_days_since_update:.2f}",
        f"- Coverage appears stable enough for backtesting: {coverage_pct_ge_5 >= 0.20}",
    ]

    if skipped_models:
        lines.extend(["", "## Skipped Models", *[f"- {item}" for item in sorted(set(skipped_models))]])

    lines.extend(
        [
            "",
            "## Test Period Leaders",
            "",
            _dataframe_to_markdown(comparison_df.head(20).round(6)),
            "",
            "## Answers",
            f"- Did historical rating-count models beat SPY on the test period? {'Yes.' if historical_best is not None and historical_best['test_period_excess_return_vs_spy'] > 0 else 'No.'}",
            f"- Did historical rating-count models beat technical_only? {'Yes.' if historical_best is not None and technical_best is not None and historical_best['test_period_excess_return_vs_spy'] > technical_best['test_period_excess_return_vs_spy'] else 'No.'}",
            f"- Did historical rating-count models beat snapshot analyst models? {'Yes.' if historical_best is not None and snapshot_best is not None and historical_best['test_period_excess_return_vs_spy'] > snapshot_best['test_period_excess_return_vs_spy'] else 'No.'}",
            f"- Did sentiment improve historical rating-count models? {'Yes.' if best_by_strategy.get('historical_rating_counts_plus_sentiment') is not None and best_by_strategy.get('historical_rating_counts_model') is not None and best_by_strategy['historical_rating_counts_plus_sentiment']['test_period_excess_return_vs_spy'] > best_by_strategy['historical_rating_counts_model']['test_period_excess_return_vs_spy'] else 'No or inconclusive.'}",
            f"- How much coverage did grades-historical have? {coverage_pct_any:.2%} of the universe had any dated rating-count snapshot, {coverage_pct_ge_5:.2%} had at least 5 ratings, and {coverage_pct_ge_10:.2%} had at least 10 ratings.",
            f"- Are results limited by sparse historical analyst data? {'Yes.' if coverage_pct_ge_5 < 0.20 else 'Not materially.'}",
        ]
    )

    report_path = config.reports_dir / "historical_analyst_model_comparison.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved historical analyst comparison table to {config.tables_dir / 'historical_analyst_model_comparison.csv'}")
    print(f"Saved historical rating-count diagnostics to {config.tables_dir / 'historical_rating_count_diagnostics.csv'}")
    print(f"Saved historical analyst comparison report to {report_path}")


if __name__ == "__main__":
    main()
