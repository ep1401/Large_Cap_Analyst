from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.backtest import run_weekly_backtest, select_rebalance_dates
from src.config import Config
from src.metrics import calculate_performance_metrics
from src.scoring import strategy_analyst_data_mode
from src.utils import load_dataframe, save_dataframe


IMPORTANT_CAVEAT = (
    "Important caveat: analyst-driven snapshot results use FMP data as a current snapshot merged "
    "across historical dates unless true point-in-time analyst history is provided. These results "
    "should be treated as research exploration, not a valid historical analyst-signal backtest."
)
HISTORICAL_GRADE_NOTE = (
    "Historical grade features are built from dated FMP grade events and use only events available on or before each rebalance date."
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
    avoid_recent_downgrades: bool,
    min_grade_events_90d: int,
    use_regime_filter: bool,
    max_names_per_sector: int | None,
) -> dict:
    full = _safe_metrics(weekly, holding_period_days)
    dev = _safe_metrics(_slice_period(weekly, end=DEV_END), holding_period_days)
    test = _safe_metrics(_slice_period(weekly, start=TEST_START), holding_period_days)
    return {
        "strategy_name": strategy_name,
        "analyst_data_mode": strategy_analyst_data_mode(strategy_name),
        "holding_period_days": holding_period_days,
        "top_n": top_n,
        "avoid_recent_downgrades": avoid_recent_downgrades,
        "min_grade_events_90d": min_grade_events_90d,
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


def _build_diagnostics(features: pd.DataFrame, selected_holdings: pd.DataFrame, benchmark: str) -> pd.DataFrame:
    selected_daily = (
        selected_holdings.groupby("date")
        .agg(
            selected_avg_net_upgrade_score_30d=("net_upgrade_score_30d", "mean"),
            selected_downgrade_count_30d=("downgrade_count_30d", "mean"),
        )
        .reset_index()
    )

    rows = []
    for date, day in features.loc[features["ticker"] != benchmark].groupby("date"):
        rows.append(
            {
                "date": pd.to_datetime(date),
                "total_candidates": len(day),
                "candidates_with_historical_grade_data": int(day["historical_grade_data_available"].fillna(False).sum()),
                "candidates_with_grade_event_90d": int((day["analyst_grade_event_count_90d"] > 0).sum()),
                "candidates_with_upgrade_30d": int((day["upgrade_count_30d"] > 0).sum()),
                "candidates_with_downgrade_30d": int((day["downgrade_count_30d"] > 0).sum()),
                "candidates_with_recent_downgrade_flag_30d": int(day["recent_downgrade_flag_30d"].fillna(False).sum()),
                "avg_net_upgrade_score_30d": float(day["net_upgrade_score_30d"].mean()),
                "avg_positive_grade_ratio_30d": float(day["positive_grade_ratio_30d"].mean()),
            }
        )

    diagnostics = pd.DataFrame(rows).merge(selected_daily, on="date", how="left")
    diagnostics["selected_avg_net_upgrade_score_30d"] = diagnostics["selected_avg_net_upgrade_score_30d"].fillna(0.0)
    diagnostics["selected_downgrade_count_30d"] = diagnostics["selected_downgrade_count_30d"].fillna(0.0)
    return diagnostics.sort_values("date").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-path", default=None)
    args = parser.parse_args()

    config = Config.from_env()
    features_path = Path(args.features_path) if args.features_path else config.final_dir / "features_panel.csv"
    features = load_dataframe(features_path, parse_dates=["date"])
    grades_path = config.processed_dir / "historical_analyst_grades.csv"
    grades_df = load_dataframe(grades_path, parse_dates=["date"]) if grades_path.exists() else pd.DataFrame()

    has_historical_grade_data = (
        "historical_grade_data_available" in features.columns
        and features["historical_grade_data_available"].fillna(False).any()
    )
    has_sentiment = (
        "relevance_weighted_sentiment_7d" in features.columns
        and "negative_news_ratio_7d" in features.columns
        and not features["sentiment_data_mode"].fillna("").eq("missing_news_sentiment").all()
    )

    strategy_specs = [
        {"strategy_name": "technical_only", "use_analyst_filters": False},
        {"strategy_name": "technical_momentum_model", "use_analyst_filters": False},
        {"strategy_name": "analyst_snapshot_model", "use_analyst_filters": True},
        {"strategy_name": "full_model", "use_analyst_filters": True},
    ]
    skipped_models: list[str] = []
    if has_historical_grade_data:
        strategy_specs.extend(
            [
                {"strategy_name": "historical_grades_model", "use_analyst_filters": False},
                {"strategy_name": "strict_historical_grades_checklist", "use_analyst_filters": False},
            ]
        )
        if has_sentiment:
            strategy_specs.append({"strategy_name": "historical_grades_plus_sentiment", "use_analyst_filters": False})
        else:
            skipped_models.append("historical_grades_plus_sentiment (missing sentiment features)")
    else:
        skipped_models.extend(
            [
                "historical_grades_model (missing historical grade data)",
                "historical_grades_plus_sentiment (missing historical grade data)",
                "strict_historical_grades_checklist (missing historical grade data)",
            ]
        )

    comparison_rows: list[dict] = []
    selected_for_diagnostics = pd.DataFrame()
    for holding_period_days in [5, 21, 63]:
        future_col = {5: "future_5d_spy_return", 21: "future_21d_spy_return", 63: "future_63d_spy_return"}[holding_period_days]
        portfolio_value = config.initial_capital
        spy_records = []
        rebalance_dates = select_rebalance_dates(features, holding_period_days=holding_period_days, benchmark=config.benchmark)
        for date in rebalance_dates:
            spy_return = float(
                features.loc[(features["ticker"] == config.benchmark) & (features["date"] == date), future_col].iloc[0]
            )
            portfolio_value *= 1 + spy_return
            spy_records.append(
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
        spy_weekly = pd.DataFrame(spy_records)
        comparison_rows.append(
            _metrics_row(
                "SPY",
                spy_weekly,
                holding_period_days=holding_period_days,
                top_n=1,
                avoid_recent_downgrades=False,
                min_grade_events_90d=0,
                use_regime_filter=False,
                max_names_per_sector=None,
            )
        )

        sector_limits = [None, 3] if features.get("sector") is not None and features["sector"].notna().any() else [None]
        for top_n in [10, 20]:
            for use_regime_filter in [False, True]:
                for max_names_per_sector in sector_limits:
                    for spec in strategy_specs:
                        strategy_name = spec["strategy_name"]
                        grade_grid = (
                            [
                                (avoid_recent_downgrades, min_grade_events_90d)
                                for avoid_recent_downgrades in [False, True]
                                for min_grade_events_90d in [1, 2, 3]
                            ]
                            if strategy_name in {"historical_grades_model", "historical_grades_plus_sentiment", "strict_historical_grades_checklist"}
                            else [(False, 1)]
                        )
                        for avoid_recent_downgrades, min_grade_events_90d in grade_grid:
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
                                analyst_count_threshold=10,
                                min_avg_dollar_volume=20_000_000,
                                strategy_name=strategy_name,
                                avoid_recent_downgrades=avoid_recent_downgrades,
                                min_grade_events_90d=min_grade_events_90d,
                                max_names_per_sector=max_names_per_sector,
                            )
                            comparison_rows.append(
                                _metrics_row(
                                    strategy_name,
                                    weekly,
                                    holding_period_days=holding_period_days,
                                    top_n=top_n,
                                    avoid_recent_downgrades=avoid_recent_downgrades,
                                    min_grade_events_90d=min_grade_events_90d,
                                    use_regime_filter=use_regime_filter,
                                    max_names_per_sector=max_names_per_sector,
                                )
                            )
                            if (
                                strategy_name == "historical_grades_model"
                                and holding_period_days == 21
                                and top_n == 10
                                and not avoid_recent_downgrades
                                and min_grade_events_90d == 1
                                and not use_regime_filter
                                and max_names_per_sector is None
                            ):
                                selected_for_diagnostics = holdings.copy()

    comparison_df = pd.DataFrame(comparison_rows).sort_values(
        ["test_period_excess_return_vs_spy", "test_sharpe_ratio", "max_drawdown"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    save_dataframe(config.tables_dir / "historical_analyst_model_comparison.csv", comparison_df)

    diagnostics_df = _build_diagnostics(features, selected_for_diagnostics, config.benchmark)
    save_dataframe(config.tables_dir / "historical_grade_diagnostics.csv", diagnostics_df)

    latest_by_strategy = {
        row["strategy_name"]: row for _, row in comparison_df.groupby("strategy_name", as_index=False).head(1).iterrows()
    }
    historical_best = comparison_df.loc[
        comparison_df["strategy_name"].isin(
            ["historical_grades_model", "historical_grades_plus_sentiment", "strict_historical_grades_checklist"]
        )
    ].head(10)
    technical_best = latest_by_strategy.get("technical_only")
    spy_best = latest_by_strategy.get("SPY")
    snapshot_best = latest_by_strategy.get("analyst_snapshot_model")
    historical_best_row = historical_best.iloc[0] if not historical_best.empty else None

    coverage_pct_any = float(features.loc[features["ticker"] != config.benchmark, "historical_grade_data_available"].fillna(False).mean())
    coverage_pct_90d = float(
        (features.loc[features["ticker"] != config.benchmark, "analyst_grade_event_count_90d"].fillna(0) > 0).mean()
    )
    average_events_per_ticker = float(grades_df.groupby("ticker").size().mean()) if not grades_df.empty else 0.0
    common_actions = grades_df["action"].fillna("unknown").value_counts().head(5).to_dict() if not grades_df.empty else {}
    no_coverage_tickers = sorted(set(features["ticker"].unique()) - {config.benchmark} - set(grades_df["ticker"].unique())) if not grades_df.empty else []

    lines = [
        "# Historical Analyst Model Comparison",
        "",
        f"- Benchmark: {config.benchmark}",
        f"- Feature panel: {features_path.name}",
        f"- {HISTORICAL_GRADE_NOTE}",
        f"- {IMPORTANT_CAVEAT}",
        "",
        "## Coverage Summary",
        f"- Percent of universe with any historical grade event: {coverage_pct_any:.2%}",
        f"- Percent with a grade event in last 90 days: {coverage_pct_90d:.2%}",
        f"- Average grade events per ticker: {average_events_per_ticker:.2f}",
        f"- Most common grade actions: {common_actions}",
        f"- Tickers with no historical grade coverage: {', '.join(no_coverage_tickers[:20]) if no_coverage_tickers else 'none'}",
    ]
    if skipped_models:
        lines.extend(["", "## Skipped Models", *[f"- {item}" for item in skipped_models]])

    lines.extend(
        [
            "",
            "## Diagnostics",
            f"- Historical results limited by sparse analyst event data: {coverage_pct_90d < 0.20}",
            "",
            "## Test Period Leaders",
            "",
            _dataframe_to_markdown(comparison_df.head(15).round(6)),
            "",
            "## Final Answers",
            f"- Did historical grade events improve over technical_only? {'Yes.' if historical_best_row is not None and technical_best is not None and historical_best_row['test_period_excess_return_vs_spy'] > technical_best['test_period_excess_return_vs_spy'] else 'No.'}",
            f"- Did historical grade events improve over SPY on the test period? {'Yes.' if historical_best_row is not None and historical_best_row['test_period_excess_return_vs_spy'] > 0 else 'No.'}",
            f"- Did historical grade events perform better or worse than snapshot analyst models? {'Better.' if historical_best_row is not None and snapshot_best is not None and historical_best_row['test_period_excess_return_vs_spy'] > snapshot_best['test_period_excess_return_vs_spy'] else 'Worse.'}",
            f"- How much coverage did historical grade data have? {coverage_pct_any:.2%} of the universe had any historical grade history and {coverage_pct_90d:.2%} had an event in the prior 90 days.",
            f"- Were results limited by sparse analyst event data? {'Yes.' if coverage_pct_90d < 0.20 else 'Not materially.'}",
        ]
    )

    report_path = config.reports_dir / "historical_analyst_model_comparison.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved historical analyst comparison report to {report_path}")


if __name__ == "__main__":
    main()
