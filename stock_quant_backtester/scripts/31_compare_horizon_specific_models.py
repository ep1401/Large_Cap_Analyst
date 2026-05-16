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
    get_strategy_filter_params,
    get_future_return_columns,
    score_rebalance_date,
    strategy_analyst_data_mode,
    strategy_display_name,
    strategy_historical_validity_group,
    strategy_uses_historical_grade_events,
    strategy_uses_historical_ratings,
    strategy_uses_sentiment,
    strategy_uses_snapshot_fields,
)
from src.utils import load_dataframe, save_dataframe


BACKTEST_CAVEAT = "Back-tested performance is hypothetical and does not reflect actual live performance."
SNAPSHOT_CAVEAT = (
    "Important caveat: snapshot analyst target models are excluded from the main historically safer ranking because current target data is not point-in-time historical data."
)
HISTORICAL_NOTE = (
    "Historical rating-count features are built from dated FMP grades-historical records and use only the latest record available on or before each rebalance date."
)
SENTIMENT_CAVEAT = (
    "News sentiment results depend on Alpha Vantage coverage, ticker relevance scoring, publication timestamps, and provider sentiment classification."
)
RESEARCH_CAVEAT = "This is a historical research backtest, not financial advice and not a live trading system."
DEV_END = pd.Timestamp("2024-12-31")
TEST_START = pd.Timestamp("2025-01-01")
FEATURE_COLUMNS_FOR_DIAGNOSTICS = [
    "historical_rating_score",
    "historical_positive_rating_ratio",
    "historical_negative_rating_ratio",
    "net_upgrade_score_30d",
    "downgrade_count_30d",
    "relevance_weighted_sentiment_7d",
    "relevance_weighted_sentiment_30d",
    "sentiment_change_7d_vs_30d",
    "negative_news_ratio_7d",
    "relative_strength_21d",
    "relative_strength_63d",
    "distance_to_63d_high",
    "volatility_21d",
    "beta_to_spy_63d",
]
HORIZON_STRATEGIES = {
    5: [
        "SPY",
        "technical_only",
        "sentiment_only",
        "historical_rating_counts_model",
        "historical_rating_counts_plus_events",
        "historical_rating_counts_plus_sentiment",
        "historical_rating_counts_plus_events_sentiment",
        "final_quant_5d_no_snapshot",
        "final_quant_5d_no_snapshot_loose",
        "final_quant_5d_no_snapshot_no_sma_filter",
    ],
    21: [
        "SPY",
        "technical_only",
        "sentiment_only",
        "technical_sentiment_model",
        "historical_rating_counts_plus_events",
        "historical_rating_counts_plus_events_sentiment",
        "final_quant_21d_no_snapshot",
        "final_quant_21d_no_snapshot_with_sma_filter",
        "final_quant_21d_no_snapshot_sector_capped",
    ],
    63: [
        "SPY",
        "sentiment_only",
        "technical_only",
        "technical_sentiment_model",
        "historical_rating_counts_plus_events",
        "historical_rating_counts_plus_sentiment",
        "final_quant_63d_no_snapshot",
        "final_quant_63d_no_snapshot_with_sma200_filter",
        "final_quant_63d_no_snapshot_sector_capped",
    ],
}


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
            "sharpe_ratio": float("nan"),
            "max_drawdown": float("nan"),
            "average_turnover": float("nan"),
            "average_selected_count": float("nan"),
            "number_of_rebalance_periods": 0,
            "weeks_beating_spy": float("nan"),
        }
    return calculate_performance_metrics(frame, holding_period_days=holding_period_days)


def _build_spy_weekly(features: pd.DataFrame, config: Config, holding_period_days: int) -> pd.DataFrame:
    future_return_column = get_future_return_columns(holding_period_days)[1]
    benchmark_dates = select_rebalance_dates(
        features,
        holding_period_days=holding_period_days,
        benchmark=config.benchmark,
    )
    benchmark_rows = (
        features.loc[
            (features["ticker"] == config.benchmark)
            & (features["date"].isin(benchmark_dates))
            & features[future_return_column].notna(),
            ["date", future_return_column],
        ]
        .drop_duplicates("date")
        .sort_values("date")
    )
    portfolio_value = config.initial_capital
    spy_value = config.initial_capital
    rows: list[dict] = []
    for row in benchmark_rows.itertuples(index=False):
        ret = float(getattr(row, future_return_column))
        portfolio_value *= 1 + ret
        spy_value *= 1 + ret
        rows.append(
            {
                "date": pd.to_datetime(row.date),
                "strategy_name": "SPY",
                "selected_count": 1,
                "qualified_count": 1,
                "gross_return": ret,
                "turnover": 0.0,
                "transaction_cost": 0.0,
                "net_return": ret,
                "spy_return": ret,
                "excess_return": 0.0,
                "portfolio_value": portfolio_value,
                "spy_value": spy_value,
                "exposure": 1.0,
                "regime_allowed": True,
            }
        )
    return pd.DataFrame(rows)


def _comparison_row(strategy_name: str, weekly: pd.DataFrame, holding_period_days: int) -> dict:
    full = _safe_metrics(weekly, holding_period_days)
    dev = _safe_metrics(_slice_period(weekly, end=DEV_END), holding_period_days)
    test = _safe_metrics(_slice_period(weekly, start=TEST_START), holding_period_days)
    return {
        "holding_period_days": holding_period_days,
        "strategy_name": strategy_name,
        "display_name": strategy_display_name(strategy_name),
        "historical_validity_group": strategy_historical_validity_group(strategy_name),
        "analyst_data_mode": strategy_analyst_data_mode(strategy_name),
        "uses_snapshot_fields": strategy_uses_snapshot_fields(strategy_name),
        "uses_sentiment": strategy_uses_sentiment(strategy_name),
        "uses_historical_ratings": strategy_uses_historical_ratings(strategy_name),
        "uses_historical_grade_events": strategy_uses_historical_grade_events(strategy_name),
        "full_period_total_return": full["total_return"],
        "development_period_total_return": dev["total_return"],
        "test_period_total_return": test["total_return"],
        "full_period_excess_return_vs_spy": full["excess_total_return"],
        "test_period_excess_return_vs_spy": test["excess_total_return"],
        "test_sharpe_ratio": test["sharpe_ratio"],
        "max_drawdown": full["max_drawdown"],
        "average_turnover": full["average_turnover"],
        "average_holdings": full["average_selected_count"],
        "number_of_rebalance_periods": full["number_of_rebalance_periods"],
        "periods_beating_spy": full["weeks_beating_spy"],
    }


def _dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "No rows."
    headers = list(df.columns)
    widths = [max(len(str(header)), *(len(str(value)) for value in df[header].tolist())) for header in headers]
    header_line = "| " + " | ".join(str(header).ljust(widths[idx]) for idx, header in enumerate(headers)) + " |"
    separator = "| " + " | ".join("-" * widths[idx] for idx in range(len(headers))) + " |"
    body = []
    for _, row in df.iterrows():
        body.append("| " + " | ".join(str(row[header]).ljust(widths[idx]) for idx, header in enumerate(headers)) + " |")
    return "\n".join([header_line, separator, *body])


def _build_feature_contribution_rows(features: pd.DataFrame, holdings: pd.DataFrame, strategy_name: str, holding_period_days: int, benchmark: str) -> list[dict]:
    if holdings.empty:
        return []
    merged = holdings[["date", "ticker"]].merge(features, on=["date", "ticker"], how="left")
    universe = features.loc[(features["ticker"] != benchmark) & features["date"].isin(merged["date"].unique())].copy()
    rows: list[dict] = []
    for column in FEATURE_COLUMNS_FOR_DIAGNOSTICS:
        if column not in merged.columns or column not in universe.columns:
            continue
        rows.append(
            {
                "strategy_name": strategy_name,
                "display_name": strategy_display_name(strategy_name),
                "holding_period_days": holding_period_days,
                "feature_name": column,
                "selected_mean": float(pd.to_numeric(merged[column], errors="coerce").mean()),
                "universe_mean": float(pd.to_numeric(universe[column], errors="coerce").mean()),
                "selected_minus_universe": float(pd.to_numeric(merged[column], errors="coerce").mean() - pd.to_numeric(universe[column], errors="coerce").mean()),
            }
        )
    return rows


def _current_recommendations(features: pd.DataFrame, config: Config) -> pd.DataFrame:
    latest_date = pd.to_datetime(features["date"].max())
    day = features.loc[features["date"] == latest_date].copy()
    if day.empty:
        return pd.DataFrame()
    for column in ["future_5d_return", "future_21d_return", "future_63d_return", "future_5d_spy_return", "future_21d_spy_return", "future_63d_spy_return"]:
        if column in day.columns:
            day[column] = day[column].fillna(0.0)
    recommendations: list[dict] = []
    strategy_specs = [
        ("final_quant_5d_no_snapshot", 5, None),
        ("final_quant_21d_no_snapshot", 21, None),
        ("final_quant_63d_no_snapshot", 63, None),
    ]
    for strategy_name, holding_period_days, max_names_per_sector in strategy_specs:
        params = get_strategy_filter_params(
            strategy_name=strategy_name,
            use_analyst_filters=False,
            analyst_count_threshold=config.analyst_count_threshold,
            min_avg_dollar_volume=config.min_avg_dollar_volume,
            min_historical_rating_count=5,
        )
        from src.scoring import apply_filters  # local import to avoid circular at module import time

        qualified, _ = apply_filters(day.loc[day["ticker"] != config.benchmark].copy(), params, holding_period_days, config.benchmark)
        scored = score_rebalance_date(qualified, strategy_name=strategy_name, use_analyst_filters=False).sort_values("score", ascending=False).head(config.top_n)
        if max_names_per_sector is not None and "sector" in scored.columns:
            scored = scored.groupby("sector", group_keys=False).head(max_names_per_sector)
        if scored.empty:
            continue
        weight = 1 / len(scored)
        for row in scored.itertuples(index=False):
            recommendations.append(
                {
                    "strategy_name": strategy_name,
                    "holding_period_days": holding_period_days,
                    "ticker": row.ticker,
                    "score": row.score,
                    "rank": row.rank,
                    "weight": weight,
                    "historical_rating_score": getattr(row, "historical_rating_score", None),
                    "historical_positive_rating_ratio": getattr(row, "historical_positive_rating_ratio", None),
                    "historical_negative_rating_ratio": getattr(row, "historical_negative_rating_ratio", None),
                    "net_upgrade_score_30d": getattr(row, "net_upgrade_score_30d", None),
                    "downgrade_count_30d": getattr(row, "downgrade_count_30d", None),
                    "relevance_weighted_sentiment_7d": getattr(row, "relevance_weighted_sentiment_7d", None),
                    "negative_news_ratio_7d": getattr(row, "negative_news_ratio_7d", None),
                    "relative_strength_21d": getattr(row, "relative_strength_21d", None),
                    "relative_strength_63d": getattr(row, "relative_strength_63d", None),
                    "above_sma_50": getattr(row, "above_sma_50", None),
                    "above_sma_200": getattr(row, "above_sma_200", None),
                }
            )
    return pd.DataFrame(recommendations)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-path", default=None)
    args = parser.parse_args()

    config = Config.from_env()
    features_path = Path(args.features_path) if args.features_path else config.final_dir / "features_panel_2023-01-01_2026-01-01.csv"
    features = load_dataframe(features_path, parse_dates=["date"])

    rows: list[dict] = []
    feature_diag_rows: list[dict] = []
    report_lines = [
        "# Horizon Specific Model Comparison",
        "",
        f"- {BACKTEST_CAVEAT}",
        f"- {SNAPSHOT_CAVEAT}",
        f"- {HISTORICAL_NOTE}",
        f"- {SENTIMENT_CAVEAT}",
        f"- {RESEARCH_CAVEAT}",
    ]

    for holding_period_days, strategies in HORIZON_STRATEGIES.items():
        horizon_rows: list[dict] = []
        report_lines.extend(["", f"## {holding_period_days}-Day Horizon", ""])
        for strategy_name in strategies:
            max_names_per_sector = 3 if strategy_name.endswith("sector_capped") else None
            if strategy_name == "SPY":
                weekly = _build_spy_weekly(features, config, holding_period_days)
                holdings = pd.DataFrame()
            else:
                weekly, holdings, _ = run_weekly_backtest(
                    features=features,
                    holding_period_days=holding_period_days,
                    benchmark=config.benchmark,
                    top_n=config.top_n,
                    initial_capital=config.initial_capital,
                    transaction_cost_bps=config.transaction_cost_bps,
                    use_regime_filter=False,
                    regime_exposure=0.0,
                    use_analyst_filters=False,
                    analyst_count_threshold=config.analyst_count_threshold,
                    min_avg_dollar_volume=config.min_avg_dollar_volume,
                    strategy_name=strategy_name,
                    max_names_per_sector=max_names_per_sector,
                    min_historical_rating_count=5,
                )
            row = _comparison_row(strategy_name, weekly, holding_period_days)
            horizon_rows.append(row)
            rows.append(row)
            if strategy_name in {
                "final_quant_5d_no_snapshot",
                "final_quant_21d_no_snapshot",
                "final_quant_63d_no_snapshot",
            }:
                feature_diag_rows.extend(
                    _build_feature_contribution_rows(features, holdings, strategy_name, holding_period_days, config.benchmark)
                )

        horizon_df = pd.DataFrame(horizon_rows).sort_values(
            ["test_period_excess_return_vs_spy", "test_sharpe_ratio", "max_drawdown"],
            ascending=[False, False, False],
        )
        report_lines.append(_dataframe_to_markdown(horizon_df.round(6)))

    comparison_df = pd.DataFrame(rows).sort_values(
        ["holding_period_days", "test_period_excess_return_vs_spy", "test_sharpe_ratio", "max_drawdown"],
        ascending=[True, False, False, False],
    )
    save_dataframe(config.tables_dir / "horizon_specific_model_comparison.csv", comparison_df)

    feature_diag_df = pd.DataFrame(feature_diag_rows)
    save_dataframe(config.tables_dir / "horizon_specific_selected_feature_means.csv", feature_diag_df)
    if not feature_diag_df.empty:
        report_lines.extend(["", "## Why the Model Selected These Stocks", "", _dataframe_to_markdown(feature_diag_df.round(6))])

    recommendations_df = _current_recommendations(features, config)
    save_dataframe(config.tables_dir / "current_recommendations_no_snapshot.csv", recommendations_df)

    (config.reports_dir / "horizon_specific_model_comparison.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Saved horizon comparison to {config.tables_dir / 'horizon_specific_model_comparison.csv'}")
    print(f"Saved horizon comparison report to {config.reports_dir / 'horizon_specific_model_comparison.md'}")
    print(f"Saved feature contribution diagnostics to {config.tables_dir / 'horizon_specific_selected_feature_means.csv'}")
    print(f"Saved current recommendations to {config.tables_dir / 'current_recommendations_no_snapshot.csv'}")


if __name__ == "__main__":
    main()
