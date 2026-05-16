from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import Config
from src.utils import load_dataframe


IMPORTANT_CAVEAT = (
    "Important caveat: analyst-driven snapshot results use FMP data as a current snapshot merged "
    "across historical dates unless true point-in-time analyst history is provided. These results "
    "should be treated as research exploration, not a valid historical analyst-signal backtest."
)


def _format_pct(value: float) -> str:
    return f"{value:.2%}"


def _dataframe_to_markdown(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    widths = [max(len(str(header)), *(len(str(value)) for value in df[header].tolist())) for header in headers]
    header_line = "| " + " | ".join(str(header).ljust(widths[idx]) for idx, header in enumerate(headers)) + " |"
    separator = "| " + " | ".join("-" * widths[idx] for idx in range(len(headers))) + " |"
    body = []
    for _, row in df.iterrows():
        body.append("| " + " | ".join(str(row[header]).ljust(widths[idx]) for idx, header in enumerate(headers)) + " |")
    return "\n".join([header_line, separator, *body])


def main() -> None:
    config = Config.from_env()
    comparison_full = load_dataframe(config.tables_dir / "strategy_comparison_full.csv")
    comparison_dev = load_dataframe(config.tables_dir / "strategy_comparison_dev.csv")
    comparison_test = load_dataframe(config.tables_dir / "strategy_comparison_test.csv")
    validation = load_dataframe(config.tables_dir / "backtest_validation.csv")
    diagnostics_path = config.tables_dir / "filter_diagnostics.csv"
    diagnostics = load_dataframe(diagnostics_path, parse_dates=["date"]) if diagnostics_path.exists() else pd.DataFrame()
    sentiment_diagnostics_path = config.tables_dir / "sentiment_diagnostics.csv"
    sentiment_diagnostics = (
        load_dataframe(sentiment_diagnostics_path, parse_dates=["date"]) if sentiment_diagnostics_path.exists() else pd.DataFrame()
    )
    weekly = load_dataframe(config.tables_dir / "weekly_portfolio_returns.csv", parse_dates=["date"])
    features = load_dataframe(config.final_dir / "features_panel.csv", parse_dates=["date"])
    chart_paths = sorted(path.name for path in Path(config.charts_dir).glob("*.png"))

    analyst_mode = (
        features["analyst_data_mode"].dropna().iloc[0]
        if "analyst_data_mode" in features.columns and features["analyst_data_mode"].notna().any()
        else "historical_backtest_without_analyst"
    )
    full_model_full = comparison_full.loc[comparison_full["strategy_name"] == "full_model"].iloc[0]
    full_model_test = comparison_test.loc[comparison_test["strategy_name"] == "full_model"].iloc[0]
    under_diversified = float(full_model_full["average_selected_count"]) < 3
    validation_diff = float(validation["absolute_difference"].iloc[0])
    diagnostics_summary: list[str] = []

    if not diagnostics.empty:
        avg_final_pass_count = float(diagnostics["final_pass_count"].mean())
        pct_fewer_than_top_n = float((diagnostics["selected_count"] < int(full_model_full["top_n"])).mean())
        drop_columns = [
            ("liquidity", "starting_universe_count", "passed_liquidity_count"),
            ("analyst_count", "passed_liquidity_count", "passed_analyst_count"),
            ("consensus_upside", "passed_analyst_count", "passed_consensus_upside_count"),
            ("low_target_upside", "passed_consensus_upside_count", "passed_low_target_upside_count"),
            ("revision_7d", "passed_low_target_upside_count", "passed_revision_7d_count"),
            ("revision_30d", "passed_revision_7d_count", "passed_revision_30d_count"),
            ("resistance_breakout", "passed_revision_30d_count", "passed_resistance_count"),
        ]

    sentiment_summary: list[str] = []
    if not sentiment_diagnostics.empty:
        coverage_pct = float(sentiment_diagnostics["coverage_pct_7d"].mean())
        avg_article_count = float(sentiment_diagnostics["avg_article_count_7d"].mean())
        selected_positive_pct = float(sentiment_diagnostics["selected_positive_sentiment_pct"].mean())
        restrictive = coverage_pct < 0.15 or float(
            (sentiment_diagnostics["candidates_with_news_7d"] < sentiment_diagnostics["total_candidates"] * 0.1).mean()
        ) > 0.5
        sentiment_summary = [
            "",
            "## Sentiment Coverage Diagnostics",
            f"- Percent of universe with at least one article in prior 7 days: {_format_pct(coverage_pct)}",
            f"- Average article_count_7d: {avg_article_count:.2f}",
            f"- Percent of selected holdings with positive sentiment: {_format_pct(selected_positive_pct)}",
            f"- Sentiment filters appear too restrictive: {restrictive}",
        ]
        drop_stats: list[tuple[str, float]] = []
        for name, before_col, after_col in drop_columns:
            if before_col in diagnostics.columns and after_col in diagnostics.columns:
                drop_stats.append((name, float((diagnostics[before_col] - diagnostics[after_col]).mean())))
        most_restrictive_filter = max(drop_stats, key=lambda item: item[1])[0] if drop_stats else "n/a"
        diagnostics_summary = [
            "",
            "## Filter Diagnostics",
            f"- Average final pass count: {avg_final_pass_count:.2f}",
            f"- Percent of periods with fewer than top_n candidates: {_format_pct(pct_fewer_than_top_n)}",
            f"- Most restrictive filter by average drop count: {most_restrictive_filter}",
            f"- Under-diversified under current settings: {under_diversified or pct_fewer_than_top_n > 0.5}",
        ]

    lines = [
        "# Backtest Summary",
        "",
        "## Strategy Settings",
        f"- Start date: {config.start_date}",
        f"- End date: {config.end_date}",
        f"- Benchmark: {config.benchmark}",
        f"- Holding period days: {int(full_model_full['holding_period_days'])}",
        f"- Top N: {int(full_model_full['top_n'])}",
        f"- Transaction cost bps: {config.transaction_cost_bps}",
        f"- Regime filter enabled: {bool(full_model_full['use_regime_filter'])}",
        f"- Regime exposure when blocked: {full_model_full['regime_exposure']}",
        f"- Analyst count threshold: {int(full_model_full['analyst_count_threshold'])}",
        f"- Minimum avg dollar volume: {full_model_full['min_avg_dollar_volume']}",
        "",
        "## Analyst Data Mode",
        f"- Analyst data mode: {analyst_mode}",
        f"- Point-in-time analyst history available: {bool(full_model_full['analyst_data_is_point_in_time'])}",
        f"- {IMPORTANT_CAVEAT}",
        "",
        "## Benchmark Validation",
        f"- First rebalance date: {validation['first_rebalance_date'].iloc[0]}",
        f"- Last rebalance date: {validation['last_rebalance_date'].iloc[0]}",
        f"- Number of rebalance periods: {int(validation['number_of_rebalance_periods'].iloc[0])}",
        f"- Compounded SPY return from backtest: {_format_pct(validation['compounded_spy_return_from_backtest'].iloc[0])}",
        f"- Direct SPY buy-and-hold return: {_format_pct(validation['direct_spy_buy_hold_return'].iloc[0])}",
        f"- Absolute difference: {_format_pct(validation_diff)}",
        "",
        "## Dataset Summary",
        f"- Universe size including benchmark rows: {features['ticker'].nunique()}",
        f"- Feature rows: {len(features)}",
        f"- Feature date range: {features['date'].min().date()} to {features['date'].max().date()}",
        "",
        "## Test Period Emphasis",
        f"- Full model test-period total return: {_format_pct(full_model_test['total_return'])}",
        f"- Full model test-period excess return vs SPY: {_format_pct(full_model_test['excess_total_return'])}",
        f"- Full model test-period Sharpe: {full_model_test['sharpe_ratio']:.2f}",
        f"- Full model test-period max drawdown: {_format_pct(full_model_test['max_drawdown'])}",
        "",
        "## Strategy Comparison - Full Period",
        "",
        _dataframe_to_markdown(comparison_full.round(6)),
        "",
        "## Strategy Comparison - Development Period",
        "",
        _dataframe_to_markdown(comparison_dev.round(6)),
        "",
        "## Strategy Comparison - Test Period",
        "",
        _dataframe_to_markdown(comparison_test.round(6)),
        "",
        "## Full Model Snapshot",
        f"- Average number of holdings: {full_model_full['average_selected_count']:.2f}",
        f"- Average turnover: {full_model_full['average_turnover']:.2f}",
        f"- Number of invested periods: {int(full_model_full['number_of_invested_periods'])}",
        f"- Full-period annualized return: {_format_pct(full_model_full['annualized_return'])}",
        f"- Test-period annualized return: {_format_pct(full_model_test['annualized_return'])}",
        f"- Full-period total return: {_format_pct(full_model_full['total_return'])}",
        f"- Test-period total return: {_format_pct(full_model_test['total_return'])}",
        f"- Full-period SPY total return: {_format_pct(full_model_full['spy_total_return'])}",
        f"- Test-period SPY total return: {_format_pct(full_model_test['spy_total_return'])}",
        f"- Full-period weeks beating SPY: {full_model_full['weeks_beating_spy']:.2%}",
        f"- Test-period weeks beating SPY: {full_model_test['weeks_beating_spy']:.2%}",
    ]

    lines.extend(diagnostics_summary)
    lines.extend(sentiment_summary)

    if under_diversified:
        lines.extend(
            [
                "",
                "## Warning",
                "- The average selected count is very low, so the strategy appears under-diversified under the current settings.",
            ]
        )

    if validation_diff > 0.05:
        lines.extend(
            [
                "",
                "## Validation Warning",
                "- Benchmark validation difference is above tolerance. Benchmark compounding may still be inconsistent.",
            ]
        )

    lines.extend(
        [
            "",
            "## Key Caveats",
            "- This project is a historical research backtest, not financial advice and not a live trading system.",
            f"- {IMPORTANT_CAVEAT}",
            "- The initial universe uses a static large-cap stock list, which introduces survivorship bias.",
            "- holding_period_days controls both the return horizon and the rebalance frequency in the corrected non-overlapping engine.",
            "- Transaction costs are modeled using turnover-based costs.",
            "- The SPY 200-day moving-average regime filter is optional and changes exposure rather than signal quality.",
            "",
            "## Charts",
        ]
    )
    lines.extend(f"- {chart}" for chart in chart_paths)

    report_path = config.reports_dir / "backtest_summary.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved report to {report_path}")


if __name__ == "__main__":
    main()
