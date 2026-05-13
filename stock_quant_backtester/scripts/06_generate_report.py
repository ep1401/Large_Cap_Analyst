from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import Config
from src.utils import load_dataframe


IMPORTANT_CAVEAT = (
    "Important caveat: analyst-driven results currently use FMP data as a current snapshot merged "
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
        body.append(
            "| " + " | ".join(str(row[header]).ljust(widths[idx]) for idx, header in enumerate(headers)) + " |"
        )
    return "\n".join([header_line, separator, *body])


def main() -> None:
    config = Config.from_env()
    comparison = load_dataframe(config.tables_dir / "strategy_comparison.csv")
    weekly = load_dataframe(config.tables_dir / "weekly_portfolio_returns.csv", parse_dates=["date"])
    holdings = load_dataframe(config.tables_dir / "weekly_holdings.csv", parse_dates=["date"])
    features = load_dataframe(config.final_dir / "features_panel.csv", parse_dates=["date"])
    chart_paths = sorted(path.name for path in Path(config.charts_dir).glob("*.png"))

    analyst_mode = (
        features["analyst_data_mode"].dropna().iloc[0]
        if "analyst_data_mode" in features.columns and features["analyst_data_mode"].notna().any()
        else "historical_backtest_without_analyst"
    )
    full_model_row = comparison.loc[comparison["strategy_name"] == "full_model"].iloc[0]
    under_diversified = float(full_model_row["average_selected_count"]) < 3

    lines = [
        "# Backtest Summary",
        "",
        "## Strategy Settings",
        f"- Start date: {config.start_date}",
        f"- End date: {config.end_date}",
        f"- Benchmark: {config.benchmark}",
        f"- Holding period days: {int(full_model_row['holding_period_days'])}",
        f"- Top N: {int(full_model_row['top_n'])}",
        f"- Transaction cost bps: {config.transaction_cost_bps}",
        f"- Regime filter enabled: {bool(full_model_row['use_regime_filter'])}",
        f"- Regime exposure when blocked: {full_model_row['regime_exposure']}",
        f"- Analyst count threshold: {int(full_model_row['analyst_count_threshold'])}",
        f"- Minimum avg dollar volume: {full_model_row['min_avg_dollar_volume']}",
        "",
        "## Analyst Data Mode",
        f"- Analyst data mode: {analyst_mode}",
        f"- Point-in-time analyst history available: {analyst_mode != 'research_current_snapshot'}",
        f'- {IMPORTANT_CAVEAT}',
        "",
        "## Dataset Summary",
        f"- Universe size including benchmark rows: {features['ticker'].nunique()}",
        f"- Feature rows: {len(features)}",
        f"- Feature date range: {features['date'].min().date()} to {features['date'].max().date()}",
        "",
        "## Strategy Comparison",
        "",
        _dataframe_to_markdown(comparison.round(6)),
        "",
        "## Full Model Snapshot",
        f"- Average number of holdings: {full_model_row['average_selected_count']:.2f}",
        f"- Average turnover: {full_model_row['average_turnover']:.2f}",
        f"- Number of invested periods: {int(full_model_row['number_of_invested_periods'])}",
        f"- Annualized return: {_format_pct(full_model_row['annualized_return'])}",
        f"- Excess total return vs SPY: {_format_pct(full_model_row['excess_total_return'])}",
        f"- Sharpe ratio: {full_model_row['sharpe_ratio']:.2f}",
        f"- Max drawdown: {_format_pct(full_model_row['max_drawdown'])}",
    ]

    if under_diversified:
        lines.extend(
            [
                "",
                "## Warning",
                "- The average selected count is very low, so the strategy appears under-diversified under the current settings.",
            ]
        )

    lines.extend(
        [
            "",
            "## Benchmark Comparison",
            f"- Portfolio total return: {_format_pct(full_model_row['total_return'])}",
            f"- SPY total return: {_format_pct(full_model_row['spy_total_return'])}",
            f"- Weeks beating SPY: {full_model_row['weeks_beating_spy']:.2%}",
            "",
            "## Key Caveats",
            "- This project is a historical research backtest, not financial advice and not a live trading system.",
            f"- {IMPORTANT_CAVEAT}",
            "- The initial universe uses a static large-cap stock list, which introduces survivorship bias.",
            "- Transaction costs are modeled using turnover-based costs rather than a brokerage-specific execution model.",
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
