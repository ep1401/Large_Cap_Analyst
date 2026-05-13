from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import Config
from src.utils import load_dataframe


def _format_pct(value: float) -> str:
    return f"{value:.2%}"


def _dataframe_to_markdown(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    rows = [headers]
    for _, row in df.iterrows():
        formatted = []
        for value in row.tolist():
            if isinstance(value, float):
                formatted.append(f"{value:.6f}")
            else:
                formatted.append(str(value))
        rows.append(formatted)
    widths = [max(len(str(row[idx])) for row in rows) for idx in range(len(headers))]
    header_line = "| " + " | ".join(str(headers[idx]).ljust(widths[idx]) for idx in range(len(headers))) + " |"
    separator = "| " + " | ".join("-" * widths[idx] for idx in range(len(headers))) + " |"
    body_lines = [
        "| " + " | ".join(str(row[idx]).ljust(widths[idx]) for idx in range(len(headers))) + " |"
        for row in rows[1:]
    ]
    return "\n".join([header_line, separator, *body_lines])


def main() -> None:
    config = Config.from_env()
    comparison = load_dataframe(config.tables_dir / "strategy_comparison.csv")
    features = load_dataframe(config.final_dir / "features_panel.csv", parse_dates=["date"])
    chart_paths = sorted(path.name for path in Path(config.charts_dir).glob("*.png"))

    lines = [
        "# Backtest Summary",
        "",
        "## Strategy Settings",
        f"- Start date: {config.start_date}",
        f"- End date: {config.end_date}",
        f"- Benchmark: {config.benchmark}",
        f"- Initial capital: {config.initial_capital}",
        f"- Top N: {config.top_n}",
        f"- Transaction cost bps: {config.transaction_cost_bps}",
        "",
        "## Dataset Summary",
        f"- Universe size: {features['ticker'].nunique()}",
        f"- Feature rows: {len(features)}",
        f"- Feature date range: {features['date'].min().date()} to {features['date'].max().date()}",
        "",
        "## Strategy Comparison",
        "",
        _dataframe_to_markdown(comparison),
        "",
        "## Best Strategy Snapshot",
    ]

    best = comparison.iloc[0]
    lines.extend(
        [
            f"- Best annualized return: {best['strategy_name']} at {_format_pct(best['annualized_return'])}",
            f"- Total return: {_format_pct(best['total_return'])}",
            f"- Excess total return vs SPY: {_format_pct(best['excess_total_return'])}",
            f"- Sharpe ratio: {best['sharpe_ratio']:.2f}",
            "",
            "## Limitations",
            "- This project is a historical research backtest, not financial advice and not a live trading system.",
            "- The largest methodological risk is point-in-time analyst data. If the analyst API returns only current consensus data, then it cannot be used for a valid historical backtest. The code therefore supports running the backtest without analyst filters, and the README clearly marks whether analyst data is point-in-time.",
            "- The initial universe uses a static large-cap stock list, which introduces survivorship bias. A production-grade backtest should use historical index constituents or historical Fortune 500 membership.",
            "- Transaction costs are simplified. Slippage, bid-ask spreads, taxes, borrow costs for shorts, and market impact are not fully modeled.",
            "- News sentiment availability may vary by ticker and date, and the current default workflow does not include Alpha Vantage sentiment data.",
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
