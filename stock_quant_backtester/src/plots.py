from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _save_plot(fig: plt.Figure, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def create_plots(
    weekly_returns: pd.DataFrame,
    holdings: pd.DataFrame,
    charts_dir: str | Path,
) -> list[Path]:
    """Create and save the standard backtest charts."""
    charts_dir = Path(charts_dir)
    df = weekly_returns.sort_values("date").copy()
    generated: list[Path] = []

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df["date"], df["portfolio_value"], label="Portfolio")
    ax.plot(df["date"], df["spy_value"], label="SPY")
    ax.set_title("Portfolio Value vs SPY")
    ax.legend()
    path = charts_dir / "equity_curve_vs_spy.png"
    _save_plot(fig, path)
    generated.append(path)

    fig, ax = plt.subplots(figsize=(10, 6))
    portfolio_drawdown = df["portfolio_value"] / df["portfolio_value"].cummax() - 1
    spy_drawdown = df["spy_value"] / df["spy_value"].cummax() - 1
    ax.plot(df["date"], portfolio_drawdown, label="Portfolio Drawdown")
    ax.plot(df["date"], spy_drawdown, label="SPY Drawdown")
    ax.set_title("Drawdown vs SPY")
    ax.legend()
    path = charts_dir / "drawdown_vs_spy.png"
    _save_plot(fig, path)
    generated.append(path)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(df["date"], df["excess_return"])
    ax.set_title("Weekly Excess Returns")
    path = charts_dir / "weekly_excess_returns.png"
    _save_plot(fig, path)
    generated.append(path)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df["date"], df["qualified_count"])
    ax.set_title("Number of Qualifying Stocks Per Week")
    path = charts_dir / "qualifying_stocks_per_week.png"
    _save_plot(fig, path)
    generated.append(path)

    if not holdings.empty:
        ticker_counts = holdings["ticker"].value_counts().head(15).sort_values()
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(ticker_counts.index, ticker_counts.values)
        ax.set_title("Top Selected Tickers Count")
        path = charts_dir / "top_selected_tickers_count.png"
        _save_plot(fig, path)
        generated.append(path)

        averages = (
            holdings.groupby("date")[
                ["consensus_upside", "news_sentiment_7d", "distance_to_30d_high", "future_5d_return"]
            ]
            .mean(numeric_only=True)
            .reset_index()
        )
        fig, ax = plt.subplots(figsize=(10, 6))
        for column in ["consensus_upside", "news_sentiment_7d", "distance_to_30d_high"]:
            if column in averages:
                ax.plot(averages["date"], averages[column], label=column)
        ax.set_title("Average Feature Values of Selected Stocks")
        ax.legend()
        path = charts_dir / "average_feature_values_selected.png"
        _save_plot(fig, path)
        generated.append(path)

    return generated
