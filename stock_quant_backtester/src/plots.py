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
    ax.plot(df["date"], df["selected_count"])
    ax.set_title("Selected Stocks Per Rebalance")
    path = charts_dir / "qualifying_stocks_per_week.png"
    _save_plot(fig, path)
    generated.append(path)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df["date"], df["turnover"])
    ax.set_title("Turnover Per Rebalance")
    path = charts_dir / "turnover_per_rebalance.png"
    _save_plot(fig, path)
    generated.append(path)

    if "exposure" in df.columns and df["exposure"].nunique() > 1:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df["date"], df["exposure"])
        ax.set_ylim(0, 1.05)
        ax.set_title("Exposure Per Rebalance")
        path = charts_dir / "exposure_per_rebalance.png"
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
                ["consensus_upside", "relative_strength_21d", "distance_to_30d_high", "future_return_used"]
            ]
            .mean(numeric_only=True)
            .reset_index()
        )
        fig, ax = plt.subplots(figsize=(10, 6))
        for column in ["consensus_upside", "relative_strength_21d", "distance_to_30d_high"]:
            if column in averages:
                ax.plot(averages["date"], averages[column], label=column)
        ax.set_title("Average Feature Values of Selected Stocks")
        ax.legend()
        path = charts_dir / "average_feature_values_selected.png"
        _save_plot(fig, path)
        generated.append(path)

    return generated


def create_sentiment_plots(
    diagnostics_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    strategy_curves_df: pd.DataFrame,
    charts_dir: str | Path,
) -> list[Path]:
    charts_dir = Path(charts_dir)
    generated: list[Path] = []

    if not diagnostics_df.empty:
        diag = diagnostics_df.sort_values("date").copy()

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(diag["date"], diag["coverage_pct_7d"], label="Universe coverage %")
        ax.set_title("Sentiment Coverage Over Time")
        ax.set_ylim(0, 1)
        ax.legend()
        path = charts_dir / "sentiment_coverage_over_time.png"
        _save_plot(fig, path)
        generated.append(path)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(diag["date"], diag["average_news_sentiment_7d"], label="Universe avg sentiment")
        ax.plot(diag["date"], diag["selected_avg_news_sentiment_7d"], label="Selected avg sentiment")
        ax.set_title("Selected vs Universe Sentiment")
        ax.legend()
        path = charts_dir / "selected_vs_universe_sentiment.png"
        _save_plot(fig, path)
        generated.append(path)

    if not strategy_curves_df.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        for strategy_name, group in strategy_curves_df.groupby("strategy_name"):
            ax.plot(group["date"], group["portfolio_value"], label=strategy_name)
        ax.set_title("Sentiment Strategy Equity Curves")
        ax.legend()
        path = charts_dir / "sentiment_strategy_equity_curves.png"
        _save_plot(fig, path)
        generated.append(path)

    if not comparison_df.empty:
        best_rows = (
            comparison_df.sort_values(
                ["test_period_excess_return_vs_spy", "test_sharpe_ratio", "max_drawdown"],
                ascending=[False, False, False],
            )
            .groupby("strategy_name", as_index=False)
            .head(1)
            .copy()
        )
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(best_rows["strategy_name"], best_rows["test_period_excess_return_vs_spy"])
        ax.set_title("Sentiment Model Test Period Comparison")
        ax.tick_params(axis="x", rotation=45)
        path = charts_dir / "sentiment_model_test_period_comparison.png"
        _save_plot(fig, path)
        generated.append(path)

    return generated
