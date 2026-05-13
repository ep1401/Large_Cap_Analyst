from __future__ import annotations

from math import sqrt

import pandas as pd


def calculate_performance_metrics(returns_df: pd.DataFrame) -> dict:
    """Calculate standard weekly performance statistics."""
    if returns_df.empty:
        raise ValueError("Cannot calculate metrics on an empty returns dataframe.")

    df = returns_df.sort_values("date").copy()
    weekly_returns = df["net_return"].fillna(0)
    spy_returns = df["spy_return"].fillna(0)
    num_weeks = len(df)
    initial_value = float(df["portfolio_value"].iloc[0] / (1 + weekly_returns.iloc[0])) if num_weeks else 1.0
    final_value = float(df["portfolio_value"].iloc[-1])
    spy_initial = float(df["spy_value"].iloc[0] / (1 + spy_returns.iloc[0])) if num_weeks else 1.0
    spy_final = float(df["spy_value"].iloc[-1])

    total_return = final_value / initial_value - 1
    spy_total_return = spy_final / spy_initial - 1
    excess_total_return = total_return - spy_total_return
    annualized_return = (1 + total_return) ** (52 / num_weeks) - 1 if num_weeks else 0.0
    spy_annualized_return = (1 + spy_total_return) ** (52 / num_weeks) - 1 if num_weeks else 0.0
    weekly_volatility = weekly_returns.std(ddof=0)
    annualized_volatility = weekly_volatility * sqrt(52) if num_weeks > 1 else 0.0
    sharpe_ratio = weekly_returns.mean() / weekly_volatility * sqrt(52) if weekly_volatility > 0 else 0.0
    drawdown = df["portfolio_value"] / df["portfolio_value"].cummax() - 1
    max_drawdown = drawdown.min()

    return {
        "total_return": total_return,
        "spy_total_return": spy_total_return,
        "excess_total_return": excess_total_return,
        "annualized_return": annualized_return,
        "spy_annualized_return": spy_annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "win_rate": float((weekly_returns > 0).mean()),
        "weeks_beating_spy": float((weekly_returns > spy_returns).mean()),
        "average_weekly_return": weekly_returns.mean(),
        "average_weekly_excess_return": df["excess_return"].mean(),
        "average_turnover": df["turnover"].mean() if "turnover" in df.columns else 0.0,
        "number_of_rebalance_periods": num_weeks,
        "number_of_invested_periods": int((df.get("exposure", 0) > 0).sum()) if "exposure" in df.columns else num_weeks,
        "average_selected_count": df["selected_count"].mean(),
    }
