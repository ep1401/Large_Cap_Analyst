from __future__ import annotations

from math import sqrt

import pandas as pd

from src.scoring import validate_holding_period_days


def calculate_performance_metrics(returns_df: pd.DataFrame, holding_period_days: int) -> dict:
    """Calculate holding-period-aware performance statistics."""
    validate_holding_period_days(holding_period_days)
    if returns_df.empty:
        raise ValueError("Cannot calculate metrics on an empty returns dataframe.")

    df = returns_df.sort_values("date").copy()
    period_returns = df["net_return"].fillna(0.0)
    spy_returns = df["spy_return"].fillna(0.0)
    num_periods = len(df)
    periods_per_year = 252 / holding_period_days

    initial_value = float(df["portfolio_value"].iloc[0] / (1 + period_returns.iloc[0])) if num_periods else 1.0
    final_value = float(df["portfolio_value"].iloc[-1])
    spy_initial = float(df["spy_value"].iloc[0] / (1 + spy_returns.iloc[0])) if num_periods else 1.0
    spy_final = float(df["spy_value"].iloc[-1])

    total_return = final_value / initial_value - 1
    spy_total_return = spy_final / spy_initial - 1
    excess_total_return = total_return - spy_total_return
    annualized_return = (1 + total_return) ** (periods_per_year / num_periods) - 1 if num_periods else 0.0
    spy_annualized_return = (1 + spy_total_return) ** (periods_per_year / num_periods) - 1 if num_periods else 0.0
    period_volatility = period_returns.std(ddof=0)
    annualized_volatility = period_volatility * sqrt(periods_per_year) if num_periods > 1 else 0.0
    sharpe_ratio = period_returns.mean() / period_volatility * sqrt(periods_per_year) if period_volatility > 0 else 0.0
    drawdown = df["portfolio_value"] / df["portfolio_value"].cummax() - 1
    max_drawdown = drawdown.min()

    invested_periods = ((df["selected_count"] > 0) & (df["exposure"] > 0)).sum() if "exposure" in df.columns else num_periods

    return {
        "total_return_decimal": total_return,
        "total_return_pct": total_return * 100,
        "spy_total_return_decimal": spy_total_return,
        "spy_total_return_pct": spy_total_return * 100,
        "excess_total_return_decimal": excess_total_return,
        "excess_total_return_pct": excess_total_return * 100,
        "total_return": total_return,
        "spy_total_return": spy_total_return,
        "excess_total_return": excess_total_return,
        "annualized_return": annualized_return,
        "spy_annualized_return": spy_annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "win_rate": float((period_returns > 0).mean()),
        "weeks_beating_spy": float((period_returns > spy_returns).mean()),
        "average_weekly_return": period_returns.mean(),
        "average_weekly_excess_return": df["excess_return"].mean(),
        "average_turnover": df["turnover"].mean() if "turnover" in df.columns else 0.0,
        "average_selected_count": df["selected_count"].mean(),
        "number_of_rebalance_periods": num_periods,
        "number_of_invested_periods": int(invested_periods),
        "periods_per_year": periods_per_year,
    }
