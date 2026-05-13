from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.scoring import apply_filters, get_strategy_filter_params, score_rebalance_date
from src.utils import save_dataframe


def _select_rebalance_dates(features: pd.DataFrame) -> list[pd.Timestamp]:
    dates = pd.Series(pd.to_datetime(features["date"]).sort_values().unique())
    weeks = dates.dt.strftime("%G-%V")
    return dates.groupby(weeks).min().tolist()


def run_weekly_backtest(
    features: pd.DataFrame,
    top_n: int = 10,
    initial_capital: float = 10000,
    transaction_cost_bps: float = 10,
    use_analyst_filters: bool = True,
    analyst_count_threshold: int = 20,
    strategy_name: str = "full_model",
    include_sentiment: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run a weekly equal-weighted cross-sectional backtest."""
    df = features.copy()
    df["date"] = pd.to_datetime(df["date"])
    rebalance_dates = _select_rebalance_dates(df)
    cost = transaction_cost_bps / 10000.0
    params = get_strategy_filter_params(
        strategy_name,
        analyst_count_threshold,
        use_analyst_filters,
        include_sentiment=include_sentiment,
    )

    weekly_rows: list[dict] = []
    holding_rows: list[dict] = []
    portfolio_value = initial_capital
    spy_value = initial_capital

    for rebalance_date in rebalance_dates:
        day_slice = df.loc[df["date"] == rebalance_date].copy()
        qualified = apply_filters(day_slice, params)
        scored = score_rebalance_date(qualified, strategy_name=strategy_name).sort_values("score", ascending=False)
        selected = scored.head(top_n).copy()

        selected_count = len(selected)
        if selected_count > 0:
            selected["weight"] = 1.0 / selected_count
            gross_return = selected["future_5d_return"].fillna(0).mean()
            transaction_cost = 2 * cost
            net_return = gross_return - transaction_cost
        else:
            gross_return = 0.0
            transaction_cost = 0.0
            net_return = 0.0

        spy_return_series = day_slice["future_5d_spy_return"].dropna()
        spy_return = float(spy_return_series.iloc[0]) if not spy_return_series.empty else 0.0
        excess_return = net_return - spy_return
        portfolio_value *= 1 + net_return
        spy_value *= 1 + spy_return

        weekly_rows.append(
            {
                "date": rebalance_date,
                "strategy_name": strategy_name,
                "selected_count": selected_count,
                "gross_return": gross_return,
                "transaction_cost": transaction_cost,
                "net_return": net_return,
                "spy_return": spy_return,
                "excess_return": excess_return,
                "portfolio_value": portfolio_value,
                "spy_value": spy_value,
                "qualified_count": len(qualified),
            }
        )

        if selected_count > 0:
            selected["date"] = rebalance_date
            selected["strategy_name"] = strategy_name
            selected["future_5d_excess_return"] = selected["future_5d_excess_return"].fillna(
                selected["future_5d_return"].fillna(0) - spy_return
            )
            holding_rows.extend(
                selected[
                    [
                        "date",
                        "strategy_name",
                        "ticker",
                        "score",
                        "weight",
                        "future_5d_return",
                        "future_5d_excess_return",
                        "consensus_upside",
                        "news_sentiment_7d",
                        "distance_to_30d_high",
                        "breakout_30d",
                    ]
                ].to_dict("records")
            )

    return pd.DataFrame(weekly_rows), pd.DataFrame(holding_rows)


def save_backtest_outputs(
    weekly_returns: pd.DataFrame,
    holdings: pd.DataFrame,
    output_dir: str | Path,
) -> None:
    """Persist weekly returns, holdings, and a simple trades table."""
    output_dir = Path(output_dir)
    save_dataframe(output_dir / "weekly_portfolio_returns.csv", weekly_returns)
    save_dataframe(output_dir / "weekly_holdings.csv", holdings)
    trades = holdings.copy()
    trades["trade_type"] = "buy_and_hold_one_week"
    save_dataframe(output_dir / "trades.csv", trades)
