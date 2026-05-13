from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.scoring import (
    apply_filters,
    get_future_return_columns,
    get_strategy_filter_params,
    score_rebalance_date,
    validate_holding_period_days,
)
from src.utils import LOGGER, save_dataframe


def _select_rebalance_dates(features: pd.DataFrame) -> list[pd.Timestamp]:
    dates = pd.Series(pd.to_datetime(features["date"]).sort_values().unique())
    weeks = dates.dt.strftime("%G-%V")
    return dates.groupby(weeks).min().tolist()


def _compute_turnover(
    previous_holdings: dict[str, float],
    current_holdings: dict[str, float],
) -> float:
    all_tickers = set(previous_holdings) | set(current_holdings)
    return sum(abs(current_holdings.get(ticker, 0.0) - previous_holdings.get(ticker, 0.0)) for ticker in all_tickers)


def _validate_holdings(holdings: pd.DataFrame, benchmark: str) -> None:
    if not holdings.empty and holdings["ticker"].eq(benchmark).any():
        raise ValueError(f"Benchmark ticker {benchmark} must not appear in weekly holdings output.")


def run_weekly_backtest(
    features: pd.DataFrame,
    holding_period_days: int = 5,
    benchmark: str = "SPY",
    top_n: int = 10,
    initial_capital: float = 10000,
    transaction_cost_bps: float = 10,
    use_regime_filter: bool = False,
    regime_exposure: float = 0.0,
    use_analyst_filters: bool = True,
    analyst_count_threshold: int = 10,
    min_avg_dollar_volume: float = 20_000_000,
    strategy_name: str = "full_model",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run a weekly cross-sectional backtest with optional regime scaling and turnover costs."""
    validate_holding_period_days(holding_period_days)
    if not 0.0 <= regime_exposure <= 1.0:
        raise ValueError(f"regime_exposure must be between 0.0 and 1.0; got {regime_exposure}")

    df = features.copy()
    df["date"] = pd.to_datetime(df["date"])
    rebalance_dates = _select_rebalance_dates(df)
    future_return_column, future_spy_return_column, future_excess_return_column = get_future_return_columns(
        holding_period_days
    )

    if strategy_name == "full_model" and use_analyst_filters and df["consensus_upside"].notna().sum() == 0:
        LOGGER.warning(
            "Analyst data is missing for full_model. Falling back to no-analyst ranking mode instead of failing."
        )
        use_analyst_filters = False

    params = get_strategy_filter_params(
        strategy_name=strategy_name,
        analyst_count_threshold=analyst_count_threshold,
        use_analyst_filters=use_analyst_filters,
        min_avg_dollar_volume=min_avg_dollar_volume,
    )

    weekly_rows: list[dict] = []
    holding_rows: list[dict] = []
    portfolio_value = initial_capital
    spy_value = initial_capital
    previous_holdings: dict[str, float] = {}

    for rebalance_date in rebalance_dates:
        day_slice = df.loc[df["date"] == rebalance_date].copy()
        qualified = apply_filters(
            day_slice,
            params=params,
            holding_period_days=holding_period_days,
            benchmark=benchmark,
        )
        scored = score_rebalance_date(
            qualified,
            strategy_name=strategy_name,
            use_analyst_filters=use_analyst_filters,
        ).sort_values("score", ascending=False)
        selected = scored.head(top_n).copy()

        regime_allowed = True
        exposure = 1.0
        if use_regime_filter:
            spy_above = bool(day_slice["spy_above_sma_200"].fillna(False).iloc[0]) if not day_slice.empty else False
            regime_allowed = spy_above
            exposure = 1.0 if spy_above else regime_exposure

        selected_count = len(selected)
        current_holdings: dict[str, float] = {}
        if selected_count > 0 and exposure > 0:
            target_weight = exposure / selected_count
            selected["weight"] = target_weight
            current_holdings = dict(zip(selected["ticker"], selected["weight"]))
            gross_return = (selected[future_return_column].fillna(0.0) * selected["weight"]).sum()
        else:
            selected["weight"] = 0.0
            gross_return = 0.0

        turnover = _compute_turnover(previous_holdings, current_holdings)
        transaction_cost = turnover * transaction_cost_bps / 10000.0
        net_return = gross_return - transaction_cost

        spy_return_series = day_slice[future_spy_return_column].dropna()
        spy_return = float(spy_return_series.iloc[0]) if not spy_return_series.empty else 0.0
        excess_return = net_return - spy_return
        portfolio_value *= 1 + net_return
        spy_value *= 1 + spy_return

        weekly_rows.append(
            {
                "date": rebalance_date,
                "strategy_name": strategy_name,
                "holding_period_days": holding_period_days,
                "selected_count": selected_count,
                "gross_return": gross_return,
                "transaction_cost": transaction_cost,
                "turnover": turnover,
                "net_return": net_return,
                "spy_return": spy_return,
                "excess_return": excess_return,
                "portfolio_value": portfolio_value,
                "spy_value": spy_value,
                "regime_allowed": regime_allowed,
                "exposure": exposure,
                "qualified_count": len(qualified),
            }
        )

        if selected_count > 0 and exposure > 0:
            selected["date"] = rebalance_date
            selected["strategy_name"] = strategy_name
            selected["holding_period_days"] = holding_period_days
            selected["future_return_used"] = selected[future_return_column]
            selected["future_excess_return_used"] = selected[future_excess_return_column].fillna(
                selected[future_return_column].fillna(0.0) - spy_return
            )
            holding_rows.extend(
                selected[
                    [
                        "date",
                        "strategy_name",
                        "ticker",
                        "weight",
                        "score",
                        "holding_period_days",
                        "future_return_used",
                        "future_excess_return_used",
                        "consensus_upside",
                        "low_target_upside",
                        "analyst_count",
                        "relative_strength_21d",
                        "distance_to_30d_high",
                        "breakout_30d",
                        "volume_spike_ratio",
                        "volatility_21d",
                    ]
                ].to_dict("records")
            )

        previous_holdings = current_holdings

    weekly_df = pd.DataFrame(weekly_rows)
    holdings_df = pd.DataFrame(holding_rows)
    _validate_holdings(holdings_df, benchmark)
    return weekly_df, holdings_df


def save_backtest_outputs(
    weekly_returns: pd.DataFrame,
    holdings: pd.DataFrame,
    output_dir: str | Path,
    benchmark: str = "SPY",
) -> None:
    """Persist weekly returns, holdings, and a simple trades table."""
    _validate_holdings(holdings, benchmark)

    output_dir = Path(output_dir)
    save_dataframe(output_dir / "weekly_portfolio_returns.csv", weekly_returns)
    save_dataframe(output_dir / "weekly_holdings.csv", holdings)
    trades = holdings.copy()
    trades["trade_type"] = "rebalance_target"
    save_dataframe(output_dir / "trades.csv", trades)
