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


VALIDATION_TOLERANCE = 0.02


def select_rebalance_dates(
    features: pd.DataFrame,
    holding_period_days: int,
    benchmark: str,
) -> list[pd.Timestamp]:
    """Select non-overlapping rebalance dates using actual trading dates."""
    validate_holding_period_days(holding_period_days)
    _, future_spy_return_column, _ = get_future_return_columns(holding_period_days)
    benchmark_dates = (
        features.loc[
            (features["ticker"] == benchmark) & features[future_spy_return_column].notna(),
            "date",
        ]
        .drop_duplicates()
        .sort_values()
    )
    unique_dates = list(pd.to_datetime(benchmark_dates).tolist())
    return unique_dates[::holding_period_days]


def _compute_turnover(
    previous_weights: dict[str, float],
    new_weights: dict[str, float],
) -> float:
    all_tickers = set(previous_weights) | set(new_weights)
    return sum(abs(new_weights.get(ticker, 0.0) - previous_weights.get(ticker, 0.0)) for ticker in all_tickers)


def _validate_holdings(holdings: pd.DataFrame, benchmark: str) -> None:
    if not holdings.empty and holdings["ticker"].eq(benchmark).any():
        raise ValueError(f"Benchmark ticker {benchmark} must not appear in weekly holdings output.")


def _build_validation_row(
    features: pd.DataFrame,
    weekly_returns: pd.DataFrame,
    benchmark: str,
    holding_period_days: int,
) -> pd.DataFrame:
    """Compare compounded benchmark return to direct buy-and-hold over the same non-overlapping window."""
    if weekly_returns.empty:
        return pd.DataFrame(
            [
                {
                    "first_rebalance_date": None,
                    "last_rebalance_date": None,
                    "holding_period_days": holding_period_days,
                    "number_of_rebalance_periods": 0,
                    "compounded_spy_return_from_backtest": 0.0,
                    "direct_spy_buy_hold_return": 0.0,
                    "absolute_difference": 0.0,
                }
            ]
        )

    benchmark_df = (
        features.loc[features["ticker"] == benchmark, ["date", "adjusted_close"]]
        .drop_duplicates("date")
        .sort_values("date")
        .reset_index(drop=True)
    )
    trading_dates = list(pd.to_datetime(benchmark_df["date"]).tolist())
    date_to_index = {date: idx for idx, date in enumerate(trading_dates)}

    first_rebalance_date = pd.to_datetime(weekly_returns["date"].iloc[0])
    last_rebalance_date = pd.to_datetime(weekly_returns["date"].iloc[-1])
    start_index = date_to_index[first_rebalance_date]
    end_index = start_index + holding_period_days * len(weekly_returns)
    if end_index >= len(trading_dates):
        end_index = len(trading_dates) - 1
    end_date = trading_dates[end_index]

    start_price = float(benchmark_df.loc[benchmark_df["date"] == first_rebalance_date, "adjusted_close"].iloc[0])
    end_price = float(benchmark_df.loc[benchmark_df["date"] == end_date, "adjusted_close"].iloc[0])
    direct_buy_hold_return = end_price / start_price - 1
    initial_spy_value = float(weekly_returns["spy_value"].iloc[0] / (1 + weekly_returns["spy_return"].iloc[0]))
    compounded_spy_return = float(weekly_returns["spy_value"].iloc[-1] / initial_spy_value - 1)
    absolute_difference = abs(compounded_spy_return - direct_buy_hold_return)

    return pd.DataFrame(
        [
            {
                "first_rebalance_date": first_rebalance_date.date(),
                "last_rebalance_date": last_rebalance_date.date(),
                "holding_period_days": holding_period_days,
                "number_of_rebalance_periods": len(weekly_returns),
                "compounded_spy_return_from_backtest": compounded_spy_return,
                "direct_spy_buy_hold_return": direct_buy_hold_return,
                "absolute_difference": absolute_difference,
            }
        ]
    )


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
    """Run a non-overlapping cross-sectional backtest with turnover-based transaction costs."""
    validate_holding_period_days(holding_period_days)
    if not 0.0 <= regime_exposure <= 1.0:
        raise ValueError(f"regime_exposure must be between 0.0 and 1.0; got {regime_exposure}")

    df = features.copy()
    df["date"] = pd.to_datetime(df["date"])
    future_return_column, future_spy_return_column, future_excess_return_column = get_future_return_columns(
        holding_period_days
    )
    rebalance_dates = select_rebalance_dates(df, holding_period_days=holding_period_days, benchmark=benchmark)

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
    previous_weights: dict[str, float] = {}

    for rebalance_date in rebalance_dates:
        day_slice = df.loc[df["date"] == rebalance_date].copy()
        day_slice = day_slice.loc[day_slice["ticker"] != benchmark].copy()
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
        base_exposure = 1.0
        if use_regime_filter:
            spy_series = df.loc[
                (df["date"] == rebalance_date) & (df["ticker"] == benchmark),
                "spy_above_sma_200",
            ]
            spy_above = bool(spy_series.fillna(False).iloc[0]) if not spy_series.empty else False
            regime_allowed = spy_above
            base_exposure = 1.0 if spy_above else regime_exposure

        selected_count = len(selected)
        new_weights: dict[str, float] = {}
        actual_exposure = 0.0
        gross_return = 0.0

        if selected_count > 0 and base_exposure > 0:
            actual_exposure = base_exposure
            target_weight = actual_exposure / selected_count
            selected["weight"] = target_weight
            new_weights = dict(zip(selected["ticker"], selected["weight"]))
            gross_return = float((selected["weight"] * selected[future_return_column].fillna(0.0)).sum())
        else:
            selected["weight"] = 0.0

        turnover = _compute_turnover(previous_weights, new_weights)
        transaction_cost = turnover * transaction_cost_bps / 10000.0
        net_return = gross_return - transaction_cost

        benchmark_slice = df.loc[(df["date"] == rebalance_date) & (df["ticker"] == benchmark), future_spy_return_column]
        spy_return = float(benchmark_slice.iloc[0]) if not benchmark_slice.empty else 0.0
        excess_return = net_return - spy_return
        portfolio_value *= 1 + net_return
        spy_value *= 1 + spy_return

        weekly_rows.append(
            {
                "date": rebalance_date,
                "strategy_name": strategy_name,
                "holding_period_days": holding_period_days,
                "top_n": top_n,
                "use_regime_filter": use_regime_filter,
                "regime_exposure": regime_exposure,
                "analyst_count_threshold": analyst_count_threshold,
                "min_avg_dollar_volume": min_avg_dollar_volume,
                "selected_count": selected_count,
                "qualified_count": len(qualified),
                "gross_return": gross_return,
                "turnover": turnover,
                "transaction_cost": transaction_cost,
                "net_return": net_return,
                "spy_return": spy_return,
                "excess_return": excess_return,
                "portfolio_value": portfolio_value,
                "spy_value": spy_value,
                "exposure": actual_exposure,
                "regime_allowed": regime_allowed,
            }
        )

        if selected_count > 0 and actual_exposure > 0:
            selected["date"] = rebalance_date
            selected["strategy_name"] = strategy_name
            selected["holding_period_days"] = holding_period_days
            selected["future_return_used"] = selected[future_return_column]
            selected["future_excess_return_used"] = selected[future_excess_return_column]
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

        previous_weights = new_weights

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


def save_backtest_validation(
    features: pd.DataFrame,
    weekly_returns: pd.DataFrame,
    output_dir: str | Path,
    benchmark: str,
    holding_period_days: int,
) -> pd.DataFrame:
    validation_df = _build_validation_row(
        features=features,
        weekly_returns=weekly_returns,
        benchmark=benchmark,
        holding_period_days=holding_period_days,
    )
    save_dataframe(Path(output_dir) / "backtest_validation.csv", validation_df)
    if float(validation_df["absolute_difference"].iloc[0]) > VALIDATION_TOLERANCE:
        LOGGER.warning(
            "Benchmark validation difference %.4f exceeds tolerance %.4f. Benchmark compounding may be inconsistent.",
            float(validation_df["absolute_difference"].iloc[0]),
            VALIDATION_TOLERANCE,
        )
    return validation_df
