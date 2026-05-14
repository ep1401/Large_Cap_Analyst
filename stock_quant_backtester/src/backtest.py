from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.scoring import (
    StrategyParams,
    apply_filters,
    get_filter_diagnostics,
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
    rebalance_frequency_days: int | None = None,
) -> list[pd.Timestamp]:
    """Select non-overlapping rebalance dates using actual trading dates."""
    step = rebalance_frequency_days if rebalance_frequency_days is not None else holding_period_days
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
    return unique_dates[::step]


def _compute_turnover(previous_weights: dict[str, float], new_weights: dict[str, float]) -> float:
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
    start_index = date_to_index[first_rebalance_date]
    end_index = start_index + holding_period_days * len(weekly_returns)
    end_index = min(end_index, len(trading_dates) - 1)
    end_date = trading_dates[end_index]

    start_price = float(benchmark_df.loc[benchmark_df["date"] == first_rebalance_date, "adjusted_close"].iloc[0])
    end_price = float(benchmark_df.loc[benchmark_df["date"] == end_date, "adjusted_close"].iloc[0])
    direct_buy_hold_return = end_price / start_price - 1
    initial_spy_value = float(weekly_returns["spy_value"].iloc[0] / (1 + weekly_returns["spy_return"].iloc[0]))
    compounded_spy_return = float(weekly_returns["spy_value"].iloc[-1] / initial_spy_value - 1)

    return pd.DataFrame(
        [
            {
                "first_rebalance_date": first_rebalance_date.date(),
                "last_rebalance_date": pd.to_datetime(weekly_returns["date"].iloc[-1]).date(),
                "holding_period_days": holding_period_days,
                "number_of_rebalance_periods": len(weekly_returns),
                "compounded_spy_return_from_backtest": compounded_spy_return,
                "direct_spy_buy_hold_return": direct_buy_hold_return,
                "absolute_difference": abs(compounded_spy_return - direct_buy_hold_return),
            }
        ]
    )


def _apply_sector_limit(selected: pd.DataFrame, max_names_per_sector: int | None) -> pd.DataFrame:
    if max_names_per_sector is None or selected.empty or "sector" not in selected.columns:
        return selected
    return selected.groupby("sector", group_keys=False).head(max_names_per_sector)


def _build_weights(
    selected: pd.DataFrame,
    exposure: float,
    use_inverse_vol_weighting: bool,
    max_single_name_weight: float,
) -> pd.DataFrame:
    selected = selected.copy()
    if selected.empty or exposure <= 0:
        selected["weight"] = 0.0
        return selected

    if use_inverse_vol_weighting:
        inv_vol = 1 / selected["volatility_21d"].replace(0, np.nan)
        inv_vol = inv_vol.fillna(inv_vol.mean()).fillna(1.0)
        weights = inv_vol / inv_vol.sum()
    else:
        weights = pd.Series(1 / len(selected), index=selected.index)

    weights = weights * exposure
    weights = weights.clip(upper=max_single_name_weight)
    if weights.sum() > 0:
        weights = weights * (exposure / weights.sum())
    selected["weight"] = weights
    return selected


def _adjust_exposure_for_drawdown(
    current_exposure: float,
    current_value: float,
    peak_value: float,
    market_above_50: bool,
) -> float:
    if peak_value <= 0:
        return current_exposure
    drawdown = current_value / peak_value - 1
    if drawdown <= -0.30 and not market_above_50:
        return 0.0
    if drawdown <= -0.20:
        return min(current_exposure, 0.5)
    return current_exposure


def _append_holding_rows(
    holding_rows: list[dict],
    selected: pd.DataFrame,
    date: pd.Timestamp,
    strategy_name: str,
    holding_period_days: int,
    future_return_column: str,
    future_excess_return_column: str,
) -> None:
    if selected.empty:
        return
    selected = selected.copy()
    selected["date"] = date
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
                "news_sentiment_7d",
                "negative_news_ratio_7d",
                "article_count_7d",
                "strong_negative_news_flag",
            ]
        ].to_dict("records")
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
    resistance_distance_threshold: float = 0.02,
    require_low_target_upside_4pct: bool = False,
    require_positive_revision_7d: bool = False,
    require_positive_revision_30d: bool = False,
    resistance_window: int = 30,
    max_names_per_sector: int | None = None,
    use_inverse_vol_weighting: bool = False,
    max_single_name_weight: float = 0.15,
    enable_drawdown_protection: bool = False,
    require_positive_sentiment: bool = False,
    avoid_strong_negative_news: bool = False,
    min_article_count_7d: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run a non-overlapping cross-sectional backtest with turnover costs and diagnostics."""
    validate_holding_period_days(holding_period_days)
    if not 0.0 <= regime_exposure <= 1.0:
        raise ValueError(f"regime_exposure must be between 0.0 and 1.0; got {regime_exposure}")

    df = features.copy()
    df["date"] = pd.to_datetime(df["date"])
    future_return_column, future_spy_return_column, future_excess_return_column = get_future_return_columns(
        holding_period_days
    )
    rebalance_dates = select_rebalance_dates(df, holding_period_days=holding_period_days, benchmark=benchmark)

    if strategy_name in {"full_model", "strict_checklist_model", "analyst_only"} and use_analyst_filters and df["consensus_upside"].notna().sum() == 0:
        LOGGER.warning(
            "Analyst data is missing for %s. Falling back to no-analyst mode when applicable.", strategy_name
        )
        use_analyst_filters = False

    params = get_strategy_filter_params(
        strategy_name=strategy_name,
        analyst_count_threshold=analyst_count_threshold,
        use_analyst_filters=use_analyst_filters,
        min_avg_dollar_volume=min_avg_dollar_volume,
        resistance_distance_threshold=resistance_distance_threshold,
        require_low_target_upside_4pct=require_low_target_upside_4pct,
        require_positive_revision_7d=require_positive_revision_7d,
        require_positive_revision_30d=require_positive_revision_30d,
        resistance_window=resistance_window,
        require_positive_sentiment=require_positive_sentiment,
        avoid_strong_negative_news=avoid_strong_negative_news,
        min_article_count_7d=min_article_count_7d,
    )

    weekly_rows: list[dict] = []
    holding_rows: list[dict] = []
    diagnostics_rows: list[dict] = []
    portfolio_value = initial_capital
    portfolio_peak = initial_capital
    spy_value = initial_capital
    previous_weights: dict[str, float] = {}

    for rebalance_date in rebalance_dates:
        day_all = df.loc[df["date"] == rebalance_date].copy()
        day_slice = day_all.loc[day_all["ticker"] != benchmark].copy()
        qualified, diagnostics = apply_filters(
            day_slice,
            params=params,
            holding_period_days=holding_period_days,
            benchmark=benchmark,
        )
        diagnostics_rows.append({"date": rebalance_date, "strategy_name": strategy_name, **diagnostics, "selected_count": 0})

        scored = score_rebalance_date(
            qualified,
            strategy_name=strategy_name,
            use_analyst_filters=use_analyst_filters,
            resistance_window=resistance_window,
        ).sort_values("score", ascending=False)
        selected = _apply_sector_limit(scored.head(top_n), max_names_per_sector=max_names_per_sector)

        regime_allowed = True
        exposure = 1.0
        if use_regime_filter:
            spy_above = bool(day_all.loc[day_all["ticker"] == benchmark, "spy_above_sma_200"].fillna(False).iloc[0])
            regime_allowed = spy_above
            exposure = 1.0 if spy_above else regime_exposure

        if enable_drawdown_protection:
            market_above_50 = (
                bool(
                    day_all.loc[day_all["ticker"] == benchmark, "spy_close"].iloc[0]
                    > day_all.loc[day_all["ticker"] == benchmark, "spy_sma_50"].iloc[0]
                )
                if not day_all.loc[day_all["ticker"] == benchmark].empty
                else True
            )
            exposure = _adjust_exposure_for_drawdown(exposure, portfolio_value, portfolio_peak, market_above_50)

        selected = _build_weights(
            selected,
            exposure=exposure if not selected.empty else 0.0,
            use_inverse_vol_weighting=use_inverse_vol_weighting,
            max_single_name_weight=max_single_name_weight,
        )
        selected_count = len(selected)
        diagnostics_rows[-1]["selected_count"] = selected_count

        new_weights = dict(zip(selected["ticker"], selected["weight"])) if selected_count else {}
        gross_return = float((selected["weight"] * selected[future_return_column].fillna(0.0)).sum()) if selected_count else 0.0
        actual_exposure = float(selected["weight"].sum()) if selected_count else 0.0
        turnover = _compute_turnover(previous_weights, new_weights)
        transaction_cost = turnover * transaction_cost_bps / 10000.0
        net_return = gross_return - transaction_cost

        benchmark_slice = day_all.loc[day_all["ticker"] == benchmark, future_spy_return_column]
        spy_return = float(benchmark_slice.iloc[0]) if not benchmark_slice.empty else 0.0
        excess_return = net_return - spy_return
        portfolio_value *= 1 + net_return
        portfolio_peak = max(portfolio_peak, portfolio_value)
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

        _append_holding_rows(
            holding_rows,
            selected,
            rebalance_date,
            strategy_name,
            holding_period_days,
            future_return_column,
            future_excess_return_column,
        )
        previous_weights = new_weights

    weekly_df = pd.DataFrame(weekly_rows)
    holdings_df = pd.DataFrame(holding_rows)
    diagnostics_df = pd.DataFrame(diagnostics_rows)
    _validate_holdings(holdings_df, benchmark)
    return weekly_df, holdings_df, diagnostics_df


def run_condition_based_backtest(
    features: pd.DataFrame,
    strategy_name: str = "strict_checklist_model",
    top_n: int = 10,
    enter_rank: int = 10,
    exit_rank: int = 30,
    max_holding_days: int = 126,
    rebalance_frequency_days: int = 5,
    initial_capital: float = 10000,
    transaction_cost_bps: float = 10,
    benchmark: str = "SPY",
    analyst_count_threshold: int = 10,
    min_avg_dollar_volume: float = 20_000_000,
    use_regime_filter: bool = False,
    regime_exposure: float = 0.0,
    use_inverse_vol_weighting: bool = False,
    max_names_per_sector: int | None = None,
    resistance_distance_threshold: float = 0.02,
    require_low_target_upside_4pct: bool = False,
    require_positive_revision_7d: bool = False,
    require_positive_revision_30d: bool = False,
    resistance_window: int = 30,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run a condition-based portfolio using realized next-rebalance returns."""
    df = features.copy()
    df["date"] = pd.to_datetime(df["date"])
    decision_dates = select_rebalance_dates(df, holding_period_days=5, benchmark=benchmark, rebalance_frequency_days=rebalance_frequency_days)
    params = get_strategy_filter_params(
        strategy_name=strategy_name,
        analyst_count_threshold=analyst_count_threshold,
        use_analyst_filters=True,
        min_avg_dollar_volume=min_avg_dollar_volume,
        resistance_distance_threshold=resistance_distance_threshold,
        require_low_target_upside_4pct=require_low_target_upside_4pct,
        require_positive_revision_7d=require_positive_revision_7d,
        require_positive_revision_30d=require_positive_revision_30d,
        resistance_window=resistance_window,
    )

    benchmark_prices = (
        df.loc[df["ticker"] == benchmark, ["date", "adjusted_close", "spy_above_sma_200", "sma_50"]]
        .drop_duplicates("date")
        .sort_values("date")
        .set_index("date")
    )
    price_lookup = (
        df[["date", "ticker", "adjusted_close", "score" if "score" in df.columns else "adjusted_close"]]
        .drop_duplicates(["date", "ticker"])
        .set_index(["date", "ticker"])
    )

    portfolio_value = initial_capital
    portfolio_peak = initial_capital
    spy_value = initial_capital
    holdings: dict[str, dict] = {}
    holdings_rows: list[dict] = []
    trades_rows: list[dict] = []
    returns_rows: list[dict] = []

    for idx in range(len(decision_dates) - 1):
        current_date = decision_dates[idx]
        next_date = decision_dates[idx + 1]
        day_all = df.loc[df["date"] == current_date].copy()
        day_slice = day_all.loc[day_all["ticker"] != benchmark].copy()
        qualified, diagnostics = apply_filters(day_slice, params=params, holding_period_days=5, benchmark=benchmark)
        ranked = score_rebalance_date(
            qualified,
            strategy_name=strategy_name,
            use_analyst_filters=True,
            resistance_window=resistance_window,
        ).sort_values("score", ascending=False)
        ranked = _apply_sector_limit(ranked, max_names_per_sector=max_names_per_sector)

        ranked_by_ticker = ranked.set_index("ticker") if not ranked.empty else pd.DataFrame()
        exposure = 1.0
        regime_allowed = True
        if use_regime_filter and current_date in benchmark_prices.index:
            spy_above = bool(benchmark_prices.loc[current_date, "spy_above_sma_200"])
            regime_allowed = spy_above
            exposure = 1.0 if spy_above else regime_exposure
        if portfolio_peak > 0:
            drawdown = portfolio_value / portfolio_peak - 1
            market_above_50 = True
            if current_date in benchmark_prices.index:
                market_above_50 = bool(benchmark_prices.loc[current_date, "adjusted_close"] > benchmark_prices.loc[current_date, "sma_50"])
            exposure = _adjust_exposure_for_drawdown(exposure, portfolio_value, portfolio_peak, market_above_50)

        exiting: set[str] = set()
        for ticker, meta in list(holdings.items()):
            reason = None
            if ticker not in ranked_by_ticker.index:
                reason = "NO_LONGER_QUALIFIED"
            else:
                row = ranked_by_ticker.loc[ticker]
                rank = int(row["rank"])
                if rank > exit_rank:
                    reason = "RANK_EXIT"
                elif row["relative_strength_21d"] < 0:
                    reason = "NEGATIVE_REL_STRENGTH"
                elif pd.notna(row["sma_50"]) and row["adjusted_close"] < row["sma_50"]:
                    reason = "BELOW_SMA_50"
                elif pd.notna(row["atr_14"]) and row["adjusted_close"] < meta["peak_price"] - 2 * row["atr_14"]:
                    reason = "ATR_TRAILING_STOP"
                elif meta["holding_days"] >= max_holding_days:
                    reason = "MAX_HOLDING_DAYS"
            if reason is not None:
                trades_rows.append(
                    {
                        "date": current_date,
                        "ticker": ticker,
                        "action": "SELL",
                        "reason": reason,
                        "price": float(meta["current_price"]),
                        "score": float(meta["score"]),
                        "rank": int(meta["rank"]),
                        "holding_days": int(meta["holding_days"]),
                        "weight_before": float(meta["weight"]),
                        "weight_after": 0.0,
                    }
                )
                exiting.add(ticker)

        for ticker in exiting:
            holdings.pop(ticker, None)

        enter_candidates = ranked.loc[ranked["rank"] <= enter_rank].copy()
        for _, row in enter_candidates.iterrows():
            if row["ticker"] not in holdings and len(holdings) < top_n:
                holdings[row["ticker"]] = {
                    "entry_date": current_date,
                    "holding_days": 0,
                    "peak_price": float(row["adjusted_close"]),
                    "current_price": float(row["adjusted_close"]),
                    "score": float(row["score"]),
                    "rank": int(row["rank"]),
                    "weight": 0.0,
                }
                trades_rows.append(
                    {
                        "date": current_date,
                        "ticker": row["ticker"],
                        "action": "BUY",
                        "reason": "ENTER_RANK",
                        "price": float(row["adjusted_close"]),
                        "score": float(row["score"]),
                        "rank": int(row["rank"]),
                        "holding_days": 0,
                        "weight_before": 0.0,
                        "weight_after": None,
                    }
                )

        holding_tickers = list(holdings)
        current_selected = ranked.loc[ranked["ticker"].isin(holding_tickers)].copy()
        current_selected = _build_weights(
            current_selected,
            exposure=exposure if not current_selected.empty else 0.0,
            use_inverse_vol_weighting=use_inverse_vol_weighting,
            max_single_name_weight=0.15,
        )
        new_weights = dict(zip(current_selected["ticker"], current_selected["weight"])) if not current_selected.empty else {}
        previous_weights = {ticker: meta["weight"] for ticker, meta in holdings.items()}
        turnover = _compute_turnover(previous_weights, new_weights)
        transaction_cost = turnover * transaction_cost_bps / 10000.0

        gross_return = 0.0
        holding_records = []
        for _, row in current_selected.iterrows():
            ticker = row["ticker"]
            current_price = float(row["adjusted_close"])
            next_price_series = df.loc[(df["date"] == next_date) & (df["ticker"] == ticker), "adjusted_close"]
            if next_price_series.empty or pd.isna(next_price_series.iloc[0]):
                realized_return = 0.0
                sell_reason = "MISSING_PRICE_DATA"
                trades_rows.append(
                    {
                        "date": current_date,
                        "ticker": ticker,
                        "action": "SELL",
                        "reason": sell_reason,
                        "price": current_price,
                        "score": float(row["score"]),
                        "rank": int(row["rank"]),
                        "holding_days": int(holdings[ticker]["holding_days"]),
                        "weight_before": float(row["weight"]),
                        "weight_after": 0.0,
                    }
                )
                holdings.pop(ticker, None)
                continue

            next_price = float(next_price_series.iloc[0])
            realized_return = next_price / current_price - 1
            gross_return += float(row["weight"]) * realized_return
            holdings[ticker].update(
                {
                    "holding_days": int(holdings[ticker]["holding_days"] + rebalance_frequency_days),
                    "peak_price": max(float(holdings[ticker]["peak_price"]), next_price),
                    "current_price": next_price,
                    "score": float(row["score"]),
                    "rank": int(row["rank"]),
                    "weight": float(row["weight"]),
                }
            )
            holding_records.append(
                {
                    "date": current_date,
                    "strategy_name": strategy_name,
                    "ticker": ticker,
                    "weight": float(row["weight"]),
                    "score": float(row["score"]),
                    "rank": int(row["rank"]),
                    "holding_days": int(holdings[ticker]["holding_days"]),
                    "adjusted_close": current_price,
                    "next_adjusted_close": next_price,
                    "realized_period_return": realized_return,
                }
            )

        net_return = gross_return - transaction_cost
        spy_next_price = float(benchmark_prices.loc[next_date, "adjusted_close"]) if next_date in benchmark_prices.index else float(benchmark_prices.loc[current_date, "adjusted_close"])
        spy_current_price = float(benchmark_prices.loc[current_date, "adjusted_close"]) if current_date in benchmark_prices.index else spy_next_price
        spy_return = spy_next_price / spy_current_price - 1 if spy_current_price else 0.0
        portfolio_value *= 1 + net_return
        portfolio_peak = max(portfolio_peak, portfolio_value)
        spy_value *= 1 + spy_return

        returns_rows.append(
            {
                "date": current_date,
                "strategy_name": strategy_name,
                "selected_count": len(current_selected),
                "gross_return": gross_return,
                "turnover": turnover,
                "transaction_cost": transaction_cost,
                "net_return": net_return,
                "spy_return": spy_return,
                "excess_return": net_return - spy_return,
                "portfolio_value": portfolio_value,
                "spy_value": spy_value,
                "exposure": float(current_selected["weight"].sum()) if not current_selected.empty else 0.0,
                "regime_allowed": regime_allowed,
                "qualified_count": len(qualified),
                **diagnostics,
            }
        )
        holdings_rows.extend(holding_records)

    returns_df = pd.DataFrame(returns_rows)
    holdings_df = pd.DataFrame(holdings_rows)
    trades_df = pd.DataFrame(trades_rows)
    _validate_holdings(holdings_df, benchmark)
    return returns_df, holdings_df, trades_df


def save_backtest_outputs(
    weekly_returns: pd.DataFrame,
    holdings: pd.DataFrame,
    output_dir: str | Path,
    benchmark: str = "SPY",
    prefix: str | None = None,
) -> None:
    _validate_holdings(holdings, benchmark)
    output_dir = Path(output_dir)
    stem = f"{prefix}_" if prefix else ""
    save_dataframe(output_dir / f"{stem}weekly_portfolio_returns.csv", weekly_returns)
    save_dataframe(output_dir / f"{stem}weekly_holdings.csv", holdings)
    trades = holdings.copy()
    trades["trade_type"] = "rebalance_target"
    save_dataframe(output_dir / f"{stem}trades.csv", trades)


def save_filter_diagnostics(diagnostics: pd.DataFrame, output_dir: str | Path, prefix: str | None = None) -> None:
    stem = f"{prefix}_" if prefix else ""
    save_dataframe(Path(output_dir) / f"{stem}filter_diagnostics.csv", diagnostics)


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
