from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.scoring import (
    StrategyParams,
    apply_filters,
    canonical_strategy_name,
    get_filter_diagnostics,
    get_future_return_columns,
    get_strategy_filter_params,
    resolve_long_short_base_strategy,
    score_rebalance_date,
    strategy_analyst_data_mode,
    validate_strategy_holding_period,
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
    position_sizing: str = "equal_weight",
) -> pd.DataFrame:
    selected = selected.copy()
    if selected.empty or exposure <= 0:
        selected["weight"] = 0.0
        return selected

    if use_inverse_vol_weighting and position_sizing == "equal_weight":
        position_sizing = "inverse_volatility"

    if position_sizing == "inverse_volatility":
        inv_vol = 1 / selected["volatility_21d"].replace(0, np.nan)
        inv_vol = inv_vol.fillna(inv_vol.mean()).fillna(1.0)
        weights = inv_vol / inv_vol.sum()
    elif position_sizing == "score_weighted":
        shifted = pd.to_numeric(selected["score"], errors="coerce")
        shifted = shifted - shifted.min() + 1e-6
        if float(shifted.sum()) <= 0:
            weights = pd.Series(1 / len(selected), index=selected.index)
        else:
            weights = shifted / shifted.sum()
    elif position_sizing == "score_over_volatility":
        shifted = pd.to_numeric(selected["score"], errors="coerce")
        shifted = shifted - shifted.min() + 1e-6
        inv_vol = 1 / selected["volatility_21d"].replace(0, np.nan)
        inv_vol = inv_vol.fillna(inv_vol.mean()).fillna(1.0)
        combined = shifted * inv_vol
        if float(combined.sum()) <= 0:
            weights = pd.Series(1 / len(selected), index=selected.index)
        else:
            weights = combined / combined.sum()
    else:
        weights = pd.Series(1 / len(selected), index=selected.index)

    weights = weights * exposure
    weights = weights.clip(upper=max_single_name_weight)
    if weights.sum() > 0:
        weights = weights * (exposure / weights.sum())
    selected["weight"] = weights
    return selected


def _compute_regime_state(
    df: pd.DataFrame,
    rebalance_date: pd.Timestamp,
    benchmark: str,
    regime_filter_type: str,
) -> bool:
    benchmark_history = (
        df.loc[df["ticker"] == benchmark, ["date", "adjusted_close", "spy_close", "spy_sma_50", "spy_sma_200"]]
        .drop_duplicates("date")
        .sort_values("date")
    )
    benchmark_history = benchmark_history.loc[benchmark_history["date"] <= rebalance_date].copy()
    if benchmark_history.empty:
        return True

    latest = benchmark_history.iloc[-1]
    close_value = float(latest["spy_close"]) if pd.notna(latest.get("spy_close")) else float(latest["adjusted_close"])

    if regime_filter_type == "spy_50d":
        sma_50 = float(latest["spy_sma_50"]) if pd.notna(latest.get("spy_sma_50")) else float("nan")
        return bool(pd.notna(sma_50) and close_value > sma_50)
    if regime_filter_type == "spy_200d":
        sma_200 = float(latest["spy_sma_200"]) if pd.notna(latest.get("spy_sma_200")) else float("nan")
        return bool(pd.notna(sma_200) and close_value > sma_200)
    if regime_filter_type == "spy_50d_return_positive":
        if len(benchmark_history) < 51:
            return False
        prior = float(benchmark_history.iloc[-51]["adjusted_close"])
        return bool(prior > 0 and close_value / prior - 1 > 0)
    if regime_filter_type == "spy_21d_return_positive":
        if len(benchmark_history) < 22:
            return False
        prior = float(benchmark_history.iloc[-22]["adjusted_close"])
        return bool(prior > 0 and close_value / prior - 1 > 0)
    raise ValueError(f"Unsupported regime_filter_type: {regime_filter_type}")


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
    book_side: str | None = None,
) -> None:
    if selected.empty:
        return
    selected = selected.copy()
    selected["date"] = date
    selected["strategy_name"] = strategy_name
    selected["holding_period_days"] = holding_period_days
    selected["future_return_used"] = selected[future_return_column]
    selected["future_excess_return_used"] = selected[future_excess_return_column]
    selected["analyst_data_mode"] = strategy_analyst_data_mode(strategy_name)
    if book_side is not None:
        selected["book_side"] = book_side
    desired_columns = [
        "date",
        "strategy_name",
        "ticker",
        "sector",
        "book_side",
        "weight",
        "score",
        "rank",
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
        "historical_rating_score",
        "historical_positive_rating_ratio",
        "historical_negative_rating_ratio",
        "historical_total_ratings",
        "historical_rating_count_data_available",
        "net_upgrade_score_30d",
        "downgrade_count_30d",
        "positive_grade_ratio_30d",
        "historical_grade_data_available",
        "analyst_data_mode",
    ]
    holding_rows.extend(selected[[column for column in desired_columns if column in selected.columns]].to_dict("records"))


def _safe_median(df: pd.DataFrame, column: str) -> float:
    if column not in df.columns:
        return float("nan")
    series = pd.to_numeric(df[column], errors="coerce").dropna()
    if series.empty:
        return float("nan")
    return float(series.median())


def _select_short_candidates(scored: pd.DataFrame) -> pd.DataFrame:
    if scored.empty:
        return scored.copy()

    rating_score_median = _safe_median(scored, "historical_rating_score")
    negative_ratio_median = _safe_median(scored, "historical_negative_rating_ratio")
    negative_news_median = _safe_median(scored, "negative_news_ratio_7d")

    bearish_mask = pd.Series(False, index=scored.index)
    if "relative_strength_21d" in scored.columns:
        bearish_mask |= pd.to_numeric(scored["relative_strength_21d"], errors="coerce").fillna(np.inf) < 0
    if "historical_rating_score" in scored.columns and not pd.isna(rating_score_median):
        bearish_mask |= pd.to_numeric(scored["historical_rating_score"], errors="coerce").fillna(np.inf) < rating_score_median
    if "historical_negative_rating_ratio" in scored.columns and not pd.isna(negative_ratio_median):
        bearish_mask |= (
            pd.to_numeric(scored["historical_negative_rating_ratio"], errors="coerce").fillna(-np.inf) > negative_ratio_median
        )
    if "recent_downgrade_flag_30d" in scored.columns:
        bearish_mask |= scored["recent_downgrade_flag_30d"].fillna(False).astype(bool)
    if "negative_news_ratio_7d" in scored.columns and not pd.isna(negative_news_median):
        bearish_mask |= pd.to_numeric(scored["negative_news_ratio_7d"], errors="coerce").fillna(-np.inf) > negative_news_median
    if "strong_negative_news_flag" in scored.columns:
        bearish_mask |= scored["strong_negative_news_flag"].fillna(False).astype(bool)

    return scored.loc[bearish_mask].copy()


def _validate_long_short_holdings(
    long_holdings: pd.DataFrame,
    short_holdings: pd.DataFrame,
    benchmark: str,
    long_exposure: float,
    short_exposure: float,
) -> None:
    _validate_holdings(long_holdings, benchmark)
    _validate_holdings(short_holdings, benchmark)

    if long_holdings.empty and short_holdings.empty:
        return

    overlap = long_holdings[["date", "ticker"]].merge(short_holdings[["date", "ticker"]], on=["date", "ticker"], how="inner")
    if not overlap.empty:
        raise ValueError("Long and short books overlap on at least one rebalance date.")

    long_by_date = long_holdings.groupby("date")["weight"].sum() if not long_holdings.empty else pd.Series(dtype=float)
    short_by_date = short_holdings.groupby("date")["weight"].sum() if not short_holdings.empty else pd.Series(dtype=float)
    all_dates = sorted(set(long_by_date.index) | set(short_by_date.index))
    for date in all_dates:
        long_sum = float(long_by_date.get(date, 0.0))
        short_sum = float(short_by_date.get(date, 0.0))
        if long_holdings.loc[long_holdings["date"] == date].empty:
            long_target = 0.0
        else:
            long_target = long_exposure
        if short_holdings.loc[short_holdings["date"] == date].empty:
            short_target = 0.0
        else:
            short_target = -short_exposure
        if not np.isclose(long_sum, long_target, atol=1e-8):
            raise ValueError(f"Long weights do not sum to exposure on {date}: {long_sum} vs {long_target}")
        if not np.isclose(short_sum, short_target, atol=1e-8):
            raise ValueError(f"Short weights do not sum to exposure on {date}: {short_sum} vs {short_target}")


def run_long_short_backtest(
    features: pd.DataFrame,
    strategy_name: str,
    holding_period_days: int = 5,
    long_n: int = 10,
    short_n: int = 10,
    long_exposure: float = 1.0,
    short_exposure: float = 0.5,
    benchmark: str = "SPY",
    transaction_cost_bps: float = 10,
    short_borrow_bps_annual: float = 300,
    extra_short_slippage_bps: float = 5,
    max_single_name_weight: float = 0.15,
    min_avg_dollar_volume: float = 20_000_000,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run a non-overlapping long/short cross-sectional backtest using an existing no-snapshot score."""
    validate_holding_period_days(holding_period_days)
    validate_strategy_holding_period(strategy_name, holding_period_days)
    if long_n <= 0 or short_n <= 0:
        raise ValueError("long_n and short_n must both be positive integers.")
    if long_exposure < 0 or short_exposure < 0:
        raise ValueError("long_exposure and short_exposure must be non-negative.")

    df = features.copy()
    df["date"] = pd.to_datetime(df["date"])
    strategy_name = canonical_strategy_name(strategy_name)
    scoring_strategy_name = resolve_long_short_base_strategy(strategy_name)
    future_return_column, future_spy_return_column, future_excess_return_column = get_future_return_columns(
        holding_period_days
    )
    rebalance_dates = select_rebalance_dates(df, holding_period_days=holding_period_days, benchmark=benchmark)
    params = get_strategy_filter_params(
        strategy_name=scoring_strategy_name,
        use_analyst_filters=False,
        analyst_count_threshold=0,
        min_avg_dollar_volume=min_avg_dollar_volume,
        min_historical_rating_count=5,
    )

    weekly_rows: list[dict] = []
    long_holding_rows: list[dict] = []
    short_holding_rows: list[dict] = []
    diagnostics_rows: list[dict] = []
    portfolio_value = 10000.0
    spy_value = 10000.0
    previous_long_weights: dict[str, float] = {}
    previous_short_weights: dict[str, float] = {}

    for rebalance_date in rebalance_dates:
        day_all = df.loc[df["date"] == rebalance_date].copy()
        day_slice = day_all.loc[day_all["ticker"] != benchmark].copy()
        qualified, diagnostics = apply_filters(
            day_slice,
            params=params,
            holding_period_days=holding_period_days,
            benchmark=benchmark,
        )
        scored = score_rebalance_date(
            qualified,
            strategy_name=scoring_strategy_name,
            use_analyst_filters=False,
        ).sort_values("score", ascending=False)
        bearish_candidates = _select_short_candidates(scored)

        selected_long = _build_weights(
            scored.head(long_n),
            exposure=long_exposure,
            use_inverse_vol_weighting=False,
            max_single_name_weight=max_single_name_weight,
        )
        selected_short = _build_weights(
            bearish_candidates.sort_values("score", ascending=True).head(short_n),
            exposure=short_exposure,
            use_inverse_vol_weighting=False,
            max_single_name_weight=max_single_name_weight,
        )
        selected_short = selected_short.copy()
        if not selected_short.empty:
            selected_short["weight"] = -selected_short["weight"].abs()

        long_tickers = set(selected_long["ticker"]) if not selected_long.empty else set()
        if not selected_short.empty and long_tickers:
            selected_short = selected_short.loc[~selected_short["ticker"].isin(long_tickers)].copy()
            if not selected_short.empty:
                selected_short = _build_weights(
                    selected_short.assign(weight=0.0),
                    exposure=short_exposure,
                    use_inverse_vol_weighting=False,
                    max_single_name_weight=max_single_name_weight,
                )
                selected_short["weight"] = -selected_short["weight"].abs()

        new_long_weights = dict(zip(selected_long["ticker"], selected_long["weight"])) if not selected_long.empty else {}
        new_short_weights = dict(zip(selected_short["ticker"], selected_short["weight"])) if not selected_short.empty else {}
        long_turnover = _compute_turnover(previous_long_weights, new_long_weights)
        short_turnover = _compute_turnover(previous_short_weights, new_short_weights)
        turnover = long_turnover + short_turnover

        long_contribution = float((selected_long["weight"] * selected_long[future_return_column].fillna(0.0)).sum()) if not selected_long.empty else 0.0
        short_contribution = float((selected_short["weight"] * selected_short[future_return_column].fillna(0.0)).sum()) if not selected_short.empty else 0.0
        short_sign_check = float(
            -(
                selected_short["weight"].abs() * selected_short[future_return_column].fillna(0.0)
            ).sum()
        ) if not selected_short.empty else 0.0
        if not np.isclose(short_contribution, short_sign_check, atol=1e-12):
            raise ValueError(f"Short P&L sign check failed on {rebalance_date}.")

        gross_return = long_contribution + short_contribution
        transaction_cost = turnover * transaction_cost_bps / 10000.0
        extra_short_slippage = short_turnover * extra_short_slippage_bps / 10000.0
        actual_short_exposure = float(selected_short["weight"].abs().sum()) if not selected_short.empty else 0.0
        borrow_cost = actual_short_exposure * short_borrow_bps_annual / 10000.0 * holding_period_days / 252.0
        if actual_short_exposure == 0 and borrow_cost != 0:
            raise ValueError(f"Borrow cost should only apply to active short exposure on {rebalance_date}.")
        total_cost = transaction_cost + extra_short_slippage + borrow_cost
        net_return = gross_return - total_cost

        benchmark_slice = day_all.loc[day_all["ticker"] == benchmark, future_spy_return_column]
        spy_return = float(benchmark_slice.iloc[0]) if not benchmark_slice.empty else 0.0
        excess_return = net_return - spy_return
        portfolio_value *= 1 + net_return
        spy_value *= 1 + spy_return

        actual_long_exposure = float(selected_long["weight"].sum()) if not selected_long.empty else 0.0
        gross_exposure = actual_long_exposure + actual_short_exposure
        net_exposure = actual_long_exposure - actual_short_exposure
        average_long_return = float(pd.to_numeric(selected_long[future_return_column], errors="coerce").mean()) if not selected_long.empty else 0.0
        average_short_book_return = float(pd.to_numeric(selected_short[future_return_column], errors="coerce").mean()) if not selected_short.empty else 0.0
        short_book_helped = bool(short_contribution > 0)
        short_book_hurt = bool(short_contribution < 0)

        diagnostics_rows.append(
            {
                "date": rebalance_date,
                "strategy_name": strategy_name,
                **diagnostics,
                "selected_long_count": len(selected_long),
                "selected_short_count": len(selected_short),
                "bearish_short_candidate_count": len(bearish_candidates),
                "final_pass_count": len(qualified),
            }
        )
        weekly_rows.append(
            {
                "date": rebalance_date,
                "strategy_name": strategy_name,
                "holding_period_days": holding_period_days,
                "long_n": long_n,
                "short_n": short_n,
                "selected_count": len(selected_long) + len(selected_short),
                "selected_long_count": len(selected_long),
                "selected_short_count": len(selected_short),
                "qualified_count": len(qualified),
                "gross_return": gross_return,
                "long_contribution": long_contribution,
                "short_contribution": short_contribution,
                "turnover": turnover,
                "long_turnover": long_turnover,
                "short_turnover": short_turnover,
                "transaction_cost": transaction_cost,
                "extra_short_slippage": extra_short_slippage,
                "borrow_cost": borrow_cost,
                "net_return": net_return,
                "spy_return": spy_return,
                "excess_return": excess_return,
                "portfolio_value": portfolio_value,
                "spy_value": spy_value,
                "exposure": net_exposure,
                "gross_exposure": gross_exposure,
                "net_exposure": net_exposure,
                "long_exposure": actual_long_exposure,
                "short_exposure": actual_short_exposure,
                "average_long_return": average_long_return,
                "average_short_book_return": average_short_book_return,
                "short_book_helped": short_book_helped,
                "short_book_hurt": short_book_hurt,
                "analyst_data_mode": strategy_analyst_data_mode(strategy_name),
            }
        )

        _append_holding_rows(
            long_holding_rows,
            selected_long,
            rebalance_date,
            strategy_name,
            holding_period_days,
            future_return_column,
            future_excess_return_column,
            book_side="long",
        )
        _append_holding_rows(
            short_holding_rows,
            selected_short,
            rebalance_date,
            strategy_name,
            holding_period_days,
            future_return_column,
            future_excess_return_column,
            book_side="short",
        )

        previous_long_weights = new_long_weights
        previous_short_weights = new_short_weights

    weekly_df = pd.DataFrame(weekly_rows)
    long_holdings_df = pd.DataFrame(long_holding_rows)
    short_holdings_df = pd.DataFrame(short_holding_rows)
    diagnostics_df = pd.DataFrame(diagnostics_rows)
    _validate_long_short_holdings(
        long_holdings_df,
        short_holdings_df,
        benchmark=benchmark,
        long_exposure=long_exposure,
        short_exposure=short_exposure,
    )
    if not weekly_df.empty:
        expected_gross = weekly_df["long_exposure"] + weekly_df["short_exposure"]
        expected_net = weekly_df["long_exposure"] - weekly_df["short_exposure"]
        if not np.allclose(weekly_df["gross_exposure"], expected_gross, atol=1e-10):
            raise ValueError("Gross exposure validation failed in long/short backtest.")
        if not np.allclose(weekly_df["net_exposure"], expected_net, atol=1e-10):
            raise ValueError("Net exposure validation failed in long/short backtest.")
    return weekly_df, long_holdings_df, short_holdings_df, diagnostics_df


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
    position_sizing: str = "equal_weight",
    max_single_name_weight: float = 0.15,
    enable_drawdown_protection: bool = False,
    regime_filter_type: str = "spy_200d",
    require_positive_sentiment: bool = False,
    avoid_strong_negative_news: bool = False,
    min_article_count_7d: int = 0,
    avoid_recent_downgrades: bool = False,
    min_grade_events_90d: int = 1,
    min_historical_rating_count: int = 5,
    min_score_threshold: float | None = None,
    allow_cash: bool = False,
    min_holdings: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run a non-overlapping cross-sectional backtest with turnover costs and diagnostics."""
    validate_holding_period_days(holding_period_days)
    validate_strategy_holding_period(strategy_name, holding_period_days)
    if not 0.0 <= regime_exposure <= 1.0:
        raise ValueError(f"regime_exposure must be between 0.0 and 1.0; got {regime_exposure}")
    if min_holdings is not None and min_holdings < 0:
        raise ValueError(f"min_holdings must be non-negative or None; got {min_holdings}")
    if top_n <= 0:
        raise ValueError(f"top_n must be positive; got {top_n}")

    df = features.copy()
    df["date"] = pd.to_datetime(df["date"])
    strategy_name = canonical_strategy_name(strategy_name)
    future_return_column, future_spy_return_column, future_excess_return_column = get_future_return_columns(
        holding_period_days
    )
    rebalance_dates = select_rebalance_dates(df, holding_period_days=holding_period_days, benchmark=benchmark)

    if strategy_name in {"full_model", "strict_checklist_model", "analyst_snapshot_model"} and use_analyst_filters and df["consensus_upside"].notna().sum() == 0:
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
        avoid_recent_downgrades=avoid_recent_downgrades,
        min_grade_events_90d=min_grade_events_90d,
        min_historical_rating_count=min_historical_rating_count,
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
        diagnostics_rows.append(
            {
                "date": rebalance_date,
                "strategy_name": strategy_name,
                **diagnostics,
                "selected_count": 0,
                "threshold_pass_count": 0,
                "target_top_n": top_n,
                "min_score_threshold": min_score_threshold,
                "allow_cash": allow_cash,
                "min_holdings": min_holdings,
                "cash_weight": 0.0,
                "percent_invested": 0.0,
            }
        )

        scored = score_rebalance_date(
            qualified,
            strategy_name=strategy_name,
            use_analyst_filters=use_analyst_filters,
            resistance_window=resistance_window,
        ).sort_values("score", ascending=False)
        threshold_passed = scored.copy()
        if min_score_threshold is not None:
            threshold_passed = threshold_passed.loc[
                pd.to_numeric(threshold_passed["score"], errors="coerce").fillna(-np.inf) > float(min_score_threshold)
            ].copy()
        threshold_pass_count = len(threshold_passed)
        selected = threshold_passed.head(top_n).copy()
        if min_holdings is not None and threshold_pass_count < min_holdings:
            if allow_cash:
                selected = threshold_passed.iloc[0:0].copy()
            else:
                selected = scored.head(top_n).copy()
        elif not allow_cash and len(selected) < top_n:
            refill = scored.loc[~scored["ticker"].isin(selected["ticker"])].head(top_n - len(selected))
            selected = pd.concat([selected, refill], ignore_index=False)
            selected = selected.sort_values("score", ascending=False).head(top_n).copy()
        selected = _apply_sector_limit(selected, max_names_per_sector=max_names_per_sector)

        regime_allowed = True
        exposure = 1.0
        if use_regime_filter:
            regime_allowed = _compute_regime_state(
                df,
                rebalance_date=rebalance_date,
                benchmark=benchmark,
                regime_filter_type=regime_filter_type,
            )
            exposure = 1.0 if regime_allowed else regime_exposure

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

        invested_exposure = exposure
        if allow_cash:
            invested_exposure = exposure * (len(selected) / top_n) if top_n > 0 else 0.0
        selected = _build_weights(
            selected,
            exposure=invested_exposure if not selected.empty else 0.0,
            use_inverse_vol_weighting=use_inverse_vol_weighting,
            max_single_name_weight=max_single_name_weight,
            position_sizing=position_sizing,
        )
        selected_count = len(selected)
        diagnostics_rows[-1]["selected_count"] = selected_count
        diagnostics_rows[-1]["threshold_pass_count"] = threshold_pass_count

        new_weights = dict(zip(selected["ticker"], selected["weight"])) if selected_count else {}
        gross_return = float((selected["weight"] * selected[future_return_column].fillna(0.0)).sum()) if selected_count else 0.0
        actual_exposure = float(selected["weight"].sum()) if selected_count else 0.0
        cash_weight = max(0.0, float(exposure) - actual_exposure)
        percent_invested = actual_exposure / float(exposure) if exposure > 0 else 0.0
        diagnostics_rows[-1]["cash_weight"] = cash_weight
        diagnostics_rows[-1]["percent_invested"] = percent_invested
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
                "regime_filter_type": regime_filter_type,
                "regime_exposure": regime_exposure,
                "position_sizing": position_sizing,
                "min_score_threshold": min_score_threshold,
                "allow_cash": allow_cash,
                "min_holdings": min_holdings,
                "analyst_count_threshold": analyst_count_threshold,
                "min_avg_dollar_volume": min_avg_dollar_volume,
                "selected_count": selected_count,
                "threshold_pass_count": threshold_pass_count,
                "target_top_n": top_n,
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
                "cash_weight": cash_weight,
                "percent_invested": percent_invested,
                "regime_allowed": regime_allowed,
                "analyst_data_mode": strategy_analyst_data_mode(strategy_name),
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


def save_long_short_backtest_outputs(
    weekly_returns: pd.DataFrame,
    long_holdings: pd.DataFrame,
    short_holdings: pd.DataFrame,
    output_dir: str | Path,
    benchmark: str = "SPY",
    prefix: str | None = None,
) -> None:
    _validate_long_short_holdings(
        long_holdings,
        short_holdings,
        benchmark=benchmark,
        long_exposure=0.0 if long_holdings.empty else float(long_holdings.groupby("date")["weight"].sum().iloc[0]),
        short_exposure=0.0 if short_holdings.empty else abs(float(short_holdings.groupby("date")["weight"].sum().iloc[0])),
    )
    output_dir = Path(output_dir)
    stem = f"{prefix}_" if prefix else ""
    save_dataframe(output_dir / f"{stem}weekly_portfolio_returns.csv", weekly_returns)
    save_dataframe(output_dir / f"{stem}long_holdings.csv", long_holdings)
    save_dataframe(output_dir / f"{stem}short_holdings.csv", short_holdings)


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
