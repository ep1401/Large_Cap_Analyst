from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


VALID_HOLDING_PERIODS = {5, 21, 63}
FUTURE_RETURN_COLUMN_MAP = {
    5: ("future_5d_return", "future_5d_spy_return", "future_5d_excess_return"),
    21: ("future_21d_return", "future_21d_spy_return", "future_21d_excess_return"),
    63: ("future_63d_return", "future_63d_spy_return", "future_63d_excess_return"),
}
SENTIMENT_STRATEGIES = {
    "full_model_with_sentiment",
    "strict_checklist_with_sentiment",
    "sentiment_only",
    "analyst_sentiment_model",
    "technical_sentiment_model",
}
SENTIMENT_COLUMNS = [
    "article_count_7d",
    "article_count_30d",
    "news_sentiment_7d",
    "news_sentiment_30d",
    "sentiment_change_7d_vs_30d",
    "positive_news_ratio_7d",
    "negative_news_ratio_7d",
    "neutral_news_ratio_7d",
    "strong_negative_news_flag",
    "strong_positive_news_flag",
]
SCORE_INPUT_COLUMNS = [
    "consensus_upside",
    "low_target_upside",
    "target_revision_7d",
    "target_revision_30d",
    "relative_strength_21d",
    "relative_strength_63d",
    "distance_to_30d_high",
    "distance_to_63d_high",
    "breakout_30d",
    "breakout_63d",
    "volume_spike_ratio",
    "volatility_21d",
    "beta_to_spy_63d",
    "above_sma_50",
    "above_sma_200",
    *SENTIMENT_COLUMNS,
]


@dataclass(slots=True)
class StrategyParams:
    strategy_name: str
    use_analyst_filters: bool = True
    analyst_count_threshold: int = 10
    min_avg_dollar_volume: float = 20_000_000
    resistance_distance_threshold: float = 0.02
    require_low_target_upside_4pct: bool = False
    require_positive_revision_7d: bool = False
    require_positive_revision_30d: bool = False
    resistance_window: int = 30
    require_positive_sentiment: bool = False
    avoid_strong_negative_news: bool = False
    min_article_count_7d: int = 0


def validate_holding_period_days(holding_period_days: int) -> None:
    if holding_period_days not in VALID_HOLDING_PERIODS:
        raise ValueError(f"holding_period_days must be one of {sorted(VALID_HOLDING_PERIODS)}; got {holding_period_days}")


def get_future_return_columns(holding_period_days: int) -> tuple[str, str, str]:
    validate_holding_period_days(holding_period_days)
    return FUTURE_RETURN_COLUMN_MAP[holding_period_days]


def validate_score_inputs() -> None:
    invalid = [column for column in SCORE_INPUT_COLUMNS if column.startswith("future_")]
    if invalid:
        raise ValueError(f"Future return columns cannot be used in score calculations: {invalid}")


def _cross_sectional_zscore(series: pd.Series, clip_range: tuple[float, float] = (-3, 3)) -> pd.Series:
    filled = pd.to_numeric(series, errors="coerce")
    std = filled.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(np.zeros(len(series)), index=series.index, dtype=float)
    zscore = (filled - filled.mean()) / std
    return zscore.clip(*clip_range).fillna(0.0)


def _safe_series(df: pd.DataFrame, column: str, default: float | bool | None = np.nan) -> pd.Series:
    if column in df.columns:
        return df[column]
    return pd.Series(default, index=df.index)


def _resistance_columns(resistance_window: int) -> tuple[str, str]:
    if resistance_window not in {30, 63, 126}:
        raise ValueError(f"resistance_window must be one of [30, 63, 126]; got {resistance_window}")
    return f"distance_to_{resistance_window}d_high", f"breakout_{resistance_window}d"


def _requires_sentiment(strategy_name: str) -> bool:
    return strategy_name.lower() in SENTIMENT_STRATEGIES


def _validate_sentiment_inputs(df: pd.DataFrame, strategy_name: str) -> None:
    strategy_name = strategy_name.lower()
    if not _requires_sentiment(strategy_name):
        return
    missing_columns = [column for column in SENTIMENT_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Strategy '{strategy_name}' requires sentiment features, but these columns are missing: {missing_columns}"
        )
    if "sentiment_data_mode" in df.columns and df["sentiment_data_mode"].fillna("").eq("missing_news_sentiment").all():
        raise ValueError(
            f"Strategy '{strategy_name}' requires sentiment features, but the feature panel was built without "
            "news sentiment data. Run scripts/12_fetch_fmp_news.py, scripts/13_build_news_sentiment.py, and "
            "scripts/04_build_features.py first."
        )


def get_strategy_filter_params(
    strategy_name: str,
    analyst_count_threshold: int = 10,
    use_analyst_filters: bool = True,
    min_avg_dollar_volume: float = 20_000_000,
    resistance_distance_threshold: float = 0.02,
    require_low_target_upside_4pct: bool = False,
    require_positive_revision_7d: bool = False,
    require_positive_revision_30d: bool = False,
    resistance_window: int = 30,
    require_positive_sentiment: bool = False,
    avoid_strong_negative_news: bool = False,
    min_article_count_7d: int = 0,
) -> StrategyParams:
    return StrategyParams(
        strategy_name=strategy_name.lower(),
        use_analyst_filters=use_analyst_filters,
        analyst_count_threshold=analyst_count_threshold,
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


def get_filter_diagnostics(
    df: pd.DataFrame,
    params: StrategyParams,
    holding_period_days: int,
    benchmark: str,
) -> dict[str, int]:
    _validate_sentiment_inputs(df, params.strategy_name)
    future_return_column, _, _ = get_future_return_columns(holding_period_days)
    distance_col, breakout_col = _resistance_columns(params.resistance_window)

    starting_mask = df["ticker"].ne(benchmark) & _safe_series(df, "adjusted_close").notna() & _safe_series(df, future_return_column).notna()
    liquidity_mask = starting_mask & (_safe_series(df, "avg_dollar_volume_21d", 0).fillna(0) >= params.min_avg_dollar_volume)
    analyst_mask = liquidity_mask & (
        (_safe_series(df, "analyst_count", -np.inf).fillna(-np.inf) >= params.analyst_count_threshold)
        if params.use_analyst_filters
        else True
    )
    consensus_mask = analyst_mask & (_safe_series(df, "consensus_upside", -np.inf).fillna(-np.inf) >= 0.04)
    low_target_threshold = 0.04 if params.require_low_target_upside_4pct else 0.0
    low_target_mask = consensus_mask & (_safe_series(df, "low_target_upside", -np.inf).fillna(-np.inf) >= low_target_threshold)

    revision_7d_mask = low_target_mask
    if params.require_positive_revision_7d:
        revision_7d_mask = revision_7d_mask & _safe_series(df, "target_revision_7d").notna() & (_safe_series(df, "target_revision_7d") >= 0)

    revision_30d_mask = revision_7d_mask
    if params.require_positive_revision_30d:
        revision_30d_mask = revision_30d_mask & _safe_series(df, "target_revision_30d").notna() & (_safe_series(df, "target_revision_30d") >= 0)

    resistance_mask = revision_30d_mask & (
        (_safe_series(df, distance_col, np.inf).fillna(np.inf) <= params.resistance_distance_threshold)
        | _safe_series(df, breakout_col, False).fillna(False)
    )

    diagnostics = {
        "starting_universe_count": int(starting_mask.sum()),
        "passed_liquidity_count": int(liquidity_mask.sum()),
        "passed_analyst_count": int(analyst_mask.sum()),
        "passed_consensus_upside_count": int(consensus_mask.sum()),
        "passed_low_target_upside_count": int(low_target_mask.sum()),
        "passed_revision_7d_count": int(revision_7d_mask.sum()),
        "passed_revision_30d_count": int(revision_30d_mask.sum()),
        "passed_resistance_count": int(resistance_mask.sum()),
        "final_pass_count": int(resistance_mask.sum()),
    }

    if params.strategy_name == "strict_checklist_with_sentiment":
        sentiment_mask = resistance_mask
        sentiment_mask &= _safe_series(df, "article_count_7d", 0).fillna(0) >= params.min_article_count_7d
        diagnostics["passed_min_article_count_7d_count"] = int(sentiment_mask.sum())
        if params.require_positive_sentiment:
            sentiment_mask &= _safe_series(df, "news_sentiment_7d", -np.inf).fillna(-np.inf) > 0
            diagnostics["passed_positive_sentiment_count"] = int(sentiment_mask.sum())
        if params.avoid_strong_negative_news:
            sentiment_mask &= ~_safe_series(df, "strong_negative_news_flag", False).fillna(False)
            diagnostics["passed_avoid_strong_negative_news_count"] = int(sentiment_mask.sum())
        diagnostics["final_pass_count"] = int(sentiment_mask.sum())

    return diagnostics


def apply_filters(
    df: pd.DataFrame,
    params: StrategyParams,
    holding_period_days: int,
    benchmark: str,
) -> tuple[pd.DataFrame, dict[str, int]]:
    _validate_sentiment_inputs(df, params.strategy_name)
    future_return_column, _, _ = get_future_return_columns(holding_period_days)
    distance_col, breakout_col = _resistance_columns(params.resistance_window)

    mask = pd.Series(True, index=df.index)
    mask &= df["ticker"].ne(benchmark)
    mask &= _safe_series(df, "adjusted_close").notna()
    mask &= _safe_series(df, future_return_column).notna()
    mask &= _safe_series(df, "avg_dollar_volume_21d", 0).fillna(0) >= params.min_avg_dollar_volume

    if params.use_analyst_filters:
        mask &= _safe_series(df, "analyst_count", -np.inf).fillna(-np.inf) >= params.analyst_count_threshold

    diagnostics = get_filter_diagnostics(df, params, holding_period_days, benchmark)

    if params.strategy_name == "technical_only":
        mask &= _safe_series(df, "relative_strength_21d", -np.inf).fillna(-np.inf) > 0
    elif params.strategy_name == "analyst_only":
        mask &= _safe_series(df, "consensus_upside").notna() & _safe_series(df, "low_target_upside").notna()
    elif params.strategy_name in {"strict_checklist_model", "strict_checklist_with_sentiment"}:
        mask &= _safe_series(df, "consensus_upside", -np.inf).fillna(-np.inf) >= 0.04
        low_target_threshold = 0.04 if params.require_low_target_upside_4pct else 0.0
        mask &= _safe_series(df, "low_target_upside", -np.inf).fillna(-np.inf) >= low_target_threshold
        if params.require_positive_revision_7d:
            mask &= _safe_series(df, "target_revision_7d").notna() & (_safe_series(df, "target_revision_7d") >= 0)
        if params.require_positive_revision_30d:
            mask &= _safe_series(df, "target_revision_30d").notna() & (_safe_series(df, "target_revision_30d") >= 0)
        mask &= (
            (_safe_series(df, distance_col, np.inf).fillna(np.inf) <= params.resistance_distance_threshold)
            | _safe_series(df, breakout_col, False).fillna(False)
        )
        if params.strategy_name == "strict_checklist_with_sentiment":
            mask &= _safe_series(df, "article_count_7d", 0).fillna(0) >= params.min_article_count_7d
            if params.require_positive_sentiment:
                mask &= _safe_series(df, "news_sentiment_7d", -np.inf).fillna(-np.inf) > 0
            if params.avoid_strong_negative_news:
                mask &= ~_safe_series(df, "strong_negative_news_flag", False).fillna(False)
    elif params.strategy_name == "sentiment_only":
        mask &= _safe_series(df, "article_count_7d", 0).fillna(0) >= 1

    return df.loc[mask].copy(), diagnostics


def score_rebalance_date(
    df_for_one_date: pd.DataFrame,
    strategy_name: str = "full_model",
    use_analyst_filters: bool = True,
    resistance_window: int = 30,
) -> pd.DataFrame:
    validate_score_inputs()
    _validate_sentiment_inputs(df_for_one_date, strategy_name)

    if df_for_one_date.empty:
        return df_for_one_date.assign(score=pd.Series(dtype=float), rank=pd.Series(dtype=float))

    strategy_name = strategy_name.lower()
    distance_col, breakout_col = _resistance_columns(resistance_window)
    df = df_for_one_date.copy()

    consensus_component = _cross_sectional_zscore(_safe_series(df, "consensus_upside"))
    low_target_component = _cross_sectional_zscore(_safe_series(df, "low_target_upside"))
    revision_7d_component = _cross_sectional_zscore(_safe_series(df, "target_revision_7d"))
    revision_30d_component = _cross_sectional_zscore(_safe_series(df, "target_revision_30d"))
    relative_strength_21_component = _cross_sectional_zscore(_safe_series(df, "relative_strength_21d"))
    relative_strength_63_component = _cross_sectional_zscore(_safe_series(df, "relative_strength_63d"))
    distance_component = _cross_sectional_zscore(-_safe_series(df, distance_col))
    distance_63_component = _cross_sectional_zscore(-_safe_series(df, "distance_to_63d_high"))
    volume_component = _cross_sectional_zscore(_safe_series(df, "volume_spike_ratio"))
    volatility_component = _cross_sectional_zscore(_safe_series(df, "volatility_21d"))
    beta_component = _cross_sectional_zscore(_safe_series(df, "beta_to_spy_63d"))
    above_sma_50_component = _cross_sectional_zscore(_safe_series(df, "above_sma_50"))
    above_sma_200_component = _cross_sectional_zscore(_safe_series(df, "above_sma_200"))
    breakout_bonus = _safe_series(df, breakout_col, False).fillna(False).astype(float) * 0.25

    news_sentiment_component = _cross_sectional_zscore(_safe_series(df, "news_sentiment_7d"))
    sentiment_change_component = _cross_sectional_zscore(_safe_series(df, "sentiment_change_7d_vs_30d"))
    positive_news_ratio_component = _cross_sectional_zscore(_safe_series(df, "positive_news_ratio_7d"))
    negative_news_ratio_component = _cross_sectional_zscore(_safe_series(df, "negative_news_ratio_7d"))
    strong_negative_news_penalty = _safe_series(df, "strong_negative_news_flag", False).fillna(False).astype(float)

    if strategy_name == "technical_only":
        df["score"] = (
            0.50 * relative_strength_21_component
            + 0.30 * _cross_sectional_zscore(-_safe_series(df, "distance_to_30d_high"))
            + 0.20 * volume_component
        )
    elif strategy_name == "technical_momentum_model":
        df["score"] = (
            0.25 * relative_strength_21_component
            + 0.20 * relative_strength_63_component
            + 0.15 * distance_component
            + 0.15 * volume_component
            + 0.10 * above_sma_50_component
            + 0.10 * above_sma_200_component
            - 0.05 * volatility_component
            + breakout_bonus
        )
    elif strategy_name == "analyst_only":
        df["score"] = 0.70 * consensus_component + 0.30 * low_target_component
    elif strategy_name == "strict_checklist_model":
        df["score"] = (
            0.35 * consensus_component
            + 0.20 * low_target_component
            + 0.15 * revision_7d_component
            + 0.10 * revision_30d_component
            + 0.10 * distance_component
            + 0.10 * relative_strength_21_component
            + breakout_bonus
        )
    elif strategy_name == "full_model_with_sentiment":
        base = (
            0.25 * consensus_component
            + 0.15 * low_target_component
            + 0.15 * relative_strength_21_component
            + 0.10 * relative_strength_63_component
            + 0.10 * distance_component
            + 0.10 * volume_component
            + 0.10 * above_sma_50_component
            + 0.05 * above_sma_200_component
            - 0.05 * volatility_component
            - 0.05 * beta_component
            + breakout_bonus
        )
        df["score"] = (
            base
            + 0.15 * news_sentiment_component
            + 0.10 * sentiment_change_component
            + 0.05 * positive_news_ratio_component
            - 0.10 * negative_news_ratio_component
            - 0.10 * strong_negative_news_penalty
        )
    elif strategy_name == "strict_checklist_with_sentiment":
        base = (
            0.35 * consensus_component
            + 0.20 * low_target_component
            + 0.15 * revision_7d_component
            + 0.10 * revision_30d_component
            + 0.10 * distance_component
            + 0.10 * relative_strength_21_component
            + breakout_bonus
        )
        df["score"] = (
            base
            + 0.15 * news_sentiment_component
            + 0.10 * sentiment_change_component
            + 0.05 * positive_news_ratio_component
            - 0.10 * negative_news_ratio_component
            - 0.10 * strong_negative_news_penalty
        )
    elif strategy_name == "sentiment_only":
        df["score"] = (
            0.50 * news_sentiment_component
            + 0.25 * sentiment_change_component
            + 0.15 * positive_news_ratio_component
            - 0.10 * negative_news_ratio_component
        )
    elif strategy_name == "analyst_sentiment_model":
        df["score"] = (
            0.35 * consensus_component
            + 0.20 * low_target_component
            + 0.20 * news_sentiment_component
            + 0.10 * sentiment_change_component
            - 0.10 * negative_news_ratio_component
            + 0.15 * relative_strength_21_component
        )
    elif strategy_name == "technical_sentiment_model":
        df["score"] = (
            0.25 * relative_strength_21_component
            + 0.20 * relative_strength_63_component
            + 0.15 * distance_63_component
            + 0.15 * news_sentiment_component
            + 0.10 * sentiment_change_component
            + 0.10 * volume_component
            - 0.05 * volatility_component
        )
    else:
        if use_analyst_filters:
            df["score"] = (
                0.25 * consensus_component
                + 0.15 * low_target_component
                + 0.15 * relative_strength_21_component
                + 0.10 * relative_strength_63_component
                + 0.10 * distance_component
                + 0.10 * volume_component
                + 0.10 * above_sma_50_component
                + 0.05 * above_sma_200_component
                - 0.05 * volatility_component
                - 0.05 * beta_component
                + breakout_bonus
            )
        else:
            df["score"] = (
                0.25 * relative_strength_21_component
                + 0.20 * relative_strength_63_component
                + 0.15 * distance_component
                + 0.15 * volume_component
                + 0.10 * above_sma_50_component
                + 0.10 * above_sma_200_component
                - 0.05 * volatility_component
                + breakout_bonus
            )

    df["rank"] = df["score"].rank(ascending=False, method="first")
    return df
