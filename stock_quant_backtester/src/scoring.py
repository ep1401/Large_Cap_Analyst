from __future__ import annotations

import numpy as np
import pandas as pd


VALID_HOLDING_PERIODS = {5, 21, 63}
FUTURE_RETURN_COLUMN_MAP = {
    5: ("future_5d_return", "future_5d_spy_return", "future_5d_excess_return"),
    21: ("future_21d_return", "future_21d_spy_return", "future_21d_excess_return"),
    63: ("future_63d_return", "future_63d_spy_return", "future_63d_excess_return"),
}
SCORE_INPUT_COLUMNS = [
    "consensus_upside",
    "low_target_upside",
    "relative_strength_21d",
    "distance_to_30d_high",
    "breakout_30d",
    "volume_spike_ratio",
    "volatility_21d",
]


def validate_holding_period_days(holding_period_days: int) -> None:
    if holding_period_days not in VALID_HOLDING_PERIODS:
        raise ValueError(
            f"holding_period_days must be one of {sorted(VALID_HOLDING_PERIODS)}; got {holding_period_days}"
        )


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


def apply_filters(
    df: pd.DataFrame,
    params: dict,
    holding_period_days: int,
    benchmark: str,
) -> pd.DataFrame:
    """Apply minimal hard filters before ranking."""
    future_return_column, _, _ = get_future_return_columns(holding_period_days)

    mask = pd.Series(True, index=df.index)
    mask &= df["ticker"].ne(benchmark)
    mask &= df["adjusted_close"].notna()
    mask &= df[future_return_column].notna()
    mask &= df["avg_dollar_volume_21d"].fillna(0) >= params.get("min_avg_dollar_volume", 20_000_000)

    if params.get("use_analyst_filters", False):
        mask &= df["analyst_count"].fillna(-np.inf) >= params.get("analyst_count_threshold", 10)

    if params.get("require_positive_relative_strength", False):
        mask &= df["relative_strength_21d"].fillna(-np.inf) > 0

    return df.loc[mask].copy()


def score_rebalance_date(
    df_for_one_date: pd.DataFrame,
    strategy_name: str = "full_model",
    use_analyst_filters: bool = True,
) -> pd.DataFrame:
    """Calculate cross-sectional scores on a single rebalance date."""
    validate_score_inputs()

    if df_for_one_date.empty:
        return df_for_one_date.assign(score=pd.Series(dtype=float))

    df = df_for_one_date.copy()
    strategy_name = strategy_name.lower()

    consensus_component = _cross_sectional_zscore(df["consensus_upside"])
    low_target_component = _cross_sectional_zscore(df["low_target_upside"])
    relative_strength_component = _cross_sectional_zscore(df["relative_strength_21d"])
    distance_component = _cross_sectional_zscore(-df["distance_to_30d_high"])
    volume_component = _cross_sectional_zscore(df["volume_spike_ratio"])
    volatility_component = _cross_sectional_zscore(df["volatility_21d"])
    breakout_bonus = df["breakout_30d"].fillna(False).astype(float) * 0.25

    if strategy_name == "technical_only":
        df["score"] = 0.50 * relative_strength_component + 0.30 * distance_component + 0.20 * volume_component
        return df

    if strategy_name == "analyst_only":
        df["score"] = 0.70 * consensus_component + 0.30 * low_target_component
        return df

    analyst_weight_scale = 1.0 if use_analyst_filters else 0.0
    analyst_total_weight = 0.30 + 0.20
    non_analyst_score = (
        0.20 * relative_strength_component
        + 0.15 * distance_component
        + 0.10 * volume_component
        - 0.05 * volatility_component
        + breakout_bonus
    )
    if analyst_weight_scale == 0:
        scale = 1.0 / (0.20 + 0.15 + 0.10 + 0.05)
        df["score"] = non_analyst_score * scale
        return df

    df["score"] = (
        0.30 * consensus_component
        + 0.20 * low_target_component
        + 0.20 * relative_strength_component
        + 0.15 * distance_component
        + 0.10 * volume_component
        - 0.05 * volatility_component
        + breakout_bonus
    )
    return df


def get_strategy_filter_params(
    strategy_name: str,
    analyst_count_threshold: int = 10,
    use_analyst_filters: bool = True,
    min_avg_dollar_volume: float = 20_000_000,
) -> dict:
    """Map strategy names to minimal hard filter settings."""
    strategy_name = strategy_name.lower()
    base = {
        "use_analyst_filters": False,
        "require_positive_relative_strength": False,
        "analyst_count_threshold": analyst_count_threshold,
        "min_avg_dollar_volume": min_avg_dollar_volume,
    }

    if strategy_name == "technical_only":
        return {**base, "require_positive_relative_strength": True}
    if strategy_name == "analyst_only":
        return {**base, "use_analyst_filters": use_analyst_filters}
    return {**base, "use_analyst_filters": use_analyst_filters}
