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
    "historical_grades_plus_sentiment",
    "historical_rating_counts_plus_sentiment",
    "historical_rating_counts_plus_events_sentiment",
    "final_quant_model_1y_no_snapshot",
    "final_quant_model_no_snapshot",
}
SNAPSHOT_ANALYST_STRATEGIES = {
    "full_model",
    "full_model_with_sentiment",
    "strict_checklist_model",
    "strict_checklist_with_sentiment",
    "analyst_only",
    "analyst_snapshot_model",
    "analyst_sentiment_model",
    "final_quant_model_1y",
    "final_quant_model_snapshot",
    "final_quant_model_1y_no_sentiment",
    "final_quant_model_1y_sentiment_risk_filter",
    "final_quant_model_1y_sector_capped",
}
HISTORICAL_GRADE_STRATEGIES = {
    "historical_grades_model",
    "historical_grades_plus_sentiment",
    "strict_historical_grades_checklist",
    "historical_rating_counts_plus_events",
    "historical_rating_counts_plus_events_sentiment",
}
HISTORICAL_RATING_COUNT_STRATEGIES = {
    "historical_rating_counts_model",
    "historical_rating_counts_plus_sentiment",
    "historical_rating_counts_plus_events",
    "historical_rating_counts_plus_events_sentiment",
    "final_quant_model_1y_no_snapshot",
    "final_quant_model_no_snapshot",
}
SNAPSHOT_FIELD_COLUMNS = {
    "consensus_upside",
    "low_target_upside",
    "high_target_upside",
    "median_target",
    "consensus_target",
    "low_target",
    "high_target",
    "analyst_count",
    "last_month_target_upside",
    "last_quarter_target_upside",
    "last_year_target_upside",
    "all_time_target_upside",
    "last_month_target_count",
    "last_quarter_target_count",
    "last_year_target_count",
    "all_time_target_count",
    "target_revision_7d",
    "target_revision_30d",
    "last_month_avg_price_target",
    "last_quarter_avg_price_target",
    "last_year_avg_price_target",
    "all_time_avg_price_target",
    "target_spread",
}
NO_SNAPSHOT_STRATEGIES = {"final_quant_model_1y_no_snapshot", "final_quant_model_no_snapshot"}
STRATEGY_DISPLAY_NAMES = {
    "final_quant_model_1y_no_snapshot": "Final Quant Model - No Snapshot",
    "final_quant_model_no_snapshot": "Final Quant Model - No Snapshot",
    "final_quant_model_1y": "Final Quant Model - Snapshot Exploratory",
    "final_quant_model_snapshot": "Final Quant Model - Snapshot Exploratory",
}
STRATEGY_SCORE_FIELDS = {
    "historical_rating_counts_model": {
        "historical_rating_score",
        "historical_positive_rating_ratio",
        "historical_negative_rating_ratio",
        "historical_rating_score_change_30d",
        "historical_negative_ratio_change_30d",
        "relative_strength_21d",
        "relative_strength_63d",
    },
    "historical_rating_counts_plus_sentiment": {
        "historical_rating_score",
        "historical_positive_rating_ratio",
        "historical_negative_rating_ratio",
        "historical_rating_score_change_30d",
        "historical_negative_ratio_change_30d",
        "relative_strength_21d",
        "relative_strength_63d",
        "relevance_weighted_sentiment_7d",
        "sentiment_change_7d_vs_30d",
        "negative_news_ratio_7d",
    },
    "historical_rating_counts_plus_events": {
        "historical_rating_score",
        "historical_positive_rating_ratio",
        "historical_negative_rating_ratio",
        "historical_rating_score_change_30d",
        "historical_negative_ratio_change_30d",
        "relative_strength_21d",
        "relative_strength_63d",
        "net_upgrade_score_30d",
        "downgrade_count_30d",
        "recent_downgrade_flag_30d",
    },
    "historical_rating_counts_plus_events_sentiment": {
        "historical_rating_score",
        "historical_positive_rating_ratio",
        "historical_negative_rating_ratio",
        "historical_rating_score_change_30d",
        "historical_negative_ratio_change_30d",
        "relative_strength_21d",
        "relative_strength_63d",
        "net_upgrade_score_30d",
        "downgrade_count_30d",
        "recent_downgrade_flag_30d",
        "relevance_weighted_sentiment_7d",
        "sentiment_change_7d_vs_30d",
        "negative_news_ratio_7d",
    },
    "final_quant_model_1y_no_snapshot": {
        "historical_rating_score",
        "historical_positive_rating_ratio",
        "historical_negative_rating_ratio",
        "historical_rating_score_change_30d",
        "historical_negative_ratio_change_30d",
        "relative_strength_21d",
        "relative_strength_63d",
        "distance_to_63d_high",
        "relevance_weighted_sentiment_7d",
        "sentiment_change_7d_vs_30d",
        "negative_news_ratio_7d",
        "strong_negative_news_flag",
        "volatility_21d",
        "beta_to_spy_63d",
        "breakout_63d",
        "net_upgrade_score_30d",
        "downgrade_count_30d",
        "recent_downgrade_flag_30d",
    },
    "final_quant_model_no_snapshot": {
        "historical_rating_score",
        "historical_positive_rating_ratio",
        "historical_negative_rating_ratio",
        "historical_rating_score_change_30d",
        "historical_negative_ratio_change_30d",
        "relative_strength_21d",
        "relative_strength_63d",
        "distance_to_63d_high",
        "relevance_weighted_sentiment_7d",
        "sentiment_change_7d_vs_30d",
        "negative_news_ratio_7d",
        "strong_negative_news_flag",
        "volatility_21d",
        "beta_to_spy_63d",
        "breakout_63d",
        "net_upgrade_score_30d",
        "downgrade_count_30d",
        "recent_downgrade_flag_30d",
    },
}
SENTIMENT_COLUMNS = [
    "article_count_7d",
    "article_count_30d",
    "news_sentiment_7d",
    "news_sentiment_30d",
    "relevance_weighted_sentiment_7d",
    "sentiment_change_7d_vs_30d",
    "positive_news_ratio_7d",
    "negative_news_ratio_7d",
    "neutral_news_ratio_7d",
    "strong_negative_news_flag",
    "strong_positive_news_flag",
]
HISTORICAL_GRADE_COLUMNS = [
    "analyst_grade_event_count_90d",
    "upgrade_count_30d",
    "downgrade_count_30d",
    "net_upgrade_score_30d",
    "avg_new_grade_score_30d",
    "positive_grade_ratio_30d",
    "negative_grade_ratio_30d",
    "recent_downgrade_flag_30d",
    "historical_grade_data_available",
]
HISTORICAL_RATING_COUNT_COLUMNS = [
    "historical_rating_count_data_available",
    "historical_total_ratings",
    "historical_positive_rating_ratio",
    "historical_negative_rating_ratio",
    "historical_neutral_rating_ratio",
    "historical_buy_hold_sell_score",
    "historical_rating_score",
    "historical_rating_score_change_30d",
    "historical_rating_score_change_90d",
    "historical_positive_ratio_change_30d",
    "historical_negative_ratio_change_30d",
    "historical_negative_rating_increase_30d",
    "historical_positive_rating_increase_30d",
    "days_since_historical_rating_update",
]
SCORE_INPUT_COLUMNS = [
    "consensus_upside",
    "low_target_upside",
    "last_month_target_upside",
    "last_quarter_target_upside",
    "last_year_target_upside",
    "all_time_target_upside",
    "last_month_target_count",
    "last_quarter_target_count",
    "last_year_target_count",
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
    *HISTORICAL_RATING_COUNT_COLUMNS,
    *HISTORICAL_GRADE_COLUMNS,
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
    avoid_recent_downgrades: bool = False
    min_grade_events_90d: int = 1
    min_historical_rating_count: int = 5


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


def canonical_strategy_name(strategy_name: str) -> str:
    name = strategy_name.lower()
    if name == "analyst_only":
        return "analyst_snapshot_model"
    if name == "final_quant_model_no_snapshot":
        return "final_quant_model_1y_no_snapshot"
    if name == "final_quant_model_snapshot":
        return "final_quant_model_1y"
    return name


def strategy_display_name(strategy_name: str) -> str:
    name = canonical_strategy_name(strategy_name)
    return STRATEGY_DISPLAY_NAMES.get(name, name)


def strategy_historical_validity_group(strategy_name: str) -> str:
    name = canonical_strategy_name(strategy_name)
    if name == "spy":
        return "benchmark"
    if name in SNAPSHOT_ANALYST_STRATEGIES:
        return "snapshot_exploratory"
    return "historically_safer"


def strategy_uses_snapshot_fields(strategy_name: str) -> bool:
    name = canonical_strategy_name(strategy_name)
    return name in SNAPSHOT_ANALYST_STRATEGIES


def strategy_uses_sentiment(strategy_name: str) -> bool:
    return canonical_strategy_name(strategy_name) in SENTIMENT_STRATEGIES


def strategy_uses_historical_ratings(strategy_name: str) -> bool:
    return canonical_strategy_name(strategy_name) in HISTORICAL_RATING_COUNT_STRATEGIES


def strategy_uses_historical_grade_events(strategy_name: str) -> bool:
    return canonical_strategy_name(strategy_name) in HISTORICAL_GRADE_STRATEGIES


def strategy_score_fields(strategy_name: str) -> set[str]:
    return set(STRATEGY_SCORE_FIELDS.get(canonical_strategy_name(strategy_name), set()))


def validate_no_snapshot_strategy_fields(strategy_name: str) -> None:
    name = canonical_strategy_name(strategy_name)
    if "no_snapshot" not in name:
        return
    used_fields = strategy_score_fields(name)
    offending = sorted(used_fields & SNAPSHOT_FIELD_COLUMNS)
    if offending:
        raise ValueError(f"Strategy '{name}' references snapshot fields: {offending}")


def strategy_analyst_data_mode(strategy_name: str) -> str:
    name = canonical_strategy_name(strategy_name)
    if name in {"historical_grades_model", "strict_historical_grades_checklist"}:
        return "historical_grade_events"
    if name == "historical_grades_plus_sentiment":
        return "historical_grade_events"
    if name == "historical_rating_counts_model":
        return "historical_rating_counts"
    if name == "historical_rating_counts_plus_sentiment":
        return "historical_rating_counts_plus_sentiment"
    if name == "historical_rating_counts_plus_events":
        return "historical_rating_counts_plus_events"
    if name in {"historical_rating_counts_plus_events_sentiment", "final_quant_model_1y_no_snapshot"}:
        return "historical_rating_counts_plus_events_sentiment"
    if name in SNAPSHOT_ANALYST_STRATEGIES:
        return "snapshot_current"
    return "none"


def _requires_sentiment(strategy_name: str) -> bool:
    return canonical_strategy_name(strategy_name) in SENTIMENT_STRATEGIES


def _requires_historical_grades(strategy_name: str) -> bool:
    return canonical_strategy_name(strategy_name) in HISTORICAL_GRADE_STRATEGIES


def _requires_historical_rating_counts(strategy_name: str) -> bool:
    return canonical_strategy_name(strategy_name) in HISTORICAL_RATING_COUNT_STRATEGIES


def _validate_sentiment_inputs(df: pd.DataFrame, strategy_name: str) -> None:
    strategy_name = canonical_strategy_name(strategy_name)
    if not _requires_sentiment(strategy_name):
        return
    if df.empty:
        return
    missing_columns = [column for column in SENTIMENT_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Strategy '{strategy_name}' requires sentiment features, but these columns are missing: {missing_columns}"
        )
    if "sentiment_data_mode" in df.columns and df["sentiment_data_mode"].fillna("").eq("missing_news_sentiment").all():
        raise ValueError(
            f"Strategy '{strategy_name}' requires sentiment features, but the feature panel was built without "
            "news sentiment data. Run scripts/12_fetch_alpha_vantage_news.py, scripts/13_build_news_sentiment.py, and "
            "scripts/04_build_features.py first."
        )


def _validate_historical_grade_inputs(df: pd.DataFrame, strategy_name: str) -> None:
    strategy_name = canonical_strategy_name(strategy_name)
    if not _requires_historical_grades(strategy_name):
        return
    if df.empty:
        return
    missing_columns = [column for column in HISTORICAL_GRADE_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Strategy '{strategy_name}' requires historical analyst grade features, but these columns are missing: {missing_columns}"
        )
    if not _safe_series(df, "historical_grade_data_available", False).fillna(False).any():
        raise ValueError(
            f"Strategy '{strategy_name}' requires historical analyst grade data, but no point-in-time grade events are available. "
            "Run scripts/16_fetch_fmp_historical_grades.py and scripts/04_build_features.py first."
        )


def _validate_historical_rating_count_inputs(df: pd.DataFrame, strategy_name: str) -> None:
    strategy_name = canonical_strategy_name(strategy_name)
    if not _requires_historical_rating_counts(strategy_name):
        return
    if df.empty:
        return
    missing_columns = [column for column in HISTORICAL_RATING_COUNT_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Strategy '{strategy_name}' requires historical analyst rating-count features, but these columns are missing: {missing_columns}"
        )
    if not _safe_series(df, "historical_rating_count_data_available", False).fillna(False).any():
        raise ValueError(
            f"Strategy '{strategy_name}' requires historical analyst rating-count data, but no point-in-time grades-historical data is available. "
            "Run scripts/16_fetch_fmp_historical_grades.py and scripts/04_build_features.py first."
        )


def _snapshot_coverage_mask(df: pd.DataFrame, threshold: int) -> pd.Series:
    analyst_count = _safe_series(df, "analyst_count", -np.inf).fillna(-np.inf)
    last_year_count = _safe_series(df, "last_year_target_count", -np.inf).fillna(-np.inf)
    return (analyst_count >= threshold) | (last_year_count >= threshold)


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
    avoid_recent_downgrades: bool = False,
    min_grade_events_90d: int = 1,
    min_historical_rating_count: int = 5,
) -> StrategyParams:
    return StrategyParams(
        strategy_name=canonical_strategy_name(strategy_name),
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
        avoid_recent_downgrades=avoid_recent_downgrades,
        min_grade_events_90d=min_grade_events_90d,
        min_historical_rating_count=min_historical_rating_count,
    )


def get_filter_diagnostics(
    df: pd.DataFrame,
    params: StrategyParams,
    holding_period_days: int,
    benchmark: str,
) -> dict[str, int]:
    _validate_sentiment_inputs(df, params.strategy_name)
    _validate_historical_grade_inputs(df, params.strategy_name)
    _validate_historical_rating_count_inputs(df, params.strategy_name)
    future_return_column, _, _ = get_future_return_columns(holding_period_days)
    distance_col, breakout_col = _resistance_columns(params.resistance_window)

    starting_mask = df["ticker"].ne(benchmark) & _safe_series(df, "adjusted_close").notna() & _safe_series(df, future_return_column).notna()
    liquidity_mask = starting_mask & (_safe_series(df, "avg_dollar_volume_21d", 0).fillna(0) >= params.min_avg_dollar_volume)
    analyst_mask = liquidity_mask & (
        (
            _snapshot_coverage_mask(df, params.analyst_count_threshold)
            if params.strategy_name in {
                "final_quant_model_1y",
                "final_quant_model_1y_no_sentiment",
                "final_quant_model_1y_sentiment_risk_filter",
                "final_quant_model_1y_sector_capped",
            }
            else _safe_series(df, "analyst_count", -np.inf).fillna(-np.inf) >= params.analyst_count_threshold
        )
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
    elif params.strategy_name in HISTORICAL_GRADE_STRATEGIES:
        grade_mask = liquidity_mask & _safe_series(df, "historical_grade_data_available", False).fillna(False)
        diagnostics["passed_historical_grade_data_count"] = int(grade_mask.sum())
        grade_mask &= _safe_series(df, "analyst_grade_event_count_90d", 0).fillna(0) >= params.min_grade_events_90d
        diagnostics["passed_min_grade_events_90d_count"] = int(grade_mask.sum())
        if params.avoid_recent_downgrades or params.strategy_name == "strict_historical_grades_checklist":
            grade_mask &= ~_safe_series(df, "recent_downgrade_flag_30d", False).fillna(False)
            diagnostics["passed_avoid_recent_downgrades_count"] = int(grade_mask.sum())
        if params.strategy_name == "strict_historical_grades_checklist":
            grade_mask &= _safe_series(df, "downgrade_count_30d", np.inf).fillna(np.inf) == 0
            grade_mask &= _safe_series(df, "negative_grade_ratio_30d", np.inf).fillna(np.inf) == 0
            grade_mask &= _safe_series(df, "positive_grade_ratio_30d", -np.inf).fillna(-np.inf) >= 0.50
            grade_mask &= (
                (_safe_series(df, distance_col, np.inf).fillna(np.inf) <= params.resistance_distance_threshold)
                | _safe_series(df, breakout_col, False).fillna(False)
            )
            diagnostics["passed_strict_historical_checklist_count"] = int(grade_mask.sum())
        diagnostics["final_pass_count"] = int(grade_mask.sum())
    if params.strategy_name in HISTORICAL_RATING_COUNT_STRATEGIES:
        rating_mask = liquidity_mask & _safe_series(df, "historical_rating_count_data_available", False).fillna(False)
        diagnostics["passed_historical_rating_count_data_count"] = int(rating_mask.sum())
        rating_mask &= _safe_series(df, "historical_total_ratings", 0).fillna(0) >= params.min_historical_rating_count
        diagnostics["passed_min_historical_rating_count_count"] = int(rating_mask.sum())
        if params.strategy_name in {"historical_rating_counts_plus_events", "historical_rating_counts_plus_events_sentiment"}:
            if "historical_grade_data_available" in df.columns:
                rating_mask &= _safe_series(df, "historical_grade_data_available", False).fillna(False)
                diagnostics["historical_grade_event_data_present_count"] = int(rating_mask.sum())
        if params.strategy_name == "final_quant_model_1y_no_snapshot" and params.avoid_recent_downgrades:
            rating_mask &= ~_safe_series(df, "recent_downgrade_flag_30d", False).fillna(False)
        diagnostics["final_pass_count"] = int(rating_mask.sum())
    elif params.strategy_name == "final_quant_model_1y_sentiment_risk_filter":
        risk_mask = analyst_mask
        risk_mask &= ~_safe_series(df, "strong_negative_news_flag", False).fillna(False)
        risk_mask &= ~(
            (_safe_series(df, "negative_news_ratio_7d", 0.0).fillna(0.0) >= 0.50)
            & (_safe_series(df, "article_count_7d", 0.0).fillna(0.0) >= 3)
        )
        diagnostics["final_pass_count"] = int(risk_mask.sum())

    return diagnostics


def apply_filters(
    df: pd.DataFrame,
    params: StrategyParams,
    holding_period_days: int,
    benchmark: str,
) -> tuple[pd.DataFrame, dict[str, int]]:
    _validate_sentiment_inputs(df, params.strategy_name)
    _validate_historical_grade_inputs(df, params.strategy_name)
    _validate_historical_rating_count_inputs(df, params.strategy_name)
    future_return_column, _, _ = get_future_return_columns(holding_period_days)
    distance_col, breakout_col = _resistance_columns(params.resistance_window)

    mask = pd.Series(True, index=df.index)
    mask &= df["ticker"].ne(benchmark)
    mask &= _safe_series(df, "adjusted_close").notna()
    mask &= _safe_series(df, future_return_column).notna()
    mask &= _safe_series(df, "avg_dollar_volume_21d", 0).fillna(0) >= params.min_avg_dollar_volume

    if params.use_analyst_filters:
        if params.strategy_name in {
            "final_quant_model_1y",
            "final_quant_model_1y_no_sentiment",
            "final_quant_model_1y_sentiment_risk_filter",
            "final_quant_model_1y_sector_capped",
        }:
            mask &= _snapshot_coverage_mask(df, params.analyst_count_threshold)
        else:
            mask &= _safe_series(df, "analyst_count", -np.inf).fillna(-np.inf) >= params.analyst_count_threshold

    diagnostics = get_filter_diagnostics(df, params, holding_period_days, benchmark)

    if params.strategy_name == "technical_only":
        mask &= _safe_series(df, "relative_strength_21d", -np.inf).fillna(-np.inf) > 0
    elif params.strategy_name == "analyst_snapshot_model":
        mask &= _safe_series(df, "consensus_upside").notna() & _safe_series(df, "low_target_upside").notna()
    elif params.strategy_name in {"historical_grades_model", "historical_grades_plus_sentiment"}:
        mask &= _safe_series(df, "historical_grade_data_available", False).fillna(False)
        mask &= _safe_series(df, "analyst_grade_event_count_90d", 0).fillna(0) >= params.min_grade_events_90d
        if params.avoid_recent_downgrades:
            mask &= ~_safe_series(df, "recent_downgrade_flag_30d", False).fillna(False)
    elif params.strategy_name == "strict_historical_grades_checklist":
        mask &= _safe_series(df, "historical_grade_data_available", False).fillna(False)
        mask &= _safe_series(df, "analyst_grade_event_count_90d", 0).fillna(0) >= params.min_grade_events_90d
        mask &= _safe_series(df, "downgrade_count_30d", np.inf).fillna(np.inf) == 0
        mask &= _safe_series(df, "negative_grade_ratio_30d", np.inf).fillna(np.inf) == 0
        mask &= _safe_series(df, "positive_grade_ratio_30d", -np.inf).fillna(-np.inf) >= 0.50
        mask &= (
            (_safe_series(df, distance_col, np.inf).fillna(np.inf) <= params.resistance_distance_threshold)
            | _safe_series(df, breakout_col, False).fillna(False)
        )
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
    elif params.strategy_name in {
        "historical_rating_counts_model",
        "historical_rating_counts_plus_sentiment",
        "historical_rating_counts_plus_events",
        "historical_rating_counts_plus_events_sentiment",
        "final_quant_model_1y_no_snapshot",
    }:
        mask &= _safe_series(df, "historical_rating_count_data_available", False).fillna(False)
        mask &= _safe_series(df, "historical_total_ratings", 0).fillna(0) >= params.min_historical_rating_count
        if params.strategy_name in {
            "historical_rating_counts_plus_events",
            "historical_rating_counts_plus_events_sentiment",
        } and "historical_grade_data_available" in df.columns:
            mask &= _safe_series(df, "historical_grade_data_available", False).fillna(False)
        if params.strategy_name == "final_quant_model_1y_no_snapshot" and params.avoid_recent_downgrades:
            mask &= ~_safe_series(df, "recent_downgrade_flag_30d", False).fillna(False)
    elif params.strategy_name == "final_quant_model_1y_sentiment_risk_filter":
        mask &= ~_safe_series(df, "strong_negative_news_flag", False).fillna(False)
        mask &= ~(
            (_safe_series(df, "negative_news_ratio_7d", 0.0).fillna(0.0) >= 0.50)
            & (_safe_series(df, "article_count_7d", 0.0).fillna(0.0) >= 3)
        )

    return df.loc[mask].copy(), diagnostics


def score_rebalance_date(
    df_for_one_date: pd.DataFrame,
    strategy_name: str = "full_model",
    use_analyst_filters: bool = True,
    resistance_window: int = 30,
) -> pd.DataFrame:
    validate_score_inputs()
    _validate_sentiment_inputs(df_for_one_date, strategy_name)
    _validate_historical_grade_inputs(df_for_one_date, strategy_name)
    _validate_historical_rating_count_inputs(df_for_one_date, strategy_name)

    if df_for_one_date.empty:
        return df_for_one_date.assign(score=pd.Series(dtype=float), rank=pd.Series(dtype=float))

    strategy_name = canonical_strategy_name(strategy_name)
    validate_no_snapshot_strategy_fields(strategy_name)
    distance_col, breakout_col = _resistance_columns(resistance_window)
    df = df_for_one_date.copy()

    consensus_component = _cross_sectional_zscore(_safe_series(df, "consensus_upside"))
    low_target_component = _cross_sectional_zscore(_safe_series(df, "low_target_upside"))
    last_month_target_upside_component = _cross_sectional_zscore(_safe_series(df, "last_month_target_upside"))
    last_quarter_target_upside_component = _cross_sectional_zscore(_safe_series(df, "last_quarter_target_upside"))
    last_year_target_upside_component = _cross_sectional_zscore(_safe_series(df, "last_year_target_upside"))
    last_month_target_count_component = _cross_sectional_zscore(_safe_series(df, "last_month_target_count"))
    last_quarter_target_count_component = _cross_sectional_zscore(_safe_series(df, "last_quarter_target_count"))
    last_year_target_count_component = _cross_sectional_zscore(_safe_series(df, "last_year_target_count"))
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
    breakout_63_component = _safe_series(df, "breakout_63d", False).fillna(False).astype(float)

    news_sentiment_component = _cross_sectional_zscore(_safe_series(df, "news_sentiment_7d"))
    relevance_weighted_sentiment_component = _cross_sectional_zscore(_safe_series(df, "relevance_weighted_sentiment_7d"))
    relevance_weighted_sentiment_30_component = _cross_sectional_zscore(_safe_series(df, "relevance_weighted_sentiment_30d"))
    sentiment_change_component = _cross_sectional_zscore(_safe_series(df, "sentiment_change_7d_vs_30d"))
    positive_news_ratio_component = _cross_sectional_zscore(_safe_series(df, "positive_news_ratio_7d"))
    negative_news_ratio_component = _cross_sectional_zscore(_safe_series(df, "negative_news_ratio_7d"))
    strong_negative_news_penalty = _safe_series(df, "strong_negative_news_flag", False).fillna(False).astype(float)
    net_upgrade_30_component = _cross_sectional_zscore(_safe_series(df, "net_upgrade_score_30d"))
    avg_new_grade_score_30_component = _cross_sectional_zscore(_safe_series(df, "avg_new_grade_score_30d"))
    positive_grade_ratio_30_component = _cross_sectional_zscore(_safe_series(df, "positive_grade_ratio_30d"))
    negative_grade_ratio_30_component = _cross_sectional_zscore(_safe_series(df, "negative_grade_ratio_30d"))
    downgrade_count_30_component = _cross_sectional_zscore(_safe_series(df, "downgrade_count_30d"))
    recent_downgrade_penalty = _safe_series(df, "recent_downgrade_flag_30d", False).fillna(False).astype(float)
    historical_rating_score_component = _cross_sectional_zscore(_safe_series(df, "historical_rating_score"))
    historical_positive_ratio_component = _cross_sectional_zscore(_safe_series(df, "historical_positive_rating_ratio"))
    historical_negative_ratio_component = _cross_sectional_zscore(_safe_series(df, "historical_negative_rating_ratio"))
    historical_rating_score_change_30_component = _cross_sectional_zscore(_safe_series(df, "historical_rating_score_change_30d"))
    historical_negative_ratio_change_30_component = _cross_sectional_zscore(
        _safe_series(df, "historical_negative_ratio_change_30d")
    )

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
    elif strategy_name == "analyst_snapshot_model":
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
    elif strategy_name == "historical_grades_model":
        df["score"] = (
            0.30 * net_upgrade_30_component
            + 0.20 * avg_new_grade_score_30_component
            + 0.20 * positive_grade_ratio_30_component
            - 0.20 * negative_grade_ratio_30_component
            - 0.20 * downgrade_count_30_component
            + 0.10 * relative_strength_21_component
            + 0.10 * relative_strength_63_component
        )
    elif strategy_name == "historical_grades_plus_sentiment":
        base = (
            0.30 * net_upgrade_30_component
            + 0.20 * avg_new_grade_score_30_component
            + 0.20 * positive_grade_ratio_30_component
            - 0.20 * negative_grade_ratio_30_component
            - 0.20 * downgrade_count_30_component
            + 0.10 * relative_strength_21_component
            + 0.10 * relative_strength_63_component
        )
        df["score"] = (
            base
            + 0.15 * relevance_weighted_sentiment_component
            + 0.10 * sentiment_change_component
            - 0.10 * negative_news_ratio_component
        )
    elif strategy_name == "historical_rating_counts_model":
        df["score"] = (
            0.35 * historical_rating_score_component
            + 0.20 * historical_positive_ratio_component
            - 0.20 * historical_negative_ratio_component
            + 0.15 * historical_rating_score_change_30_component
            - 0.10 * historical_negative_ratio_change_30_component
            + 0.10 * relative_strength_21_component
            + 0.10 * relative_strength_63_component
        )
    elif strategy_name == "historical_rating_counts_plus_sentiment":
        base = (
            0.35 * historical_rating_score_component
            + 0.20 * historical_positive_ratio_component
            - 0.20 * historical_negative_ratio_component
            + 0.15 * historical_rating_score_change_30_component
            - 0.10 * historical_negative_ratio_change_30_component
            + 0.10 * relative_strength_21_component
            + 0.10 * relative_strength_63_component
        )
        df["score"] = (
            base
            + 0.15 * relevance_weighted_sentiment_component
            + 0.10 * sentiment_change_component
            - 0.10 * negative_news_ratio_component
        )
    elif strategy_name == "historical_rating_counts_plus_events":
        base = (
            0.35 * historical_rating_score_component
            + 0.20 * historical_positive_ratio_component
            - 0.20 * historical_negative_ratio_component
            + 0.15 * historical_rating_score_change_30_component
            - 0.10 * historical_negative_ratio_change_30_component
            + 0.10 * relative_strength_21_component
            + 0.10 * relative_strength_63_component
        )
        df["score"] = base + 0.10 * net_upgrade_30_component - 0.10 * downgrade_count_30_component - 0.10 * recent_downgrade_penalty
    elif strategy_name == "historical_rating_counts_plus_events_sentiment":
        base = (
            0.35 * historical_rating_score_component
            + 0.20 * historical_positive_ratio_component
            - 0.20 * historical_negative_ratio_component
            + 0.15 * historical_rating_score_change_30_component
            - 0.10 * historical_negative_ratio_change_30_component
            + 0.10 * relative_strength_21_component
            + 0.10 * relative_strength_63_component
        )
        df["score"] = (
            base
            + 0.10 * net_upgrade_30_component
            - 0.10 * downgrade_count_30_component
            - 0.10 * recent_downgrade_penalty
            + 0.15 * relevance_weighted_sentiment_component
            + 0.10 * sentiment_change_component
            - 0.10 * negative_news_ratio_component
        )
    elif strategy_name == "strict_historical_grades_checklist":
        df["score"] = (
            0.35 * net_upgrade_30_component
            + 0.25 * avg_new_grade_score_30_component
            + 0.15 * positive_grade_ratio_30_component
            + 0.15 * relative_strength_21_component
            + 0.10 * distance_component
            + 0.25 * breakout_bonus
        )
    elif strategy_name in {
        "final_quant_model_1y",
        "final_quant_model_1y_sentiment_risk_filter",
        "final_quant_model_1y_sector_capped",
    }:
        score = (
            0.20 * consensus_component
            + 0.10 * low_target_component
            + 0.10 * last_quarter_target_upside_component
            + 0.03 * last_month_target_upside_component
            + 0.02 * last_year_target_upside_component
            + 0.02 * last_month_target_count_component
            + 0.02 * last_quarter_target_count_component
            + 0.02 * last_year_target_count_component
            + 0.10 * relative_strength_21_component
            + 0.10 * relative_strength_63_component
            + 0.08 * distance_63_component
            + 0.06 * above_sma_50_component
            + 0.04 * above_sma_200_component
            + 0.08 * relevance_weighted_sentiment_component
            + 0.04 * relevance_weighted_sentiment_30_component
            + 0.06 * sentiment_change_component
            - 0.06 * negative_news_ratio_component
            - 0.06 * strong_negative_news_penalty
            - 0.05 * volatility_component
            - 0.04 * beta_component
            + 0.10 * breakout_63_component
        )
        if "historical_grade_data_available" in df.columns and _safe_series(df, "historical_grade_data_available", False).fillna(False).any():
            score = (
                score
                + 0.05 * net_upgrade_30_component
                - 0.07 * downgrade_count_30_component
                - 0.05 * recent_downgrade_penalty
            )
        df["score"] = score
    elif strategy_name == "final_quant_model_1y_no_snapshot":
        score = (
            0.18 * historical_rating_score_component
            + 0.12 * historical_positive_ratio_component
            - 0.12 * historical_negative_ratio_component
            + 0.10 * historical_rating_score_change_30_component
            - 0.08 * historical_negative_ratio_change_30_component
            + 0.12 * relative_strength_21_component
            + 0.10 * relative_strength_63_component
            + 0.08 * distance_63_component
            + 0.08 * relevance_weighted_sentiment_component
            + 0.06 * sentiment_change_component
            - 0.06 * negative_news_ratio_component
            - 0.06 * strong_negative_news_penalty
            - 0.04 * volatility_component
            - 0.04 * beta_component
            + 0.08 * breakout_63_component
        )
        if "historical_grade_data_available" in df.columns and _safe_series(df, "historical_grade_data_available", False).fillna(False).any():
            score = (
                score
                + 0.05 * net_upgrade_30_component
                - 0.05 * downgrade_count_30_component
                - 0.05 * recent_downgrade_penalty
            )
        df["score"] = score
    elif strategy_name == "final_quant_model_1y_no_sentiment":
        score = (
            0.24 * consensus_component
            + 0.12 * low_target_component
            + 0.10 * last_quarter_target_upside_component
            + 0.03 * last_month_target_upside_component
            + 0.03 * last_year_target_upside_component
            + 0.02 * last_month_target_count_component
            + 0.02 * last_quarter_target_count_component
            + 0.02 * last_year_target_count_component
            + 0.12 * relative_strength_21_component
            + 0.12 * relative_strength_63_component
            + 0.10 * distance_63_component
            + 0.06 * above_sma_50_component
            + 0.05 * above_sma_200_component
            - 0.05 * volatility_component
            - 0.04 * beta_component
            + 0.10 * breakout_63_component
        )
        if "historical_grade_data_available" in df.columns and _safe_series(df, "historical_grade_data_available", False).fillna(False).any():
            score = (
                score
                + 0.04 * net_upgrade_30_component
                - 0.05 * downgrade_count_30_component
                - 0.04 * recent_downgrade_penalty
            )
        df["score"] = score
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
