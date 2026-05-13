from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(slots=True)
class StrategyParams:
    name: str
    analyst_count_threshold: int = 20
    use_analyst_filters: bool = True
    use_sentiment_filters: bool = True
    use_technical_filters: bool = True


def _cross_sectional_zscore(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - series.mean()) / std


def apply_filters(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Apply configurable hard filters for the chosen strategy variant."""
    mask = pd.Series(True, index=df.index)

    if params.get("use_analyst_filters", False):
        mask &= df["analyst_count"].fillna(-np.inf) >= params.get("analyst_count_threshold", 20)
        mask &= df["consensus_upside"].fillna(-np.inf) >= 0.04
        mask &= df["low_target_upside"].fillna(-np.inf) >= 0

    if params.get("use_sentiment_filters", False):
        mask &= df["news_sentiment_7d"].fillna(0) > 0
        mask &= df["news_article_count_7d"].fillna(0) >= 1

    if params.get("use_technical_filters", False):
        mask &= (
            (df["distance_to_30d_high"].fillna(np.inf) <= 0.02)
            | (df["breakout_30d"].fillna(False))
        )
        mask &= df["volume_spike_ratio"].fillna(0) >= 0.8

    if params.get("require_positive_relative_strength", False):
        mask &= df["relative_strength_21d"].fillna(-np.inf) > 0

    return df.loc[mask].copy()


def score_rebalance_date(df_for_one_date: pd.DataFrame, strategy_name: str = "full_model") -> pd.DataFrame:
    """Calculate cross-sectional scores on a single rebalance date."""
    if df_for_one_date.empty:
        return df_for_one_date.assign(score=pd.Series(dtype=float))

    df = df_for_one_date.copy()

    if strategy_name == "technical_only":
        df["score"] = (
            0.50 * _cross_sectional_zscore(df["relative_strength_21d"].fillna(0))
            + 0.30 * _cross_sectional_zscore((-df["distance_to_30d_high"]).fillna(0))
            + 0.20 * _cross_sectional_zscore(df["volume_spike_ratio"].fillna(0))
        )
        return df

    if strategy_name == "sentiment_only":
        df["score"] = (
            0.70 * _cross_sectional_zscore(df["news_sentiment_7d"].fillna(0))
            + 0.30 * _cross_sectional_zscore(df["news_sentiment_change"].fillna(0))
        )
        return df

    if strategy_name == "analyst_only":
        df["score"] = (
            0.70 * _cross_sectional_zscore(df["consensus_upside"].fillna(0))
            + 0.30 * _cross_sectional_zscore(df["low_target_upside"].fillna(0))
        )
        return df

    if strategy_name == "technical_plus_sentiment":
        df["score"] = (
            0.35 * _cross_sectional_zscore(df["news_sentiment_7d"].fillna(0))
            + 0.15 * _cross_sectional_zscore(df["news_sentiment_change"].fillna(0))
            + 0.25 * _cross_sectional_zscore(df["relative_strength_21d"].fillna(0))
            + 0.15 * _cross_sectional_zscore((-df["distance_to_30d_high"]).fillna(0))
            + 0.10 * _cross_sectional_zscore(df["volume_spike_ratio"].fillna(0))
        )
        df.loc[df["breakout_30d"].fillna(False), "score"] += 0.25
        return df

    df["score"] = (
        0.30 * _cross_sectional_zscore(df["consensus_upside"].fillna(0))
        + 0.20 * _cross_sectional_zscore(df["news_sentiment_7d"].fillna(0))
        + 0.15 * _cross_sectional_zscore(df["news_sentiment_change"].fillna(0))
        + 0.15 * _cross_sectional_zscore(df["relative_strength_21d"].fillna(0))
        + 0.10 * _cross_sectional_zscore((-df["distance_to_30d_high"]).fillna(0))
        + 0.10 * _cross_sectional_zscore(df["volume_spike_ratio"].fillna(0))
    )
    df.loc[df["breakout_30d"].fillna(False), "score"] += 0.25
    return df


def get_strategy_filter_params(
    strategy_name: str,
    analyst_count_threshold: int = 20,
    use_analyst_filters: bool = True,
    include_sentiment: bool = True,
) -> dict:
    """Map strategy names to filter settings."""
    strategy_name = strategy_name.lower()
    if strategy_name == "technical_only":
        return {
            "use_analyst_filters": False,
            "use_sentiment_filters": False,
            "use_technical_filters": True,
            "require_positive_relative_strength": True,
            "analyst_count_threshold": analyst_count_threshold,
        }
    if strategy_name == "sentiment_only":
        if not include_sentiment:
            return {
                "use_analyst_filters": False,
                "use_sentiment_filters": False,
                "use_technical_filters": False,
                "require_positive_relative_strength": False,
                "analyst_count_threshold": analyst_count_threshold,
            }
        return {
            "use_analyst_filters": False,
            "use_sentiment_filters": True,
            "use_technical_filters": False,
            "require_positive_relative_strength": False,
            "analyst_count_threshold": analyst_count_threshold,
        }
    if strategy_name == "analyst_only":
        return {
            "use_analyst_filters": use_analyst_filters,
            "use_sentiment_filters": False,
            "use_technical_filters": False,
            "require_positive_relative_strength": False,
            "analyst_count_threshold": analyst_count_threshold,
        }
    if strategy_name == "technical_plus_sentiment":
        if not include_sentiment:
            return {
                "use_analyst_filters": False,
                "use_sentiment_filters": False,
                "use_technical_filters": True,
                "require_positive_relative_strength": False,
                "analyst_count_threshold": analyst_count_threshold,
            }
        return {
            "use_analyst_filters": False,
            "use_sentiment_filters": True,
            "use_technical_filters": True,
            "require_positive_relative_strength": False,
            "analyst_count_threshold": analyst_count_threshold,
        }
    return {
        "use_analyst_filters": use_analyst_filters,
        "use_sentiment_filters": include_sentiment,
        "use_technical_filters": True,
        "require_positive_relative_strength": False,
        "analyst_count_threshold": analyst_count_threshold,
    }
