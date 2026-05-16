from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.build_historical_grade_features import HISTORICAL_GRADE_FEATURE_COLUMNS, build_historical_grade_features
from src.build_historical_rating_count_features import (
    HISTORICAL_RATING_COUNT_FEATURE_COLUMNS,
    build_historical_rating_count_features,
)
from src.utils import LOGGER, load_dataframe, save_dataframe


SENTIMENT_DAILY_COLUMNS = [
    "article_count_1d",
    "avg_sentiment_1d",
    "median_sentiment_1d",
    "positive_news_count_1d",
    "negative_news_count_1d",
    "neutral_news_count_1d",
    "positive_news_ratio_1d",
    "negative_news_ratio_1d",
    "neutral_news_ratio_1d",
    "max_negative_prob_1d",
    "max_positive_prob_1d",
    "relevance_weighted_sentiment_1d",
]


def _load_optional_dataframe(path: str | Path, parse_dates: list[str]) -> pd.DataFrame | None:
    file_path = Path(path)
    if not file_path.exists():
        if file_path.name == "news_sentiment_daily.csv":
            LOGGER.warning("No sentiment file found; building features without sentiment.")
        else:
            LOGGER.warning("Optional dataset not found: %s", file_path)
        return None
    return load_dataframe(file_path, parse_dates=parse_dates)


def _prepare_prices(prices: pd.DataFrame) -> pd.DataFrame:
    df = prices.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    return df


def _merge_analyst_snapshot(
    features: pd.DataFrame,
    analyst_df: pd.DataFrame | None,
    use_current_snapshot: bool,
) -> pd.DataFrame:
    analyst_columns = [
        "consensus_target",
        "low_target",
        "high_target",
        "median_target",
        "analyst_count",
        "consensus_upside",
        "low_target_upside",
        "high_target_upside",
        "last_month_target_count",
        "last_month_avg_price_target",
        "last_quarter_target_count",
        "last_quarter_avg_price_target",
        "last_year_target_count",
        "last_year_avg_price_target",
        "all_time_target_count",
        "all_time_avg_price_target",
        "analyst_publishers",
        "last_month_target_upside",
        "last_quarter_target_upside",
        "last_year_target_upside",
        "all_time_target_upside",
        "target_spread",
        "target_revision_7d",
        "target_revision_30d",
    ]
    if analyst_df is None or analyst_df.empty:
        for column in analyst_columns:
            features[column] = np.nan
        return features

    analyst_df = analyst_df.copy()
    analyst_df["date"] = pd.to_datetime(analyst_df["date"])
    if use_current_snapshot:
        snapshot = analyst_df.sort_values("date").groupby("ticker").tail(1)
        merged = features.merge(snapshot[["ticker", *analyst_columns]], on="ticker", how="left")
        merged["analyst_data_mode"] = "snapshot_current"
        return merged

    merged = features.merge(analyst_df[["date", "ticker", *analyst_columns]], on=["date", "ticker"], how="left")
    merged["analyst_data_mode"] = "none"
    return merged


def _merge_sentiment(features: pd.DataFrame, sentiment_df: pd.DataFrame | None) -> pd.DataFrame:
    sentiment_mode = "daily_fmp_news" if sentiment_df is not None and not sentiment_df.empty else "missing_news_sentiment"

    if sentiment_df is None or sentiment_df.empty:
        for column in SENTIMENT_DAILY_COLUMNS:
            features[column] = 0.0
        features["article_count_1d"] = features["article_count_1d"].astype(int)
        features["positive_news_count_1d"] = features["positive_news_count_1d"].astype(int)
        features["negative_news_count_1d"] = features["negative_news_count_1d"].astype(int)
        features["neutral_news_count_1d"] = features["neutral_news_count_1d"].astype(int)
        features["article_count_7d"] = 0.0
        features["article_count_30d"] = 0.0
        features["news_sentiment_7d"] = 0.0
        features["news_sentiment_30d"] = 0.0
        features["relevance_weighted_sentiment_7d"] = 0.0
        features["relevance_weighted_sentiment_30d"] = 0.0
        features["sentiment_change_7d_vs_30d"] = 0.0
        features["positive_news_ratio_7d"] = 0.0
        features["negative_news_ratio_7d"] = 0.0
        features["neutral_news_ratio_7d"] = 0.0
        features["strong_negative_news_flag"] = False
        features["strong_positive_news_flag"] = False
        features["sentiment_data_mode"] = sentiment_mode
        return features

    sentiment_df = sentiment_df.copy()
    sentiment_df["date"] = pd.to_datetime(sentiment_df["date"])
    sentiment_df = sentiment_df.sort_values(["ticker", "date"]).reset_index(drop=True)

    min_date = features["date"].min()
    max_date = features["date"].max()
    tickers = sorted(features["ticker"].dropna().unique().tolist())
    calendar_index = pd.MultiIndex.from_product(
        [tickers, pd.date_range(min_date, max_date, freq="D")],
        names=["ticker", "date"],
    )
    sentiment_calendar = pd.DataFrame(index=calendar_index).reset_index()
    sentiment_calendar = sentiment_calendar.merge(sentiment_df, on=["ticker", "date"], how="left")

    for column in SENTIMENT_DAILY_COLUMNS:
        if column not in sentiment_calendar.columns:
            sentiment_calendar[column] = 0.0

    fill_zero_columns = [column for column in SENTIMENT_DAILY_COLUMNS if column in sentiment_calendar.columns]
    for column in fill_zero_columns:
        sentiment_calendar[column] = pd.to_numeric(sentiment_calendar[column], errors="coerce").fillna(0.0)

    for column in ["article_count_1d", "positive_news_count_1d", "negative_news_count_1d", "neutral_news_count_1d"]:
        sentiment_calendar[column] = sentiment_calendar[column].astype(int)

    grouped = sentiment_calendar.groupby("ticker", group_keys=False)
    sentiment_calendar["article_count_7d"] = grouped["article_count_1d"].transform(lambda s: s.rolling(7, min_periods=1).sum())
    sentiment_calendar["article_count_30d"] = grouped["article_count_1d"].transform(lambda s: s.rolling(30, min_periods=1).sum())
    sentiment_calendar["news_sentiment_7d"] = grouped["avg_sentiment_1d"].transform(lambda s: s.rolling(7, min_periods=1).mean())
    sentiment_calendar["news_sentiment_30d"] = grouped["avg_sentiment_1d"].transform(lambda s: s.rolling(30, min_periods=1).mean())
    sentiment_calendar["relevance_weighted_sentiment_7d"] = grouped["relevance_weighted_sentiment_1d"].transform(
        lambda s: s.rolling(7, min_periods=1).mean()
    )
    sentiment_calendar["relevance_weighted_sentiment_30d"] = grouped["relevance_weighted_sentiment_1d"].transform(
        lambda s: s.rolling(30, min_periods=1).mean()
    )
    sentiment_calendar["sentiment_change_7d_vs_30d"] = (
        sentiment_calendar["news_sentiment_7d"] - sentiment_calendar["news_sentiment_30d"]
    )

    positive_7d = grouped["positive_news_count_1d"].transform(lambda s: s.rolling(7, min_periods=1).sum())
    negative_7d = grouped["negative_news_count_1d"].transform(lambda s: s.rolling(7, min_periods=1).sum())
    neutral_7d = grouped["neutral_news_count_1d"].transform(lambda s: s.rolling(7, min_periods=1).sum())
    denom_7d = sentiment_calendar["article_count_7d"].replace(0, np.nan)
    sentiment_calendar["positive_news_ratio_7d"] = (positive_7d / denom_7d).fillna(0.0)
    sentiment_calendar["negative_news_ratio_7d"] = (negative_7d / denom_7d).fillna(0.0)
    sentiment_calendar["neutral_news_ratio_7d"] = (neutral_7d / denom_7d).fillna(0.0)
    sentiment_calendar["strong_negative_news_flag"] = (
        (sentiment_calendar["negative_news_ratio_7d"] >= 0.50) & (sentiment_calendar["article_count_7d"] >= 3)
    )
    sentiment_calendar["strong_positive_news_flag"] = (
        (sentiment_calendar["positive_news_ratio_7d"] >= 0.50) & (sentiment_calendar["article_count_7d"] >= 3)
    )
    sentiment_calendar["sentiment_data_mode"] = sentiment_mode

    merged = features.merge(
        sentiment_calendar[
            [
                "date",
                "ticker",
                *SENTIMENT_DAILY_COLUMNS,
                "article_count_7d",
                "article_count_30d",
                "news_sentiment_7d",
                "news_sentiment_30d",
                "relevance_weighted_sentiment_7d",
                "relevance_weighted_sentiment_30d",
                "sentiment_change_7d_vs_30d",
                "positive_news_ratio_7d",
                "negative_news_ratio_7d",
                "neutral_news_ratio_7d",
                "strong_negative_news_flag",
                "strong_positive_news_flag",
                "sentiment_data_mode",
            ]
        ],
        on=["date", "ticker"],
        how="left",
    )

    for column in SENTIMENT_DAILY_COLUMNS + [
        "article_count_7d",
        "article_count_30d",
        "news_sentiment_7d",
        "news_sentiment_30d",
        "relevance_weighted_sentiment_7d",
        "relevance_weighted_sentiment_30d",
        "sentiment_change_7d_vs_30d",
        "positive_news_ratio_7d",
        "negative_news_ratio_7d",
        "neutral_news_ratio_7d",
    ]:
        merged[column] = pd.to_numeric(merged[column], errors="coerce").fillna(0.0)

    for column in ["article_count_1d", "positive_news_count_1d", "negative_news_count_1d", "neutral_news_count_1d"]:
        merged[column] = merged[column].astype(int)

    merged["strong_negative_news_flag"] = merged["strong_negative_news_flag"].fillna(False).astype(bool)
    merged["strong_positive_news_flag"] = merged["strong_positive_news_flag"].fillna(False).astype(bool)
    merged["sentiment_data_mode"] = merged["sentiment_data_mode"].fillna(sentiment_mode)
    return merged


def _merge_historical_rating_count_features(
    features: pd.DataFrame,
    historical_rating_counts_path: str | Path | None,
    processed_output_path: str | Path | None,
) -> pd.DataFrame:
    if not historical_rating_counts_path or not Path(historical_rating_counts_path).exists():
        LOGGER.warning("No historical analyst rating counts found; skipping rating-count features.")
        for column in HISTORICAL_RATING_COUNT_FEATURE_COLUMNS:
            if column in {"historical_rating_count_data_available", "historical_negative_rating_increase_30d", "historical_positive_rating_increase_30d"}:
                features[column] = False
            elif column == "historical_rating_record_date":
                features[column] = pd.NaT
            elif column == "days_since_historical_rating_update":
                features[column] = np.nan
            elif column == "historical_rating_score":
                features[column] = 3.0
            else:
                features[column] = 0.0
        return features

    historical_features = build_historical_rating_count_features(
        historical_rating_counts_path,
        features[["date", "ticker"]].copy(),
        processed_output_path,
    )
    return features.merge(historical_features, on=["date", "ticker"], how="left")


def _merge_historical_grade_features(
    features: pd.DataFrame,
    historical_grade_events_path: str | Path | None,
    processed_output_path: str | Path | None,
) -> pd.DataFrame:
    if not historical_grade_events_path or not Path(historical_grade_events_path).exists():
        LOGGER.warning("No historical analyst grade events found; skipping grade-event features.")
        for column in HISTORICAL_GRADE_FEATURE_COLUMNS:
            if column.startswith("recent_") or column == "historical_grade_data_available":
                features[column] = False
            elif column.startswith("avg_new_grade_score"):
                features[column] = 3.0
            elif column.startswith("days_since_"):
                features[column] = np.nan
            else:
                features[column] = 0.0
        return features

    historical_features = build_historical_grade_features(
        historical_grade_events_path,
        features[["date", "ticker"]].copy(),
        processed_output_path,
    )
    return features.merge(historical_features, on=["date", "ticker"], how="left")


def build_feature_panel(
    prices_path: str | Path,
    universe_path: str | Path,
    output_path: str | Path,
    analyst_path: str | Path | None = None,
    sentiment_path: str | Path | None = None,
    historical_rating_counts_path: str | Path | None = None,
    historical_grade_events_path: str | Path | None = None,
    historical_rating_count_features_output_path: str | Path | None = None,
    historical_grade_features_output_path: str | Path | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    benchmark_ticker: str = "SPY",
    use_current_snapshot_analyst: bool = False,
) -> pd.DataFrame:
    prices = _prepare_prices(load_dataframe(prices_path, parse_dates=["date"]))
    universe = pd.read_csv(universe_path)
    analyst_df = _load_optional_dataframe(analyst_path, parse_dates=["date"]) if analyst_path else None
    sentiment_df = _load_optional_dataframe(sentiment_path, parse_dates=["date"]) if sentiment_path else None

    price_features = prices.copy()
    price_features["daily_return"] = price_features.groupby("ticker")["adjusted_close"].pct_change()
    price_features["return_5d"] = price_features.groupby("ticker")["adjusted_close"].pct_change(5)
    price_features["return_21d"] = price_features.groupby("ticker")["adjusted_close"].pct_change(21)
    price_features["return_63d"] = price_features.groupby("ticker")["adjusted_close"].pct_change(63)
    price_features["volatility_21d"] = price_features.groupby("ticker")["daily_return"].transform(
        lambda s: s.rolling(21).std()
    )
    price_features["volatility_63d"] = price_features.groupby("ticker")["daily_return"].transform(
        lambda s: s.rolling(63).std()
    )
    price_features["high_30d_prev"] = price_features.groupby("ticker")["high"].transform(lambda s: s.shift(1).rolling(30).max())
    price_features["high_63d_prev"] = price_features.groupby("ticker")["high"].transform(lambda s: s.shift(1).rolling(63).max())
    price_features["high_126d_prev"] = price_features.groupby("ticker")["high"].transform(lambda s: s.shift(1).rolling(126).max())
    price_features["high_252d_prev"] = price_features.groupby("ticker")["high"].transform(lambda s: s.shift(1).rolling(252).max())
    price_features["distance_to_30d_high"] = (price_features["high_30d_prev"] - price_features["close"]) / price_features["close"]
    price_features["distance_to_63d_high"] = (price_features["high_63d_prev"] - price_features["close"]) / price_features["close"]
    price_features["distance_to_126d_high"] = (price_features["high_126d_prev"] - price_features["close"]) / price_features["close"]
    price_features["distance_to_252d_high"] = (price_features["high_252d_prev"] - price_features["close"]) / price_features["close"]
    price_features["breakout_30d"] = price_features["close"] > price_features["high_30d_prev"]
    price_features["breakout_63d"] = price_features["close"] > price_features["high_63d_prev"]
    price_features["breakout_126d"] = price_features["close"] > price_features["high_126d_prev"]
    price_features["breakout_252d"] = price_features["close"] > price_features["high_252d_prev"]
    price_features["volume_avg_21d"] = price_features.groupby("ticker")["volume"].transform(lambda s: s.shift(1).rolling(21).mean())
    price_features["volume_spike_ratio"] = price_features["volume"] / price_features["volume_avg_21d"]
    price_features["dollar_volume"] = price_features["adjusted_close"] * price_features["volume"]
    price_features["avg_dollar_volume_21d"] = price_features.groupby("ticker")["dollar_volume"].transform(
        lambda s: s.shift(1).rolling(21).mean()
    )
    price_features["sma_20"] = price_features.groupby("ticker")["adjusted_close"].transform(lambda s: s.rolling(20).mean())
    price_features["sma_50"] = price_features.groupby("ticker")["adjusted_close"].transform(lambda s: s.rolling(50).mean())
    price_features["sma_200"] = price_features.groupby("ticker")["adjusted_close"].transform(lambda s: s.rolling(200).mean())
    price_features["above_sma_20"] = (price_features["adjusted_close"] > price_features["sma_20"]).astype(float)
    price_features["above_sma_50"] = (price_features["adjusted_close"] > price_features["sma_50"]).astype(float)
    price_features["above_sma_200"] = (price_features["adjusted_close"] > price_features["sma_200"]).astype(float)
    price_features["close_to_sma_20"] = (price_features["adjusted_close"] - price_features["sma_20"]) / price_features["sma_20"]
    price_features["close_to_sma_50"] = (price_features["adjusted_close"] - price_features["sma_50"]) / price_features["sma_50"]
    price_features["close_to_sma_200"] = (price_features["adjusted_close"] - price_features["sma_200"]) / price_features["sma_200"]

    prev_close = price_features.groupby("ticker")["adjusted_close"].shift(1)
    true_range = pd.concat(
        [
            price_features["high"] - price_features["low"],
            (price_features["high"] - prev_close).abs(),
            (price_features["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    price_features["atr_14"] = true_range.groupby(price_features["ticker"]).transform(lambda s: s.rolling(14).mean())

    delta = price_features.groupby("ticker")["adjusted_close"].diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.groupby(price_features["ticker"]).transform(lambda s: s.rolling(14).mean())
    avg_loss = losses.groupby(price_features["ticker"]).transform(lambda s: s.rolling(14).mean())
    rs = avg_gain / avg_loss.replace(0, np.nan)
    price_features["rsi_14"] = pd.to_numeric(100 - (100 / (1 + rs)), errors="coerce").fillna(50.0)

    benchmark = (
        price_features.loc[
            price_features["ticker"] == benchmark_ticker,
            ["date", "daily_return", "return_21d", "return_63d", "adjusted_close"],
        ]
        .rename(
            columns={
                "daily_return": "spy_daily_return",
                "return_21d": "spy_return_21d",
                "return_63d": "spy_return_63d",
                "adjusted_close": "spy_adjusted_close",
            }
        )
        .sort_values("date")
    )
    benchmark["spy_close"] = benchmark["spy_adjusted_close"]
    benchmark["spy_sma_50"] = benchmark["spy_adjusted_close"].rolling(50).mean()
    benchmark["spy_sma_200"] = benchmark["spy_adjusted_close"].rolling(200).mean()
    benchmark["spy_above_sma_200"] = benchmark["spy_adjusted_close"] > benchmark["spy_sma_200"]
    benchmark["future_5d_spy_return"] = benchmark["spy_adjusted_close"].shift(-5) / benchmark["spy_adjusted_close"] - 1
    benchmark["future_21d_spy_return"] = benchmark["spy_adjusted_close"].shift(-21) / benchmark["spy_adjusted_close"] - 1
    benchmark["future_63d_spy_return"] = benchmark["spy_adjusted_close"].shift(-63) / benchmark["spy_adjusted_close"] - 1

    features = price_features.merge(benchmark, on="date", how="left")
    features["relative_strength_21d"] = features["return_21d"] - features["spy_return_21d"]
    features["relative_strength_63d"] = features["return_63d"] - features["spy_return_63d"]
    features["future_5d_return"] = features.groupby("ticker")["adjusted_close"].shift(-5) / features["adjusted_close"] - 1
    features["future_21d_return"] = features.groupby("ticker")["adjusted_close"].shift(-21) / features["adjusted_close"] - 1
    features["future_63d_return"] = features.groupby("ticker")["adjusted_close"].shift(-63) / features["adjusted_close"] - 1
    features["future_5d_excess_return"] = features["future_5d_return"] - features["future_5d_spy_return"]
    features["future_21d_excess_return"] = features["future_21d_return"] - features["future_21d_spy_return"]
    features["future_63d_excess_return"] = features["future_63d_return"] - features["future_63d_spy_return"]
    features["stock_drawdown_63d"] = (
        features["adjusted_close"] / features.groupby("ticker")["adjusted_close"].transform(lambda s: s.rolling(63).max()) - 1
    )
    features["beta_to_spy_63d"] = (
        features.groupby("ticker")["daily_return"].transform(
            lambda s: s.rolling(63).cov(features.loc[s.index, "spy_daily_return"])
        )
        / features["spy_daily_return"].rolling(63).var()
    )

    features = _merge_sentiment(features, sentiment_df)
    features = _merge_analyst_snapshot(features, analyst_df, use_current_snapshot_analyst)
    features = _merge_historical_rating_count_features(
        features,
        historical_rating_counts_path,
        historical_rating_count_features_output_path,
    )
    features = _merge_historical_grade_features(features, historical_grade_events_path, historical_grade_features_output_path)
    features = features.merge(universe[["ticker", "sector"]], on="ticker", how="left")

    final_columns = [
        "date",
        "ticker",
        "sector",
        "close",
        "adjusted_close",
        "volume",
        "dollar_volume",
        "avg_dollar_volume_21d",
        "return_5d",
        "return_21d",
        "return_63d",
        "relative_strength_21d",
        "relative_strength_63d",
        "volatility_21d",
        "volatility_63d",
        "high_30d_prev",
        "high_63d_prev",
        "high_126d_prev",
        "high_252d_prev",
        "distance_to_30d_high",
        "distance_to_63d_high",
        "distance_to_126d_high",
        "distance_to_252d_high",
        "breakout_30d",
        "breakout_63d",
        "breakout_126d",
        "breakout_252d",
        "volume_avg_21d",
        "volume_spike_ratio",
        "sma_20",
        "sma_50",
        "sma_200",
        "above_sma_20",
        "above_sma_50",
        "above_sma_200",
        "close_to_sma_20",
        "close_to_sma_50",
        "close_to_sma_200",
        "rsi_14",
        "atr_14",
        "beta_to_spy_63d",
        "stock_drawdown_63d",
        *SENTIMENT_DAILY_COLUMNS,
        "article_count_7d",
        "article_count_30d",
        "news_sentiment_7d",
        "news_sentiment_30d",
        "relevance_weighted_sentiment_7d",
        "relevance_weighted_sentiment_30d",
        "sentiment_change_7d_vs_30d",
        "positive_news_ratio_7d",
        "negative_news_ratio_7d",
        "neutral_news_ratio_7d",
        "strong_negative_news_flag",
        "strong_positive_news_flag",
        "sentiment_data_mode",
        "consensus_target",
        "low_target",
        "high_target",
        "median_target",
        "analyst_count",
        "consensus_upside",
        "low_target_upside",
        "high_target_upside",
        "last_month_target_count",
        "last_month_avg_price_target",
        "last_quarter_target_count",
        "last_quarter_avg_price_target",
        "last_year_target_count",
        "last_year_avg_price_target",
        "all_time_target_count",
        "all_time_avg_price_target",
        "analyst_publishers",
        "last_month_target_upside",
        "last_quarter_target_upside",
        "last_year_target_upside",
        "all_time_target_upside",
        "target_spread",
        "target_revision_7d",
        "target_revision_30d",
        *HISTORICAL_RATING_COUNT_FEATURE_COLUMNS,
        *HISTORICAL_GRADE_FEATURE_COLUMNS,
        "spy_close",
        "spy_sma_50",
        "spy_sma_200",
        "spy_above_sma_200",
        "future_5d_return",
        "future_21d_return",
        "future_63d_return",
        "future_5d_spy_return",
        "future_21d_spy_return",
        "future_63d_spy_return",
        "future_5d_excess_return",
        "future_21d_excess_return",
        "future_63d_excess_return",
        "analyst_data_mode",
    ]
    features = features[final_columns].sort_values(["date", "ticker"]).reset_index(drop=True)
    if start_date is not None:
        features = features.loc[features["date"] >= pd.Timestamp(start_date)].copy()
    if end_date is not None:
        features = features.loc[features["date"] < pd.Timestamp(end_date)].copy()
    save_dataframe(output_path, features)
    return features
