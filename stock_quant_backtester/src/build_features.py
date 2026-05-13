from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import LOGGER, load_dataframe, save_dataframe


def _load_optional_dataframe(path: str | Path, parse_dates: list[str]) -> pd.DataFrame | None:
    file_path = Path(path)
    if not file_path.exists():
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
        merged["analyst_data_mode"] = "research_current_snapshot"
        return merged

    merged = features.merge(analyst_df[["date", "ticker", *analyst_columns]], on=["date", "ticker"], how="left")
    merged["analyst_data_mode"] = "historical_backtest_without_analyst"
    return merged


def _merge_sentiment(features: pd.DataFrame, sentiment_df: pd.DataFrame | None) -> pd.DataFrame:
    sentiment_columns = [
        "article_count",
        "avg_sentiment_score",
        "weighted_sentiment_score",
        "positive_article_ratio",
        "negative_article_ratio",
        "neutral_article_ratio",
        "avg_relevance_score",
    ]
    if sentiment_df is None or sentiment_df.empty:
        for column in sentiment_columns:
            features[column] = 0.0
        return features

    sentiment_df = sentiment_df.copy()
    sentiment_df["date"] = pd.to_datetime(sentiment_df["date"])
    merged = features.merge(
        sentiment_df[["date", "ticker", *sentiment_columns]],
        on=["date", "ticker"],
        how="left",
    )
    for column in sentiment_columns:
        merged[column] = merged[column].fillna(0.0)
    return merged


def build_feature_panel(
    prices_path: str | Path,
    universe_path: str | Path,
    output_path: str | Path,
    analyst_path: str | Path | None = None,
    sentiment_path: str | Path | None = None,
    benchmark_ticker: str = "SPY",
    use_current_snapshot_analyst: bool = False,
) -> pd.DataFrame:
    """Build a point-in-time feature panel for all tickers and dates."""
    prices = _prepare_prices(load_dataframe(prices_path, parse_dates=["date"]))
    universe = pd.read_csv(universe_path)
    analyst_df = _load_optional_dataframe(analyst_path, parse_dates=["date"]) if analyst_path else None
    sentiment_df = _load_optional_dataframe(sentiment_path, parse_dates=["date"]) if sentiment_path else None

    price_features = prices.copy()
    price_features["daily_return"] = price_features.groupby("ticker")["adjusted_close"].pct_change()
    price_features["return_5d"] = price_features.groupby("ticker")["adjusted_close"].pct_change(5)
    price_features["return_21d"] = price_features.groupby("ticker")["adjusted_close"].pct_change(21)
    price_features["volatility_21d"] = (
        price_features.groupby("ticker")["daily_return"].transform(lambda s: s.rolling(21).std())
    )
    price_features["high_30d_prev"] = (
        price_features.groupby("ticker")["high"].transform(lambda s: s.shift(1).rolling(30).max())
    )
    price_features["distance_to_30d_high"] = (
        (price_features["high_30d_prev"] - price_features["close"]) / price_features["close"]
    )
    price_features["breakout_30d"] = price_features["close"] > price_features["high_30d_prev"]
    price_features["volume_avg_21d"] = (
        price_features.groupby("ticker")["volume"].transform(lambda s: s.shift(1).rolling(21).mean())
    )
    price_features["volume_spike_ratio"] = price_features["volume"] / price_features["volume_avg_21d"]

    benchmark = (
        price_features.loc[price_features["ticker"] == benchmark_ticker, ["date", "return_21d", "adjusted_close"]]
        .rename(columns={"return_21d": "spy_return_21d", "adjusted_close": "spy_adjusted_close"})
        .sort_values("date")
    )
    benchmark["future_5d_spy_return"] = benchmark["spy_adjusted_close"].shift(-5) / benchmark["spy_adjusted_close"] - 1
    benchmark["future_21d_spy_return"] = (
        benchmark["spy_adjusted_close"].shift(-21) / benchmark["spy_adjusted_close"] - 1
    )

    features = price_features.merge(benchmark, on="date", how="left")
    features["relative_strength_21d"] = features["return_21d"] - features["spy_return_21d"]
    features["future_5d_return"] = (
        features.groupby("ticker")["adjusted_close"].shift(-5) / features["adjusted_close"] - 1
    )
    features["future_21d_return"] = (
        features.groupby("ticker")["adjusted_close"].shift(-21) / features["adjusted_close"] - 1
    )
    features["future_5d_excess_return"] = features["future_5d_return"] - features["future_5d_spy_return"]
    features["future_21d_excess_return"] = features["future_21d_return"] - features["future_21d_spy_return"]

    features = _merge_sentiment(features, sentiment_df)
    features["news_sentiment_7d"] = (
        features.groupby("ticker")["weighted_sentiment_score"].transform(lambda s: s.rolling(7).mean())
    )
    features["news_sentiment_30d"] = (
        features.groupby("ticker")["weighted_sentiment_score"].transform(lambda s: s.rolling(30).mean())
    )
    features["news_sentiment_change"] = features["news_sentiment_7d"] - features["news_sentiment_30d"]
    features["news_article_count_7d"] = (
        features.groupby("ticker")["article_count"].transform(lambda s: s.rolling(7).sum())
    )
    features["positive_news_ratio_7d"] = (
        features.groupby("ticker")["positive_article_ratio"].transform(lambda s: s.rolling(7).mean())
    )
    features["negative_news_ratio_7d"] = (
        features.groupby("ticker")["negative_article_ratio"].transform(lambda s: s.rolling(7).mean())
    )

    features = _merge_analyst_snapshot(features, analyst_df, use_current_snapshot_analyst)
    features = features.merge(universe[["ticker", "sector"]], on="ticker", how="left")

    final_columns = [
        "date",
        "ticker",
        "sector",
        "close",
        "adjusted_close",
        "volume",
        "return_5d",
        "return_21d",
        "relative_strength_21d",
        "volatility_21d",
        "high_30d_prev",
        "distance_to_30d_high",
        "breakout_30d",
        "volume_avg_21d",
        "volume_spike_ratio",
        "news_sentiment_7d",
        "news_sentiment_30d",
        "news_sentiment_change",
        "news_article_count_7d",
        "positive_news_ratio_7d",
        "negative_news_ratio_7d",
        "consensus_target",
        "low_target",
        "high_target",
        "median_target",
        "analyst_count",
        "consensus_upside",
        "low_target_upside",
        "target_spread",
        "target_revision_7d",
        "target_revision_30d",
        "future_5d_return",
        "future_21d_return",
        "future_5d_spy_return",
        "future_21d_spy_return",
        "future_5d_excess_return",
        "future_21d_excess_return",
    ]
    features = features[final_columns].sort_values(["date", "ticker"]).reset_index(drop=True)
    save_dataframe(output_path, features)
    return features

