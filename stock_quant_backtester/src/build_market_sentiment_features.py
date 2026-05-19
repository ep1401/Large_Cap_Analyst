from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import load_dataframe, save_dataframe


MARKET_SENTIMENT_OUTPUT_COLUMNS = [
    "date",
    "market_article_count_1d",
    "market_article_count_7d",
    "market_article_count_30d",
    "market_sentiment_1d",
    "market_sentiment_7d",
    "market_sentiment_30d",
    "market_sentiment_change_7d_vs_30d",
    "market_positive_news_ratio_7d",
    "market_negative_news_ratio_7d",
    "market_strong_negative_news_ratio_7d",
    "market_news_breadth_7d",
    "percent_tickers_positive_sentiment_7d",
    "percent_tickers_negative_sentiment_7d",
    "sentiment_dispersion_7d",
]


def build_market_sentiment_features(
    sentiment_daily_path: str | Path,
    universe_path: str | Path,
    output_path: str | Path,
) -> pd.DataFrame:
    sentiment_daily = load_dataframe(sentiment_daily_path, parse_dates=["date"])
    universe = pd.read_csv(universe_path)
    universe_tickers = sorted(universe["ticker"].dropna().astype(str).unique().tolist())

    if sentiment_daily.empty or not universe_tickers:
        empty = pd.DataFrame(columns=MARKET_SENTIMENT_OUTPUT_COLUMNS)
        save_dataframe(output_path, empty)
        return empty

    sentiment_daily = sentiment_daily.copy()
    sentiment_daily["date"] = pd.to_datetime(sentiment_daily["date"])
    min_date = pd.Timestamp(sentiment_daily["date"].min())
    max_date = pd.Timestamp(sentiment_daily["date"].max())
    calendar = pd.MultiIndex.from_product(
        [universe_tickers, pd.date_range(min_date, max_date, freq="D")],
        names=["ticker", "date"],
    )
    ticker_calendar = pd.DataFrame(index=calendar).reset_index()

    keep_columns = [
        "date",
        "ticker",
        "article_count_1d",
        "positive_news_count_1d",
        "negative_news_count_1d",
        "neutral_news_count_1d",
        "relevance_weighted_sentiment_1d",
        "strong_negative_news_flag",
    ]
    for column in keep_columns:
        if column not in sentiment_daily.columns:
            if column == "strong_negative_news_flag":
                sentiment_daily[column] = False
            else:
                sentiment_daily[column] = 0.0

    ticker_calendar = ticker_calendar.merge(sentiment_daily[keep_columns], on=["date", "ticker"], how="left")
    for column in [
        "article_count_1d",
        "positive_news_count_1d",
        "negative_news_count_1d",
        "neutral_news_count_1d",
        "relevance_weighted_sentiment_1d",
    ]:
        ticker_calendar[column] = pd.to_numeric(ticker_calendar[column], errors="coerce").fillna(0.0)
    ticker_calendar["strong_negative_news_flag"] = ticker_calendar["strong_negative_news_flag"].fillna(False).astype(bool)

    ticker_calendar["sentiment_x_articles_1d"] = (
        ticker_calendar["relevance_weighted_sentiment_1d"] * ticker_calendar["article_count_1d"]
    )

    grouped = ticker_calendar.groupby("ticker", group_keys=False)
    ticker_calendar["article_count_7d"] = grouped["article_count_1d"].transform(lambda s: s.rolling(7, min_periods=1).sum())
    ticker_calendar["article_count_30d"] = grouped["article_count_1d"].transform(lambda s: s.rolling(30, min_periods=1).sum())
    ticker_calendar["positive_news_count_7d"] = grouped["positive_news_count_1d"].transform(lambda s: s.rolling(7, min_periods=1).sum())
    ticker_calendar["negative_news_count_7d"] = grouped["negative_news_count_1d"].transform(lambda s: s.rolling(7, min_periods=1).sum())
    ticker_calendar["strong_negative_news_flag_7d"] = grouped["strong_negative_news_flag"].transform(
        lambda s: s.rolling(7, min_periods=1).max()
    ).astype(bool)
    ticker_calendar["sentiment_x_articles_7d"] = grouped["sentiment_x_articles_1d"].transform(
        lambda s: s.rolling(7, min_periods=1).sum()
    )
    ticker_calendar["sentiment_x_articles_30d"] = grouped["sentiment_x_articles_1d"].transform(
        lambda s: s.rolling(30, min_periods=1).sum()
    )
    ticker_calendar["ticker_sentiment_7d"] = (
        ticker_calendar["sentiment_x_articles_7d"] / ticker_calendar["article_count_7d"].replace(0, np.nan)
    ).fillna(0.0)
    ticker_calendar["ticker_sentiment_30d"] = (
        ticker_calendar["sentiment_x_articles_30d"] / ticker_calendar["article_count_30d"].replace(0, np.nan)
    ).fillna(0.0)

    market_daily = (
        ticker_calendar.groupby("date", as_index=False)
        .agg(
            market_article_count_1d=("article_count_1d", "sum"),
            market_article_count_7d=("article_count_7d", "sum"),
            market_article_count_30d=("article_count_30d", "sum"),
            market_positive_news_count_7d=("positive_news_count_7d", "sum"),
            market_negative_news_count_7d=("negative_news_count_7d", "sum"),
            market_sentiment_x_articles_1d=("sentiment_x_articles_1d", "sum"),
            market_sentiment_x_articles_7d=("sentiment_x_articles_7d", "sum"),
            market_sentiment_x_articles_30d=("sentiment_x_articles_30d", "sum"),
            tickers_with_news_7d=("article_count_7d", lambda s: int((pd.to_numeric(s, errors="coerce").fillna(0.0) > 0).sum())),
            tickers_positive_sentiment_7d=("ticker_sentiment_7d", lambda s: int((pd.to_numeric(s, errors="coerce").fillna(0.0) > 0).sum())),
            tickers_negative_sentiment_7d=("ticker_sentiment_7d", lambda s: int((pd.to_numeric(s, errors="coerce").fillna(0.0) < 0).sum())),
            strong_negative_tickers_7d=("strong_negative_news_flag_7d", lambda s: int(pd.Series(s).fillna(False).astype(bool).sum())),
            sentiment_dispersion_7d=("ticker_sentiment_7d", lambda s: float(pd.to_numeric(s, errors="coerce").fillna(0.0).std(ddof=0))),
        )
        .sort_values("date")
        .reset_index(drop=True)
    )

    universe_count = max(len(universe_tickers), 1)
    market_daily["market_sentiment_1d"] = (
        market_daily["market_sentiment_x_articles_1d"] / market_daily["market_article_count_1d"].replace(0, np.nan)
    ).fillna(0.0)
    market_daily["market_sentiment_7d"] = (
        market_daily["market_sentiment_x_articles_7d"] / market_daily["market_article_count_7d"].replace(0, np.nan)
    ).fillna(0.0)
    market_daily["market_sentiment_30d"] = (
        market_daily["market_sentiment_x_articles_30d"] / market_daily["market_article_count_30d"].replace(0, np.nan)
    ).fillna(0.0)
    market_daily["market_sentiment_change_7d_vs_30d"] = (
        market_daily["market_sentiment_7d"] - market_daily["market_sentiment_30d"]
    )
    market_daily["market_positive_news_ratio_7d"] = (
        market_daily["market_positive_news_count_7d"] / market_daily["market_article_count_7d"].replace(0, np.nan)
    ).fillna(0.0)
    market_daily["market_negative_news_ratio_7d"] = (
        market_daily["market_negative_news_count_7d"] / market_daily["market_article_count_7d"].replace(0, np.nan)
    ).fillna(0.0)
    market_daily["market_strong_negative_news_ratio_7d"] = (
        market_daily["strong_negative_tickers_7d"] / universe_count
    ).fillna(0.0)
    market_daily["market_news_breadth_7d"] = (
        market_daily["tickers_with_news_7d"] / universe_count
    ).fillna(0.0)
    market_daily["percent_tickers_positive_sentiment_7d"] = (
        market_daily["tickers_positive_sentiment_7d"] / universe_count
    ).fillna(0.0)
    market_daily["percent_tickers_negative_sentiment_7d"] = (
        market_daily["tickers_negative_sentiment_7d"] / universe_count
    ).fillna(0.0)

    market_daily = market_daily[MARKET_SENTIMENT_OUTPUT_COLUMNS].copy()
    save_dataframe(output_path, market_daily)
    return market_daily
