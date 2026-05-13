from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.utils import LOGGER, safe_get_json, save_dataframe, save_json


ALPHA_VANTAGE_URL = "https://www.alphavantage.co/query"


def fetch_alpha_vantage_news_sentiment(
    ticker: str,
    start_datetime: str,
    end_datetime: str,
    api_key: str,
) -> dict:
    """Fetch Alpha Vantage ticker-specific news sentiment for a time window."""
    if not api_key:
        raise ValueError("ALPHA_VANTAGE_API_KEY is required to fetch news sentiment.")
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": ticker,
        "time_from": start_datetime,
        "time_to": end_datetime,
        "limit": 1000,
        "apikey": api_key,
    }
    return safe_get_json(ALPHA_VANTAGE_URL, params=params)


def _month_starts(start_date: str, end_date: str) -> list[pd.Timestamp]:
    return list(pd.date_range(pd.Timestamp(start_date), pd.Timestamp(end_date), freq="MS"))


def _empty_sentiment_row(ticker: str, date: pd.Timestamp) -> dict:
    return {
        "date": date,
        "ticker": ticker,
        "article_count": 0,
        "avg_sentiment_score": 0.0,
        "weighted_sentiment_score": 0.0,
        "positive_article_ratio": 0.0,
        "negative_article_ratio": 0.0,
        "neutral_article_ratio": 0.0,
        "avg_relevance_score": 0.0,
    }


def build_news_sentiment_dataset(
    tickers: list[str],
    start_date: str,
    end_date: str,
    api_key: str,
    raw_output_dir: str | Path,
    processed_output_path: str | Path,
) -> pd.DataFrame:
    """Fetch monthly windows and aggregate article-level sentiment to daily features."""
    raw_output_dir = Path(raw_output_dir)
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    month_starts = _month_starts(start_date, end_date)
    rows: list[dict] = []

    for ticker in tqdm(tickers, desc="Fetching Alpha Vantage sentiment"):
        article_rows: list[dict] = []
        for month_start in month_starts:
            window_start = max(month_start, start_ts)
            window_end = min(month_start + pd.offsets.MonthEnd(1), end_ts)
            if window_start > window_end:
                continue
            start_str = window_start.strftime("%Y%m%dT0000")
            end_str = window_end.strftime("%Y%m%dT2359")
            try:
                payload = fetch_alpha_vantage_news_sentiment(ticker, start_str, end_str, api_key)
                save_json(raw_output_dir / f"{ticker}_{start_str}_{end_str}.json", payload)
            except Exception as exc:  # noqa: BLE001
                LOGGER.error("Failed to fetch sentiment for %s %s-%s: %s", ticker, start_str, end_str, exc)
                payload = {}

            for article in payload.get("feed", []):
                published = article.get("time_published")
                if not published:
                    continue
                article_date = pd.to_datetime(published[:8], format="%Y%m%d", errors="coerce")
                if pd.isna(article_date):
                    continue
                ticker_payload = next(
                    (item for item in article.get("ticker_sentiment", []) if item.get("ticker") == ticker),
                    None,
                )
                if ticker_payload is None:
                    continue
                sentiment_score = float(ticker_payload.get("ticker_sentiment_score", 0.0))
                relevance_score = float(ticker_payload.get("relevance_score", 0.0))
                label = str(ticker_payload.get("ticker_sentiment_label", "Neutral")).lower()
                article_rows.append(
                    {
                        "date": article_date.normalize(),
                        "ticker": ticker,
                        "sentiment_score": sentiment_score,
                        "relevance_score": relevance_score,
                        "label": label,
                    }
                )

        ticker_days = pd.date_range(start_ts, end_ts, freq="D")
        if article_rows:
            article_df = pd.DataFrame(article_rows)
            for date, daily in article_df.groupby("date"):
                article_count = len(daily)
                score_sum = daily["sentiment_score"].sum()
                relevance_sum = daily["relevance_score"].sum()
                rows.append(
                    {
                        "date": date,
                        "ticker": ticker,
                        "article_count": article_count,
                        "avg_sentiment_score": daily["sentiment_score"].mean(),
                        "weighted_sentiment_score": (
                            (daily["sentiment_score"] * daily["relevance_score"]).sum() / relevance_sum
                            if relevance_sum != 0
                            else 0.0
                        ),
                        "positive_article_ratio": (daily["label"].eq("bullish").mean()),
                        "negative_article_ratio": (daily["label"].eq("bearish").mean()),
                        "neutral_article_ratio": (daily["label"].eq("neutral").mean()),
                        "avg_relevance_score": daily["relevance_score"].mean(),
                    }
                )
            existing_dates = set(pd.to_datetime(pd.DataFrame(rows)[lambda df: df["ticker"] == ticker]["date"]))
            for date in ticker_days:
                if date.normalize() not in existing_dates:
                    rows.append(_empty_sentiment_row(ticker, date.normalize()))
        else:
            rows.extend(_empty_sentiment_row(ticker, date.normalize()) for date in ticker_days)

    df = pd.DataFrame(rows).sort_values(["ticker", "date"]).reset_index(drop=True)
    save_dataframe(processed_output_path, df)
    return df

