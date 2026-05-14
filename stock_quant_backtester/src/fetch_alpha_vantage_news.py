from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
import math
import time

import pandas as pd
import requests

from src.cache_utils import cache_exists, load_cached_json, save_csv_cache, save_json_cache, should_use_cache
from src.utils import LOGGER, RateLimiter


ALPHA_VANTAGE_URL = "https://www.alphavantage.co/query"
ERROR_KEYS = {"Note", "Information", "Error Message"}


@dataclass(slots=True)
class NewsRequestWindow:
    ticker: str
    month_label: str
    start_ts: pd.Timestamp
    end_ts: pd.Timestamp
    cache_path: Path


def _normalize_window_bounds(start_date: str, end_date: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    if end_ts <= start_ts:
        raise ValueError(f"end_date must be after start_date; got start={start_date} end={end_date}")
    return start_ts.normalize(), end_ts.normalize()


def build_monthly_windows(
    tickers: list[str],
    start_date: str,
    end_date: str,
    raw_news_dir: str | Path,
) -> list[NewsRequestWindow]:
    start_ts, end_ts = _normalize_window_bounds(start_date, end_date)
    raw_news_dir = Path(raw_news_dir)
    month_starts = list(pd.date_range(start=start_ts, end=end_ts - pd.Timedelta(days=1), freq="MS"))
    if start_ts.day != 1:
        month_starts = [start_ts.replace(day=1)] + [month for month in month_starts if month != start_ts.replace(day=1)]

    windows: list[NewsRequestWindow] = []
    for ticker in tickers:
        for month_start in month_starts:
            month_end_exclusive = min(month_start + pd.offsets.MonthBegin(1), end_ts)
            if month_end_exclusive <= start_ts or month_start >= end_ts:
                continue
            effective_start = max(month_start, start_ts)
            month_label = effective_start.strftime("%Y_%m")
            cache_path = raw_news_dir / ticker / f"{month_label}.json"
            windows.append(
                NewsRequestWindow(
                    ticker=ticker,
                    month_label=month_label,
                    start_ts=effective_start,
                    end_ts=month_end_exclusive,
                    cache_path=cache_path,
                )
            )
    return windows


def _cached_response_is_error(payload: dict) -> bool:
    if payload.get("error"):
        return True
    response = payload.get("response")
    return isinstance(response, dict) and any(key in response for key in ERROR_KEYS)


def _build_error_payload(window: NewsRequestWindow, message: str, response: dict | None = None) -> dict:
    return {
        "ticker": window.ticker,
        "month": window.month_label,
        "error": True,
        "message": message,
        "response": response or {},
        "fetched_at_utc": datetime.now(UTC).isoformat(),
    }


def _request_news_payload(
    window: NewsRequestWindow,
    api_key: str,
    limit: int,
    requests_per_minute: int,
    rate_limiter: RateLimiter,
) -> dict:
    if not api_key:
        raise ValueError("ALPHA_VANTAGE_API_KEY is required to fetch news sentiment.")

    time_from = window.start_ts.strftime("%Y%m%dT0000")
    time_to = (window.end_ts - pd.Timedelta(minutes=1)).strftime("%Y%m%dT2359") if window.end_ts.hour == 0 else (
        window.end_ts - pd.Timedelta(minutes=1)
    ).strftime("%Y%m%dT%H%M")

    last_response: dict | None = None
    for attempt in range(1, 6):
        rate_limiter.wait()
        time.sleep(max(1.05, 60.0 / max(requests_per_minute, 1)))
        response = requests.get(
            ALPHA_VANTAGE_URL,
            params={
                "function": "NEWS_SENTIMENT",
                "tickers": window.ticker,
                "time_from": time_from,
                "time_to": time_to,
                "limit": limit,
                "apikey": api_key,
            },
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        last_response = payload if isinstance(payload, dict) else {}
        if isinstance(payload, dict) and "Note" in payload:
            LOGGER.warning("Alpha Vantage rate-limit or note for %s %s: %s", window.ticker, window.month_label, payload["Note"])
            time.sleep(2**attempt)
            continue
        return {
            "ticker": window.ticker,
            "month": window.month_label,
            "error": False,
            "response": payload,
            "fetched_at_utc": datetime.now(UTC).isoformat(),
        }
    return _build_error_payload(window, "Alpha Vantage returned repeated rate-limit or note responses.", last_response)


def fetch_alpha_vantage_news_cache(
    windows: list[NewsRequestWindow],
    *,
    api_key: str,
    cache_enabled: bool,
    force: bool,
    limit: int,
    requests_per_minute: int,
) -> list[dict]:
    rate_limiter = RateLimiter(calls_per_minute=max(requests_per_minute, 1))
    cached_payloads: list[dict] = []

    for window in windows:
        if should_use_cache(window.cache_path, force=force, cache_enabled=cache_enabled):
            cached_payloads.append(load_cached_json(window.cache_path))
            continue

        if cache_enabled and not force and cache_exists(window.cache_path):
            cached_payloads.append(load_cached_json(window.cache_path))
            continue

        payload = _request_news_payload(
            window,
            api_key=api_key,
            limit=limit,
            requests_per_minute=requests_per_minute,
            rate_limiter=rate_limiter,
        )
        response = payload.get("response", {})
        if isinstance(response, dict) and any(key in response for key in ERROR_KEYS):
            LOGGER.warning("Alpha Vantage returned warning payload for %s %s", window.ticker, window.month_label)
            payload["error"] = True
        if cache_enabled:
            save_json_cache(window.cache_path, payload)
        cached_payloads.append(payload)

    return cached_payloads


def summarize_request_plan(windows: list[NewsRequestWindow], *, force: bool, cache_enabled: bool, requests_per_minute: int) -> dict[str, int | float]:
    cached = 0
    missing = 0
    for window in windows:
        if cache_enabled and not force and window.cache_path.exists():
            cached += 1
        else:
            missing += 1
    est_runtime_minutes = missing / max(requests_per_minute, 1)
    return {
        "tickers": len({window.ticker for window in windows}),
        "months": len({window.month_label for window in windows}),
        "total_possible_requests": len(windows),
        "cached_requests": cached,
        "missing_requests": missing,
        "estimated_runtime_minutes": est_runtime_minutes,
    }


def _normalize_provider_label(label: str | None) -> str:
    label_text = str(label or "").strip().lower()
    if label_text in {"bullish", "positive"}:
        return "positive"
    if label_text in {"bearish", "negative"}:
        return "negative"
    return "neutral"


def normalize_alpha_vantage_news_cache(
    windows: list[NewsRequestWindow],
    *,
    processed_output_path: str | Path,
    combined_output_path: str | Path,
) -> pd.DataFrame:
    rows: list[dict] = []
    for window in windows:
        if not window.cache_path.exists():
            continue
        payload = load_cached_json(window.cache_path)
        if _cached_response_is_error(payload):
            continue
        response = payload.get("response", {})
        if not isinstance(response, dict):
            continue
        feed = response.get("feed", [])
        if not isinstance(feed, list):
            continue

        for article in feed:
            published_raw = article.get("time_published")
            if not published_raw:
                continue
            published_date = pd.to_datetime(published_raw, format="%Y%m%dT%H%M%S", errors="coerce", utc=True)
            if pd.isna(published_date):
                published_date = pd.to_datetime(published_raw, errors="coerce", utc=True)
            if pd.isna(published_date):
                continue

            ticker_payload = next(
                (item for item in article.get("ticker_sentiment", []) if str(item.get("ticker")).upper() == window.ticker.upper()),
                {},
            )
            rows.append(
                {
                    "published_date": published_date,
                    "date": published_date.tz_convert(None).normalize(),
                    "ticker": window.ticker,
                    "title": str(article.get("title") or "").strip(),
                    "text": str(article.get("summary") or article.get("title") or "").strip(),
                    "source": str(article.get("source") or "").strip(),
                    "url": str(article.get("url") or "").strip(),
                    "provider": "alpha_vantage",
                    "provider_sentiment_score": pd.to_numeric(ticker_payload.get("ticker_sentiment_score"), errors="coerce"),
                    "provider_sentiment_label": _normalize_provider_label(ticker_payload.get("ticker_sentiment_label")),
                    "provider_relevance_score": pd.to_numeric(ticker_payload.get("relevance_score"), errors="coerce"),
                    "overall_sentiment_score": pd.to_numeric(article.get("overall_sentiment_score"), errors="coerce"),
                    "overall_sentiment_label": _normalize_provider_label(article.get("overall_sentiment_label")),
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(
            columns=[
                "published_date",
                "date",
                "ticker",
                "title",
                "text",
                "source",
                "url",
                "provider",
                "provider_sentiment_score",
                "provider_sentiment_label",
                "provider_relevance_score",
                "overall_sentiment_score",
                "overall_sentiment_label",
            ]
        )
    else:
        df = (
            df.sort_values(["ticker", "published_date", "url", "title"])
            .drop_duplicates(subset=["ticker", "url", "published_date"], keep="first")
            .reset_index(drop=True)
        )

    save_csv_cache(Path(processed_output_path), df)
    save_csv_cache(Path(combined_output_path), df)
    return df
