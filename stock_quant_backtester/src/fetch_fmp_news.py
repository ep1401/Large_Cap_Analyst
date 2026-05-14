from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from time import sleep

import pandas as pd

from src.utils import LOGGER, RateLimiter, load_json, safe_get_json, save_dataframe, save_json, standardize_ticker_for_fmp


FMP_BASE_URL = "https://financialmodelingprep.com/stable"
FMP_STOCK_NEWS_URL = f"{FMP_BASE_URL}/news/stock"


def fetch_fmp_stock_news(
    ticker: str,
    api_key: str,
    *,
    start_date: str,
    end_date: str,
    page: int = 0,
    limit: int = 200,
    rate_limiter: RateLimiter | None = None,
) -> list[dict]:
    if not api_key:
        raise ValueError("FMP_API_KEY is required to fetch stock news.")
    payload = safe_get_json(
        FMP_STOCK_NEWS_URL,
        params={
            "symbols": standardize_ticker_for_fmp(ticker),
            "from": start_date,
            "to": end_date,
            "page": page,
            "limit": limit,
            "apikey": api_key,
        },
        rate_limiter=rate_limiter,
    )
    if isinstance(payload, list):
        return payload
    LOGGER.warning("Unexpected FMP news payload type for %s page %s: %s", ticker, page, type(payload).__name__)
    return []


def _extract_published_date(article: dict) -> pd.Timestamp | None:
    for key in ["publishedDate", "published_date", "date", "publishedAt"]:
        value = article.get(key)
        if value:
            ts = pd.to_datetime(value, errors="coerce", utc=True)
            if pd.notna(ts):
                return ts
    return None


def _extract_text(article: dict) -> str:
    title = str(article.get("title") or "").strip()
    text = str(article.get("text") or article.get("snippet") or article.get("content") or "").strip()
    if title and text:
        return f"{title}\n\n{text}"
    if title:
        return title
    return text


def _extract_publisher(article: dict) -> str:
    return str(article.get("publisher") or article.get("site") or article.get("source") or "").strip()


def _normalize_article(article: dict, ticker: str, raw_json_path: Path) -> dict | None:
    published_date = _extract_published_date(article)
    if published_date is None:
        return None

    title = str(article.get("title") or "").strip()
    body_text = _extract_text(article)
    site = str(article.get("site") or article.get("source") or "").strip()
    publisher = _extract_publisher(article)
    url = str(article.get("url") or article.get("link") or "").strip()
    symbol = str(article.get("symbol") or article.get("symbols") or ticker).strip()

    return {
        "published_date": published_date,
        "date": published_date.tz_convert(None).normalize(),
        "ticker": ticker,
        "title": title,
        "text": body_text,
        "site": site,
        "publisher": publisher,
        "url": url,
        "symbol": symbol,
        "raw_json_path": str(raw_json_path),
    }


def _deduplicate_articles(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    dedupe_key = df["url"].fillna("").str.strip()
    url_mask = dedupe_key.ne("")
    url_df = df.loc[url_mask].drop_duplicates(subset=["ticker", "url"], keep="first")
    fallback_df = df.loc[~url_mask].copy()
    fallback_df["fallback_key"] = (
        fallback_df["title"].fillna("").str.strip() + "|" + fallback_df["published_date"].astype(str)
    )
    fallback_df = fallback_df.drop_duplicates(subset=["ticker", "fallback_key"], keep="first").drop(
        columns=["fallback_key"]
    )
    combined = pd.concat([url_df, fallback_df], ignore_index=True)
    return combined.sort_values(["ticker", "published_date", "title"]).reset_index(drop=True)


def _load_cached_articles(raw_path: Path, ticker: str) -> list[dict]:
    payload = load_json(raw_path)
    if isinstance(payload, dict) and isinstance(payload.get("articles"), list):
        articles = payload["articles"]
    elif isinstance(payload, list):
        articles = payload
    else:
        articles = []

    normalized = []
    for article in articles:
        row = _normalize_article(article, ticker=ticker, raw_json_path=raw_path)
        if row is not None:
            normalized.append(row)
    return normalized


def build_fmp_news_dataset(
    tickers: list[str],
    *,
    api_key: str,
    start_date: str,
    end_date: str,
    raw_output_dir: str | Path,
    processed_output_path: str | Path,
    calls_per_minute: int = 300,
    force: bool = False,
    page_limit: int = 200,
    max_pages_per_ticker: int = 50,
    request_sleep_seconds: float = 0.2,
) -> pd.DataFrame:
    raw_output_dir = Path(raw_output_dir)
    raw_output_dir.mkdir(parents=True, exist_ok=True)
    start_ts = pd.Timestamp(start_date).tz_localize("UTC")
    end_ts = pd.Timestamp(end_date).tz_localize("UTC") + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    rate_limiter = RateLimiter(calls_per_minute=max(calls_per_minute, 1))

    processed_rows: list[dict] = []

    for ticker in tickers:
        raw_path = raw_output_dir / f"{ticker}.json"
        if raw_path.exists() and not force:
            processed_rows.extend(_load_cached_articles(raw_path, ticker))
            continue

        all_articles: list[dict] = []
        seen_keys: set[str] = set()
        pages_payload: list[dict] = []

        for page in range(max_pages_per_ticker):
            try:
                batch = fetch_fmp_stock_news(
                    ticker,
                    api_key,
                    start_date=start_date,
                    end_date=end_date,
                    page=page,
                    limit=page_limit,
                    rate_limiter=rate_limiter,
                )
            except Exception as exc:  # noqa: BLE001
                LOGGER.error("Failed to fetch FMP news for %s page %s: %s", ticker, page, exc)
                break

            pages_payload.append({"page": page, "articles": batch})
            if not batch:
                break

            stop_after_page = False
            for article in batch:
                published_ts = _extract_published_date(article)
                if published_ts is None:
                    continue
                published_ts = published_ts.tz_convert("UTC")
                if published_ts < start_ts:
                    stop_after_page = True
                if published_ts > end_ts or published_ts < start_ts:
                    continue

                article_key = str(article.get("url") or f"{article.get('title')}|{published_ts.isoformat()}")
                if article_key in seen_keys:
                    continue
                seen_keys.add(article_key)
                all_articles.append(article)

            if len(batch) < page_limit or stop_after_page:
                break
            sleep(request_sleep_seconds)

        save_json(
            raw_path,
            {
                "ticker": ticker,
                "start_date": start_date,
                "end_date": end_date,
                "fetched_at_utc": datetime.now(UTC).isoformat(),
                "pages": pages_payload,
                "articles": all_articles,
            },
        )
        processed_rows.extend(_load_cached_articles(raw_path, ticker))

    df = pd.DataFrame(processed_rows)
    if df.empty:
        df = pd.DataFrame(
            columns=[
                "published_date",
                "date",
                "ticker",
                "title",
                "text",
                "site",
                "publisher",
                "url",
                "symbol",
                "raw_json_path",
            ]
        )
    else:
        df["published_date"] = pd.to_datetime(df["published_date"], utc=True, errors="coerce")
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.loc[df["published_date"].notna() & df["date"].notna()].copy()
        df = _deduplicate_articles(df)

    save_dataframe(processed_output_path, df)
    return df
