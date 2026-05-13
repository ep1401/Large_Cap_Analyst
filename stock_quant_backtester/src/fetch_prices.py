from __future__ import annotations

from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.utils import LOGGER, RateLimiter, safe_get_json, save_dataframe, standardize_ticker_for_eodhd


EODHD_BASE_URL = "https://eodhd.com/api/eod"


def fetch_eodhd_prices(
    ticker: str,
    start_date: str,
    end_date: str,
    api_key: str,
    rate_limiter: RateLimiter | None = None,
) -> pd.DataFrame:
    """Fetch daily adjusted OHLCV data from EODHD."""
    if not api_key:
        raise ValueError("EODHD_API_KEY is required to fetch price data.")

    eodhd_symbol = standardize_ticker_for_eodhd(ticker)
    url = f"{EODHD_BASE_URL}/{eodhd_symbol}"
    params = {
        "from": start_date,
        "to": end_date,
        "api_token": api_key,
        "fmt": "json",
    }
    payload = safe_get_json(url, params=params, rate_limiter=rate_limiter)
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"No EODHD price payload returned for {ticker}")

    df = pd.DataFrame(payload)
    rename_map = {
        "adjusted_close": "adjusted_close",
        "adjustedClose": "adjusted_close",
    }
    df = df.rename(columns=rename_map)
    expected = ["date", "open", "high", "low", "close", "adjusted_close", "volume"]
    if "adjusted_close" not in df.columns and "close" in df.columns:
        LOGGER.warning("Adjusted close missing for %s; falling back to close.", ticker)
        df["adjusted_close"] = df["close"]
    missing = [column for column in expected if column not in df.columns]
    if missing:
        raise ValueError(f"Price payload for {ticker} missing columns: {missing}")

    df = df[expected].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = ticker
    numeric_columns = ["open", "high", "low", "close", "adjusted_close", "volume"]
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors="coerce")
    return df.sort_values("date").reset_index(drop=True)


def fetch_and_save_prices(
    tickers: list[str],
    start_date: str,
    end_date: str,
    api_key: str,
    raw_prices_dir: str | Path,
    combined_output_path: str | Path,
    calls_per_minute: int = 1000,
) -> pd.DataFrame:
    """Fetch prices for a list of tickers and persist raw plus combined datasets."""
    raw_prices_dir = Path(raw_prices_dir)
    frames: list[pd.DataFrame] = []
    rate_limiter = RateLimiter(calls_per_minute=calls_per_minute)

    for ticker in tqdm(tickers, desc="Fetching EODHD prices"):
        try:
            df = fetch_eodhd_prices(ticker, start_date, end_date, api_key, rate_limiter=rate_limiter)
            save_dataframe(raw_prices_dir / f"{ticker}.csv", df)
            frames.append(df)
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("Failed to fetch prices for %s: %s", ticker, exc)

    if not frames:
        raise RuntimeError("No price data was fetched successfully.")

    combined = pd.concat(frames, ignore_index=True).sort_values(["ticker", "date"])
    save_dataframe(combined_output_path, combined)
    return combined
