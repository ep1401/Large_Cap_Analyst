from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from src.cache_utils import cache_exists, load_cached_json, save_json_cache, should_use_cache
from src.utils import LOGGER, RateLimiter, save_dataframe, standardize_ticker_for_fmp


FMP_BASE_URL = "https://financialmodelingprep.com/stable"
BLOCKED_STATUS_CODES = {402, 403}


def _request_fmp_json(
    endpoint: str,
    ticker: str,
    api_key: str,
    rate_limiter: RateLimiter | None = None,
) -> dict | list:
    if not api_key:
        raise ValueError("FMP_API_KEY is required to fetch analyst data.")
    if rate_limiter is not None:
        rate_limiter.wait()
    response = requests.get(
        f"{FMP_BASE_URL}/{endpoint}",
        params={"symbol": standardize_ticker_for_fmp(ticker), "apikey": api_key},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def _error_payload(ticker: str, endpoint: str, exc: Exception, status_code: int | None = None) -> dict:
    return {
        "ticker": ticker,
        "endpoint": endpoint,
        "error": True,
        "status_code": status_code,
        "message": str(exc),
        "fetched_at_utc": datetime.now(UTC).isoformat(),
    }


def _load_or_fetch_endpoint(
    ticker: str,
    endpoint: str,
    cache_path: Path,
    api_key: str,
    *,
    rate_limiter: RateLimiter,
    force: bool,
    cache_enabled: bool,
    legacy_path: Path | None = None,
) -> dict | list:
    if should_use_cache(cache_path, force=force, cache_enabled=cache_enabled):
        return load_cached_json(cache_path)
    if cache_enabled and not force and legacy_path is not None and legacy_path.exists():
        payload = load_cached_json(legacy_path)
        save_json_cache(cache_path, payload if isinstance(payload, dict) else {"data": payload})
        return payload

    try:
        payload = _request_fmp_json(endpoint, ticker, api_key, rate_limiter=rate_limiter)
    except requests.HTTPError as exc:
        status_code = exc.response.status_code if exc.response is not None else None
        if cache_enabled and status_code in BLOCKED_STATUS_CODES:
            payload = _error_payload(ticker, endpoint, exc, status_code=status_code)
            save_json_cache(cache_path, payload)
            return payload
        raise
    except Exception as exc:  # noqa: BLE001
        if cache_enabled:
            payload = _error_payload(ticker, endpoint, exc)
            save_json_cache(cache_path, payload)
            return payload
        raise

    if cache_enabled:
        save_json_cache(cache_path, payload if isinstance(payload, dict) else {"data": payload})
    return payload


def fetch_price_target_consensus(
    ticker: str,
    api_key: str,
    rate_limiter: RateLimiter | None = None,
) -> dict:
    payload = _request_fmp_json("price-target-consensus", ticker, api_key, rate_limiter=rate_limiter)
    return payload[0] if isinstance(payload, list) and payload else {}


def fetch_price_target_summary(
    ticker: str,
    api_key: str,
    rate_limiter: RateLimiter | None = None,
) -> dict:
    payload = _request_fmp_json("price-target-summary", ticker, api_key, rate_limiter=rate_limiter)
    return payload[0] if isinstance(payload, list) and payload else {}


def _extract_float(record: dict, keys: list[str]) -> float | None:
    for key in keys:
        value = record.get(key)
        if value is not None and value != "":
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return None


def _normalize_payload(payload: dict | list) -> dict:
    if isinstance(payload, dict) and payload.get("error"):
        return {}
    if isinstance(payload, dict) and "data" in payload and isinstance(payload["data"], list):
        data = payload["data"]
        return data[0] if data else {}
    if isinstance(payload, list):
        return payload[0] if payload else {}
    return payload if isinstance(payload, dict) else {}


def build_analyst_snapshot(
    tickers: list[str],
    api_key: str,
    raw_output_dir: str | Path,
    processed_output_path: str | Path,
    prices_path: str | Path | None = None,
    calls_per_minute: int = 300,
    force: bool = False,
    cache_enabled: bool = True,
) -> pd.DataFrame:
    raw_output_dir = Path(raw_output_dir)
    snapshot_date = pd.Timestamp(datetime.now(UTC).date())
    rows: list[dict] = []
    rate_limiter = RateLimiter(calls_per_minute=calls_per_minute)
    latest_close_by_ticker: dict[str, float] = {}

    if prices_path is not None and Path(prices_path).exists():
        prices = pd.read_csv(prices_path, parse_dates=["date"])
        latest_prices = prices.sort_values(["ticker", "date"]).groupby("ticker").tail(1)[["ticker", "adjusted_close"]]
        latest_close_by_ticker = dict(zip(latest_prices["ticker"], latest_prices["adjusted_close"]))

    for ticker in tqdm(tickers, desc="Fetching FMP analyst data"):
        endpoint_map = {
            "price-target-consensus": raw_output_dir / f"{ticker}_price_target_consensus.json",
            "price-target-summary": raw_output_dir / f"{ticker}_price_target_summary.json",
            "ratings-snapshot": raw_output_dir / f"{ticker}_ratings_snapshot.json",
            "ratings-historical": raw_output_dir / f"{ticker}_historical_ratings.json",
        }
        legacy_map = {
            "price-target-consensus": raw_output_dir.parent / f"{ticker}_price_target_consensus.json",
            "price-target-summary": raw_output_dir.parent / f"{ticker}_price_target_summary.json",
        }

        payloads: dict[str, dict | list] = {}
        for endpoint, cache_path in endpoint_map.items():
            try:
                payloads[endpoint] = _load_or_fetch_endpoint(
                    ticker,
                    endpoint,
                    cache_path,
                    api_key,
                    rate_limiter=rate_limiter,
                    force=force,
                    cache_enabled=cache_enabled,
                    legacy_path=legacy_map.get(endpoint),
                )
            except Exception as exc:  # noqa: BLE001
                LOGGER.error("Failed to fetch analyst endpoint %s for %s: %s", endpoint, ticker, exc)
                payloads[endpoint] = _error_payload(ticker, endpoint, exc)
                if cache_enabled:
                    save_json_cache(cache_path, payloads[endpoint])

        consensus = _normalize_payload(payloads["price-target-consensus"])
        summary = _normalize_payload(payloads["price-target-summary"])

        close_proxy = _extract_float(consensus, ["price", "currentPrice", "current_price"])
        if close_proxy is None:
            close_proxy = latest_close_by_ticker.get(ticker)
        consensus_target = _extract_float(consensus, ["targetConsensus", "target_consensus", "consensusPriceTarget"])
        low_target = _extract_float(consensus, ["targetLow", "target_low", "lowPriceTarget"])
        high_target = _extract_float(consensus, ["targetHigh", "target_high", "highPriceTarget"])
        median_target = _extract_float(consensus, ["targetMedian", "target_median", "medianPriceTarget"])
        analyst_count = _extract_float(
            summary,
            ["lastYearCount", "lastQuarterCount", "lastMonthCount", "allTimeCount", "analystCount"],
        )

        target_7d = _extract_float(summary, ["last7DaysAverage", "average7Days", "target7d"])
        target_30d = _extract_float(summary, ["last30DaysAverage", "average30Days", "target30d", "lastMonthAvgPriceTarget"])
        last_month_count = _extract_float(summary, ["lastMonthCount"])
        last_month_avg_price_target = _extract_float(summary, ["lastMonthAvgPriceTarget"])
        last_quarter_count = _extract_float(summary, ["lastQuarterCount"])
        last_quarter_avg_price_target = _extract_float(summary, ["lastQuarterAvgPriceTarget"])
        last_year_count = _extract_float(summary, ["lastYearCount"])
        last_year_avg_price_target = _extract_float(summary, ["lastYearAvgPriceTarget"])
        all_time_count = _extract_float(summary, ["allTimeCount"])
        all_time_avg_price_target = _extract_float(summary, ["allTimeAvgPriceTarget"])
        publishers = summary.get("publishers")
        if isinstance(publishers, list):
            analyst_publishers = ",".join(str(item).strip() for item in publishers if str(item).strip())
        else:
            analyst_publishers = str(publishers or "").strip()

        def upside(target: float | None) -> float | None:
            if target is None or close_proxy in (None, 0):
                return None
            return target / close_proxy - 1

        rows.append(
            {
                "date": snapshot_date,
                "ticker": ticker,
                "snapshot_mode": "snapshot_current",
                "consensus_target": consensus_target,
                "low_target": low_target,
                "high_target": high_target,
                "median_target": median_target,
                "analyst_count": analyst_count,
                "consensus_upside": upside(consensus_target),
                "low_target_upside": upside(low_target),
                "high_target_upside": upside(high_target),
                "last_month_target_count": last_month_count,
                "last_month_avg_price_target": last_month_avg_price_target,
                "last_quarter_target_count": last_quarter_count,
                "last_quarter_avg_price_target": last_quarter_avg_price_target,
                "last_year_target_count": last_year_count,
                "last_year_avg_price_target": last_year_avg_price_target,
                "all_time_target_count": all_time_count,
                "all_time_avg_price_target": all_time_avg_price_target,
                "analyst_publishers": analyst_publishers,
                "last_month_target_upside": upside(last_month_avg_price_target),
                "last_quarter_target_upside": upside(last_quarter_avg_price_target),
                "last_year_target_upside": upside(last_year_avg_price_target),
                "all_time_target_upside": upside(all_time_avg_price_target),
                "target_spread": (
                    (high_target - low_target) / close_proxy
                    if None not in (high_target, low_target, close_proxy) and close_proxy != 0
                    else None
                ),
                "target_revision_7d": (consensus_target - target_7d if None not in (consensus_target, target_7d) else None),
                "target_revision_30d": (
                    consensus_target - target_30d if None not in (consensus_target, target_30d) else None
                ),
            }
        )

    df = pd.DataFrame(rows).sort_values("ticker").reset_index(drop=True)
    numeric_columns = [column for column in df.columns if column not in {"date", "ticker", "snapshot_mode", "analyst_publishers"}]
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors="coerce")
    save_dataframe(processed_output_path, df)
    return df
