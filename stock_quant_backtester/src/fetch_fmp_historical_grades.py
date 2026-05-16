from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

from src.cache_utils import load_cached_json, save_json_cache, should_use_cache
from src.utils import LOGGER, RateLimiter, save_dataframe, standardize_ticker_for_fmp


FMP_BASE_URL = "https://financialmodelingprep.com/stable"
BLOCKED_STATUS_CODES = {402, 403}
PRIMARY_EVENT_ENDPOINT = "grades"
SECONDARY_SUMMARY_ENDPOINT = "grades-historical"


def _error_payload(ticker: str, endpoint: str, exc: Exception, status_code: int | None = None) -> dict:
    return {
        "ticker": ticker,
        "endpoint": endpoint,
        "error": True,
        "status_code": status_code,
        "message": str(exc),
        "fetched_at_utc": datetime.now(UTC).isoformat(),
    }


def _request_endpoint(endpoint: str, ticker: str, api_key: str, rate_limiter: RateLimiter) -> dict:
    if not api_key:
        raise ValueError("FMP_API_KEY is required to fetch historical analyst grades.")
    rate_limiter.wait()
    response = requests.get(
        f"{FMP_BASE_URL}/{endpoint}",
        params={"symbol": standardize_ticker_for_fmp(ticker), "apikey": api_key},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    return {
        "endpoint": endpoint,
        "status_code": response.status_code,
        "data": payload,
    }


def _fetch_ticker_payload(
    ticker: str,
    api_key: str,
    cache_path: Path,
    *,
    force: bool,
    cache_enabled: bool,
    rate_limiter: RateLimiter,
) -> dict:
    if should_use_cache(cache_path, force=force, cache_enabled=cache_enabled):
        return load_cached_json(cache_path)

    payload = {
        "ticker": ticker,
        "provider": "fmp",
        "fetched_at_utc": datetime.now(UTC).isoformat(),
        "grades_endpoint": {},
        "grades_historical_endpoint": {},
    }
    for endpoint, key in (
        (SECONDARY_SUMMARY_ENDPOINT, "grades_historical_endpoint"),
        (PRIMARY_EVENT_ENDPOINT, "grades_endpoint"),
    ):
        try:
            endpoint_payload = _request_endpoint(endpoint, ticker, api_key, rate_limiter)
            if isinstance(endpoint_payload.get("data"), dict) and any(
                marker in endpoint_payload["data"] for marker in ("Error Message", "error", "message", "Information", "Note")
            ):
                payload[key] = _error_payload(
                    ticker,
                    endpoint,
                    RuntimeError(str(endpoint_payload["data"])),
                    status_code=endpoint_payload.get("status_code"),
                )
            elif endpoint_payload.get("data") in (None, [], {}):
                payload[key] = _error_payload(
                    ticker,
                    endpoint,
                    RuntimeError("Empty response"),
                    status_code=endpoint_payload.get("status_code"),
                )
            else:
                payload[key] = endpoint_payload
        except requests.HTTPError as exc:
            status_code = exc.response.status_code if exc.response is not None else None
            payload[key] = _error_payload(ticker, endpoint, exc, status_code=status_code)
            LOGGER.warning("Historical grade endpoint %s failed for %s with status %s", endpoint, ticker, status_code)
        except Exception as exc:  # noqa: BLE001
            payload[key] = _error_payload(ticker, endpoint, exc)
            LOGGER.warning("Historical grade endpoint %s failed for %s: %s", endpoint, ticker, exc)

    if cache_enabled:
        save_json_cache(cache_path, payload)
    return payload


def _normalize_action(raw_action: str | None, derived: dict) -> str:
    action = (raw_action or "").strip().lower()
    if "upgrade" in action or derived["is_upgrade"]:
        return "upgrade"
    if "downgrade" in action or derived["is_downgrade"]:
        return "downgrade"
    if any(token in action for token in {"maintain", "reiterate", "initiated", "initiate"}) or derived["is_maintain"]:
        return "maintain"
    return action or "unknown"


def _extract_event_records(payload: dict) -> list[dict]:
    endpoint_payload = payload.get("grades_endpoint", {})
    if endpoint_payload.get("error"):
        return []
    data = endpoint_payload.get("data", [])
    return data if isinstance(data, list) else []


def build_historical_grade_dataset(
    tickers: list[str],
    api_key: str,
    raw_output_dir: str | Path,
    processed_output_path: str | Path,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    calls_per_minute: int = 300,
    force: bool = False,
    cache_enabled: bool = True,
) -> pd.DataFrame:
    from src.analyst_grade_utils import classify_grade_action

    raw_output_dir = Path(raw_output_dir)
    processed_output_path = Path(processed_output_path)
    rate_limiter = RateLimiter(calls_per_minute=calls_per_minute)
    start_ts = pd.Timestamp(start_date) if start_date else None
    end_ts = pd.Timestamp(end_date) if end_date else None

    rows: list[dict] = []
    for ticker in tqdm(tickers, desc="Fetching FMP historical grades"):
        cache_path = raw_output_dir / f"{ticker}.json"
        payload = _fetch_ticker_payload(
            ticker,
            api_key,
            cache_path,
            force=force,
            cache_enabled=cache_enabled,
            rate_limiter=rate_limiter,
        )
        event_records = _extract_event_records(payload)
        if not event_records:
            LOGGER.warning("No usable historical grade events for %s. Historical-grade models may skip this ticker.", ticker)
            continue

        for record in event_records:
            event_date = pd.to_datetime(record.get("date"), errors="coerce")
            if pd.isna(event_date):
                continue
            if start_ts is not None and event_date < start_ts:
                continue
            if end_ts is not None and event_date >= end_ts:
                continue

            raw_action = record.get("action")
            derived = classify_grade_action(record.get("previousGrade"), record.get("newGrade"), raw_action)
            rows.append(
                {
                    "date": event_date.normalize(),
                    "ticker": ticker,
                    "grading_company": str(record.get("gradingCompany") or "").strip(),
                    "previous_grade": str(record.get("previousGrade") or "").strip(),
                    "new_grade": str(record.get("newGrade") or "").strip(),
                    "action": _normalize_action(raw_action, derived),
                    "raw_action": str(raw_action or "").strip(),
                    "analyst_name": str(record.get("analystName") or record.get("analyst") or "").strip(),
                    "url": str(record.get("url") or record.get("newsURL") or record.get("articleURL") or "").strip(),
                    "provider": "fmp",
                    "raw_json_path": str(cache_path),
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(
            columns=[
                "date",
                "ticker",
                "grading_company",
                "previous_grade",
                "new_grade",
                "action",
                "raw_action",
                "analyst_name",
                "url",
                "provider",
                "raw_json_path",
            ]
        )
    else:
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        df = (
            df.sort_values(["ticker", "date", "grading_company", "new_grade", "raw_action"])
            .drop_duplicates(
                subset=["ticker", "date", "grading_company", "previous_grade", "new_grade", "raw_action"],
                keep="first",
            )
            .reset_index(drop=True)
        )

    save_dataframe(processed_output_path, df)
    return df
