from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from tqdm import tqdm

from src.analyst_grade_utils import classify_grade_action
from src.cache_utils import load_cached_json, save_json_cache, should_use_cache
from src.utils import LOGGER, RateLimiter, save_dataframe, standardize_ticker_for_fmp


FMP_BASE_URL = "https://financialmodelingprep.com/stable"
BLOCKED_STATUS_CODES = {402, 403}
GRADES_HISTORICAL_ENDPOINT = "grades-historical"
GRADES_EVENTS_ENDPOINT = "grades"


def _error_payload(
    ticker: str,
    endpoint: str,
    exc: Exception | str,
    status_code: int | None = None,
    raw_data: Any | None = None,
) -> dict[str, Any]:
    return {
        "ticker": ticker,
        "endpoint": endpoint,
        "error": True,
        "status_code": status_code,
        "message": str(exc),
        "raw_data": raw_data,
        "fetched_at_utc": datetime.now(UTC).isoformat(),
    }


def _request_endpoint(
    endpoint: str,
    ticker: str,
    api_key: str,
    rate_limiter: RateLimiter,
    limit: int,
) -> dict[str, Any]:
    if not api_key:
        raise ValueError("FMP_API_KEY is required to fetch historical analyst grades.")
    rate_limiter.wait()
    response = requests.get(
        f"{FMP_BASE_URL}/{endpoint}",
        params={"symbol": standardize_ticker_for_fmp(ticker), "limit": limit, "apikey": api_key},
        timeout=30,
    )
    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError(f"Malformed JSON returned from {endpoint} for {ticker}") from exc

    if response.status_code in BLOCKED_STATUS_CODES:
        raise requests.HTTPError(str(payload), response=response)
    response.raise_for_status()
    return {
        "ticker": ticker,
        "endpoint": endpoint,
        "status_code": response.status_code,
        "data": payload,
        "fetched_at_utc": datetime.now(UTC).isoformat(),
    }


def _looks_like_api_error(payload: dict[str, Any]) -> bool:
    data = payload.get("data")
    if isinstance(data, dict):
        lowered_keys = {str(key).lower() for key in data}
        return bool({"error", "message", "note", "information"} & lowered_keys)
    return False


def _normalize_endpoint_payload(
    ticker: str,
    endpoint: str,
    raw_payload: dict[str, Any],
) -> dict[str, Any]:
    if _looks_like_api_error(raw_payload):
        return _error_payload(
            ticker,
            endpoint,
            f"API error payload returned from {endpoint}",
            status_code=raw_payload.get("status_code"),
            raw_data=raw_payload.get("data"),
        )

    data = raw_payload.get("data")
    if not isinstance(data, list):
        return _error_payload(
            ticker,
            endpoint,
            f"Malformed JSON payload returned from {endpoint}",
            status_code=raw_payload.get("status_code"),
            raw_data=data,
        )

    if len(data) == 0:
        return _error_payload(
            ticker,
            endpoint,
            "Empty response",
            status_code=raw_payload.get("status_code"),
            raw_data=data,
        )

    return raw_payload


def _fetch_endpoint_payload(
    ticker: str,
    endpoint: str,
    api_key: str,
    cache_path: Path,
    *,
    force: bool,
    cache_enabled: bool,
    rate_limiter: RateLimiter,
    limit: int,
) -> dict[str, Any]:
    if should_use_cache(cache_path, force=force, cache_enabled=cache_enabled):
        return load_cached_json(cache_path)

    try:
        payload = _request_endpoint(endpoint, ticker, api_key, rate_limiter, limit)
        payload = _normalize_endpoint_payload(ticker, endpoint, payload)
    except requests.HTTPError as exc:
        status_code = exc.response.status_code if exc.response is not None else None
        raw_data = None
        if exc.response is not None:
            try:
                raw_data = exc.response.json()
            except ValueError:
                raw_data = exc.response.text
        payload = _error_payload(ticker, endpoint, exc, status_code=status_code, raw_data=raw_data)
        LOGGER.warning("Historical analyst endpoint %s failed for %s with status %s", endpoint, ticker, status_code)
    except Exception as exc:  # noqa: BLE001
        payload = _error_payload(ticker, endpoint, exc)
        LOGGER.warning("Historical analyst endpoint %s failed for %s: %s", endpoint, ticker, exc)

    save_json_cache(cache_path, payload)
    return payload


def _normalize_action(raw_action: str | None, derived: dict[str, Any]) -> str:
    action = (raw_action or "").strip().lower()
    if "upgrade" in action or derived["is_upgrade"]:
        return "upgrade"
    if "downgrade" in action or derived["is_downgrade"]:
        return "downgrade"
    if any(token in action for token in {"maintain", "reiterate", "initiated", "initiate"}) or derived["is_maintain"]:
        return "maintain"
    return action or "unknown"


def _extract_list_data(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if payload.get("error"):
        return []
    data = payload.get("data", [])
    return data if isinstance(data, list) else []


def _safe_int(value: Any) -> int:
    numeric = pd.to_numeric(value, errors="coerce")
    return 0 if pd.isna(numeric) else int(numeric)


def _rating_count_rows(
    ticker: str,
    payload: dict[str, Any],
    cache_path: Path,
    start_ts: pd.Timestamp | None,
    end_ts: pd.Timestamp | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in _extract_list_data(payload):
        rating_date = pd.to_datetime(record.get("date"), errors="coerce")
        if pd.isna(rating_date):
            continue
        rating_date = rating_date.normalize()
        if start_ts is not None and rating_date < start_ts:
            continue
        if end_ts is not None and rating_date >= end_ts:
            continue

        strong_buy = _safe_int(record.get("analystRatingsStrongBuy", 0))
        buy = _safe_int(record.get("analystRatingsBuy", 0))
        hold = _safe_int(record.get("analystRatingsHold", 0))
        sell = _safe_int(record.get("analystRatingsSell", 0))
        strong_sell = _safe_int(record.get("analystRatingsStrongSell", 0))

        total = strong_buy + buy + hold + sell + strong_sell
        positive = strong_buy + buy
        negative = sell + strong_sell
        if total == 0:
            positive_ratio = 0.0
            negative_ratio = 0.0
            neutral_ratio = 0.0
            buy_hold_sell_score = 0.0
            rating_score = 3.0
        else:
            positive_ratio = positive / total
            negative_ratio = negative / total
            neutral_ratio = hold / total
            buy_hold_sell_score = (positive - negative) / total
            rating_score = ((5 * strong_buy) + (4 * buy) + (3 * hold) + (2 * sell) + (1 * strong_sell)) / total

        rows.append(
            {
                "date": rating_date,
                "ticker": ticker,
                "analyst_ratings_strong_buy": strong_buy,
                "analyst_ratings_buy": buy,
                "analyst_ratings_hold": hold,
                "analyst_ratings_sell": sell,
                "analyst_ratings_strong_sell": strong_sell,
                "historical_total_ratings": total,
                "historical_positive_ratings": positive,
                "historical_negative_ratings": negative,
                "historical_positive_rating_ratio": positive_ratio,
                "historical_negative_rating_ratio": negative_ratio,
                "historical_neutral_rating_ratio": neutral_ratio,
                "historical_buy_hold_sell_score": buy_hold_sell_score,
                "historical_rating_score": rating_score,
                "provider": "fmp_grades_historical",
                "raw_json_path": str(cache_path),
            }
        )
    return rows


def _event_rows(
    ticker: str,
    payload: dict[str, Any],
    cache_path: Path,
    start_ts: pd.Timestamp | None,
    end_ts: pd.Timestamp | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in _extract_list_data(payload):
        event_date = pd.to_datetime(record.get("date"), errors="coerce")
        if pd.isna(event_date):
            continue
        event_date = event_date.normalize()
        if start_ts is not None and event_date < start_ts:
            continue
        if end_ts is not None and event_date >= end_ts:
            continue

        raw_action = record.get("action")
        derived = classify_grade_action(record.get("previousGrade"), record.get("newGrade"), raw_action)
        rows.append(
            {
                "date": event_date,
                "ticker": ticker,
                "grading_company": str(record.get("gradingCompany") or "").strip(),
                "previous_grade": str(record.get("previousGrade") or "").strip(),
                "new_grade": str(record.get("newGrade") or "").strip(),
                "action": _normalize_action(raw_action, derived),
                "raw_action": str(raw_action or "").strip(),
                "analyst_name": str(record.get("analystName") or record.get("analyst") or "").strip(),
                "provider": "fmp_grades",
                "raw_json_path": str(cache_path),
                "previous_grade_score": derived["previous_grade_score"],
                "new_grade_score": derived["new_grade_score"],
                "grade_delta": derived["grade_delta"],
                "is_upgrade": derived["is_upgrade"],
                "is_downgrade": derived["is_downgrade"],
                "is_maintain": derived["is_maintain"],
                "is_positive_grade": derived["is_positive_grade"],
                "is_negative_grade": derived["is_negative_grade"],
            }
        )
    return rows


def build_historical_grade_datasets(
    tickers: list[str],
    api_key: str,
    raw_output_dir: str | Path,
    rating_counts_output_path: str | Path,
    grade_events_output_path: str | Path,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    calls_per_minute: int = 300,
    force: bool = False,
    cache_enabled: bool = True,
    limit: int = 1000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_output_dir = Path(raw_output_dir)
    rating_counts_output_path = Path(rating_counts_output_path)
    grade_events_output_path = Path(grade_events_output_path)
    rate_limiter = RateLimiter(calls_per_minute=calls_per_minute)
    start_ts = pd.Timestamp(start_date).normalize() if start_date else None
    end_ts = pd.Timestamp(end_date).normalize() if end_date else None

    rating_count_records: list[dict[str, Any]] = []
    grade_event_records: list[dict[str, Any]] = []

    for ticker in tqdm(tickers, desc="Fetching FMP historical analyst data"):
        historical_cache_path = raw_output_dir / f"{ticker}_grades_historical.json"
        events_cache_path = raw_output_dir / f"{ticker}_grades_events.json"

        historical_payload = _fetch_endpoint_payload(
            ticker,
            GRADES_HISTORICAL_ENDPOINT,
            api_key,
            historical_cache_path,
            force=force,
            cache_enabled=cache_enabled,
            rate_limiter=rate_limiter,
            limit=limit,
        )
        if historical_payload.get("error"):
            LOGGER.warning(
                "Skipping malformed or unavailable grades-historical data for %s. See %s",
                ticker,
                historical_cache_path,
            )
        else:
            rating_count_records.extend(
                _rating_count_rows(ticker, historical_payload, historical_cache_path, start_ts, end_ts)
            )

        events_payload = _fetch_endpoint_payload(
            ticker,
            GRADES_EVENTS_ENDPOINT,
            api_key,
            events_cache_path,
            force=force,
            cache_enabled=cache_enabled,
            rate_limiter=rate_limiter,
            limit=limit,
        )
        if events_payload.get("error"):
            LOGGER.warning(
                "Skipping malformed or unavailable grades event data for %s. See %s",
                ticker,
                events_cache_path,
            )
        else:
            grade_event_records.extend(_event_rows(ticker, events_payload, events_cache_path, start_ts, end_ts))

    rating_counts = pd.DataFrame(rating_count_records)
    if rating_counts.empty:
        rating_counts = pd.DataFrame(
            columns=[
                "date",
                "ticker",
                "analyst_ratings_strong_buy",
                "analyst_ratings_buy",
                "analyst_ratings_hold",
                "analyst_ratings_sell",
                "analyst_ratings_strong_sell",
                "historical_total_ratings",
                "historical_positive_ratings",
                "historical_negative_ratings",
                "historical_positive_rating_ratio",
                "historical_negative_rating_ratio",
                "historical_neutral_rating_ratio",
                "historical_buy_hold_sell_score",
                "historical_rating_score",
                "provider",
                "raw_json_path",
            ]
        )
    else:
        rating_counts["date"] = pd.to_datetime(rating_counts["date"]).dt.normalize()
        rating_counts = (
            rating_counts.sort_values(["ticker", "date", "historical_total_ratings"], ascending=[True, True, False])
            .drop_duplicates(subset=["ticker", "date"], keep="first")
            .reset_index(drop=True)
        )

    grade_events = pd.DataFrame(grade_event_records)
    if grade_events.empty:
        grade_events = pd.DataFrame(
            columns=[
                "date",
                "ticker",
                "grading_company",
                "previous_grade",
                "new_grade",
                "action",
                "raw_action",
                "analyst_name",
                "provider",
                "raw_json_path",
                "previous_grade_score",
                "new_grade_score",
                "grade_delta",
                "is_upgrade",
                "is_downgrade",
                "is_maintain",
                "is_positive_grade",
                "is_negative_grade",
            ]
        )
    else:
        grade_events["date"] = pd.to_datetime(grade_events["date"]).dt.normalize()
        grade_events = (
            grade_events.sort_values(["ticker", "date", "grading_company", "new_grade", "raw_action"])
            .drop_duplicates(
                subset=["ticker", "date", "grading_company", "previous_grade", "new_grade", "raw_action"],
                keep="first",
            )
            .reset_index(drop=True)
        )

    save_dataframe(rating_counts_output_path, rating_counts)
    save_dataframe(grade_events_output_path, grade_events)
    return rating_counts, grade_events
