from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
import logging
import time
from typing import Any

import pandas as pd
import requests


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOGGER = logging.getLogger(__name__)

SENSITIVE_PARAM_KEYS = {"api_key", "api_token", "apikey", "token"}


@dataclass
class RateLimiter:
    """Simple sequential rate limiter based on calls per minute."""

    calls_per_minute: int
    _last_call_time: float = field(default=0.0, init=False)

    def wait(self) -> None:
        if self.calls_per_minute <= 0:
            return
        min_interval = 60.0 / self.calls_per_minute
        now = time.monotonic()
        elapsed = now - self._last_call_time
        if self._last_call_time and elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self._last_call_time = time.monotonic()


def safe_get_json(
    url: str,
    params: dict[str, Any] | None = None,
    timeout: int = 30,
    max_retries: int = 3,
    retry_sleep_seconds: float = 1.0,
    rate_limiter: RateLimiter | None = None,
) -> Any:
    """Request JSON data with simple retry handling and rate-limit pacing."""
    def redact_params(raw_params: dict[str, Any] | None) -> dict[str, Any] | None:
        if raw_params is None:
            return None
        redacted: dict[str, Any] = {}
        for key, value in raw_params.items():
            key_text = str(key).lower()
            if key_text in SENSITIVE_PARAM_KEYS or key_text.endswith("_key"):
                redacted[key] = "***REDACTED***"
            else:
                redacted[key] = value
        return redacted

    def redact_text(text: str, raw_params: dict[str, Any] | None) -> str:
        redacted = text
        if raw_params is None:
            return redacted
        for key, value in raw_params.items():
            key_text = str(key).lower()
            if key_text in SENSITIVE_PARAM_KEYS or key_text.endswith("_key"):
                value_text = str(value).strip()
                if value_text:
                    redacted = redacted.replace(value_text, "***REDACTED***")
        return redacted

    last_error: Exception | None = None
    redacted_params = redact_params(params)
    for attempt in range(1, max_retries + 1):
        try:
            if rate_limiter is not None:
                rate_limiter.wait()
            response = requests.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            LOGGER.warning(
                "Request failed on attempt %s/%s for %s with params=%s: %s",
                attempt,
                max_retries,
                url,
                redacted_params,
                redact_text(str(exc), params),
            )
            time.sleep(retry_sleep_seconds * attempt)
    detail = f": {redact_text(str(last_error), params)}" if last_error is not None else ""
    raise RuntimeError(f"Failed to fetch JSON from {url}{detail}") from last_error


def save_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_dataframe(path: str | Path, df: pd.DataFrame, index: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)


def load_dataframe(path: str | Path, parse_dates: list[str] | None = None) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=parse_dates)


def standardize_ticker_for_eodhd(ticker: str) -> str:
    if ticker == "BRK-B":
        return "BRK-B.US"
    return f"{ticker}.US"


def standardize_ticker_for_fmp(ticker: str) -> str:
    if ticker == "BRK-B":
        return "BRK-B"
    return ticker


def str_to_bool(value: str | bool | None, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "t", "yes", "y"}
