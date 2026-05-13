from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils import LOGGER, RateLimiter, safe_get_json, save_dataframe, save_json, standardize_ticker_for_fmp


FMP_BASE_URL = "https://financialmodelingprep.com/stable"


def fetch_price_target_consensus(
    ticker: str,
    api_key: str,
    rate_limiter: RateLimiter | None = None,
) -> dict:
    """Fetch FMP price target consensus snapshot."""
    if not api_key:
        raise ValueError("FMP_API_KEY is required to fetch analyst data.")
    url = f"{FMP_BASE_URL}/price-target-consensus"
    payload = safe_get_json(
        url,
        params={"symbol": standardize_ticker_for_fmp(ticker), "apikey": api_key},
        rate_limiter=rate_limiter,
    )
    return payload[0] if isinstance(payload, list) and payload else {}


def fetch_price_target_summary(
    ticker: str,
    api_key: str,
    rate_limiter: RateLimiter | None = None,
) -> dict:
    """Fetch FMP price target summary snapshot."""
    if not api_key:
        raise ValueError("FMP_API_KEY is required to fetch analyst data.")
    url = f"{FMP_BASE_URL}/price-target-summary"
    payload = safe_get_json(
        url,
        params={"symbol": standardize_ticker_for_fmp(ticker), "apikey": api_key},
        rate_limiter=rate_limiter,
    )
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


def build_analyst_snapshot(
    tickers: list[str],
    api_key: str,
    raw_output_dir: str | Path,
    processed_output_path: str | Path,
    prices_path: str | Path | None = None,
    calls_per_minute: int = 300,
) -> pd.DataFrame:
    """Build a research snapshot of analyst target data.

    This is intentionally stored as a snapshot because many plans do not expose
    point-in-time historical target data. The backtester can skip these features.
    """
    raw_output_dir = Path(raw_output_dir)
    snapshot_date = pd.Timestamp(datetime.utcnow().date())
    rows: list[dict] = []
    rate_limiter = RateLimiter(calls_per_minute=calls_per_minute)
    latest_close_by_ticker: dict[str, float] = {}

    if prices_path is not None and Path(prices_path).exists():
        prices = pd.read_csv(prices_path, parse_dates=["date"])
        latest_prices = (
            prices.sort_values(["ticker", "date"])
            .groupby("ticker")
            .tail(1)[["ticker", "adjusted_close"]]
        )
        latest_close_by_ticker = dict(zip(latest_prices["ticker"], latest_prices["adjusted_close"]))

    for ticker in tqdm(tickers, desc="Fetching FMP analyst data"):
        try:
            consensus = fetch_price_target_consensus(ticker, api_key, rate_limiter=rate_limiter)
            summary = fetch_price_target_summary(ticker, api_key, rate_limiter=rate_limiter)
            save_json(raw_output_dir / f"{ticker}_price_target_consensus.json", consensus)
            save_json(raw_output_dir / f"{ticker}_price_target_summary.json", summary)
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("Failed to fetch analyst data for %s: %s", ticker, exc)
            consensus, summary = {}, {}

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
        target_30d = _extract_float(
            summary,
            ["last30DaysAverage", "average30Days", "target30d", "lastMonthAvgPriceTarget"],
        )

        def upside(target: float | None) -> float | None:
            if target is None or close_proxy in (None, 0):
                return None
            return target / close_proxy - 1

        rows.append(
            {
                "date": snapshot_date,
                "ticker": ticker,
                "snapshot_mode": "research_current_snapshot",
                "consensus_target": consensus_target,
                "low_target": low_target,
                "high_target": high_target,
                "median_target": median_target,
                "analyst_count": analyst_count,
                "consensus_upside": upside(consensus_target),
                "low_target_upside": upside(low_target),
                "high_target_upside": upside(high_target),
                "target_spread": (
                    (high_target - low_target) / close_proxy
                    if None not in (high_target, low_target, close_proxy) and close_proxy != 0
                    else None
                ),
                "target_revision_7d": (
                    consensus_target - target_7d if None not in (consensus_target, target_7d) else None
                ),
                "target_revision_30d": (
                    consensus_target - target_30d if None not in (consensus_target, target_30d) else None
                ),
            }
        )

    df = pd.DataFrame(rows).sort_values("ticker").reset_index(drop=True)
    numeric_columns = [column for column in df.columns if column not in {"date", "ticker", "snapshot_mode"}]
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors="coerce")
    save_dataframe(processed_output_path, df)
    return df
