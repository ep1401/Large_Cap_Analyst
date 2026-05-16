from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import Config
from src.fetch_fmp_historical_grades import build_historical_grade_datasets
from src.universe import get_tickers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", default=None)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--lookback-days", type=int, default=None)
    args = parser.parse_args()

    config = Config.from_env()
    tickers = (
        [ticker.strip().upper() for ticker in args.tickers.split(",") if ticker.strip()]
        if args.tickers
        else get_tickers(config.universe_path)
    )
    requested_start_date = args.start_date or config.start_date
    requested_end_date = args.end_date or config.end_date
    lookback_days = args.lookback_days if args.lookback_days is not None else config.historical_analyst_lookback_days
    fetch_start_date = (pd.Timestamp(requested_start_date) - pd.Timedelta(days=lookback_days)).strftime("%Y-%m-%d")

    rating_counts, grade_events = build_historical_grade_datasets(
        tickers=tickers,
        api_key=config.fmp_api_key,
        raw_output_dir=config.raw_dir / "analyst" / "fmp_historical_grades",
        rating_counts_output_path=config.processed_dir / "historical_analyst_rating_counts.csv",
        grade_events_output_path=config.processed_dir / "historical_analyst_grade_events.csv",
        start_date=fetch_start_date,
        end_date=requested_end_date,
        calls_per_minute=config.fmp_calls_per_minute,
        force=args.force or config.force_refresh,
        cache_enabled=not args.no_cache and config.cache_enabled,
        limit=args.limit,
    )
    print(f"Requested feature window: {requested_start_date} to {requested_end_date}")
    print(f"Historical analyst fetch window with lookback: {fetch_start_date} to {requested_end_date}")
    print(f"Saved historical analyst rating-count rows: {len(rating_counts)}")
    print(f"Saved historical analyst grade-event rows: {len(grade_events)}")


if __name__ == "__main__":
    main()
