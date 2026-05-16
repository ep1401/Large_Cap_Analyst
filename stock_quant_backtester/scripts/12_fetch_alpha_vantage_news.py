from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import Config
from src.fetch_alpha_vantage_news import (
    build_monthly_windows,
    fetch_alpha_vantage_news_cache,
    normalize_alpha_vantage_news_cache,
    summarize_request_plan,
)
from src.universe import get_tickers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--lookback-years", type=int, default=None)
    parser.add_argument("--tickers", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--requests-per-minute", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=1000)
    args = parser.parse_args()

    config = Config.from_env()
    if not args.dry_run and not config.alpha_vantage_api_key:
        raise SystemExit("ALPHA_VANTAGE_API_KEY is missing. Add it to .env before fetching Alpha Vantage news.")
    if args.start_date and args.end_date:
        start_date = args.start_date
        end_date = args.end_date
    elif args.lookback_years is not None:
        end_ts = config.sentiment_end_ts
        start_ts = end_ts - pd.DateOffset(years=args.lookback_years)
        start_date = start_ts.strftime("%Y-%m-%d")
        end_date = end_ts.strftime("%Y-%m-%d")
    else:
        start_date = config.sentiment_start_date
        end_date = config.sentiment_end_date

    tickers = (
        [ticker.strip().upper() for ticker in args.tickers.split(",") if ticker.strip()]
        if args.tickers
        else get_tickers(config.universe_path)
    )
    windows = build_monthly_windows(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        raw_news_dir=config.raw_dir / "news" / "alpha_vantage",
    )
    requests_per_minute = args.requests_per_minute or config.alpha_vantage_requests_per_minute
    cache_enabled = config.cache_enabled and not args.no_cache
    force = args.force or config.force_refresh
    plan = summarize_request_plan(
        windows,
        force=force,
        cache_enabled=cache_enabled,
        requests_per_minute=requests_per_minute,
    )

    print(f"Requested sentiment start date: {start_date}")
    print(f"Requested sentiment end date: {end_date}")
    print(f"Number of months: {plan['months']}")
    print(f"Number of tickers: {plan['tickers']}")
    print(f"Expected ticker-month files: {plan['total_possible_requests']}")
    print(f"Cached ticker-month files: {plan['cached_requests']}")
    print(f"Missing ticker-month files: {plan['missing_requests']}")
    print(f"Estimated API calls: {plan['missing_requests']}")
    print(f"Estimated runtime (minutes): {plan['estimated_runtime_minutes']:.2f}")

    if args.dry_run:
        return

    if plan["missing_requests"] == 0 and not force:
        print("All Alpha Vantage ticker-month files are cached. No API calls required.")
    else:
        fetch_alpha_vantage_news_cache(
            windows,
            api_key=config.alpha_vantage_api_key,
            cache_enabled=cache_enabled,
            force=force,
            limit=args.limit,
            requests_per_minute=requests_per_minute,
        )

    df = normalize_alpha_vantage_news_cache(
        windows,
        processed_output_path=config.processed_dir / "stock_news_alpha_vantage.csv",
        combined_output_path=config.processed_dir / "stock_news.csv",
    )
    print(f"Saved normalized Alpha Vantage news rows: {len(df)}")


if __name__ == "__main__":
    main()
