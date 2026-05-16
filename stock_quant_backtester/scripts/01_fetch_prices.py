from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import Config
from src.fetch_prices import fetch_and_save_prices
from src.universe import get_tickers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--tickers", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    config = Config.from_env()
    print(config.describe_analysis_windows())
    tickers = (
        [ticker.strip().upper() for ticker in args.tickers.split(",") if ticker.strip()]
        if args.tickers
        else get_tickers(config.universe_path)
    )
    if config.benchmark not in tickers:
        tickers = tickers + [config.benchmark]
    fetch_and_save_prices(
        tickers=tickers,
        start_date=args.start_date or config.start_date,
        end_date=args.end_date or config.end_date,
        api_key=config.eodhd_api_key,
        raw_prices_dir=config.raw_dir / "prices" / "eodhd",
        combined_output_path=config.processed_dir / "prices_all.csv",
        calls_per_minute=config.eodhd_calls_per_minute,
        force=args.force or config.force_refresh,
        cache_enabled=config.cache_enabled and not args.no_cache,
    )


if __name__ == "__main__":
    main()
