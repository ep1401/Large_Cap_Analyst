from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import Config
from src.fetch_analyst_data import build_analyst_snapshot
from src.universe import get_tickers


def main() -> None:
    parser = argparse.ArgumentParser()
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
    df = build_analyst_snapshot(
        tickers=tickers,
        api_key=config.fmp_api_key,
        raw_output_dir=config.raw_dir / "analyst" / "fmp",
        processed_output_path=config.processed_dir / "analyst_features.csv",
        prices_path=config.processed_dir / "prices_all.csv",
        calls_per_minute=config.fmp_calls_per_minute,
        force=args.force or config.force_refresh,
        cache_enabled=config.cache_enabled and not args.no_cache,
    )
    print(
        "Warning: analyst_features.csv is stored as a current research snapshot unless your FMP plan provides"
        " point-in-time history. These snapshot fields are exploratory and should not be treated as historically valid analyst signals."
    )
    print(f"Saved analyst snapshot rows: {len(df)}")


if __name__ == "__main__":
    main()
