from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import Config
from src.fetch_prices import fetch_and_save_prices
from src.universe import get_tickers


def main() -> None:
    config = Config.from_env()
    tickers = get_tickers(config.universe_path)
    if config.benchmark not in tickers:
        tickers = tickers + [config.benchmark]
    fetch_and_save_prices(
        tickers=tickers,
        start_date=config.start_date,
        end_date=config.end_date,
        api_key=config.eodhd_api_key,
        raw_prices_dir=config.raw_dir / "prices",
        combined_output_path=config.processed_dir / "prices_all.csv",
        calls_per_minute=config.eodhd_calls_per_minute,
    )


if __name__ == "__main__":
    main()
