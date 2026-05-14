from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import Config
from src.fetch_fmp_news import build_fmp_news_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = Config.from_env()
    if not config.fmp_api_key:
        raise SystemExit("FMP_API_KEY is missing. Add it to .env before running scripts/12_fetch_fmp_news.py.")
    universe = pd.read_csv(config.universe_path)
    tickers = sorted(universe["ticker"].dropna().astype(str).unique().tolist())

    news_df = build_fmp_news_dataset(
        tickers=tickers,
        api_key=config.fmp_api_key,
        start_date=config.start_date,
        end_date=config.end_date,
        raw_output_dir=config.raw_dir / "news",
        processed_output_path=config.processed_dir / "stock_news.csv",
        calls_per_minute=config.fmp_calls_per_minute,
        force=args.force,
    )
    print(f"Saved processed stock news rows: {len(news_df)}")


if __name__ == "__main__":
    main()
