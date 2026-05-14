from __future__ import annotations

from pathlib import Path
import json
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import Config
from src.fetch_alpha_vantage_news import build_monthly_windows, summarize_request_plan
from src.universe import get_tickers


def _print_processed_status(config: Config) -> None:
    print("- Processed files:")
    for name in [
        "prices_all.csv",
        "analyst_features.csv",
        "stock_news.csv",
        "news_sentiment_daily.csv",
    ]:
        path = config.processed_dir / name
        print(f"  - {name}: {'exists' if path.exists() else 'missing'}")
    print(f"  - features_panel.csv: {'exists' if (config.final_dir / 'features_panel.csv').exists() else 'missing'}")


def main() -> None:
    config = Config.from_env()
    tickers = get_tickers(config.universe_path)
    benchmark_and_universe = sorted(set(tickers + [config.benchmark]))

    print("Cache Summary")
    print("")

    price_dir = config.raw_dir / "prices" / "eodhd"
    price_files = list(price_dir.glob("*.csv"))
    legacy_price_files = list((config.raw_dir / "prices").glob("*.csv"))
    price_tickers = sorted(
        {path.name.split("_")[0] for path in price_files}
        | {path.stem for path in legacy_price_files}
    )
    missing_price_tickers = sorted(set(benchmark_and_universe) - set(price_tickers))
    date_range = "n/a"
    if (config.processed_dir / "prices_all.csv").exists():
        prices_df = pd.read_csv(config.processed_dir / "prices_all.csv", parse_dates=["date"])
        if not prices_df.empty:
            date_range = f"{prices_df['date'].min().date()} to {prices_df['date'].max().date()}"
    print("- EODHD price cache:")
    print(f"  - number of tickers cached: {len(price_tickers)}")
    print(f"  - date range available: {date_range}")
    print(f"  - missing tickers: {', '.join(missing_price_tickers) if missing_price_tickers else 'none'}")

    analyst_dir = config.raw_dir / "analyst" / "fmp"
    analyst_consensus = list(analyst_dir.glob("*_price_target_consensus.json"))
    legacy_analyst_consensus = list((config.raw_dir / "analyst").glob("*_price_target_consensus.json"))
    analyst_tickers = sorted(
        {path.name.split("_price_target_consensus.json")[0] for path in analyst_consensus}
        | {path.name.split("_price_target_consensus.json")[0] for path in legacy_analyst_consensus}
    )
    missing_analyst_tickers = sorted(set(tickers) - set(analyst_tickers))
    blocked_count = 0
    error_count = 0
    for path in list(analyst_dir.glob("*.json")) + list((config.raw_dir / "analyst").glob("*.json")):
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict) and payload.get("error"):
            error_count += 1
            if payload.get("status_code") in {402, 403}:
                blocked_count += 1
    print("- FMP analyst cache:")
    print(f"  - number of tickers cached: {len(analyst_tickers)}")
    print(f"  - missing tickers: {', '.join(missing_analyst_tickers) if missing_analyst_tickers else 'none'}")
    print(f"  - blocked endpoint/error counts: blocked={blocked_count}, total_errors={error_count}")

    windows = build_monthly_windows(
        tickers=tickers,
        start_date=config.sentiment_start_date,
        end_date=config.sentiment_end_date,
        raw_news_dir=config.raw_dir / "news" / "alpha_vantage",
    )
    av_plan = summarize_request_plan(
        windows,
        force=False,
        cache_enabled=config.cache_enabled,
        requests_per_minute=config.alpha_vantage_requests_per_minute,
    )
    print("- Alpha Vantage news cache:")
    print(f"  - sentiment window: {config.sentiment_start_date} to {config.sentiment_end_date}")
    print(f"  - expected ticker-month files: {av_plan['total_possible_requests']}")
    print(f"  - cached ticker-month files: {av_plan['cached_requests']}")
    print(f"  - missing ticker-month files: {av_plan['missing_requests']}")
    print(f"  - estimated API calls needed: {av_plan['missing_requests']}")
    print(f"  - estimated runtime: {av_plan['estimated_runtime_minutes']:.2f} minutes")

    _print_processed_status(config)


if __name__ == "__main__":
    main()
