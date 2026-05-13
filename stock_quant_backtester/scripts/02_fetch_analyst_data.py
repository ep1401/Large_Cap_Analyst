from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import Config
from src.fetch_analyst_data import build_analyst_snapshot
from src.universe import get_tickers


def main() -> None:
    config = Config.from_env()
    tickers = get_tickers(config.universe_path)
    df = build_analyst_snapshot(
        tickers=tickers,
        api_key=config.fmp_api_key,
        raw_output_dir=config.raw_dir / "analyst",
        processed_output_path=config.processed_dir / "analyst_features.csv",
        prices_path=config.processed_dir / "prices_all.csv",
        calls_per_minute=config.fmp_calls_per_minute,
    )
    print(
        "Warning: analyst_features.csv is stored as a current research snapshot unless your FMP plan provides"
        " point-in-time history. Use historical_backtest_without_analyst for valid historical backtests."
    )
    print(f"Saved analyst snapshot rows: {len(df)}")


if __name__ == "__main__":
    main()
