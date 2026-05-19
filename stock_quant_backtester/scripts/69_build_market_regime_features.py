from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.build_market_regime_features import build_market_regime_features
from src.config import Config


def main() -> None:
    config = Config.from_env()
    prices_path = config.processed_dir / "prices_all.csv"
    market_sentiment_path = config.processed_dir / "market_sentiment_daily.csv"
    if not prices_path.exists():
        raise SystemExit("Missing data/processed/prices_all.csv. Build prices before market regime features.")
    if not market_sentiment_path.exists():
        raise SystemExit("Missing data/processed/market_sentiment_daily.csv. Run scripts/68_build_market_sentiment_features.py first.")
    output_path = config.processed_dir / "market_regime_daily.csv"
    market_regime = build_market_regime_features(
        prices_path=prices_path,
        market_sentiment_path=market_sentiment_path,
        output_path=output_path,
        benchmark_ticker=config.benchmark,
        secondary_ticker="QQQ",
    )
    print(f"Saved {output_path}")
    print(f"Rows: {len(market_regime)}")
    if not market_regime.empty:
        print(f"Date range: {market_regime['date'].min().date()} to {market_regime['date'].max().date()}")


if __name__ == "__main__":
    main()
