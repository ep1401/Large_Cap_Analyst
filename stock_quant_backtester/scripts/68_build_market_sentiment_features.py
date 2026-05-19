from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.build_market_sentiment_features import build_market_sentiment_features
from src.config import Config


def main() -> None:
    config = Config.from_env()
    sentiment_path = config.processed_dir / "news_sentiment_daily.csv"
    if not sentiment_path.exists():
        raise SystemExit("Missing data/processed/news_sentiment_daily.csv. Build ticker news sentiment before market sentiment features.")
    output_path = config.processed_dir / "market_sentiment_daily.csv"
    market_sentiment = build_market_sentiment_features(
        sentiment_daily_path=sentiment_path,
        universe_path=config.universe_path,
        output_path=output_path,
    )
    print(f"Saved {output_path}")
    print(f"Rows: {len(market_sentiment)}")
    if not market_sentiment.empty:
        print(f"Date range: {market_sentiment['date'].min().date()} to {market_sentiment['date'].max().date()}")


if __name__ == "__main__":
    main()
