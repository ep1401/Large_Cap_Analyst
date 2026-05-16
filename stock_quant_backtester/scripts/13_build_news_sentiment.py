from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.build_news_sentiment import build_news_sentiment_outputs
from src.config import Config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--rescore-with-finbert", action="store_true")
    parser.add_argument("--prefer-finbert", action="store_true")
    args = parser.parse_args()

    config = Config.from_env()
    news_input_path = config.processed_dir / "stock_news.csv"
    if not news_input_path.exists():
        raise SystemExit(
            "Missing data/processed/stock_news.csv. Run scripts/12_fetch_alpha_vantage_news.py before scripts/13_build_news_sentiment.py."
        )
    start_date = args.start_date or config.sentiment_start_date
    end_date = args.end_date or config.sentiment_end_date
    window_label = f"{start_date}_{end_date}"
    articles_df, daily_df = build_news_sentiment_outputs(
        news_input_path=news_input_path,
        articles_output_path=config.processed_dir / "news_sentiment_articles.csv",
        daily_output_path=config.processed_dir / "news_sentiment_daily.csv",
        start_date=start_date,
        end_date=end_date,
        force=args.force,
        rescore_with_finbert=args.rescore_with_finbert,
        prefer_finbert=args.prefer_finbert,
    )
    articles_window_path = config.processed_dir / f"news_sentiment_articles_{window_label}.csv"
    daily_window_path = config.processed_dir / f"news_sentiment_daily_{window_label}.csv"
    articles_df.to_csv(articles_window_path, index=False)
    daily_df.to_csv(daily_window_path, index=False)
    print(f"Sentiment window: {start_date} to {end_date}")
    print(f"Saved article-level sentiment rows: {len(articles_df)}")
    print(f"Saved daily sentiment rows: {len(daily_df)}")


if __name__ == "__main__":
    main()
