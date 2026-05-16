from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.build_features import build_feature_panel
from src.config import Config
from src.utils import str_to_bool


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-current-snapshot-analyst", default="true")
    args = parser.parse_args()

    config = Config.from_env()
    print(config.describe_analysis_windows())
    prices_path = config.processed_dir / "prices_all.csv"
    if not prices_path.exists():
        raise SystemExit("Missing data/processed/prices_all.csv. Run scripts/01_fetch_prices.py before building features.")
    features = build_feature_panel(
        prices_path=prices_path,
        universe_path=config.universe_path,
        analyst_path=config.processed_dir / "analyst_features.csv",
        sentiment_path=config.processed_dir / "news_sentiment_daily.csv",
        historical_rating_counts_path=config.processed_dir / "historical_analyst_rating_counts.csv",
        historical_grade_events_path=config.processed_dir / "historical_analyst_grade_events.csv",
        historical_rating_count_features_output_path=config.processed_dir / "historical_rating_count_features.csv",
        historical_grade_features_output_path=config.processed_dir / "historical_grade_features.csv",
        output_path=config.final_dir / "features_panel.csv",
        benchmark_ticker=config.benchmark,
        use_current_snapshot_analyst=str_to_bool(args.use_current_snapshot_analyst),
    )
    sentiment_panel = features.loc[
        (features["date"] >= config.sentiment_start_ts) & (features["date"] < config.sentiment_end_ts)
    ].copy()
    sentiment_panel.to_csv(config.final_dir / "features_panel_sentiment_1y.csv", index=False)
    print(f"Saved features rows: {len(features)}")
    print(f"Saved 1-year sentiment feature rows: {len(sentiment_panel)}")


if __name__ == "__main__":
    main()
